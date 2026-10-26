using System.Diagnostics;
using Parakeet.Audio;
using Parakeet.Chunking;
using Parakeet.IO;
using Parakeet.Model;
using Parakeet.Numerics;
using Parakeet.Text;

namespace Parakeet;

/// <summary>Options for loading a model.</summary>
public sealed record ParakeetOptions
{
    /// <summary>Worker threads (0 = all cores, 1 = single core).</summary>
    public int Threads { get; init; }

    /// <summary>STFT padding. <see cref="StftPadMode.Reflect"/> matches the ONNX export; NeMo itself zero-pads.</summary>
    public StftPadMode PadMode { get; init; } = StftPadMode.Reflect;

    /// <summary>Back the 2.4 GB of packed weights with transparent huge pages (Linux). Off by default: the kernel may
    /// have to compact memory to provide them, which can add seconds to loading on a machine with a busy page cache.</summary>
    public bool HugePages { get; init; }

    /// <summary>Longest input (in 80 ms encoder frames) whose relative positional projections are cached across calls.
    /// They depend only on the input length, so caching them removes ~7% of the encoder's work for every input up to
    /// this length, at <c>~200 KB × frames</c> of memory (default 512 frames ≈ 41 s of audio, ≈ 100 MB at most).
    /// 0 disables the cache.</summary>
    public int PositionCacheFrames { get; init; } = 512;

    /// <summary>Precision of the encoder's large projections. <see cref="WeightPrecision.Int8"/> (opt-in) quantizes them
    /// for AVX-VNNI: roughly twice the GEMM throughput and a quarter of the weight memory, with a small accuracy cost.</summary>
    public WeightPrecision Precision { get; init; } = WeightPrecision.Float32;

    /// <summary>Longest stretch of audio (seconds) the encoder runs over in one pass. Longer input is split into
    /// overlapping windows of this size whose seams are placed on the quietest point of the overlap and de-duplicated, so
    /// any length can be transcribed in bounded time and memory per window. Input that fits one window takes the
    /// unchunked path and is unaffected. 0 turns chunking off (the whole input is encoded in one pass).</summary>
    public double ChunkSeconds { get; init; } = 300;

    /// <summary>Audio (seconds) shared by neighbouring windows. It gives the encoder context on both sides of a seam and
    /// the decoder time to warm up before the window's own region starts. Must be smaller than <see cref="ChunkSeconds"/>.</summary>
    public double ChunkOverlapSeconds { get; init; } = 15;
}

/// <summary>A decoded token with its time span.</summary>
public sealed record TimedToken(int Id, string Text, int Frame, int Duration, double Start, double End);

/// <summary>A word (tokens from one ▁-prefixed piece up to the next) with its time span.</summary>
public sealed record TimedWord(string Text, double Start, double End);

/// <summary>Result of a transcription, with per-stage timings.</summary>
public sealed record Transcription(string Text, IReadOnlyList<TimedToken> Tokens, IReadOnlyList<TimedWord> Words, double AudioSeconds,
                                   TimeSpan FeatureTime, TimeSpan EncoderTime, TimeSpan DecoderTime)
{
    public TimeSpan TotalTime => FeatureTime + EncoderTime + DecoderTime;
    public double RealTimeFactor => AudioSeconds > 0 ? TotalTime.TotalSeconds / AudioSeconds : double.NaN;
}

/// <summary>
/// NVIDIA Parakeet TDT speech recognition in pure C#: log-mel front end, FastConformer encoder and greedy TDT decoding.
/// Loads the official <c>.nemo</c> checkpoint (e.g. <c>nvidia/parakeet-tdt-0.6b-v2</c>, <c>-v3</c>) or the ONNX export
/// directory (<c>encoder-model.onnx</c>, <c>decoder_joint-model.onnx</c>, <c>vocab.txt</c>). Thread-safe: one instance
/// can serve concurrent <see cref="Transcribe(ReadOnlySpan{float}, CancellationToken, Action{string}?)"/> calls.
/// </summary>
public sealed class ParakeetAsr : IDisposable
{
    /// <summary>Sample rate the model expects.</summary>
    public const int SampleRate = 16000;

    private ParakeetAsr(ITensorSource source, ParakeetConfig config, Vocabulary vocabulary, ParakeetOptions options)
    {
        Config = config with { Mel = config.Mel with { PadMode = options.PadMode } };
        Options = options;
        if (options.ChunkSeconds > 0)
            ChunkPlanner.Validate(SecondsToMelFrames(options.ChunkSeconds), SecondsToMelFrames(options.ChunkOverlapSeconds), Config.SubsamplingFactor);
        Vocabulary = vocabulary;
        Parallelism = new Parallelism(options.Threads);
        var w = new WeightReader(source, options.HugePages, options.Precision);
        Precision = options.Precision == WeightPrecision.Int8 && Int8Gemm.IsSupported ? WeightPrecision.Int8 : WeightPrecision.Float32;
        float[]? fb = null, window = null;
        if (source.Contains("preprocessor.featurizer.fb")) fb = source.Get("preprocessor.featurizer.fb").ToArray();
        if (source.Contains("preprocessor.featurizer.window")) window = source.Get("preprocessor.featurizer.window").ToArray();
        Frontend = new MelSpectrogram(Config.Mel, fb, window);
        Encoder = new ConformerEncoder(w, Config, options.PositionCacheFrames);
        try
        {
            Decoder = new TdtDecoder(w, Config);
        }
        catch
        {
            Encoder.Dispose();
            throw;
        }
    }

    public ParakeetConfig Config { get; }
    public ParakeetOptions Options { get; }
    public Vocabulary Vocabulary { get; }

    /// <summary>Effective precision of the encoder projections (Int8 falls back to Float32 without AVX-VNNI).</summary>
    public WeightPrecision Precision { get; }
    public Parallelism Parallelism { get; }
    public MelSpectrogram Frontend { get; }
    public ConformerEncoder Encoder { get; }
    public TdtDecoder Decoder { get; }

    /// <summary>Loads a model from a <c>.nemo</c> file, an extracted NeMo directory, a directory holding a <c>.nemo</c>
    /// file, or an ONNX export directory.</summary>
    public static ParakeetAsr Load(string path, ParakeetOptions? options = null)
    {
        options ??= new ParakeetOptions();
        if (Directory.Exists(path) && !File.Exists(Path.Combine(path, "model_config.yaml")))
        {
            if (File.Exists(Path.Combine(path, "encoder-model.onnx")))
            {
                using var onnx = OnnxModelSource.Open(path);
                return new ParakeetAsr(onnx, onnx.Config, new Vocabulary(onnx.Vocabulary), options);
            }
            var nemo = Directory.GetFiles(path, "*.nemo");
            if (nemo.Length == 0) throw new FileNotFoundException($"No .nemo checkpoint or ONNX export found in {path}.");
            path = nemo[0];
        }
        using var archive = NemoArchive.Open(path);
        return new ParakeetAsr(archive, archive.Config, new Vocabulary(archive.Vocabulary), options);
    }

    /// <summary>
    /// Downloads a published checkpoint (once: it is cached at <paramref name="downloadToPath"/>, by default
    /// <see cref="ParakeetModels.GetDefaultCachePath"/>) and loads it.
    /// </summary>
    /// <param name="model">Which checkpoint: English v2 (default) or multilingual v3.</param>
    /// <param name="options">Load options (threads, precision, ...).</param>
    /// <param name="modelUrl">Override the download URL (defaults to <see cref="ParakeetModels.GetDownloadUrl"/>). The size of
    /// the published file is only enforced for the default URL.</param>
    /// <param name="downloadToPath">Where to cache the <c>.nemo</c> file.</param>
    /// <param name="reportProgress">Optional download progress callback (~2 Hz).</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    public static async Task<ParakeetAsr> LoadAsync(ParakeetModel model = ParakeetModel.TdtV2, ParakeetOptions? options = null,
                                                    string? modelUrl = null, string? downloadToPath = null,
                                                    Action<DownloadProgress>? reportProgress = null,
                                                    CancellationToken cancellationToken = default)
    {
        string path = downloadToPath ?? ParakeetModels.GetDefaultCachePath(model);
        string url = modelUrl ?? ParakeetModels.GetDownloadUrl(model);
        long? size = modelUrl is null ? ParakeetModels.GetFileSize(model) : null;
        await ModelDownloader.DownloadFileAsync(url, path, reportProgress, size, cancellationToken).ConfigureAwait(false);
        try
        {
            return Load(path, options);
        }
        catch (InvalidDataException)
        {
            // A corrupt cached checkpoint (e.g. truncated by an older version): fetch it again once.
            try { File.Delete(path); } catch { /* ignore */ }
            await ModelDownloader.DownloadFileAsync(url, path, reportProgress, size, cancellationToken).ConfigureAwait(false);
            return Load(path, options);
        }
    }

    /// <summary>Loads a local model (see <see cref="Load"/>), or downloads and loads a published one named by
    /// <paramref name="modelOrPath"/> (<c>v2</c>, <c>v3</c>, <c>nvidia/parakeet-tdt-0.6b-v3</c>, ... see
    /// <see cref="ParakeetModels.TryParse"/>).</summary>
    public static Task<ParakeetAsr> LoadAsync(string modelOrPath, ParakeetOptions? options = null,
                                              Action<DownloadProgress>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (File.Exists(modelOrPath) || Directory.Exists(modelOrPath)) return Task.FromResult(Load(modelOrPath, options));
        if (ParakeetModels.TryParse(modelOrPath, out var model))
            return LoadAsync(model, options, reportProgress: reportProgress, cancellationToken: cancellationToken);
        throw new ArgumentException($"'{modelOrPath}' is neither a local model path nor a known model " +
                                    $"({string.Join(", ", Enum.GetValues<ParakeetModel>().Select(ParakeetModels.GetRepository))}).",
                                    nameof(modelOrPath));
    }

    /// <summary>Downloads a published checkpoint without loading it (e.g. to pre-populate a cache) and returns its path.</summary>
    public static async Task<string> DownloadModelAsync(ParakeetModel model, string? downloadToPath = null,
                                                        Action<DownloadProgress>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        string path = downloadToPath ?? ParakeetModels.GetDefaultCachePath(model);
        await ModelDownloader.DownloadFileAsync(ParakeetModels.GetDownloadUrl(model), path, reportProgress,
                                                ParakeetModels.GetFileSize(model), cancellationToken).ConfigureAwait(false);
        return path;
    }

    /// <summary>Transcribes a WAV file (any sample rate / channel count).</summary>
    public Transcription TranscribeFile(string path, CancellationToken ct = default) => Transcribe(WaveFile.ReadMono(path, SampleRate), ct);

    /// <summary>
    /// Transcribes 16 kHz mono audio of any length. <paramref name="onText"/> receives text deltas as tokens are decoded
    /// (in order, also across chunk seams). Audio longer than <see cref="ParakeetOptions.ChunkSeconds"/> is transcribed
    /// in overlapping windows; shorter audio is encoded in one pass.
    /// </summary>
    public Transcription Transcribe(ReadOnlySpan<float> audio, CancellationToken ct = default, Action<string>? onText = null)
    {
        double seconds = audio.Length / (double)SampleRate;
        if (audio.Length < Config.Mel.NFft)
            return new Transcription("", [], [], seconds, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero);
        var sw = Stopwatch.StartNew();
        float[] features = Frontend.Compute(audio, Parallelism);
        int melFrames = Frontend.FrameCount(audio.Length);
        var tFeat = sw.Elapsed;
        ct.ThrowIfCancellationRequested();

        int chunkFrames = Options.ChunkSeconds > 0 ? SecondsToMelFrames(Options.ChunkSeconds) : int.MaxValue;
        if (melFrames > chunkFrames)
            return TranscribeChunked(features, melFrames, seconds, tFeat, ct, onText);

        sw.Restart();
        float[] enc = Encoder.Forward(features, melFrames, Parallelism);
        int frames = Config.EncoderFrames(melFrames);
        var tEnc = sw.Elapsed;
        ct.ThrowIfCancellationRequested();

        sw.Restart();
        Action<DecodedToken>? stream = onText == null ? null : tok => EmitText(tok, onText);
        var decoded = Decoder.Decode(enc, frames, Parallelism, ct, stream);
        var tDec = sw.Elapsed;
        return BuildResult(decoded, seconds, tFeat, tEnc, tDec);
    }

    private int SecondsToMelFrames(double seconds) => (int)Math.Round(seconds * Config.Mel.SampleRate / Config.Mel.HopLength);

    private void EmitText(DecodedToken tok, Action<string> onText)
    {
        string s = Vocabulary.PieceText(tok.Id);
        if (s.Length > 0) onText(s);
    }

    /// <summary>
    /// Long audio: features are computed once for the whole input (the per-feature normalization stays global), then each
    /// window is encoded and decoded on its own. A window decodes everything it covers, so its decoder state stays
    /// coherent, but keeps only the tokens in its own region of the overlap. The first few kept tokens of every window
    /// after the first wait for <see cref="SeamDeduper"/>; everything else streams as it is decoded.
    /// </summary>
    private Transcription TranscribeChunked(float[] features, int melFrames, double seconds, TimeSpan tFeat, CancellationToken ct, Action<string>? onText)
    {
        int bins = Config.Mel.Features, sub = Config.SubsamplingFactor;
        var oracle = new ChainBoundaryOracle(new MelEnergyBoundaryOracle(features, melFrames, bins), new MidpointBoundaryOracle());
        var plan = ChunkPlanner.Plan(melFrames, SecondsToMelFrames(Options.ChunkSeconds), SecondsToMelFrames(Options.ChunkOverlapSeconds), sub, oracle);

        var all = new List<DecodedToken>();
        TimeSpan tEnc = TimeSpan.Zero, tDec = TimeSpan.Zero;
        for (int i = 0; i < plan.Count; i++)
        {
            var win = plan[i];
            int windowMel = win.End - win.Start;
            // Window starts are multiples of the subsampling factor, so these are exact.
            int frameOffset = win.Start / sub;
            int emitStart = (win.EmitStart - win.Start) / sub;
            int emitEnd = i == plan.Count - 1 ? int.MaxValue : (win.EmitEnd - win.Start) / sub;

            var sw = Stopwatch.StartNew();
            float[] enc = Encoder.Forward(features.AsSpan((int)((long)win.Start * bins), windowMel * bins), windowMel, Parallelism);
            int frames = Config.EncoderFrames(windowMel);
            tEnc += sw.Elapsed;
            ct.ThrowIfCancellationRequested();

            sw.Restart();
            var previous = all.Count > 0 ? all.GetRange(Math.Max(0, all.Count - SeamDeduper.HeadTokens), Math.Min(all.Count, SeamDeduper.HeadTokens)) : [];
            var head = new List<DecodedToken>();
            bool resolved = i == 0;

            void Keep(DecodedToken tok)
            {
                all.Add(tok);
                if (onText != null) EmitText(tok, onText);
            }
            void FlushHead()
            {
                foreach (var tok in SeamDeduper.Resolve(previous, head)) Keep(tok);
                head.Clear();
                resolved = true;
            }

            Decoder.Decode(enc, frames, Parallelism, ct, tok =>
            {
                if (tok.Frame < emitStart || tok.Frame >= emitEnd) return;
                var absolute = tok with { Frame = tok.Frame + frameOffset };
                if (resolved) { Keep(absolute); return; }
                head.Add(absolute);
                if (head.Count >= SeamDeduper.HeadTokens) FlushHead();
            });
            if (!resolved) FlushHead();
            tDec += sw.Elapsed;
        }
        return BuildResult(all, seconds, tFeat, tEnc, tDec);
    }

    private Transcription BuildResult(List<DecodedToken> decoded, double seconds, TimeSpan tFeat, TimeSpan tEnc, TimeSpan tDec)
    {
        double fs = Config.FrameSeconds;
        var tokens = decoded.Select(d => new TimedToken(d.Id, Vocabulary.PieceText(d.Id), d.Frame, d.Duration,
                                                        d.Frame * fs, (d.Frame + Math.Max(1, d.Duration)) * fs)).ToList();
        var words = new List<TimedWord>();
        string cur = "";
        double start = 0, end = 0;
        foreach (var t in tokens)
        {
            if (t.Text.Length == 0) continue;
            if (t.Text[0] == ' ' && cur.Length > 0)
            {
                words.Add(new TimedWord(cur, start, end));
                cur = "";
            }
            if (cur.Length == 0) start = t.Start;
            cur += t.Text.Trim();
            end = t.End;
        }
        if (cur.Length > 0) words.Add(new TimedWord(cur, start, end));
        return new Transcription(Vocabulary.Decode(decoded.Select(d => d.Id)), tokens, words, seconds, tFeat, tEnc, tDec);
    }

    public void Dispose()
    {
        Encoder.Dispose();
        Decoder.Dispose();
        Parallelism.Dispose();
    }
}
