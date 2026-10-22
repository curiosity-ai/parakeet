using System.Diagnostics;
using Parakeet.Audio;
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
        Vocabulary = vocabulary;
        Parallelism = new Parallelism(options.Threads);
        var w = new WeightReader(source, options.HugePages);
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
    public Vocabulary Vocabulary { get; }
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

    /// <summary>Loads a local model, or downloads a Hugging Face repository's <c>.nemo</c> checkpoint first.</summary>
    public static async Task<ParakeetAsr> LoadAsync(string modelOrRepo = ModelDownloader.DefaultRepo, ParakeetOptions? options = null,
                                                    IProgress<string>? progress = null, CancellationToken ct = default)
    {
        string path = await ModelDownloader.EnsureAsync(modelOrRepo, progress: progress, ct: ct).ConfigureAwait(false);
        return Load(path, options);
    }

    /// <summary>Transcribes a WAV file (any sample rate / channel count).</summary>
    public Transcription TranscribeFile(string path, CancellationToken ct = default) => Transcribe(WaveFile.ReadMono(path, SampleRate), ct);

    /// <summary>Transcribes 16 kHz mono audio. <paramref name="onText"/> receives text deltas as tokens are decoded.</summary>
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

        sw.Restart();
        float[] enc = Encoder.Forward(features, melFrames, Parallelism);
        int frames = Config.EncoderFrames(melFrames);
        var tEnc = sw.Elapsed;
        ct.ThrowIfCancellationRequested();

        sw.Restart();
        Action<DecodedToken>? stream = onText == null ? null : tok =>
        {
            string s = Vocabulary.PieceText(tok.Id);
            if (s.Length > 0) onText(s);
        };
        var decoded = Decoder.Decode(enc, frames, Parallelism, ct, stream);
        var tDec = sw.Elapsed;
        return BuildResult(decoded, seconds, tFeat, tEnc, tDec);
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
