using System.Diagnostics;
using System.Text.Json;
using Parakeet;
using Parakeet.Audio;
using Parakeet.IO;

// parakeet transcribe [--model v2|v3|path] [--threads n] [--json] [--words] file.wav ...
// parakeet bench      [--model v2|v3|path] [--threads n] [--runs n] file.wav ...
// parakeet download   [--model v2|v3] [--out path]

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("""
        usage: parakeet transcribe [options] <file.wav>...
               parakeet bench      [options] <file.wav>...
               parakeet download   [--model v2|v3] [--out <path>]

        options:
          --model <name|path>   v2 (English, default) or v3 (multilingual), also as nvidia/parakeet-tdt-0.6b-v3,
                                downloaded once to ~/.cache/parakeet ($PARAKEET_CACHE); or a local .nemo file,
                                extracted NeMo directory or ONNX export directory
          --threads <n>         worker threads (default: all cores)
          --runs <n>            bench: timed runs per file (default 3, after one warm-up)
          --chunk-seconds <s>   longest audio encoded in one pass; longer input is split into overlapping windows
                                (default 300, 0 = never split)
          --chunk-overlap <s>   overlap between windows (default 15)
          --zero-pad            NeMo-style zero padding for the STFT (default: reflect, like the ONNX export)
          --int8                int8 encoder projections on AVX-VNNI (faster, approximate)
          --json                transcribe: print JSON with token and word timestamps
          --words               transcribe: print word timestamps
        """);
    return 1;
}

string command = args[0];
var files = new List<string>();
string model = "v2";
string? outPath = null;
int threads = 0, runs = 3;
double chunkSeconds = 300, chunkOverlap = 15;
bool json = false, words = false, zeroPad = false, int8 = false;
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": model = args[++i]; break;
        case "--out": outPath = args[++i]; break;
        case "--threads": threads = int.Parse(args[++i]); break;
        case "--chunk-seconds": chunkSeconds = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--chunk-overlap": chunkOverlap = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--runs": runs = int.Parse(args[++i]); break;
        case "--json": json = true; break;
        case "--words": words = true; break;
        case "--zero-pad": zeroPad = true; break;
        case "--int8": int8 = true; break;
        default: files.Add(args[i]); break;
    }
}
// Download progress on one stderr line, refreshed by the downloader (~2 Hz).
Action<DownloadProgress> progress = p =>
{
    string total = p.TotalBytes is long t ? $" of {t / 1e9:F2} GB ({p.Fraction:P0})" : "";
    Console.Error.Write($"\rDownloading {p.FileName}: {p.DownloadedBytes / 1e9:F2} GB{total}   ");
    if (p.TotalBytes is long all && p.DownloadedBytes >= all) Console.Error.WriteLine();
};

if (command == "download")
{
    if (!ParakeetModels.TryParse(model, out var which))
    {
        Console.Error.WriteLine($"unknown model '{model}' (use v2 or v3)");
        return 1;
    }
    string path = await ParakeetAsr.DownloadModelAsync(which, outPath, progress);
    Console.WriteLine(path);
    return 0;
}
if (files.Count == 0)
{
    Console.Error.WriteLine("no input files");
    return 1;
}

var sw = Stopwatch.StartNew();
ParakeetAsr asr;
try
{
    asr = await ParakeetAsr.LoadAsync(model, new ParakeetOptions
    {
        Threads = threads,
        ChunkSeconds = chunkSeconds,
        ChunkOverlapSeconds = chunkOverlap,
        PadMode = zeroPad ? StftPadMode.Constant : StftPadMode.Reflect,
        Precision = int8 ? Parakeet.Numerics.WeightPrecision.Int8 : Parakeet.Numerics.WeightPrecision.Float32,
    }, progress);
}
catch (ArgumentException e)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
using var _ = asr;
Console.Error.WriteLine($"Loaded in {sw.Elapsed.TotalSeconds:F1}s ({asr.Config.Layers} layers, d={asr.Config.DModel}, vocab {asr.Config.VocabSize}, {asr.Parallelism.MaxThreads} threads, {asr.Precision})");

switch (command)
{
    case "transcribe":
        foreach (var f in files)
        {
            var r = asr.TranscribeFile(f);
            if (json)
            {
                Console.WriteLine(JsonSerializer.Serialize(new { file = f, r.Text, r.AudioSeconds, r.RealTimeFactor, r.Words, r.Tokens },
                                                           new JsonSerializerOptions { WriteIndented = true }));
                continue;
            }
            Console.WriteLine(files.Count > 1 ? $"{Path.GetFileName(f)}: {r.Text}" : r.Text);
            if (words)
                foreach (var w in r.Words) Console.WriteLine($"  [{w.Start,7:F2} - {w.End,7:F2}] {w.Text}");
            Console.Error.WriteLine($"  {r.AudioSeconds:F1}s audio: features {r.FeatureTime.TotalMilliseconds:F0} ms, encoder {r.EncoderTime.TotalMilliseconds:F0} ms, " +
                                    $"decoder {r.DecoderTime.TotalMilliseconds:F0} ms, RTF {r.RealTimeFactor:F4}");
        }
        break;
    case "bench":
        foreach (var f in files)
        {
            var audio = WaveFile.ReadMono(f);
            asr.Transcribe(audio);
            Parakeet.Numerics.Profiler.Reset();
            var all = Enumerable.Range(0, runs).Select(_ => asr.Transcribe(audio)).ToList();
            var best = all.MinBy(r => r.TotalTime)!;
            Console.WriteLine($"{Path.GetFileNameWithoutExtension(f)}: {best.AudioSeconds:F1}s audio  features {all.Min(r => r.FeatureTime.TotalMilliseconds):F0} ms  " +
                              $"encoder {all.Min(r => r.EncoderTime.TotalMilliseconds):F0} ms  decoder {all.Min(r => r.DecoderTime.TotalMilliseconds):F0} ms  " +
                              $"total {best.TotalTime.TotalMilliseconds:F0} ms  RTF {best.RealTimeFactor:F4}");
            if (Parakeet.Numerics.Profiler.Enabled) Console.WriteLine(Parakeet.Numerics.Profiler.Report());
        }
        break;
    default:
        Console.Error.WriteLine($"unknown command '{command}'");
        return 1;
}
return 0;
