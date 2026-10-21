using System.Diagnostics;
using System.Text.Json;
using Parakeet;
using Parakeet.Audio;
using Parakeet.IO;

// parakeet transcribe [--model path|repo] [--threads n] [--json] [--words] file.wav ...
// parakeet bench      [--model path|repo] [--threads n] [--runs n] file.wav ...

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("""
        usage: parakeet transcribe [options] <file.wav>...
               parakeet bench      [options] <file.wav>...

        options:
          --model <path|repo>   .nemo file, NeMo / ONNX export directory, or Hugging Face repo
                                (default: nvidia/parakeet-tdt-0.6b-v2, downloaded to ~/.cache/parakeet)
          --threads <n>         worker threads (default: all cores)
          --runs <n>            bench: timed runs per file (default 3, after one warm-up)
          --zero-pad            NeMo-style zero padding for the STFT (default: reflect, like the ONNX export)
          --json                transcribe: print JSON with token and word timestamps
          --words               transcribe: print word timestamps
        """);
    return 1;
}

string command = args[0];
var files = new List<string>();
string model = ModelDownloader.DefaultRepo;
int threads = 0, runs = 3;
bool json = false, words = false, zeroPad = false;
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": model = args[++i]; break;
        case "--threads": threads = int.Parse(args[++i]); break;
        case "--runs": runs = int.Parse(args[++i]); break;
        case "--json": json = true; break;
        case "--words": words = true; break;
        case "--zero-pad": zeroPad = true; break;
        default: files.Add(args[i]); break;
    }
}
if (files.Count == 0)
{
    Console.Error.WriteLine("no input files");
    return 1;
}

var progress = new Progress<string>(m => Console.Error.WriteLine(m));
var sw = Stopwatch.StartNew();
using var asr = await ParakeetAsr.LoadAsync(model, new ParakeetOptions { Threads = threads, PadMode = zeroPad ? StftPadMode.Constant : StftPadMode.Reflect }, progress);
Console.Error.WriteLine($"Loaded in {sw.Elapsed.TotalSeconds:F1}s ({asr.Config.Layers} layers, d={asr.Config.DModel}, vocab {asr.Config.VocabSize}, {asr.Parallelism.MaxThreads} threads)");

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
            var all = Enumerable.Range(0, runs).Select(_ => asr.Transcribe(audio)).ToList();
            var best = all.MinBy(r => r.TotalTime)!;
            Console.WriteLine($"{Path.GetFileNameWithoutExtension(f)}: {best.AudioSeconds:F1}s audio  features {all.Min(r => r.FeatureTime.TotalMilliseconds):F0} ms  " +
                              $"encoder {all.Min(r => r.EncoderTime.TotalMilliseconds):F0} ms  decoder {all.Min(r => r.DecoderTime.TotalMilliseconds):F0} ms  " +
                              $"total {best.TotalTime.TotalMilliseconds:F0} ms  RTF {best.RealTimeFactor:F4}");
        }
        break;
    default:
        Console.Error.WriteLine($"unknown command '{command}'");
        return 1;
}
return 0;
