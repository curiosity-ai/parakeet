using System.Diagnostics;
using Parakeet;
using Parakeet.Audio;
using Parakeet.Benchmarks;

// Pure C# Parakeet vs the ONNX Runtime pipeline of the original Go server, same machine, same thread count.
//
//   dotnet run -c Release --project benchmarks/Parakeet.Benchmarks -- \
//       --model model.nemo --onnx onnx-dir [--threads 4] [--runs 5] [--int8] [--only csharp|onnx] file.wav ...

string? model = null, onnx = null, only = null;
int threads = Environment.ProcessorCount, runs = 5;
bool int8 = false;
var files = new List<string>();
if (args.Contains("--gemm"))
{
    KernelBench.Run(args.Contains("--threads") ? int.Parse(args[Array.IndexOf(args, "--threads") + 1]) : threads);
    return 0;
}
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": model = args[++i]; break;
        case "--onnx": onnx = args[++i]; break;
        case "--threads": threads = int.Parse(args[++i]); break;
        case "--runs": runs = int.Parse(args[++i]); break;
        case "--int8": int8 = true; break;
        case "--only": only = args[++i]; break;
        default: files.Add(args[i]); break;
    }
}
if (files.Count == 0 || (model == null && onnx == null))
{
    Console.Error.WriteLine("usage: Parakeet.Benchmarks --model <.nemo|dir> --onnx <onnx dir> [--threads n] [--runs n] [--int8] [--only csharp|onnx] <file.wav>...");
    return 1;
}

var clips = files.Select(f => (Name: Path.GetFileNameWithoutExtension(f), Audio: WaveFile.ReadMono(f))).ToList();
double totalAudio = clips.Sum(c => c.Audio.Length / 16000.0);
Console.WriteLine($"{clips.Count} clips, {totalAudio:F1}s audio, {threads} threads, best of {runs} runs after a warm-up, {Environment.ProcessorCount} cores");
Console.WriteLine();

var csharp = new Dictionary<string, (double Feat, double Enc, double Dec, double Total, string Text)>();
var ort = new Dictionary<string, (double Feat, double Enc, double Dec, double Total, string Text)>();

if (model != null && only is null or "csharp")
{
    var sw = Stopwatch.StartNew();
    using var asr = ParakeetAsr.Load(model ?? onnx!, new ParakeetOptions { Threads = threads });
    Console.WriteLine($"Parakeet (pure C#) loaded in {sw.Elapsed.TotalSeconds:F1}s");
    foreach (var (name, audio) in clips)
    {
        asr.Transcribe(audio);
        var rs = Enumerable.Range(0, runs).Select(_ => asr.Transcribe(audio)).ToList();
        csharp[name] = (rs.Min(r => r.FeatureTime.TotalMilliseconds), rs.Min(r => r.EncoderTime.TotalMilliseconds),
                        rs.Min(r => r.DecoderTime.TotalMilliseconds), rs.Min(r => r.TotalTime.TotalMilliseconds), rs[0].Text);
        Print("C#  ", name, audio.Length, csharp[name]);
    }
    Console.WriteLine();
}

if (onnx != null && only is null or "onnx")
{
    var sw = Stopwatch.StartNew();
    using var pipe = new OnnxPipeline(onnx, threads, int8);
    Console.WriteLine($"ONNX Runtime {typeof(Microsoft.ML.OnnxRuntime.InferenceSession).Assembly.GetName().Version} ({pipe.EncoderFile}) loaded in {sw.Elapsed.TotalSeconds:F1}s");
    foreach (var (name, audio) in clips)
    {
        pipe.Transcribe(audio);
        var rs = Enumerable.Range(0, runs).Select(_ => pipe.Transcribe(audio)).ToList();
        ort[name] = (rs.Min(r => r.Features.TotalMilliseconds), rs.Min(r => r.Encoder.TotalMilliseconds),
                     rs.Min(r => r.Decoder.TotalMilliseconds), rs.Min(r => r.Total.TotalMilliseconds), rs[0].Text);
        Print("ORT ", name, audio.Length, ort[name]);
    }
    Console.WriteLine();
}

if (csharp.Count > 0 && ort.Count > 0)
{
    double cs = csharp.Values.Sum(v => v.Total), os = ort.Values.Sum(v => v.Total);
    double ce = csharp.Values.Sum(v => v.Enc), oe = ort.Values.Sum(v => v.Enc);
    double cd = csharp.Values.Sum(v => v.Dec), od = ort.Values.Sum(v => v.Dec);
    double cf = csharp.Values.Sum(v => v.Feat), of = ort.Values.Sum(v => v.Feat);
    int same = clips.Count(c => csharp[c.Name].Text == ort[c.Name].Text);
    Console.WriteLine($"{"",-10} {"features",10} {"encoder",10} {"decoder",10} {"total",10} {"RTF",8}");
    Console.WriteLine($"{"C#",-10} {cf,8:F0}ms {ce,8:F0}ms {cd,8:F0}ms {cs,8:F0}ms {cs / 1000 / totalAudio,8:F4}");
    Console.WriteLine($"{"ORT",-10} {of,8:F0}ms {oe,8:F0}ms {od,8:F0}ms {os,8:F0}ms {os / 1000 / totalAudio,8:F4}");
    Console.WriteLine($"{"speed-up",-10} {of / cf,9:F2}x {oe / ce,9:F2}x {od / cd,9:F2}x {os / cs,9:F2}x");
    Console.WriteLine($"identical transcripts: {same}/{clips.Count}");
}
return 0;

static void Print(string who, string name, int samples, (double Feat, double Enc, double Dec, double Total, string Text) r) =>
    Console.WriteLine($"{who} {name,-40} {samples / 16000.0,5:F1}s  feat {r.Feat,6:F0} ms  enc {r.Enc,6:F0} ms  dec {r.Dec,5:F0} ms  " +
                      $"total {r.Total,6:F0} ms  RTF {r.Total / 1000 / (samples / 16000.0):F4}");
