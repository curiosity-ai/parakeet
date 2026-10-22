using System.Diagnostics;
using Parakeet;
using Parakeet.Audio;
using Parakeet.Benchmarks;

// Pure C# Parakeet vs the ONNX Runtime pipeline of the original Go server, same machine, same thread count.
// Runs of the two engines are interleaved per clip (C#, ORT, C#, ORT, ...) so slow drifts of a shared machine hit both
// equally; medians are reported.
//
//   dotnet run -c Release --project benchmarks/Parakeet.Benchmarks -- \
//       --model model.nemo --onnx onnx-dir [--threads 4] [--runs 7] [--int8] [--only csharp|onnx] file.wav ...
//   dotnet run -c Release --project benchmarks/Parakeet.Benchmarks -- --gemm [--threads 4]

string? model = null, onnx = null, only = null;
int threads = Environment.ProcessorCount, runs = 7;
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
Console.WriteLine($"{clips.Count} clips, {totalAudio:F1}s audio, {threads} threads, median of {runs} interleaved runs after a warm-up, " +
                  $"{Environment.ProcessorCount} cores, .NET {Environment.Version}");

ParakeetAsr? asr = null;
OnnxPipeline? pipe = null;
var sw = Stopwatch.StartNew();
if (model != null && only is null or "csharp")
{
    asr = ParakeetAsr.Load(model, new ParakeetOptions { Threads = threads });
    Console.WriteLine($"Parakeet (pure C#) loaded in {sw.Elapsed.TotalSeconds:F1}s");
}
sw.Restart();
if (onnx != null && only is null or "onnx")
{
    pipe = new OnnxPipeline(onnx, threads, int8);
    Console.WriteLine($"ONNX Runtime {typeof(Microsoft.ML.OnnxRuntime.InferenceSession).Assembly.GetName().Version} ({pipe.EncoderFile}) loaded in {sw.Elapsed.TotalSeconds:F1}s");
}
Console.WriteLine();

var cs = new List<Stage>();
var ort = new List<Stage>();
int same = 0;
foreach (var (name, audio) in clips)
{
    var c = new List<Stage>();
    var o = new List<Stage>();
    string? ct = null, ot = null;
    asr?.Transcribe(audio);
    pipe?.Transcribe(audio);
    for (int r = 0; r < runs; r++)
    {
        if (asr != null)
        {
            var x = asr.Transcribe(audio);
            ct = x.Text;
            c.Add(new Stage(x.FeatureTime.TotalMilliseconds, x.EncoderTime.TotalMilliseconds, x.DecoderTime.TotalMilliseconds));
        }
        if (pipe != null)
        {
            var y = pipe.Transcribe(audio);
            ot = y.Text;
            o.Add(new Stage(y.Features.TotalMilliseconds, y.Encoder.TotalMilliseconds, y.Decoder.TotalMilliseconds));
        }
    }
    double secs = audio.Length / 16000.0;
    if (c.Count > 0) { var m = Stage.Median(c); cs.Add(m); Print("C#  ", name, secs, m); }
    if (o.Count > 0) { var m = Stage.Median(o); ort.Add(m); Print("ORT ", name, secs, m); }
    if (ct != null && ot != null)
    {
        if (ct == ot) same++;
        else Console.WriteLine($"     transcripts differ:\n       C#:  {ct}\n       ORT: {ot}");
    }
}

if (cs.Count > 0 && ort.Count > 0)
{
    var c = Stage.Sum(cs);
    var o = Stage.Sum(ort);
    Console.WriteLine();
    Console.WriteLine($"{"",-10} {"features",10} {"encoder",10} {"decoder",10} {"total",10} {"RTF",8}");
    Console.WriteLine($"{"C#",-10} {c.Features,8:F0}ms {c.Encoder,8:F0}ms {c.Decoder,8:F0}ms {c.Total,8:F0}ms {c.Total / 1000 / totalAudio,8:F4}");
    Console.WriteLine($"{"ORT",-10} {o.Features,8:F0}ms {o.Encoder,8:F0}ms {o.Decoder,8:F0}ms {o.Total,8:F0}ms {o.Total / 1000 / totalAudio,8:F4}");
    Console.WriteLine($"{"speed-up",-10} {o.Features / c.Features,9:F2}x {o.Encoder / c.Encoder,9:F2}x {o.Decoder / c.Decoder,9:F2}x {o.Total / c.Total,9:F2}x");
    Console.WriteLine($"identical transcripts: {same}/{clips.Count}");
}
asr?.Dispose();
pipe?.Dispose();
return 0;

static void Print(string who, string name, double secs, Stage r) =>
    Console.WriteLine($"{who} {name,-40} {secs,5:F1}s  feat {r.Features,6:F0} ms  enc {r.Encoder,6:F0} ms  dec {r.Decoder,5:F0} ms  " +
                      $"total {r.Total,6:F0} ms  RTF {r.Total / 1000 / secs:F4}");

internal readonly record struct Stage(double Features, double Encoder, double Decoder)
{
    public double Total => Features + Encoder + Decoder;

    public static Stage Median(List<Stage> runs)
    {
        static double Med(IEnumerable<double> v)
        {
            var a = v.OrderBy(x => x).ToArray();
            return a.Length % 2 == 1 ? a[a.Length / 2] : (a[a.Length / 2 - 1] + a[a.Length / 2]) / 2;
        }
        return new Stage(Med(runs.Select(r => r.Features)), Med(runs.Select(r => r.Encoder)), Med(runs.Select(r => r.Decoder)));
    }

    public static Stage Sum(List<Stage> s) => new(s.Sum(x => x.Features), s.Sum(x => x.Encoder), s.Sum(x => x.Decoder));
}
