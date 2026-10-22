using System.Diagnostics;
using Parakeet.Numerics;

namespace Parakeet.Benchmarks;

/// <summary>GEMM throughput on the encoder's shapes (<c>--gemm</c>).</summary>
public static unsafe class KernelBench
{
    public static void Run(int threads)
    {
        if (Environment.GetEnvironmentVariable("PARAKEET_TUNE") == "1")
        {
            Tune(threads);
            return;
        }
        if (Environment.GetEnvironmentVariable("PARAKEET_SWEEP") == "1")
        {
            Sweep(threads);
            return;
        }
        (int M, int K, int N, string What)[] shapes =
        [
            (87, 1024, 4096, "ff linear1, 7s"), (87, 4096, 1024, "ff linear2, 7s"), (87, 1024, 3072, "qkv, 7s"),
            (462, 1024, 4096, "ff linear1, 37s"), (462, 4096, 1024, "ff linear2, 37s"), (923, 1024, 1024, "pos, 37s"),
        ];
        foreach (int th in new[] { 1, threads })
        {
            var par = new Parallelism(th);
            foreach (var (m, k, n, what) in shapes)
            {
                using var a = new NativeBuffer<float>((long)m * k);
                using var w = new NativeBuffer<float>((long)n * k);
                using var c = new NativeBuffer<float>((long)m * n);
                var rnd = new Random(1);
                for (long i = 0; i < a.Length; i++) a.Ptr[i] = (float)rnd.NextDouble();
                for (long i = 0; i < w.Length; i++) w.Ptr[i] = (float)rnd.NextDouble();
                using var b = PackedMatrix.FromLinear(w.Ptr, n, k);
                // Cold weights: cycle through enough distinct matrices (≈ 512 MB) that none stays cached, like the
                // encoder streaming 2.4 GB of weights per forward pass.
                int copies = (int)Math.Max(2, (512L << 20) / b.ByteSize);
                var cold = Enumerable.Range(0, copies).Select(_ => PackedMatrix.FromLinear(w.Ptr, n, k)).ToArray();
                Gemm.Multiply(a.Ptr, k, m, b, c.Ptr, n, par);
                double best = double.MaxValue;
                for (int r = 0; r < 7; r++)
                {
                    var sw = Stopwatch.StartNew();
                    Gemm.Multiply(a.Ptr, k, m, b, c.Ptr, n, par);
                    best = Math.Min(best, sw.Elapsed.TotalSeconds);
                }
                var swc = Stopwatch.StartNew();
                foreach (var cb in cold) Gemm.Multiply(a.Ptr, k, m, cb, c.Ptr, n, par);
                double coldT = swc.Elapsed.TotalSeconds / copies;
                foreach (var cb in cold) cb.Dispose();
                double gflop = 2.0 * m * k * n / 1e9;
                Console.WriteLine($"threads {th}  M={m,4} K={k,4} N={n,4} ({what,-16}) hot {best * 1000,7:F2} ms {gflop / best,6:F1} GFLOP/s" +
                                  $"   cold {coldT * 1000,7:F2} ms {gflop / coldT,6:F1} GFLOP/s {b.ByteSize / coldT / 1e9,5:F1} GB/s");
            }
        }
    }

    private static void Sweep(int threads)
    {
        var par = new Parallelism(threads);
        int m = 462, k = 1024, n = 4096;
        using var a = new NativeBuffer<float>((long)m * k);
        using var w = new NativeBuffer<float>((long)n * k);
        using var c = new NativeBuffer<float>((long)m * n);
        for (long i = 0; i < a.Length; i++) a.Ptr[i] = 0.5f;
        for (long i = 0; i < w.Length; i++) w.Ptr[i] = 0.25f;
        foreach (int mb in new[] { 16, 32, 64, 128, 256, 512, 1024, 2048 })
        {
            int copies = mb / 16;
            var mats = Enumerable.Range(0, copies).Select(_ => PackedMatrix.FromLinear(w.Ptr, n, k)).ToArray();
            foreach (var x in mats) Gemm.Multiply(a.Ptr, k, m, x, c.Ptr, n, par);
            double best = double.MaxValue;
            for (int r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                foreach (var x in mats) Gemm.Multiply(a.Ptr, k, m, x, c.Ptr, n, par);
                best = Math.Min(best, sw.Elapsed.TotalSeconds / copies);
            }
            foreach (var x in mats) x.Dispose();
            Console.WriteLine($"working set {mb,5} MB: {2.0 * m * k * n / 1e9 / best,6:F1} GFLOP/s");
        }
        Console.WriteLine(File.ReadAllLines("/proc/meminfo").First(l => l.StartsWith("AnonHugePages")));
    }

    /// <summary>One-line cold-weight summary over the encoder's main shapes (for blocking-parameter tuning).</summary>
    private static void Tune(int threads)
    {
        var par = new Parallelism(threads);
        var parts = new List<string>();
        double sumT = 0, sumF = 0;
        foreach (var (m, k, n) in new[] { (87, 1024, 4096), (87, 4096, 1024), (462, 1024, 4096), (462, 4096, 1024) })
        {
            using var a = new NativeBuffer<float>((long)m * k);
            using var w = new NativeBuffer<float>((long)n * k);
            using var c = new NativeBuffer<float>((long)m * n);
            for (long i = 0; i < a.Length; i++) a.Ptr[i] = 0.5f;
            for (long i = 0; i < w.Length; i++) w.Ptr[i] = 0.25f;
            var mats = Enumerable.Range(0, 32).Select(_ => PackedMatrix.FromLinear(w.Ptr, n, k)).ToArray();
            foreach (var x in mats) Gemm.Multiply(a.Ptr, k, m, x, c.Ptr, n, par);
            double best = double.MaxValue;
            for (int r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                foreach (var x in mats) Gemm.Multiply(a.Ptr, k, m, x, c.Ptr, n, par);
                best = Math.Min(best, sw.Elapsed.TotalSeconds / mats.Length);
            }
            foreach (var x in mats) x.Dispose();
            double f = 2.0 * m * k * n / 1e9;
            sumT += best; sumF += f;
            parts.Add($"{f / best,5:F0}");
        }
        Console.WriteLine($"KC={PackedMatrix.KBlock,3} NCP={Environment.GetEnvironmentVariable("PARAKEET_NCP") ?? "16",2} MCT={Environment.GetEnvironmentVariable("PARAKEET_MCT") ?? "40",2}: " +
                          string.Join(" ", parts) + $"  | avg {sumF / sumT:F0} GFLOP/s");
    }
}
