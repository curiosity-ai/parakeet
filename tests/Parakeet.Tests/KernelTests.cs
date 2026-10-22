using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Parakeet.Audio;
using Parakeet.Numerics;
using Xunit.Abstractions;

namespace Parakeet.Tests;

/// <summary>Numerical kernels against straightforward double-precision references.</summary>
public sealed unsafe class KernelTests(ITestOutputHelper output)
{
    private static float[] Random(int n, int seed, float scale = 1f)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 17, 5)]
    [InlineData(4, 1030, 640)]
    [InlineData(5, 64, 300)]
    [InlineData(12, 32, 256)]
    [InlineData(13, 33, 257)]
    [InlineData(137, 1024, 1024)]
    [InlineData(250, 96, 4096)]
    [InlineData(500, 600, 40)]
    public void GemmMatchesReference(int m, int n, int k)
    {
        foreach (var threads in new[] { 1, 4 })
            foreach (var mode in new[] { "plain", "bias", "silu", "acc", "relu-acc-bias" })
            {
                var a = Random(m * k, 1);
                var w = Random(n * k, 2);          // PyTorch [n, k]
                var bias = Random(n, 3);
                var c0 = Random(m * n, 4);
                var c = (float[])c0.Clone();
                bool useBias = mode.Contains("bias"), acc = mode.Contains("acc");
                var act = mode.Contains("silu") ? Activation.Silu : mode.Contains("relu") ? Activation.Relu : Activation.None;
                fixed (float* wp = w, ap = a, cp = c, bp = bias)
                {
                    using var packed = PackedMatrix.FromLinear(wp, n, k);
                    using var par = new Parallelism(threads);
                    Gemm.Multiply(ap, k, m, packed, cp, n, par, useBias ? bp : null, acc, act);
                }
                double maxErr = 0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double s = 0;
                        for (int p = 0; p < k; p++) s += (double)a[i * k + p] * w[j * k + p];
                        if (useBias) s += bias[j];
                        if (acc) s += c0[i * n + j];
                        if (act == Activation.Silu) s = Silu(s);
                        if (act == Activation.Relu) s = Math.Max(0, s);
                        maxErr = Math.Max(maxErr, Math.Abs(s - c[i * n + j]));
                    }
                Assert.True(maxErr < 1e-4 * Math.Sqrt(k), $"m={m} n={n} k={k} {mode} threads={threads}: max error {maxErr}");
            }
    }

    [Fact]
    public void GemmPanelRangeWritesOnlyThoseColumns()
    {
        int m = 20, n = 200, k = 64, nb0 = 2, nb1 = 5;
        var a = Random(m * k, 5);
        var w = Random(n * k, 6);
        int cols = (nb1 - nb0) * PackedMatrix.NR;
        var c = new float[m * cols];
        fixed (float* wp = w, ap = a, cp = c)
        {
            using var packed = PackedMatrix.FromLinear(wp, n, k);
            Gemm.Multiply(ap, k, m, packed, nb0, nb1, cp, cols, Parallelism.Single);
        }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < cols; j++)
            {
                int col = nb0 * PackedMatrix.NR + j;
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * w[col * k + p];
                Assert.True(Math.Abs(s - c[i * cols + j]) < 1e-4, $"({i}, {col})");
            }
    }

    [Fact]
    public void ExpIsAccurate()
    {
        if (!Avx512F.IsSupported && !Avx2.IsSupported) return;
        double worst = 0;
        for (float x = -87f; x < 88f; x += 0.0137f)
        {
            float v = Avx512F.IsSupported ? VMath.Exp(Vector512.Create(x)).GetElement(3) : VMath.Exp(Vector256.Create(x)).GetElement(3);
            double rel = Math.Abs(v - Math.Exp(x)) / Math.Exp(x);
            worst = Math.Max(worst, rel);
        }
        output.WriteLine($"max relative error {worst:E2}");
        Assert.True(worst < 3e-7, $"exp relative error {worst}");
    }

    [Fact]
    public void LayerNormMatchesReference()
    {
        int rows = 7, dim = 1024;
        var x = Random(rows * dim, 7, 3f);
        var g = Random(dim, 8);
        var b = Random(dim, 9);
        var y = new float[rows * dim];
        fixed (float* xp = x, yp = y, gp = g, bp = b) Ops.LayerNorm(xp, yp, rows, dim, gp, bp, 1e-5f, Parallelism.Single);
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, var = 0;
            for (int i = 0; i < dim; i++) mean += x[r * dim + i];
            mean /= dim;
            for (int i = 0; i < dim; i++) var += Math.Pow(x[r * dim + i] - mean, 2);
            var /= dim;
            for (int i = 0; i < dim; i++)
            {
                double e = (x[r * dim + i] - mean) / Math.Sqrt(var + 1e-5) * g[i] + b[i];
                Assert.True(Math.Abs(e - y[r * dim + i]) < 1e-5, $"row {r} col {i}");
            }
        }
    }

    [Fact]
    public void SoftmaxGluAndDepthwiseConv()
    {
        var s = Random(77, 10, 20f);
        var sm = (float[])s.Clone();
        fixed (float* p = sm) Ops.SoftmaxInPlace(p, sm.Length);
        double max = s.Max(), sum = s.Sum(v => Math.Exp(v - max));
        for (int i = 0; i < s.Length; i++) Assert.True(Math.Abs(Math.Exp(s[i] - max) / sum - sm[i]) < 1e-6);

        int rows = 11, ch = 40, taps = 9;
        var x = Random(rows * 2 * ch, 11);
        var glu = new float[rows * ch];
        fixed (float* xp = x, gp = glu) Ops.Glu(xp, gp, rows, ch, Parallelism.Single);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < ch; c++)
                Assert.True(Math.Abs(x[r * 2 * ch + c] / (1 + Math.Exp(-x[r * 2 * ch + ch + c])) - glu[r * ch + c]) < 1e-6);

        var w = Random(taps * ch, 12);
        var b = Random(ch, 13);
        var y = new float[rows * ch];
        fixed (float* xp = glu, yp = y, wp = w, bp = b) Ops.DepthwiseConv1d(xp, yp, rows, ch, wp, taps, bp, silu: true, Parallelism.Single);
        for (int t = 0; t < rows; t++)
            for (int c = 0; c < ch; c++)
            {
                double acc = b[c];
                for (int k = 0; k < taps; k++)
                {
                    int tt = t + k - taps / 2;
                    if (tt >= 0 && tt < rows) acc += w[k * ch + c] * glu[tt * ch + c];
                }
                Assert.True(Math.Abs(Silu(acc) - y[t * ch + c]) < 1e-5, $"t={t} c={c}");
            }
    }

    [Fact]
    public void FftMatchesDft()
    {
        int n = 512;
        var x = Random(n, 14);
        var re = x.Select(v => (double)v).ToArray();
        var im = new double[n];
        new Fft(n).Forward(re, im);
        for (int k = 0; k < n; k += 37)
        {
            double sr = 0, si = 0;
            for (int t = 0; t < n; t++)
            {
                sr += x[t] * Math.Cos(2 * Math.PI * k * t / n);
                si -= x[t] * Math.Sin(2 * Math.PI * k * t / n);
            }
            Assert.True(Math.Abs(sr - re[k]) < 1e-9 && Math.Abs(si - im[k]) < 1e-9, $"bin {k}");
        }
    }

    [Fact]
    public void ResamplerPreservesATone()
    {
        int from = 44100, to = 16000;
        var tone = new float[from];
        for (int i = 0; i < tone.Length; i++) tone[i] = (float)Math.Sin(2 * Math.PI * 440 * i / from);
        var y = Resampler.Resample(tone, from, to);
        Assert.InRange(y.Length, to - 1, to + 1);
        double err = 0;
        for (int i = 200; i < y.Length - 200; i++) err = Math.Max(err, Math.Abs(y[i] - Math.Sin(2 * Math.PI * 440 * i / to)));
        output.WriteLine($"max error {err:E2}");
        Assert.True(err < 1e-3);
    }
}
