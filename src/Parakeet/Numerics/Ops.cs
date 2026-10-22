using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Parakeet.Numerics;

/// <summary>Row-wise and element-wise kernels over channels-last activations <c>x[t, c]</c>.</summary>
public static unsafe class Ops
{
    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseAvx2 = Avx2.IsSupported && Fma.IsSupported;

    /// <summary><c>y[t] = (x[t] - mean) / sqrt(var + eps) · gamma + beta</c> over rows of length <paramref name="dim"/>.</summary>
    public static void LayerNorm(float* x, float* y, int rows, int dim, float* gamma, float* beta, float eps, Parallelism par)
    {
        par.ForRange(rows, 16, (r0, r1) =>
        {
            for (int r = r0; r < r1; r++) LayerNormRow(x + (long)r * dim, y + (long)r * dim, dim, gamma, beta, eps);
        });
    }

    private static void LayerNormRow(float* x, float* y, int dim, float* gamma, float* beta, float eps)
    {
        int i = 0;
        float mean, var;
        if (UseAvx512 && dim % 16 == 0)
        {
            var s0 = Vector512<float>.Zero; var s1 = Vector512<float>.Zero;
            for (; i + 32 <= dim; i += 32) { s0 += Vector512.Load(x + i); s1 += Vector512.Load(x + i + 16); }
            for (; i < dim; i += 16) s0 += Vector512.Load(x + i);
            mean = Vector512.Sum(s0 + s1) / dim;
            var m = Vector512.Create(mean);
            var q0 = Vector512<float>.Zero; var q1 = Vector512<float>.Zero;
            for (i = 0; i + 32 <= dim; i += 32)
            {
                var d0 = Vector512.Load(x + i) - m; var d1 = Vector512.Load(x + i + 16) - m;
                q0 = Avx512F.FusedMultiplyAdd(d0, d0, q0); q1 = Avx512F.FusedMultiplyAdd(d1, d1, q1);
            }
            for (; i < dim; i += 16) { var d0 = Vector512.Load(x + i) - m; q0 = Avx512F.FusedMultiplyAdd(d0, d0, q0); }
            var = Vector512.Sum(q0 + q1) / dim;
            var inv = Vector512.Create(1f / MathF.Sqrt(var + eps));
            for (i = 0; i < dim; i += 16)
            {
                var v = (Vector512.Load(x + i) - m) * inv;
                Vector512.Store(Avx512F.FusedMultiplyAdd(v, Vector512.Load(gamma + i), Vector512.Load(beta + i)), y + i);
            }
            return;
        }
        double s = 0;
        for (i = 0; i < dim; i++) s += x[i];
        mean = (float)(s / dim);
        double q = 0;
        for (i = 0; i < dim; i++) { float d = x[i] - mean; q += d * d; }
        var = (float)(q / dim);
        float invs = 1f / MathF.Sqrt(var + eps);
        for (i = 0; i < dim; i++) y[i] = (x[i] - mean) * invs * gamma[i] + beta[i];
    }

    /// <summary>Gated linear unit over the channel axis: <c>y[t, c] = x[t, c] · sigmoid(x[t, c + C])</c>, x rows 2C wide.</summary>
    public static void Glu(float* x, float* y, int rows, int channels, Parallelism par)
    {
        par.ForRange(rows, 16, (r0, r1) =>
        {
            for (int r = r0; r < r1; r++)
            {
                float* a = x + (long)r * 2 * channels, g = a + channels, o = y + (long)r * channels;
                int c = 0;
                if (UseAvx512)
                    for (; c + 16 <= channels; c += 16)
                        Vector512.Store(Vector512.Load(a + c) * VMath.Sigmoid(Vector512.Load(g + c)), o + c);
                else if (UseAvx2)
                    for (; c + 8 <= channels; c += 8)
                        Vector256.Store(Vector256.Load(a + c) * VMath.Sigmoid(Vector256.Load(g + c)), o + c);
                for (; c < channels; c++) o[c] = a[c] * VMath.Sigmoid(g[c]);
            }
        });
    }

    /// <summary>
    /// Depthwise 1-D convolution over time with "same" zero padding, followed by an optional SiLU:
    /// <c>y[t, c] = act(bias[c] + Σ_k w[k, c] · x[t + k - pad, c])</c>. <paramref name="wTap"/> is tap-major
    /// (<c>[taps, channels]</c>) so a vector of channels shares one load per tap.
    /// </summary>
    public static void DepthwiseConv1d(float* x, float* y, int rows, int channels, float* wTap, int taps, float* bias,
                                       bool silu, Parallelism par)
    {
        int pad = (taps - 1) / 2;
        par.ForRange(rows, 8, (r0, r1) =>
        {
            for (int t = r0; t < r1; t++)
            {
                int k0 = Math.Max(0, pad - t), k1 = Math.Min(taps, rows + pad - t);
                float* o = y + (long)t * channels;
                int c = 0;
                if (UseAvx512)
                {
                    for (; c + 16 <= channels; c += 16)
                    {
                        var acc = Vector512.Load(bias + c);
                        for (int k = k0; k < k1; k++)
                            acc = Avx512F.FusedMultiplyAdd(Vector512.Load(wTap + k * channels + c),
                                                           Vector512.Load(x + (long)(t + k - pad) * channels + c), acc);
                        Vector512.Store(silu ? VMath.Silu(acc) : acc, o + c);
                    }
                }
                else if (UseAvx2)
                {
                    for (; c + 8 <= channels; c += 8)
                    {
                        var acc = Vector256.Load(bias + c);
                        for (int k = k0; k < k1; k++)
                            acc = Fma.MultiplyAdd(Vector256.Load(wTap + k * channels + c),
                                                  Vector256.Load(x + (long)(t + k - pad) * channels + c), acc);
                        Vector256.Store(silu ? VMath.Silu(acc) : acc, o + c);
                    }
                }
                for (; c < channels; c++)
                {
                    float acc = bias[c];
                    for (int k = k0; k < k1; k++) acc = MathF.FusedMultiplyAdd(wTap[k * channels + c], x[(long)(t + k - pad) * channels + c], acc);
                    o[c] = silu ? VMath.Silu(acc) : acc;
                }
            }
        });
    }

    /// <summary>In-place softmax of <paramref name="n"/> values; returns nothing, the row sums to 1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SoftmaxInPlace(float* x, int n)
    {
        int i = 0;
        float max = float.NegativeInfinity;
        if (UseAvx512 && n >= 16)
        {
            var vm = Vector512.Create(float.NegativeInfinity);
            for (; i + 16 <= n; i += 16) vm = Vector512.Max(vm, Vector512.Load(x + i));
            for (int l = 0; l < 16; l++) max = MathF.Max(max, vm.GetElement(l));
        }
        for (; i < n; i++) max = MathF.Max(max, x[i]);
        float sum = 0;
        i = 0;
        if (UseAvx512)
        {
            var m = Vector512.Create(max);
            var s = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var e = VMath.Exp(Vector512.Load(x + i) - m);
                Vector512.Store(e, x + i);
                s += e;
            }
            sum = Vector512.Sum(s);
        }
        for (; i < n; i++) { float e = MathF.Exp(x[i] - max); x[i] = e; sum += e; }
        float inv = 1f / sum;
        i = 0;
        if (UseAvx512)
        {
            var vi = Vector512.Create(inv);
            for (; i + 16 <= n; i += 16) Vector512.Store(Vector512.Load(x + i) * vi, x + i);
        }
        for (; i < n; i++) x[i] *= inv;
    }

    /// <summary>Index of the largest of <paramref name="n"/> values (first one on ties, like numpy / torch).</summary>
    public static int ArgMax(float* x, int n)
    {
        int best = 0;
        float bv = x[0];
        for (int i = 1; i < n; i++)
            if (x[i] > bv) { bv = x[i]; best = i; }
        return best;
    }

    /// <summary>Transposes a row-major <c>[rows, cols]</c> matrix into <c>[cols, rows]</c>.</summary>
    public static void Transpose(float* src, float* dst, int rows, int cols)
    {
        const int B = 32;
        for (int r0 = 0; r0 < rows; r0 += B)
            for (int c0 = 0; c0 < cols; c0 += B)
                for (int r = r0; r < Math.Min(rows, r0 + B); r++)
                    for (int c = c0; c < Math.Min(cols, c0 + B); c++)
                        dst[(long)c * rows + r] = src[(long)r * cols + c];
    }
}
