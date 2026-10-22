using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Parakeet.Numerics;

namespace Parakeet.Model;

/// <summary>
/// NeMo <c>ConvSubsampling</c> with <c>subsampling: dw_striding</c> (factor 8): Conv2d(1→C, 3×3, s2) → ReLU →
/// [depthwise Conv2d(3×3, s2) → pointwise Conv2d(1×1) → ReLU] × 2 → Linear(C·F/8 → d_model), over the
/// (time, mel) image. Activations are channels-last <c>[time, freq, channel]</c>, so every pointwise convolution is a
/// GEMM and the depthwise ones vectorize over channels; the final linear's columns are permuted at load time to
/// match that flattening order (PyTorch flattens <c>[channel, freq]</c>).
/// </summary>
internal sealed unsafe class Subsampling : IDisposable
{
    private readonly int _c, _fIn, _d, _stages;
    private readonly NativeBuffer<float> _w0, _b0;                  // conv.0: [9][C] tap-major, bias
    private readonly NativeBuffer<float>[] _dw, _dwb;               // depthwise: [9][C], bias
    private readonly PackedMatrix[] _pw;                            // pointwise C → C
    private readonly NativeBuffer<float>[] _pwb;
    private readonly LinearWeight _out;
    private readonly NativeBuffer<float> _outB;

    public Subsampling(WeightReader w, ParakeetConfig cfg)
    {
        _c = cfg.SubsamplingChannels;
        _fIn = cfg.Mel.Features;
        _d = cfg.DModel;
        _stages = 0;
        for (int s = cfg.SubsamplingFactor; s > 1; s >>= 1) _stages++;
        if (_stages < 1 || (1 << _stages) != cfg.SubsamplingFactor) throw new NotSupportedException("Subsampling factor must be a power of two.");
        const string p = "encoder.pre_encode.";
        _w0 = TapMajor(w.Get(p + "conv.0.weight", _c, 1, 3, 3).Span);
        _b0 = w.Vector(p + "conv.0.bias", _c);
        _dw = new NativeBuffer<float>[_stages - 1];
        _dwb = new NativeBuffer<float>[_stages - 1];
        _pw = new PackedMatrix[_stages - 1];
        _pwb = new NativeBuffer<float>[_stages - 1];
        for (int s = 0; s < _stages - 1; s++)
        {
            int dwIdx = 2 + 3 * s, pwIdx = 3 + 3 * s;   // conv.2/conv.3, conv.5/conv.6 (ReLUs at 1, 4, 7)
            _dw[s] = TapMajor(w.Get($"{p}conv.{dwIdx}.weight", _c, 1, 3, 3).Span);
            _dwb[s] = w.Vector($"{p}conv.{dwIdx}.bias", _c);
            _pw[s] = w.Linear($"{p}conv.{pwIdx}.weight", _c, _c);
            _pwb[s] = w.Vector($"{p}conv.{pwIdx}.bias", _c);
        }
        int fOut = FreqAfter(_fIn);
        int k = _c * fOut;
        var src = w.Get(p + "out.weight", _d, k).Span;
        using var perm = new NativeBuffer<float>((long)_d * k, zero: false);
        for (int o = 0; o < _d; o++)
            for (int c = 0; c < _c; c++)
                for (int f = 0; f < fOut; f++)
                    perm.Ptr[(long)o * k + f * _c + c] = src[o * k + c * fOut + f];
        _out = LinearWeight.FromLinear(perm.Ptr, _d, k, 1f, w.Precision, w.HugePages);
        _outB = w.Vector(p + "out.bias", _d);
    }

    private int FreqAfter(int f)
    {
        for (int s = 0; s < _stages; s++) f = (f - 1) / 2 + 1;
        return f;
    }

    private static int Half(int n) => (n - 1) / 2 + 1;

    /// <summary>Floats of the flattened <c>[EncoderFrames(frames), C·F/8]</c> buffer the final linear reads.</summary>
    public long ScratchSize(int frames)
    {
        int t = frames, w = _fIn;
        for (int s = 0; s < _stages; s++) { t = Half(t); w = Half(w); }
        return (long)t * w * _c;
    }

    [ThreadStatic] private static NativeBuffer<float>? t_chunk;

    /// <summary>
    /// Maps <c>[frames, features]</c> to <c>[EncoderFrames(frames), d_model]</c>. The convolutions run in independent
    /// bands of output frames, one band per work item: each band recomputes the few rows its 3×3 receptive field
    /// shares with its neighbours and keeps its intermediate activations in a small per-thread buffer, so the large
    /// full-resolution feature maps never stream through memory. The final linear is one GEMM over all frames.
    /// </summary>
    public void Forward(float* feats, int frames, float* output, float* scratch, Parallelism par)
    {
        var heights = new int[_stages + 1];
        var widths = new int[_stages + 1];
        heights[0] = frames;
        widths[0] = _fIn;
        for (int s = 1; s <= _stages; s++)
        {
            heights[s] = Half(heights[s - 1]);
            widths[s] = Half(widths[s - 1]);
        }
        int tOut = heights[_stages];
        long rowOut = (long)widths[_stages] * _c;
        int band = Math.Clamp((tOut + 2 * par.MaxThreads - 1) / (2 * par.MaxThreads), 1, 8);
        int bands = (tOut + band - 1) / band;
        par.For(bands, bi =>
        {
            int a = bi * band, b = Math.Min(tOut, a + band);
            // Row ranges needed at every level (row r at level s+1 reads rows 2r-1 .. 2r+1 at level s).
            Span<int> lo = stackalloc int[_stages + 1], hi = stackalloc int[_stages + 1];
            lo[_stages] = a;
            hi[_stages] = b;
            for (int s = _stages - 1; s >= 1; s--)
            {
                lo[s] = Math.Max(0, 2 * lo[s + 1] - 1);
                hi[s] = Math.Min(heights[s], 2 * hi[s + 1]);
            }
            long need = 0;
            for (int s = 1; s < _stages; s++) need += 2L * (hi[s] - lo[s]) * widths[s] * _c;
            var buf = t_chunk;
            if (buf == null || buf.Length < need)
            {
                buf?.Dispose();
                buf = t_chunk = new NativeBuffer<float>(Math.Max(need, 1 << 16), zero: false);
            }
            float* cur = buf.Ptr;                      // level-1 activations (conv.0 + ReLU)
            Conv0(feats, heights[0], widths[0], lo[1], hi[1], cur);
            float* free = cur + (long)(hi[1] - lo[1]) * widths[1] * _c;
            for (int s = 1; s < _stages; s++)
            {
                int rows = hi[s + 1] - lo[s + 1];
                float* dw = free;
                Depthwise(cur, lo[s], heights[s], widths[s], lo[s + 1], hi[s + 1], dw, _dw[s - 1].Ptr, _dwb[s - 1].Ptr);
                // The last stage writes straight into the flattened buffer of the final linear.
                float* dst = s + 1 == _stages ? scratch + lo[s + 1] * rowOut : dw + (long)rows * widths[s + 1] * _c;
                Gemm.Multiply(dw, _c, rows * widths[s + 1], _pw[s - 1], dst, _c, Parallelism.Single, _pwb[s - 1].Ptr, act: Activation.Relu);
                cur = dst;
                free = dst + (long)rows * widths[s + 1] * _c;
            }
        });
        _out.Multiply(scratch, rowOut, tOut, output, _d, par, _outB.Ptr);
    }

    /// <summary>conv.0 (1 → C channels, 3×3, stride 2, pad 1) + ReLU for output rows [<paramref name="t0"/>, <paramref name="t1"/>)
    /// of the <c>[h, w]</c> input image; row t is written at <c>y + (t - t0)·wo·C</c>.</summary>
    private void Conv0(float* x, int h, int w, int t0, int t1, float* y)
    {
        int wo = Half(w), C = _c;
        float* wt = _w0.Ptr, bias = _b0.Ptr;
        float* tap = stackalloc float[9];
        for (int t = t0; t < t1; t++)
            for (int f = 0; f < wo; f++)
            {
                for (int dy = 0; dy < 3; dy++)
                    for (int dx = 0; dx < 3; dx++)
                    {
                        int r = 2 * t + dy - 1, c = 2 * f + dx - 1;
                        tap[dy * 3 + dx] = r >= 0 && r < h && c >= 0 && c < w ? x[(long)r * w + c] : 0f;
                    }
                float* o = y + ((long)(t - t0) * wo + f) * C;
                int ch = 0;
                if (Avx512F.IsSupported)
                {
                    var k0 = Vector512.Create(tap[0]); var k1 = Vector512.Create(tap[1]); var k2 = Vector512.Create(tap[2]);
                    var k3 = Vector512.Create(tap[3]); var k4 = Vector512.Create(tap[4]); var k5 = Vector512.Create(tap[5]);
                    var k6 = Vector512.Create(tap[6]); var k7 = Vector512.Create(tap[7]); var k8 = Vector512.Create(tap[8]);
                    for (; ch + 16 <= C; ch += 16)
                    {
                        var acc = Vector512.Load(bias + ch);
                        acc = Avx512F.FusedMultiplyAdd(k0, Vector512.Load(wt + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k1, Vector512.Load(wt + C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k2, Vector512.Load(wt + 2 * C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k3, Vector512.Load(wt + 3 * C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k4, Vector512.Load(wt + 4 * C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k5, Vector512.Load(wt + 5 * C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k6, Vector512.Load(wt + 6 * C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k7, Vector512.Load(wt + 7 * C + ch), acc);
                        acc = Avx512F.FusedMultiplyAdd(k8, Vector512.Load(wt + 8 * C + ch), acc);
                        Vector512.Store(Vector512.Max(acc, Vector512<float>.Zero), o + ch);
                    }
                }
                for (; ch < C; ch++)
                {
                    float acc = bias[ch];
                    for (int k = 0; k < 9; k++) acc = MathF.FusedMultiplyAdd(tap[k], wt[k * C + ch], acc);
                    o[ch] = MathF.Max(acc, 0f);
                }
            }
    }

    /// <summary>Depthwise 3×3, stride 2, pad 1 over channels-last <c>[h, w, C]</c> for output rows [<paramref name="t0"/>, <paramref name="t1"/>).
    /// Input row r (global index) lives at <c>x + (r - xRow0)·w·C</c>; output row t at <c>y + (t - t0)·wo·C</c>.</summary>
    private void Depthwise(float* x, int xRow0, int h, int w, int t0, int t1, float* y, float* wt, float* bias)
    {
        int wo = Half(w), C = _c;
        for (int t = t0; t < t1; t++)
            for (int f = 0; f < wo; f++)
            {
                float* o = y + ((long)(t - t0) * wo + f) * C;
                int ch = 0;
                if (Avx512F.IsSupported)
                {
                    for (; ch + 16 <= C; ch += 16)
                    {
                        var acc = Vector512.Load(bias + ch);
                        for (int dy = 0; dy < 3; dy++)
                        {
                            int r = 2 * t + dy - 1;
                            if (r < 0 || r >= h) continue;
                            float* row = x + (long)(r - xRow0) * w * C;
                            for (int dx = 0; dx < 3; dx++)
                            {
                                int c = 2 * f + dx - 1;
                                if (c < 0 || c >= w) continue;
                                acc = Avx512F.FusedMultiplyAdd(Vector512.Load(wt + (dy * 3 + dx) * C + ch), Vector512.Load(row + (long)c * C + ch), acc);
                            }
                        }
                        Vector512.Store(acc, o + ch);
                    }
                }
                for (; ch < C; ch++)
                {
                    float acc = bias[ch];
                    for (int dy = 0; dy < 3; dy++)
                    {
                        int r = 2 * t + dy - 1;
                        if (r < 0 || r >= h) continue;
                        for (int dx = 0; dx < 3; dx++)
                        {
                            int c = 2 * f + dx - 1;
                            if (c < 0 || c >= w) continue;
                            acc = MathF.FusedMultiplyAdd(wt[(dy * 3 + dx) * C + ch], x[((long)(r - xRow0) * w + c) * C + ch], acc);
                        }
                    }
                    o[ch] = acc;
                }
            }
    }

    /// <summary>Reorders a <c>[C, 1, 3, 3]</c> kernel to tap-major <c>[9, C]</c>.</summary>
    private static NativeBuffer<float> TapMajor(ReadOnlySpan<float> w)
    {
        int c = w.Length / 9;
        var buf = new NativeBuffer<float>(w.Length);
        for (int ch = 0; ch < c; ch++)
            for (int k = 0; k < 9; k++) buf.Ptr[k * c + ch] = w[ch * 9 + k];
        return buf;
    }

    public void Dispose()
    {
        _w0.Dispose(); _b0.Dispose(); _out.Dispose(); _outB.Dispose();
        foreach (var x in _dw) x.Dispose();
        foreach (var x in _dwb) x.Dispose();
        foreach (var x in _pw) x.Dispose();
        foreach (var x in _pwb) x.Dispose();
    }
}
