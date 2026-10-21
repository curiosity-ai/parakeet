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
    private readonly PackedMatrix _out;
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
        _out = PackedMatrix.FromLinear(perm.Ptr, _d, k);
        _outB = w.Vector(p + "out.bias", _d);
    }

    private int FreqAfter(int f)
    {
        for (int s = 0; s < _stages; s++) f = (f - 1) / 2 + 1;
        return f;
    }

    private static int Half(int n) => (n - 1) / 2 + 1;

    /// <summary>Floats of scratch needed for <paramref name="frames"/> feature frames (two ping-pong buffers).</summary>
    public long ScratchSize(int frames) => 2L * Half(frames) * Half(_fIn) * _c;

    /// <summary>Maps <c>[frames, features]</c> to <c>[EncoderFrames(frames), d_model]</c>.</summary>
    public void Forward(float* feats, int frames, float* output, float* scratch, Parallelism par)
    {
        int h = Half(frames), wdt = Half(_fIn);
        float* a = scratch, b = scratch + (long)h * wdt * _c;
        Conv0(feats, frames, _fIn, a, par);
        for (int s = 0; s < _stages - 1; s++)
        {
            int h2 = Half(h), w2 = Half(wdt);
            Depthwise(a, h, wdt, b, _dw[s].Ptr, _dwb[s].Ptr, par);
            Gemm.Multiply(b, _c, h2 * w2, _pw[s], a, _c, par, _pwb[s].Ptr, act: Activation.Relu);
            h = h2;
            wdt = w2;
        }
        Gemm.Multiply(a, (long)wdt * _c, h, _out, output, _d, par, _outB.Ptr);
    }

    /// <summary>conv.0 (1 → C channels, 3×3, stride 2, pad 1) + ReLU over the <c>[h, w]</c> input image.</summary>
    private void Conv0(float* x, int h, int w, float* y, Parallelism par)
    {
        int ho = Half(h), wo = Half(w), C = _c;
        float* wt = _w0.Ptr, bias = _b0.Ptr;
        par.ForRange(ho, 4, (t0, t1) =>
        {
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
                    float* o = y + ((long)t * wo + f) * C;
                    int ch = 0;
                    if (Avx512F.IsSupported)
                    {
                        for (; ch + 16 <= C; ch += 16)
                        {
                            var acc = Vector512.Load(bias + ch);
                            for (int k = 0; k < 9; k++) acc = Avx512F.FusedMultiplyAdd(Vector512.Create(tap[k]), Vector512.Load(wt + k * C + ch), acc);
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
        });
    }

    /// <summary>Depthwise 3×3, stride 2, pad 1 over channels-last <c>[h, w, C]</c>.</summary>
    private void Depthwise(float* x, int h, int w, float* y, float* wt, float* bias, Parallelism par)
    {
        int ho = Half(h), wo = Half(w), C = _c;
        par.ForRange(ho, 2, (t0, t1) =>
        {
            for (int t = t0; t < t1; t++)
                for (int f = 0; f < wo; f++)
                {
                    float* o = y + ((long)t * wo + f) * C;
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
                                for (int dx = 0; dx < 3; dx++)
                                {
                                    int c = 2 * f + dx - 1;
                                    if (c < 0 || c >= w) continue;
                                    acc = Avx512F.FusedMultiplyAdd(Vector512.Load(wt + (dy * 3 + dx) * C + ch),
                                                                   Vector512.Load(x + ((long)r * w + c) * C + ch), acc);
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
                                acc = MathF.FusedMultiplyAdd(wt[(dy * 3 + dx) * C + ch], x[((long)r * w + c) * C + ch], acc);
                            }
                        }
                        o[ch] = acc;
                    }
                }
        });
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
