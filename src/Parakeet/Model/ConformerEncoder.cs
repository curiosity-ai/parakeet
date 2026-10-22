using Parakeet.Numerics;

namespace Parakeet.Model;

/// <summary>Receives intermediate activations (row-major <c>[rows, cols]</c>) for layer-by-layer validation.</summary>
public unsafe delegate void TraceCallback(string name, float* data, int rows, int cols);

/// <summary>One FastConformer block (macaron feed-forwards around self-attention and convolution).</summary>
internal sealed unsafe class ConformerLayer : IDisposable
{
    public NativeBuffer<float> NormFf1G = null!, NormFf1B = null!, NormAttG = null!, NormAttB = null!;
    public NativeBuffer<float> NormConvG = null!, NormConvB = null!, NormFf2G = null!, NormFf2B = null!;
    public NativeBuffer<float> NormOutG = null!, NormOutB = null!;
    public LinearWeight Ff1W1 = null!, Ff1W2 = null!, Ff2W1 = null!, Ff2W2 = null!;   // W2 carry the macaron ½
    public NativeBuffer<float>? Ff1B1, Ff1B2, Ff2B1, Ff2B2;
    public LinearWeight Qkv = null!, Out = null!;
    public PackedMatrix Pos = null!;                                                  // cached per length, kept FP32
    public NativeBuffer<float>? QkvB, OutB;
    public NativeBuffer<float> BiasU = null!, BiasV = null!;
    public LinearWeight Pw1 = null!, Pw2 = null!;
    public NativeBuffer<float>? Pw1B, Pw2B;
    public NativeBuffer<float> DwW = null!, DwB = null!;                              // [kernel, d] with batch norm folded in

    public static ConformerLayer Load(WeightReader w, ParakeetConfig cfg, int index)
    {
        string p = $"encoder.layers.{index}.";
        int d = cfg.DModel, ff = cfg.FfHidden, k = cfg.ConvKernel;
        var l = new ConformerLayer
        {
            NormFf1G = w.Vector(p + "norm_feed_forward1.weight", d), NormFf1B = w.Vector(p + "norm_feed_forward1.bias", d),
            NormAttG = w.Vector(p + "norm_self_att.weight", d), NormAttB = w.Vector(p + "norm_self_att.bias", d),
            NormConvG = w.Vector(p + "norm_conv.weight", d), NormConvB = w.Vector(p + "norm_conv.bias", d),
            NormFf2G = w.Vector(p + "norm_feed_forward2.weight", d), NormFf2B = w.Vector(p + "norm_feed_forward2.bias", d),
            NormOutG = w.Vector(p + "norm_out.weight", d), NormOutB = w.Vector(p + "norm_out.bias", d),
            Ff1W1 = w.Projection(p + "feed_forward1.linear1.weight", ff, d),
            Ff1W2 = w.Projection(p + "feed_forward1.linear2.weight", d, ff, 0.5f),
            Ff1B1 = w.OptionalVector(p + "feed_forward1.linear1.bias", ff),
            Ff1B2 = w.OptionalVector(p + "feed_forward1.linear2.bias", d, 0.5f),
            Ff2W1 = w.Projection(p + "feed_forward2.linear1.weight", ff, d),
            Ff2W2 = w.Projection(p + "feed_forward2.linear2.weight", d, ff, 0.5f),
            Ff2B1 = w.OptionalVector(p + "feed_forward2.linear1.bias", ff),
            Ff2B2 = w.OptionalVector(p + "feed_forward2.linear2.bias", d, 0.5f),
            Qkv = w.FusedProjection(d, (p + "self_attn.linear_q.weight", d), (p + "self_attn.linear_k.weight", d), (p + "self_attn.linear_v.weight", d)),
            QkvB = w.FusedBias((p + "self_attn.linear_q.bias", d), (p + "self_attn.linear_k.bias", d), (p + "self_attn.linear_v.bias", d)),
            Pos = w.Linear(p + "self_attn.linear_pos.weight", d, d),
            Out = w.Projection(p + "self_attn.linear_out.weight", d, d),
            OutB = w.OptionalVector(p + "self_attn.linear_out.bias", d),
            BiasU = w.Vector(p + "self_attn.pos_bias_u", d),
            BiasV = w.Vector(p + "self_attn.pos_bias_v", d),
            Pw1 = w.Projection(p + "conv.pointwise_conv1.weight", 2 * d, d),
            Pw1B = w.OptionalVector(p + "conv.pointwise_conv1.bias", 2 * d),
            Pw2 = w.Projection(p + "conv.pointwise_conv2.weight", d, d),
            Pw2B = w.OptionalVector(p + "conv.pointwise_conv2.bias", d),
        };
        (l.DwW, l.DwB) = LoadDepthwise(w, p + "conv.", d, k, cfg.BatchNormEps);
        return l;
    }

    /// <summary>Depthwise conv weights as tap-major <c>[k, d]</c> with the inference-mode batch norm folded in:
    /// <c>w' = w·γ/√(σ²+ε)</c>, <c>b' = (b - μ)·γ/√(σ²+ε) + β</c>. ONNX exports ship it already folded (no batch_norm.*).</summary>
    private static (NativeBuffer<float>, NativeBuffer<float>) LoadDepthwise(WeightReader w, string p, int d, int k, float eps)
    {
        var wt = w.Get(p + "depthwise_conv.weight", d, 1, k).Span;
        var bias = w.Has(p + "depthwise_conv.bias") ? w.Get(p + "depthwise_conv.bias", d).ToArray() : new float[d];
        var scale = new float[d];
        var shift = new float[d];
        if (w.Has(p + "batch_norm.weight"))
        {
            var g = w.Get(p + "batch_norm.weight", d).Span;
            var b = w.Get(p + "batch_norm.bias", d).Span;
            var mean = w.Get(p + "batch_norm.running_mean", d).Span;
            var var = w.Get(p + "batch_norm.running_var", d).Span;
            for (int c = 0; c < d; c++)
            {
                float s = g[c] / MathF.Sqrt(var[c] + eps);
                scale[c] = s;
                shift[c] = (bias[c] - mean[c]) * s + b[c];
            }
        }
        else
        {
            Array.Fill(scale, 1f);
            Array.Copy(bias, shift, d);
        }
        var dw = new NativeBuffer<float>((long)k * d);
        var db = new NativeBuffer<float>(d);
        for (int c = 0; c < d; c++)
        {
            for (int t = 0; t < k; t++) dw.Ptr[t * d + c] = wt[c * k + t] * scale[c];
            db.Ptr[c] = shift[c];
        }
        return (dw, db);
    }

    public void Dispose()
    {
        foreach (var f in GetType().GetFields())
            if (f.GetValue(this) is IDisposable x) x.Dispose();
    }
}

/// <summary>
/// FastConformer encoder: dw-striding subsampling (8×), then <c>n_layers</c> conformer blocks over <c>[T, d_model]</c>
/// time-major activations. All projections are GEMMs against pre-packed weights with fused epilogues (bias, SiLU,
/// residual add, macaron ½ folded into the weights); attention uses <see cref="RelPositionAttention"/>.
/// </summary>
public sealed unsafe class ConformerEncoder : IDisposable
{
    private readonly ParakeetConfig _cfg;
    private readonly Subsampling _sub;
    private readonly ConformerLayer[] _layers;

    internal ConformerEncoder(WeightReader w, ParakeetConfig cfg, int positionCacheFrames = 0)
    {
        _cfg = cfg;
        _positionCacheFrames = positionCacheFrames;
        _sub = new Subsampling(w, cfg);
        _layers = new ConformerLayer[cfg.Layers];
        try
        {
            // Layers are independent: read and pack them concurrently (the checkpoint is memory-mapped, so this is
            // mostly memory-bound transposition plus page faults, which scale with cores).
            Parallel.For(0, cfg.Layers, i => _layers[i] = ConformerLayer.Load(w, cfg, i));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public ParakeetConfig Config => _cfg;

    /// <summary>Encodes <c>[melFrames, features]</c> features into <c>[EncoderFrames(melFrames), d_model]</c>.</summary>
    public float[] Forward(ReadOnlySpan<float> features, int melFrames, Parallelism par, TraceCallback? trace = null)
    {
        int t = _cfg.EncoderFrames(melFrames);
        var result = new float[(long)t * _cfg.DModel];
        var ws = RentWorkspace(t, _sub.ScratchSize(melFrames));
        try
        {
            fixed (float* f = features)
            fixed (float* r = result)
            {
                Forward(f, melFrames, r, ws, par, trace);
            }
        }
        finally
        {
            ReturnWorkspace(ws);
        }
        return result;
    }

    internal void Forward(float* features, int melFrames, float* output, EncoderWorkspace ws, Parallelism par, TraceCallback? trace)
    {
        int t = _cfg.EncoderFrames(melFrames), d = _cfg.DModel, ff = _cfg.FfHidden;
        float eps = _cfg.LayerNormEps;
        float* x = output;     // the residual stream lives in the output buffer
        Profiler.Start();
        _sub.Forward(features, melFrames, x, ws.Sub.Ptr, par);
        Profiler.Mark("subsampling");
        if (_cfg.XScaling)
        {
            float xs = MathF.Sqrt(d);
            for (long i = 0; i < (long)t * d; i++) x[i] *= xs;
        }
        trace?.Invoke("pre_encode", x, t, d);
        // Relative positional encodings and their per-layer projections depend only on T: take them from the cache
        // when it covers T (a slice of the table built for the longest input so far), else compute them here.
        var table = Positions(t, par);
        long posOff = table == null ? 0 : (long)(table.Frames - t) * d;
        float* pe = table == null ? ws.PosEmb.Ptr : table.Pe.Ptr + posOff;
        if (table == null) RelPositionAttention.PositionalEncoding(t, d, pe);
        Profiler.Mark("pos_enc");
        trace?.Invoke("pos_emb", pe, 2 * t - 1, d);

        float* h = ws.Norm.Ptr, big = ws.Big.Ptr, ctx = ws.Ctx.Ptr;
        for (int li = 0; li < _layers.Length; li++)
        {
            var L = _layers[li];
            bool tr0 = trace != null && li == 0;

            // ½ · FF1
            Ops.LayerNorm(x, h, t, d, L.NormFf1G.Ptr, L.NormFf1B.Ptr, eps, par);
            Profiler.Mark("layernorm");
            if (tr0) trace!("layer0.norm_ff1", h, t, d);
            Profiler.Mark("trace");
            L.Ff1W1.Multiply(h, d, t, big, ff, par, Ptr(L.Ff1B1), act: Activation.Silu);
            L.Ff1W2.Multiply(big, ff, t, x, d, par, Ptr(L.Ff1B2), accumulate: true);
            Profiler.Mark("ff gemm");
            if (tr0) trace!("layer0.ff1", x, t, d);

            // Relative-position multi-head self-attention
            Ops.LayerNorm(x, h, t, d, L.NormAttG.Ptr, L.NormAttB.Ptr, eps, par);
            Profiler.Mark("layernorm");
            L.Qkv.Multiply(h, d, t, big, 3 * d, par, Ptr(L.QkvB));
            Profiler.Mark("qkv gemm");
            float* pp = ws.P.Ptr;
            if (table == null) Gemm.Multiply(pe, d, 2 * t - 1, L.Pos, pp, d, par);
            else pp = table.P[li].Ptr + posOff;
            Profiler.Mark("pos gemm");
            if (tr0)
            {
                using var q = new NativeBuffer<float>((long)t * d, zero: false);
                for (int r = 0; r < t; r++) Buffer.MemoryCopy(big + (long)r * 3 * d, q.Ptr + (long)r * d, d * 4, d * 4);
                trace!("layer0.q", q.Ptr, t, d);
                trace!("layer0.p", pp, 2 * t - 1, d);
            }
            Profiler.Mark("trace");
            RelPositionAttention.Forward(big, pp, t, d, _cfg.Heads, L.BiasU.Ptr, L.BiasV.Ptr, ctx, ws.Attn.Ptr, par);
            Profiler.Mark("attention");
            L.Out.Multiply(ctx, d, t, x, d, par, Ptr(L.OutB), accumulate: true);
            Profiler.Mark("out gemm");
            if (tr0) trace!("layer0.attn", x, t, d);

            // Convolution module: pointwise → GLU → depthwise (+BN) → SiLU → pointwise
            Ops.LayerNorm(x, h, t, d, L.NormConvG.Ptr, L.NormConvB.Ptr, eps, par);
            Profiler.Mark("layernorm");
            L.Pw1.Multiply(h, d, t, big, 2 * d, par, Ptr(L.Pw1B));
            Profiler.Mark("conv gemm");
            Ops.Glu(big, ctx, t, d, par);
            Profiler.Mark("glu");
            if (tr0) trace!("layer0.glu", ctx, t, d);
            Profiler.Mark("trace");
            Ops.DepthwiseConv1d(ctx, h, t, d, L.DwW.Ptr, _cfg.ConvKernel, L.DwB.Ptr, silu: true, par);
            Profiler.Mark("depthwise");
            L.Pw2.Multiply(h, d, t, x, d, par, Ptr(L.Pw2B), accumulate: true);
            Profiler.Mark("conv gemm");
            if (tr0) trace!("layer0.conv", x, t, d);

            // ½ · FF2
            Ops.LayerNorm(x, h, t, d, L.NormFf2G.Ptr, L.NormFf2B.Ptr, eps, par);
            Profiler.Mark("layernorm");
            L.Ff2W1.Multiply(h, d, t, big, ff, par, Ptr(L.Ff2B1), act: Activation.Silu);
            L.Ff2W2.Multiply(big, ff, t, x, d, par, Ptr(L.Ff2B2), accumulate: true);
            Profiler.Mark("ff gemm");
            if (tr0) trace!("layer0.ff2", x, t, d);

            Ops.LayerNorm(x, x, t, d, L.NormOutG.Ptr, L.NormOutB.Ptr, eps, par);
            Profiler.Mark("layernorm");
            trace?.Invoke($"layer{li}", x, t, d);
        }
        GC.KeepAlive(table);
    }

    /// <summary>Encodings <c>[2F-1, d]</c> and per-layer projections <c>PE · W_posᵀ</c> for up to F frames. A T-frame input
    /// uses rows [F - T, F + T - 1): row k holds distance F-1-k, and every row is computed independently, so the slice is
    /// exactly what a T-frame computation would produce.</summary>
    private sealed class PositionTable
    {
        public required int Frames { get; init; }
        public required NativeBuffer<float> Pe { get; init; }
        public required NativeBuffer<float>[] P { get; init; }
    }

    private readonly int _positionCacheFrames;
    private readonly object _positionLock = new();
    private volatile PositionTable? _positions;

    private PositionTable? Positions(int t, Parallelism par)
    {
        var table = _positions;
        if (table != null && table.Frames >= t) return table;
        if (t > _positionCacheFrames) return null;
        lock (_positionLock)
        {
            table = _positions;
            if (table != null && table.Frames >= t) return table;
            // Grow geometrically (rounded to 64 frames) so a sequence of longer inputs rebuilds only a few times.
            int frames = Math.Min(_positionCacheFrames, Math.Max((t + 63) / 64 * 64, 2 * (table?.Frames ?? 0)));
            int d = _cfg.DModel;
            long rows = 2L * frames - 1;
            var pe = new NativeBuffer<float>(rows * d, zero: false);
            RelPositionAttention.PositionalEncoding(frames, d, pe.Ptr);
            var p = new NativeBuffer<float>[_layers.Length];
            for (int li = 0; li < _layers.Length; li++)
            {
                p[li] = new NativeBuffer<float>(rows * d, zero: false);
                Gemm.Multiply(pe.Ptr, d, (int)rows, _layers[li].Pos, p[li].Ptr, d, par);
            }
            // The previous table is released by its finalizers once no running forward pass references it.
            _positions = table = new PositionTable { Frames = frames, Pe = pe, P = p };
            return table;
        }
    }

    private static float* Ptr(NativeBuffer<float>? b) => b == null ? null : b.Ptr;

    // Activation buffers are pooled: allocating them per call costs a page fault and kernel zeroing per page
    // (hundreds of MB for long audio). Each concurrent caller rents its own workspace; capacities grow in 64-frame steps.
    private readonly System.Collections.Concurrent.ConcurrentBag<EncoderWorkspace> _pool = new();

    private EncoderWorkspace RentWorkspace(int frames, long subScratch)
    {
        while (_pool.TryTake(out var ws))
        {
            if (ws.Frames >= frames && ws.Sub.Length >= subScratch) return ws;
            ws.Dispose();
        }
        int cap = (frames + 63) / 64 * 64;
        return new EncoderWorkspace(_cfg, cap, Math.Max(subScratch, (long)cap * _cfg.Mel.Features / 8 * _cfg.SubsamplingChannels));
    }

    private void ReturnWorkspace(EncoderWorkspace ws)
    {
        if (_pool.Count < Environment.ProcessorCount) _pool.Add(ws);
        else ws.Dispose();
    }

    /// <summary>Releases pooled activation buffers (they are re-created on demand).</summary>
    public void TrimWorkspaces()
    {
        while (_pool.TryTake(out var ws)) ws.Dispose();
    }

    public void Dispose()
    {
        TrimWorkspaces();
        if (_positions is { } pt)
        {
            pt.Pe.Dispose();
            foreach (var b in pt.P) b.Dispose();
            _positions = null;
        }
        _sub?.Dispose();
        if (_layers != null)
            foreach (var l in _layers) l?.Dispose();
    }
}

/// <summary>Activation buffers for one encoder pass over T frames.</summary>
internal sealed class EncoderWorkspace : IDisposable
{
    public EncoderWorkspace(ParakeetConfig cfg, int t, long subScratch)
    {
        int d = cfg.DModel;
        Frames = t;
        Sub = new NativeBuffer<float>(subScratch, zero: false);
        Norm = new NativeBuffer<float>((long)t * d, zero: false);
        Big = new NativeBuffer<float>((long)t * Math.Max(cfg.FfHidden, 3 * d), zero: false);
        Ctx = new NativeBuffer<float>((long)t * d, zero: false);
        PosEmb = new NativeBuffer<float>((2L * t - 1) * d, zero: false);
        P = new NativeBuffer<float>((2L * t - 1) * d, zero: false);
        Attn = new NativeBuffer<float>(RelPositionAttention.ScratchSize(t, d, cfg.Heads), zero: false);
    }

    public int Frames { get; }
    public NativeBuffer<float> Sub { get; }
    public NativeBuffer<float> Norm { get; }
    public NativeBuffer<float> Big { get; }
    public NativeBuffer<float> Ctx { get; }
    public NativeBuffer<float> PosEmb { get; }
    public NativeBuffer<float> P { get; }
    public NativeBuffer<float> Attn { get; }

    public void Dispose()
    {
        Sub.Dispose(); Norm.Dispose(); Big.Dispose(); Ctx.Dispose(); PosEmb.Dispose(); P.Dispose(); Attn.Dispose();
    }
}
