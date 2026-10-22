using Parakeet.Numerics;

namespace Parakeet.Model;

/// <summary>One emitted token: id, the encoder frame where it was emitted and its predicted duration (frames).</summary>
public readonly record struct DecodedToken(int Id, int Frame, int Duration, float LogitScore);

/// <summary>
/// Token-and-Duration Transducer decoder: an LSTM prediction network over the previous non-blank token, and a joint
/// network <c>W_out · relu(W_enc · enc_t + W_pred · g)</c> that predicts both the next token (or blank) and how many
/// encoder frames to skip. Greedy decoding follows NeMo's <c>GreedyTDTInfer</c>.
/// <para>
/// Unlike a per-step graph run (what the ONNX decoder_joint model does), the work is split by what actually changes:
/// the encoder half of the joint is one GEMM over all frames up front; the embedding lookup and the first LSTM
/// layer's input projection are folded into a precomputed <c>[vocab, 4H]</c> table; the prediction network only runs
/// after a non-blank token; and a blank or a skip only re-evaluates the joint's output layer.
/// </para>
/// </summary>
public sealed unsafe class TdtDecoder : IDisposable
{
    private readonly ParakeetConfig _cfg;
    private readonly int _h, _layers, _j, _outDim;
    private readonly NativeBuffer<float> _embedIn;           // [vocab + 1, 4H] = E · W_ih0ᵀ + b_ih0 + b_hh0
    private readonly PackedMatrix _whh0;                     // layer 0 recurrent weights (K = H, N = 4H)
    private readonly PackedMatrix[] _wcat;                   // layers ≥ 1: [W_ih | W_hh] (K = 2H, N = 4H)
    private readonly NativeBuffer<float>[] _bcat;
    private readonly PackedMatrix _jointEnc, _jointPred, _jointOut;
    private readonly NativeBuffer<float> _jointEncB, _jointPredB, _jointOutB;

    internal TdtDecoder(WeightReader w, ParakeetConfig cfg)
    {
        _cfg = cfg;
        _h = cfg.PredHidden;
        _layers = cfg.PredLayers;
        _j = cfg.JointHidden;
        _outDim = cfg.VocabSize + 1 + cfg.Durations.Length;
        int h = _h, g = 4 * h, vocab = cfg.VocabSize + 1;
        const string p = "decoder.prediction.";

        // Embedding folded into the first LSTM layer's input projection.
        var embed = w.Get(p + "embed.weight", vocab, h).Span;
        var bih0 = w.Get(p + "dec_rnn.lstm.bias_ih_l0", g).Span;
        var bhh0 = w.Get(p + "dec_rnn.lstm.bias_hh_l0", g).Span;
        _embedIn = new NativeBuffer<float>((long)vocab * g);
        using (var wPacked = w.Linear(p + "dec_rnn.lstm.weight_ih_l0", g, h))
        using (var bias = new NativeBuffer<float>(g))
        {
            for (int i = 0; i < g; i++) bias.Ptr[i] = bih0[i] + bhh0[i];
            using var par = new Parallelism(0);
            fixed (float* e = embed)
                Gemm.Multiply(e, h, vocab, wPacked, _embedIn.Ptr, g, par, bias.Ptr);
        }
        _whh0 = w.Linear(p + "dec_rnn.lstm.weight_hh_l0", g, h);
        _wcat = new PackedMatrix[Math.Max(0, _layers - 1)];
        _bcat = new NativeBuffer<float>[_wcat.Length];
        for (int l = 1; l < _layers; l++)
        {
            var wih = w.Get($"{p}dec_rnn.lstm.weight_ih_l{l}", g, h).Span;
            var whh = w.Get($"{p}dec_rnn.lstm.weight_hh_l{l}", g, h).Span;
            using var cat = new NativeBuffer<float>((long)g * 2 * h, zero: false);
            for (int r = 0; r < g; r++)
            {
                wih.Slice(r * h, h).CopyTo(new Span<float>(cat.Ptr + (long)r * 2 * h, h));
                whh.Slice(r * h, h).CopyTo(new Span<float>(cat.Ptr + (long)r * 2 * h + h, h));
            }
            _wcat[l - 1] = PackedMatrix.FromLinear(cat.Ptr, g, 2 * h);
            var bi = w.Get($"{p}dec_rnn.lstm.bias_ih_l{l}", g).Span;
            var bh = w.Get($"{p}dec_rnn.lstm.bias_hh_l{l}", g).Span;
            _bcat[l - 1] = new NativeBuffer<float>(g);
            for (int i = 0; i < g; i++) _bcat[l - 1].Ptr[i] = bi[i] + bh[i];
        }
        _jointEnc = w.Linear("joint.enc.weight", _j, cfg.DModel);
        _jointEncB = w.Vector("joint.enc.bias", _j);
        _jointPred = w.Linear("joint.pred.weight", _j, h);
        _jointPredB = w.Vector("joint.pred.bias", _j);
        _jointOut = w.Linear("joint.joint_net.2.weight", _outDim, _j);
        _jointOutB = w.Vector("joint.joint_net.2.bias", _outDim);
    }

    public int OutputSize => _outDim;

    /// <summary>
    /// Greedy TDT decoding of <c>[frames, d_model]</c> encoder output. <paramref name="onStep"/> (optional) receives the
    /// joint logits of every evaluation (validation); <paramref name="onToken"/> streams tokens as they are decoded.
    /// </summary>
    public List<DecodedToken> Decode(ReadOnlySpan<float> encoder, int frames, Parallelism par, CancellationToken ct = default,
                                     Action<DecodedToken>? onToken = null, StepCallback? onStep = null)
    {
        var tokens = new List<DecodedToken>();
        if (frames <= 0) return tokens;
        int h = _h, g = 4 * h, blank = _cfg.BlankId, vocab = _cfg.VocabSize + 1, nDur = _cfg.Durations.Length;
        using var encProj = new NativeBuffer<float>((long)frames * _j, zero: false);
        using var state = new NativeBuffer<float>(2L * _layers * h);            // h and c per layer (zero init)
        using var next = new NativeBuffer<float>(2L * _layers * h);
        using var gates = new NativeBuffer<float>(g, zero: false);
        using var cat = new NativeBuffer<float>(2L * h, zero: false);
        using var predProj = new NativeBuffer<float>(_j, zero: false);
        using var hidden = new NativeBuffer<float>(_j, zero: false);
        using var logits = new NativeBuffer<float>(_outDim, zero: false);
        var single = Parallelism.Single;

        fixed (float* enc = encoder)
            Gemm.Multiply(enc, _cfg.DModel, frames, _jointEnc, encProj.Ptr, _j, par, _jointEncB.Ptr);

        // Prediction network on the start symbol (blank): its embedding row is all zeros (blank_as_pad).
        int last = blank;
        Predict(last, state.Ptr, next.Ptr, gates.Ptr, cat.Ptr);
        Gemm.Multiply(next.Ptr + (2L * (_layers - 1)) * h, h, 1, _jointPred, predProj.Ptr, _j, single, _jointPredB.Ptr);

        int t = 0;
        while (t < frames)
        {
            ct.ThrowIfCancellationRequested();
            int symbols = 0, skip;
            while (true)
            {
                // joint: relu(enc_t + pred) → logits
                float* e = encProj.Ptr + (long)t * _j;
                for (int i = 0; i < _j; i++) hidden.Ptr[i] = MathF.Max(e[i] + predProj.Ptr[i], 0f);
                Gemm.Multiply(hidden.Ptr, _j, 1, _jointOut, logits.Ptr, _outDim, single, _jointOutB.Ptr);
                onStep?.Invoke(t, logits.Ptr, _outDim);
                int k = Ops.ArgMax(logits.Ptr, vocab);
                skip = _cfg.Durations[Ops.ArgMax(logits.Ptr + vocab, nDur)];
                symbols++;
                if (k != blank)
                {
                    var tok = new DecodedToken(k, t, skip, logits.Ptr[k]);
                    tokens.Add(tok);
                    onToken?.Invoke(tok);
                    // Commit the state computed for `last` and advance the prediction network with the new token.
                    Buffer.MemoryCopy(next.Ptr, state.Ptr, 2L * _layers * h * 4, 2L * _layers * h * 4);
                    last = k;
                    Predict(last, state.Ptr, next.Ptr, gates.Ptr, cat.Ptr);
                    Gemm.Multiply(next.Ptr + (2L * (_layers - 1)) * h, h, 1, _jointPred, predProj.Ptr, _j, single, _jointPredB.Ptr);
                }
                t += skip;
                // A blank with duration 0 would re-evaluate the same inputs until max_symbols: stop right away.
                if (skip != 0 || k == blank || symbols >= _cfg.MaxSymbolsPerStep) break;
            }
            if (skip == 0) t++;
        }
        return tokens;
    }

    /// <summary>One prediction-network step: feeds <paramref name="token"/> with state <paramref name="s"/>
    /// (<c>[layer][h, c]</c>) and writes the new state to <paramref name="o"/>; the output is layer L-1's h.</summary>
    private void Predict(int token, float* s, float* o, float* gates, float* cat)
    {
        int h = _h, g = 4 * h;
        var single = Parallelism.Single;
        for (int l = 0; l < _layers; l++)
        {
            float* hPrev = s + 2L * l * h, cPrev = hPrev + h;
            float* hOut = o + 2L * l * h, cOut = hOut + h;
            if (l == 0)
            {
                Buffer.MemoryCopy(_embedIn.Ptr + (long)token * g, gates, g * 4, g * 4);
                Gemm.Multiply(hPrev, h, 1, _whh0, gates, g, single, accumulate: true);
            }
            else
            {
                Buffer.MemoryCopy(o + 2L * (l - 1) * h, cat, h * 4, h * 4);
                Buffer.MemoryCopy(hPrev, cat + h, h * 4, h * 4);
                Gemm.Multiply(cat, 2 * h, 1, _wcat[l - 1], gates, g, single, _bcat[l - 1].Ptr);
            }
            // PyTorch gate order: input, forget, cell, output.
            for (int i = 0; i < h; i++)
            {
                float ig = VMath.Sigmoid(gates[i]);
                float fg = VMath.Sigmoid(gates[h + i]);
                float cg = MathF.Tanh(gates[2 * h + i]);
                float og = VMath.Sigmoid(gates[3 * h + i]);
                float c = fg * cPrev[i] + ig * cg;
                cOut[i] = c;
                hOut[i] = og * MathF.Tanh(c);
            }
        }
    }

    public void Dispose()
    {
        _embedIn.Dispose();
        _whh0.Dispose();
        foreach (var m in _wcat) m.Dispose();
        foreach (var b in _bcat) b.Dispose();
        _jointEnc.Dispose(); _jointPred.Dispose(); _jointOut.Dispose();
        _jointEncB.Dispose(); _jointPredB.Dispose(); _jointOutB.Dispose();
    }
}

/// <summary>Receives the joint output of one decoder evaluation at encoder frame <paramref name="frame"/>.</summary>
public unsafe delegate void StepCallback(int frame, float* logits, int length);
