using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Parakeet.Numerics;

namespace Parakeet.Model;

/// <summary>
/// Transformer-XL style multi-head self-attention with relative positional encodings (NeMo
/// <c>RelPositionMultiHeadAttention</c>, untied biases):
/// <code>
/// score[i, j] = ((q_i + u) · k_j + (q_i + v) · p_{i-j}) / √d_k
/// </code>
/// where <c>p_r</c> is the projected sinusoidal encoding of relative distance <c>r</c>. NeMo materializes the full
/// <c>(q + v) · Pᵀ</c> matrix over all 2T-1 distances and "rel-shifts" it; here each block of query rows multiplies only
/// the band of distances it needs, so memory stays O(T · block) and no shift copy is made. Heads × row blocks run in
/// parallel, each block doing score GEMMs, softmax and the value GEMM back to back while its rows are cache-hot.
/// </summary>
internal static unsafe class RelPositionAttention
{
    private const int RowBlock = 64;
    private const int NR = PackedMatrix.NR;

    [ThreadStatic] private static NativeBuffer<float>? t_scratch;

    /// <summary>
    /// <paramref name="qkv"/>: <c>[T, 3·d]</c> projected queries, keys and values; <paramref name="p"/>: <c>[2T-1, d]</c>
    /// projected positional encodings (row k ↔ distance T-1-k); writes <c>[T, d]</c> head-concatenated context.
    /// </summary>
    public static void Forward(float* qkv, float* p, int t, int dModel, int heads, float* biasU, float* biasV,
                               float* ctx, float* shared, Parallelism par)
    {
        int dk = dModel / heads;
        long ld = 3L * dModel;
        long kSize = PackedMatrix.SizeOf(dk, t), vSize = PackedMatrix.SizeOf(t, dk), pSize = PackedMatrix.SizeOf(dk, 2 * t - 1);
        long perHead = kSize + vSize + pSize;
        // Pack K (as the Wᵀ of a [T, d_k] "weight"), V ([T, d_k] row-major) and P per head.
        par.For(heads * 3, item =>
        {
            int h = item / 3, which = item % 3;
            float* basePtr = shared + h * perHead;
            if (which == 0) PackedMatrix.PackLinear(qkv + dModel + h * dk, t, dk, ld, basePtr);
            else if (which == 1) PackedMatrix.PackKN(qkv + 2 * dModel + h * dk, t, dk, ld, basePtr + kSize);
            else PackedMatrix.PackLinear(p + h * dk, 2 * t - 1, dk, dModel, basePtr + kSize + vSize);
        });

        Profiler.Mark("attention pack");
        int blocks = (t + RowBlock - 1) / RowBlock;
        float scale = 1f / MathF.Sqrt(dk);
        par.For(heads * blocks, item =>
        {
            int h = item / blocks, blk = item % blocks;
            int i0 = blk * RowBlock, i1 = Math.Min(t, i0 + RowBlock), rows = i1 - i0;
            float* basePtr = shared + h * perHead;
            var kMat = new PackedMatrix(basePtr, dk, t);
            var vMat = new PackedMatrix(basePtr + kSize, t, dk);
            var pMat = new PackedMatrix(basePtr + kSize + vSize, dk, 2 * t - 1);

            // Distances needed by rows [i0, i1): p index T-1-i+j ∈ [T-i1, 2T-2-i0], rounded out to whole panels.
            int pb0 = (t - i1) / NR, pb1 = Math.Min(pMat.Blocks, (2 * t - 2 - i0) / NR + 1);
            int bdCols = (pb1 - pb0) * NR;
            long need = 2L * rows * dk + (long)rows * t + (long)rows * bdCols;
            var buf = t_scratch;
            if (buf == null || buf.Length < need)
            {
                buf?.Dispose();
                buf = t_scratch = new NativeBuffer<float>(Math.Max(need, 1 << 16), zero: false);
            }
            float* qu = buf.Ptr, qv = qu + rows * dk, s = qv + rows * dk, bd = s + (long)rows * t;

            // (q + u) / √d_k and (q + v) / √d_k for this block.
            for (int r = 0; r < rows; r++)
            {
                float* q = qkv + (i0 + r) * ld + h * dk;
                float* u = biasU + h * dk, v = biasV + h * dk;
                for (int c = 0; c < dk; c++)
                {
                    qu[r * dk + c] = (q[c] + u[c]) * scale;
                    qv[r * dk + c] = (q[c] + v[c]) * scale;
                }
            }
            Gemm.Multiply(qu, dk, rows, kMat, s, t, Parallelism.Single);
            Gemm.Multiply(qv, dk, rows, pMat, pb0, pb1, bd, bdCols, Parallelism.Single);
            for (int r = 0; r < rows; r++)
            {
                int i = i0 + r;
                float* srow = s + (long)r * t;
                float* brow = bd + (long)r * bdCols + (t - 1 - i - pb0 * NR);
                int j = 0;
                if (Avx512F.IsSupported)
                    for (; j + 16 <= t; j += 16) Vector512.Store(Vector512.Load(srow + j) + Vector512.Load(brow + j), srow + j);
                for (; j < t; j++) srow[j] += brow[j];
                Ops.SoftmaxInPlace(srow, t);
            }
            Gemm.Multiply(s, t, rows, vMat, ctx + (long)i0 * dModel + h * dk, dModel, Parallelism.Single);
        });
    }

    /// <summary>Floats of shared scratch for <see cref="Forward"/>.</summary>
    public static long ScratchSize(int t, int dModel, int heads)
    {
        int dk = dModel / heads;
        return heads * (PackedMatrix.SizeOf(dk, t) + PackedMatrix.SizeOf(t, dk) + PackedMatrix.SizeOf(dk, 2 * t - 1));
    }

    /// <summary>Relative sinusoidal encodings <c>[2T-1, d]</c>, row k for distance T-1-k (NeMo <c>RelPositionalEncoding</c>),
    /// computed in float32 like the PyTorch reference.</summary>
    public static void PositionalEncoding(int t, int dModel, float* pe)
    {
        float neg = (float)(-Math.Log(10000.0) / dModel);
        var div = new float[dModel / 2];
        for (int i = 0; i < div.Length; i++) div[i] = MathF.Exp((2 * i) * neg);
        Parallel.For(0, 2 * t - 1, k =>
        {
            float pos = t - 1 - k;
            float* row = pe + (long)k * dModel;
            for (int i = 0; i < div.Length; i++)
            {
                float arg = pos * div[i];
                row[2 * i] = (float)Math.Sin(arg);
                row[2 * i + 1] = (float)Math.Cos(arg);
            }
        });
    }
}
