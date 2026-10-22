namespace Parakeet.Numerics;

/// <summary>
/// The right-hand operand <c>B[K, N]</c> of a GEMM, pre-packed into column panels of <see cref="NR"/> = 32 outputs,
/// split along K into blocks of <see cref="KBlock"/> rows and stored k-block-major: block <c>kb</c> holds, for every
/// panel in turn, its <c>kc × 32</c> tile (<c>kc = min(KBlock, K - kb·KBlock)</c>), with the last panel zero-padded.
/// A thread sweeping consecutive panels of one k-block therefore reads one contiguous stream, which keeps the
/// hardware and software prefetchers effective when the weights come from DRAM (each weight is used once per pass).
/// A linear layer with PyTorch weight <c>W[out, in]</c> is <c>B = Wᵀ</c> (K = in, N = out). Weights are reordered
/// once at load time and never again.
/// A matrix either owns its buffer or is a view over caller-provided scratch memory (attention packs K/V per head).
/// </summary>
public sealed unsafe class PackedMatrix : IDisposable
{
    public const int NR = 32;

    /// <summary>K rows per block (the GEMM's KC).</summary>
    public const int KBlock = 256;

    private readonly NativeBuffer<float>? _owned;
    private readonly float* _data;

    private PackedMatrix(int k, int n, NativeBuffer<float> owned)
    {
        K = k;
        N = n;
        _owned = owned;
        _data = owned.Ptr;
    }

    /// <summary>A non-owning view over <see cref="SizeOf"/>(k, n) floats already in panel layout.</summary>
    public PackedMatrix(float* data, int k, int n)
    {
        K = k;
        N = n;
        _data = data;
    }

    public int K { get; }
    public int N { get; }
    public int Blocks => (N + NR - 1) / NR;
    internal float* Data => _data;

    /// <summary>The <c>kc × 32</c> tile of panel <paramref name="nb"/> for the k-block starting at <paramref name="k0"/>.</summary>
    internal float* Panel(int nb, int k0) => _data + (long)k0 * Blocks * NR + (long)nb * Math.Min(KBlock, K - k0) * NR;

    /// <summary>Offset of element (k, n) in a packed <c>k × n</c> matrix.</summary>
    private static long Offset(int kk, int col, int k, int n)
    {
        int k0 = kk / KBlock * KBlock;
        int kc = Math.Min(KBlock, k - k0);
        int blocks = (n + NR - 1) / NR;
        return (long)k0 * blocks * NR + ((long)(col / NR) * kc + (kk - k0)) * NR + col % NR;
    }
    public long ByteSize => (long)Blocks * K * NR * sizeof(float);

    /// <summary>Number of floats needed to pack a <c>k × n</c> matrix.</summary>
    public static long SizeOf(int k, int n) => (long)((n + NR - 1) / NR) * k * NR;

    /// <summary>Packs a PyTorch linear weight <c>W[n, k]</c> (row-major, rows <paramref name="ldw"/> apart), scaled by
    /// <paramref name="scale"/> (used to fold constant output scales such as the macaron ½ into the weight).</summary>
    public static PackedMatrix FromLinear(float* w, int n, int k, long ldw = -1, float scale = 1f, bool hugePages = false)
    {
        if (ldw < 0) ldw = k;
        var buf = new NativeBuffer<float>(SizeOf(k, n), zero: false, hugePages);
        PackLinear(w, n, k, ldw, buf.Ptr, scale);
        return new PackedMatrix(k, n, buf);
    }

    /// <summary>Packs a row-major matrix <c>B[k, n]</c> (rows <paramref name="ldb"/> apart), e.g. an ONNX MatMul weight.</summary>
    public static PackedMatrix FromKN(float* b, int k, int n, long ldb = -1, float scale = 1f, bool hugePages = false)
    {
        if (ldb < 0) ldb = n;
        var buf = new NativeBuffer<float>(SizeOf(k, n), zero: false, hugePages);
        PackKN(b, k, n, ldb, buf.Ptr, scale);
        return new PackedMatrix(k, n, buf);
    }

    /// <summary>Packs <c>W[n, k]</c> (B = Wᵀ) into <paramref name="dst"/> (<see cref="SizeOf"/> floats).</summary>
    public static void PackLinear(float* w, int n, int k, long ldw, float* dst, float scale = 1f)
    {
        int blocks = (n + NR - 1) / NR;
        if (n % NR != 0) new Span<float>(dst, checked((int)SizeOf(k, n))).Clear();
        for (int k0 = 0; k0 < k; k0 += KBlock)
        {
            int kc = Math.Min(KBlock, k - k0);
            for (int nb = 0; nb < blocks; nb++)
            {
                float* d = dst + (long)k0 * blocks * NR + (long)nb * kc * NR;
                int cols = Math.Min(NR, n - nb * NR);
                for (int j = 0; j < cols; j++)
                {
                    float* row = w + (long)(nb * NR + j) * ldw + k0;
                    for (int kk = 0; kk < kc; kk++) d[kk * NR + j] = row[kk] * scale;
                }
            }
        }
    }

    /// <summary>Packs a row-major <c>B[k, n]</c> into <paramref name="dst"/> (<see cref="SizeOf"/> floats).</summary>
    public static void PackKN(float* b, int k, int n, long ldb, float* dst, float scale = 1f)
    {
        int blocks = (n + NR - 1) / NR;
        if (n % NR != 0) new Span<float>(dst, checked((int)SizeOf(k, n))).Clear();
        for (int k0 = 0; k0 < k; k0 += KBlock)
        {
            int kc = Math.Min(KBlock, k - k0);
            for (int nb = 0; nb < blocks; nb++)
            {
                float* d = dst + (long)k0 * blocks * NR + (long)nb * kc * NR;
                int cols = Math.Min(NR, n - nb * NR);
                for (int kk = 0; kk < kc; kk++)
                {
                    float* src = b + (k0 + kk) * ldb + nb * NR;
                    if (cols == NR && scale == 1f)
                        Buffer.MemoryCopy(src, d + kk * NR, NR * sizeof(float), NR * sizeof(float));
                    else
                        for (int j = 0; j < cols; j++) d[kk * NR + j] = src[j] * scale;
                }
            }
        }
    }

    /// <summary>Unpacks back to a row-major <c>B[K, N]</c> (tests).</summary>
    public float[] ToKN()
    {
        var r = new float[(long)K * N];
        for (int kk = 0; kk < K; kk++)
            for (int j = 0; j < N; j++) r[(long)kk * N + j] = _data[Offset(kk, j, K, N)];
        return r;
    }

    public void Dispose() => _owned?.Dispose();
}
