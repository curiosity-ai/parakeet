namespace Parakeet.Numerics;

/// <summary>
/// The right-hand operand <c>B[K, N]</c> of a GEMM, pre-packed into column panels of <see cref="NR"/> = 32 outputs:
/// element <c>(k, n)</c> lives at <c>[(n / 32) * K + k] * 32 + n % 32</c>, with the last panel zero-padded.
/// A linear layer with PyTorch weight <c>W[out, in]</c> is <c>B = Wᵀ</c> (K = in, N = out). The panel is the exact
/// layout the GEMM micro-kernel streams, so weights are reordered once at load time and never again.
/// A matrix either owns its buffer or is a view over caller-provided scratch memory (attention packs K/V per head).
/// </summary>
public sealed unsafe class PackedMatrix : IDisposable
{
    public const int NR = 32;

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
    public long ByteSize => (long)Blocks * K * NR * sizeof(float);

    /// <summary>Number of floats needed to pack a <c>k × n</c> matrix.</summary>
    public static long SizeOf(int k, int n) => (long)((n + NR - 1) / NR) * k * NR;

    /// <summary>Packs a PyTorch linear weight <c>W[n, k]</c> (row-major, rows <paramref name="ldw"/> apart), scaled by
    /// <paramref name="scale"/> (used to fold constant output scales such as the macaron ½ into the weight).</summary>
    public static PackedMatrix FromLinear(float* w, int n, int k, long ldw = -1, float scale = 1f)
    {
        if (ldw < 0) ldw = k;
        var buf = new NativeBuffer<float>(SizeOf(k, n));
        PackLinear(w, n, k, ldw, buf.Ptr, scale);
        return new PackedMatrix(k, n, buf);
    }

    /// <summary>Packs a row-major matrix <c>B[k, n]</c> (rows <paramref name="ldb"/> apart), e.g. an ONNX MatMul weight.</summary>
    public static PackedMatrix FromKN(float* b, int k, int n, long ldb = -1, float scale = 1f)
    {
        if (ldb < 0) ldb = n;
        var buf = new NativeBuffer<float>(SizeOf(k, n));
        PackKN(b, k, n, ldb, buf.Ptr, scale);
        return new PackedMatrix(k, n, buf);
    }

    /// <summary>Packs <c>W[n, k]</c> (B = Wᵀ) into <paramref name="dst"/> (<see cref="SizeOf"/> floats).</summary>
    public static void PackLinear(float* w, int n, int k, long ldw, float* dst, float scale = 1f)
    {
        int blocks = (n + NR - 1) / NR;
        for (int nb = 0; nb < blocks; nb++)
        {
            float* d = dst + (long)nb * k * NR;
            int cols = Math.Min(NR, n - nb * NR);
            if (cols < NR) new Span<float>(d, k * NR).Clear();
            for (int j = 0; j < cols; j++)
            {
                float* row = w + (long)(nb * NR + j) * ldw;
                for (int kk = 0; kk < k; kk++) d[(long)kk * NR + j] = row[kk] * scale;
            }
        }
    }

    /// <summary>Packs a row-major <c>B[k, n]</c> into <paramref name="dst"/> (<see cref="SizeOf"/> floats).</summary>
    public static void PackKN(float* b, int k, int n, long ldb, float* dst, float scale = 1f)
    {
        int blocks = (n + NR - 1) / NR;
        for (int nb = 0; nb < blocks; nb++)
        {
            float* d = dst + (long)nb * k * NR;
            int cols = Math.Min(NR, n - nb * NR);
            if (cols == NR && scale == 1f)
            {
                for (int kk = 0; kk < k; kk++)
                    Buffer.MemoryCopy(b + kk * ldb + nb * NR, d + (long)kk * NR, NR * sizeof(float), NR * sizeof(float));
                continue;
            }
            if (cols < NR) new Span<float>(d, k * NR).Clear();
            for (int kk = 0; kk < k; kk++)
                for (int j = 0; j < cols; j++) d[(long)kk * NR + j] = b[kk * ldb + nb * NR + j] * scale;
        }
    }

    /// <summary>Concatenates linear weights <c>W_i[n_i, k]</c> along the output dimension (e.g. fused q/k/v).</summary>
    public static PackedMatrix FromLinears(int k, params (IntPtr Ptr, int Rows)[] parts)
    {
        int n = parts.Sum(p => p.Rows);
        using var tmp = new NativeBuffer<float>((long)n * k, zero: false);
        long off = 0;
        foreach (var (ptr, rows) in parts)
        {
            Buffer.MemoryCopy((void*)ptr, tmp.Ptr + off, (long)rows * k * sizeof(float), (long)rows * k * sizeof(float));
            off += (long)rows * k;
        }
        return FromLinear(tmp.Ptr, n, k);
    }

    /// <summary>Unpacks back to a row-major <c>B[K, N]</c> (tests).</summary>
    public float[] ToKN()
    {
        var r = new float[(long)K * N];
        for (int kk = 0; kk < K; kk++)
            for (int j = 0; j < N; j++) r[(long)kk * N + j] = _data[((long)(j / NR) * K + kk) * NR + j % NR];
        return r;
    }

    public void Dispose() => _owned?.Dispose();
}
