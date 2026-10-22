namespace Parakeet.Numerics;

/// <summary>
/// An int8 GEMM operand <c>B[K, N]</c> for the AVX-VNNI kernels of <see cref="Int8Gemm"/>: each output column is
/// quantized symmetrically, <c>B[k, n] ≈ Scale[n] · q[k, n]</c> with <c>q ∈ [-127, 127]</c>, and packed in panels of
/// <see cref="NR"/> = 32 columns laid out <c>[panel][k / 4][32][4]</c>, so one 32-byte load feeds a <c>vpdpbusd</c> with
/// four consecutive k values of eight columns. <see cref="Correction"/> holds <c>128 · Σ_k q[k, n]</c>, which removes the
/// +128 offset that makes the (symmetric, per-row) quantized activations unsigned.
/// </summary>
public sealed unsafe class QuantizedMatrix : IDisposable
{
    public const int NR = 32;

    private readonly NativeBuffer<sbyte> _q;
    private readonly NativeBuffer<float> _scale;
    private readonly NativeBuffer<int> _corr;

    private QuantizedMatrix(int k, int n, NativeBuffer<sbyte> q, NativeBuffer<float> scale, NativeBuffer<int> corr)
    {
        K = k;
        N = n;
        _q = q;
        _scale = scale;
        _corr = corr;
    }

    public int K { get; }
    public int N { get; }
    public int Panels => N / NR;
    internal sbyte* Data => _q.Ptr;
    internal float* Scale => _scale.Ptr;
    internal int* Correction => _corr.Ptr;
    public long ByteSize => _q.Length + _scale.Length * 4 + _corr.Length * 4;

    /// <summary>Panel <paramref name="p"/> (all K).</summary>
    internal sbyte* Panel(int p) => _q.Ptr + (long)p * K * NR;

    /// <summary>Whether a <c>k × n</c> matrix fits the kernel's shape constraints.</summary>
    public static bool Supports(int k, int n) => k % 4 == 0 && n % NR == 0;

    /// <summary>Quantizes and packs a PyTorch linear weight <c>W[n, k]</c> (rows <paramref name="ldw"/> apart), scaled by
    /// <paramref name="scale"/> first.</summary>
    public static QuantizedMatrix FromLinear(float* w, int n, int k, long ldw = -1, float scale = 1f)
    {
        if (!Supports(k, n)) throw new ArgumentException($"Int8 weights need K % 4 == 0 and N % {NR} == 0 (got {k} × {n}).");
        if (ldw < 0) ldw = k;
        var q = new NativeBuffer<sbyte>((long)k * n, zero: false);
        var sc = new NativeBuffer<float>(n, zero: false);
        var corr = new NativeBuffer<int>(n, zero: false);
        sbyte* dst = q.Ptr;
        float* scp = sc.Ptr;
        int* cp = corr.Ptr;
        int groups = k / 4;
        Parallel.For(0, n / NR, p =>
        {
            for (int lane = 0; lane < NR; lane++)
            {
                int col = p * NR + lane;
                float* row = w + col * ldw;
                float amax = 0;
                for (int kk = 0; kk < k; kk++) amax = MathF.Max(amax, MathF.Abs(row[kk] * scale));
                float s = amax > 0 ? amax / 127f : 1f, inv = 1f / s;
                int sum = 0;
                sbyte* d = dst + (long)p * k * NR + lane * 4;
                for (int g = 0; g < groups; g++)
                    for (int j = 0; j < 4; j++)
                    {
                        int v = Math.Clamp((int)MathF.Round(row[4 * g + j] * scale * inv, MidpointRounding.ToEven), -127, 127);
                        d[(long)g * NR * 4 + j] = (sbyte)v;
                        sum += v;
                    }
                scp[col] = s;
                cp[col] = 128 * sum;
            }
        });
        return new QuantizedMatrix(k, n, q, sc, corr);
    }

    /// <summary>Dequantized row-major <c>B[K, N]</c> (tests).</summary>
    public float[] ToKN()
    {
        var r = new float[(long)K * N];
        for (int col = 0; col < N; col++)
        {
            int p = col / NR, lane = col % NR;
            for (int kk = 0; kk < K; kk++)
                r[(long)kk * N + col] = _scale.Ptr[col] * _q.Ptr[(long)p * K * NR + ((long)(kk / 4) * NR + lane) * 4 + kk % 4];
        }
        return r;
    }

    public void Dispose()
    {
        _q.Dispose();
        _scale.Dispose();
        _corr.Dispose();
    }
}
