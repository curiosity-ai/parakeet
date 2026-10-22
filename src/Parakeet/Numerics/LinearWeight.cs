namespace Parakeet.Numerics;

/// <summary>Weight precision of the encoder's large projections.</summary>
public enum WeightPrecision
{
    /// <summary>Full FP32, numerically equivalent to the reference implementation (default).</summary>
    Float32,
    /// <summary>Int8 weights (per output channel) with dynamically quantized int8 activations (per row) on AVX-VNNI.
    /// Faster, with a small accuracy cost; falls back to <see cref="Float32"/> on CPUs without AVX-VNNI.</summary>
    Int8,
}

/// <summary>A linear layer's weight as the right-hand GEMM operand, packed for FP32 or quantized to int8.</summary>
public sealed unsafe class LinearWeight : IDisposable
{
    private LinearWeight(PackedMatrix? f32, QuantizedMatrix? i8)
    {
        F32 = f32;
        I8 = i8;
    }

    public PackedMatrix? F32 { get; }
    public QuantizedMatrix? I8 { get; }
    public int K => F32?.K ?? I8!.K;
    public int N => F32?.N ?? I8!.N;
    public long ByteSize => F32?.ByteSize ?? I8!.ByteSize;

    /// <summary>Packs (or quantizes) a PyTorch linear weight <c>W[n, k]</c>; shapes the int8 kernel cannot take stay FP32.</summary>
    public static LinearWeight FromLinear(float* w, int n, int k, float scale, WeightPrecision precision, bool hugePages)
    {
        if (precision == WeightPrecision.Int8 && Int8Gemm.IsSupported && QuantizedMatrix.Supports(k, n))
            return new LinearWeight(null, QuantizedMatrix.FromLinear(w, n, k, k, scale));
        return new LinearWeight(PackedMatrix.FromLinear(w, n, k, k, scale, hugePages), null);
    }

    /// <summary><c>C = act(A·W + bias)</c>, or <c>C = act(C + A·W + bias)</c> when <paramref name="accumulate"/>.</summary>
    public void Multiply(float* a, long lda, int m, float* c, long ldc, Parallelism par, float* bias = null,
                         bool accumulate = false, Activation act = Activation.None)
    {
        if (I8 != null) Int8Gemm.Multiply(a, lda, m, I8, c, ldc, par, bias, accumulate, act);
        else Gemm.Multiply(a, lda, m, F32!, c, ldc, par, bias, accumulate, act);
    }

    public void Dispose()
    {
        F32?.Dispose();
        I8?.Dispose();
    }
}
