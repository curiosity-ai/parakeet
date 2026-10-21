using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Parakeet.Numerics;

/// <summary>Vectorized transcendental functions used inside fused kernels (GEMM epilogues, softmax, conv).
/// <c>exp</c> uses the Cephes range reduction <c>e^x = 2^n · e^r</c>, |r| ≤ ln2/2, with a degree-6 polynomial
/// (≈1 ulp over the float range), so SiLU/sigmoid match a libm-based reference to float precision.</summary>
public static class VMath
{
    private const float Log2E = 1.44269504088896341f;
    private const float Ln2Hi = 0.693359375f;
    private const float Ln2Lo = -2.12194440e-4f;
    private const float ExpMax = 88.3762626647949f;
    private const float ExpMin = -87.3365447505531f;
    private const float P0 = 1.9875691500E-4f, P1 = 1.3981999507E-3f, P2 = 8.3334519073E-3f;
    private const float P3 = 4.1665795894E-2f, P4 = 1.6666665459E-1f, P5 = 5.0000001201E-1f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        x = Vector512.Min(Vector512.Max(x, Vector512.Create(ExpMin)), Vector512.Create(ExpMax));
        var n = Avx512F.RoundScale(x * Vector512.Create(Log2E), 0x08);   // round to nearest, no exceptions
        var r = Avx512F.FusedMultiplyAddNegated(n, Vector512.Create(Ln2Hi), x);
        r = Avx512F.FusedMultiplyAddNegated(n, Vector512.Create(Ln2Lo), r);
        var p = Avx512F.FusedMultiplyAdd(Vector512.Create(P0), r, Vector512.Create(P1));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P2));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P3));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P4));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P5));
        p = Avx512F.FusedMultiplyAdd(p, r * r, r + Vector512<float>.One);
        return Avx512F.Scale(p, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        x = Vector256.Min(Vector256.Max(x, Vector256.Create(ExpMin)), Vector256.Create(ExpMax));
        var n = Avx.RoundToNearestInteger(x * Vector256.Create(Log2E));
        var r = Fma.MultiplyAddNegated(n, Vector256.Create(Ln2Hi), x);
        r = Fma.MultiplyAddNegated(n, Vector256.Create(Ln2Lo), r);
        var p = Fma.MultiplyAdd(Vector256.Create(P0), r, Vector256.Create(P1));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P2));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P3));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P4));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P5));
        p = Fma.MultiplyAdd(p, r * r, r + Vector256<float>.One);
        var pow2 = Avx2.ShiftLeftLogical(Avx.ConvertToVector256Int32(n) + Vector256.Create(127), 23).AsSingle();
        return p * pow2;
    }

    /// <summary>x · sigmoid(x) (Swish / SiLU).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Silu(Vector512<float> x) => x / (Vector512<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Silu(Vector256<float> x) => x / (Vector256<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Sigmoid(Vector512<float> x) => Vector512<float>.One / (Vector512<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x) => Vector256<float>.One / (Vector256<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
