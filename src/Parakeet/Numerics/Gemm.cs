using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Parakeet.Numerics;

/// <summary>Element-wise activation fused into a GEMM epilogue.</summary>
public enum Activation { None, Relu, Silu }

/// <summary>
/// Single-precision GEMM <c>C[m, N] (+)= act(A[m, K] · B[K, N] + bias)</c> with B a <see cref="PackedMatrix"/>.
/// <para>
/// Goto-style blocking (adapted from Voxwerk): A is packed per (MC × KC) block into MR-row micro-panels (L2
/// resident), each 32-column × KC panel of B is streamed from L2 while an MR × 32 register tile of FMA accumulators
/// sweeps it - 12 × 32 on AVX-512 (24 of the 32 zmm registers), 2 × 32 on AVX2. The epilogue (bias, residual
/// accumulation, SiLU / ReLU) is applied in registers on the last K block, so no extra pass over C is needed.
/// </para>
/// <para>For m ≤ 4 (the TDT prediction network and joint) the product is memory-bound, so a separate path streams each
/// weight panel exactly once and broadcasts the few activation rows against it.</para>
/// </summary>
public static unsafe class Gemm
{
    private const int InitZero = 0, InitLoad = 1, InitBias = 2, InitLoadBias = 3;
    private const int NR = PackedMatrix.NR;
    private const int KC = 256;
    private const int NC = 16 * NR;

    internal static readonly bool UseAvx512 = Avx512F.IsSupported;
    internal static readonly bool UseAvx2 = !UseAvx512 && Avx2.IsSupported && Fma.IsSupported;
    internal static readonly int MR = UseAvx512 ? 12 : UseAvx2 ? 2 : 1;
    private static readonly int MC = MR * 20;

    [ThreadStatic] private static Scratch? t_scratch;

    private sealed class Scratch
    {
        public readonly NativeBuffer<float> APack = new((long)MC * KC);
        public readonly NativeBuffer<float> CTile = new((long)MR * NR);
        public readonly NativeBuffer<float> Bias = new(NC);
    }

    /// <summary><c>C = act(A·B + bias)</c>, or <c>C = act(C + A·B + bias)</c> when <paramref name="accumulate"/>
    /// (a fused residual add). A rows are <paramref name="lda"/> apart, C rows <paramref name="ldc"/> apart.</summary>
    public static void Multiply(float* a, long lda, int m, PackedMatrix b, float* c, long ldc, Parallelism par,
                                float* bias = null, bool accumulate = false, Activation act = Activation.None)
        => Multiply(a, lda, m, b, 0, b.Blocks, c, ldc, par, bias, accumulate, act);

    /// <summary>
    /// Like <see cref="Multiply(float*, long, int, PackedMatrix, float*, long, Parallelism, float*, bool, Activation)"/> but only
    /// for the 32-column panels [<paramref name="nb0"/>, <paramref name="nb1"/>); <paramref name="c"/> addresses output column
    /// <c>nb0 · 32</c> and <paramref name="bias"/> (when given) column 0.
    /// </summary>
    public static void Multiply(float* a, long lda, int m, PackedMatrix b, int nb0, int nb1, float* c, long ldc, Parallelism par,
                                float* bias = null, bool accumulate = false, Activation act = Activation.None)
    {
        if (m <= 0 || nb1 <= nb0) return;
        float* cBase = c - nb0 * NR;
        if (m <= 4)
        {
            SmallM(a, lda, m, b, nb0, nb1, cBase, ldc, par, bias, accumulate, act);
            return;
        }
        int blocks = nb1 - nb0;
        int mChunks = (m + MC - 1) / MC;
        int nGroups = 1;
        if (par.MaxThreads > 1)
        {
            int target = par.MaxThreads * 3;
            nGroups = Math.Clamp((target + mChunks - 1) / mChunks, 1, Math.Max(1, blocks / 4));
            if (mChunks >= target) nGroups = 1;
        }
        int items = mChunks * nGroups;
        if (items == 1)
        {
            Block(a, lda, 0, m, b, nb0, nb1, cBase, ldc, bias, accumulate, act);
            return;
        }
        par.For(items, item =>
        {
            int mc = item / nGroups, g = item % nGroups;
            int g0 = nb0 + (int)((long)g * blocks / nGroups), g1 = nb0 + (int)((long)(g + 1) * blocks / nGroups);
            int m0 = mc * MC, m1 = Math.Min(m, m0 + MC);
            Block(a, lda, m0, m1, b, g0, g1, cBase, ldc, bias, accumulate, act);
        });
    }

    private static void Block(float* a, long lda, int m0, int m1, PackedMatrix b, int nb0, int nb1, float* c, long ldc,
                              float* bias, bool accumulate, Activation act)
    {
        var s = t_scratch ??= new Scratch();
        float* aPack = s.APack.Ptr, cTile = s.CTile.Ptr, biasTile = s.Bias.Ptr;
        int K = b.K, N = b.N;
        int rows = m1 - m0;
        int tiles = (rows + MR - 1) / MR;
        const int panelsPerBlock = NC / NR;
        for (int k0 = 0; k0 < K; k0 += KC)
        {
            int kc = Math.Min(KC, K - k0);
            bool last = k0 + kc >= K;
            Activation tileAct = last ? act : Activation.None;
            PackA(a + (long)m0 * lda + k0, lda, rows, kc, aPack);
            int mode = k0 > 0 ? InitLoad : accumulate ? (bias != null ? InitLoadBias : InitLoad) : bias != null ? InitBias : InitZero;
            for (int nbBlock = nb0; nbBlock < nb1; nbBlock += panelsPerBlock)
            {
                int nbEnd = Math.Min(nb1, nbBlock + panelsPerBlock);
                if (mode == InitBias || mode == InitLoadBias)
                {
                    for (int j = 0; j < (nbEnd - nbBlock) * NR; j++)
                    {
                        int col = nbBlock * NR + j;
                        biasTile[j] = col < N ? bias[col] : 0f;
                    }
                }
                for (int t = 0; t < tiles; t++)
                {
                    int r0 = m0 + t * MR;
                    int tr = Math.Min(MR, m1 - r0);
                    float* ap = aPack + (long)t * kc * MR;
                    for (int nb = nbBlock; nb < nbEnd; nb++)
                    {
                        float* panel = b.Data + ((long)nb * K + k0) * NR;
                        float* bp = biasTile + (nb - nbBlock) * NR;
                        int cols = Math.Min(NR, N - nb * NR);
                        float* cp = c + (long)r0 * ldc + nb * NR;
                        if (tr == MR && cols == NR)
                        {
                            Kernel(ap, panel, kc, cp, ldc, mode, bp, tileAct);
                        }
                        else
                        {
                            if (mode == InitLoad || mode == InitLoadBias)
                            {
                                new Span<float>(cTile, MR * NR).Clear();
                                for (int i = 0; i < tr; i++)
                                    for (int j = 0; j < cols; j++) cTile[i * NR + j] = cp[i * ldc + j];
                            }
                            Kernel(ap, panel, kc, cTile, NR, mode, bp, tileAct);
                            for (int i = 0; i < tr; i++)
                                for (int j = 0; j < cols; j++) cp[i * ldc + j] = cTile[i * NR + j];
                        }
                    }
                }
            }
        }
    }

    /// <summary>Packs <paramref name="rows"/> × <paramref name="kc"/> of A into MR-row micro-panels laid out
    /// [tile][k][MR] (zero-padded), so the kernel's broadcasts are sequential loads.</summary>
    private static void PackA(float* a, long lda, int rows, int kc, float* dst)
    {
        int tiles = (rows + MR - 1) / MR;
        for (int t = 0; t < tiles; t++)
        {
            float* d = dst + (long)t * kc * MR;
            int r0 = t * MR;
            int tr = Math.Min(MR, rows - r0);
            if (tr < MR) new Span<float>(d, kc * MR).Clear();
            for (int i = 0; i < tr; i++)
            {
                float* src = a + (r0 + i) * lda;
                float* di = d + i;
                int p = 0;
                for (; p <= kc - 4; p += 4)
                {
                    di[p * MR] = src[p];
                    di[(p + 1) * MR] = src[p + 1];
                    di[(p + 2) * MR] = src[p + 2];
                    di[(p + 3) * MR] = src[p + 3];
                }
                for (; p < kc; p++) di[p * MR] = src[p];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Kernel(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias, Activation act)
    {
        if (UseAvx512) Kernel512(a, b, kc, c, ldc, mode, bias, act);
        else if (UseAvx2) Kernel256(a, b, kc, c, ldc, mode, bias, act);
        else KernelScalar(a, b, kc, c, ldc, mode, bias, act);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel512(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias, Activation act)
    {
        Vector512<float> c0a, c0b, c1a, c1b, c2a, c2b, c3a, c3b, c4a, c4b, c5a, c5b, c6a, c6b, c7a, c7b, c8a, c8b, c9a, c9b, c10a, c10b, c11a, c11b;
        if (mode == InitZero)
        {
            c0a = Vector512<float>.Zero; c0b = Vector512<float>.Zero;
            c1a = Vector512<float>.Zero; c1b = Vector512<float>.Zero;
            c2a = Vector512<float>.Zero; c2b = Vector512<float>.Zero;
            c3a = Vector512<float>.Zero; c3b = Vector512<float>.Zero;
            c4a = Vector512<float>.Zero; c4b = Vector512<float>.Zero;
            c5a = Vector512<float>.Zero; c5b = Vector512<float>.Zero;
            c6a = Vector512<float>.Zero; c6b = Vector512<float>.Zero;
            c7a = Vector512<float>.Zero; c7b = Vector512<float>.Zero;
            c8a = Vector512<float>.Zero; c8b = Vector512<float>.Zero;
            c9a = Vector512<float>.Zero; c9b = Vector512<float>.Zero;
            c10a = Vector512<float>.Zero; c10b = Vector512<float>.Zero;
            c11a = Vector512<float>.Zero; c11b = Vector512<float>.Zero;
        }
        else if (mode == InitBias)
        {
            var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);
            c0a = ba; c0b = bb;
            c1a = ba; c1b = bb;
            c2a = ba; c2b = bb;
            c3a = ba; c3b = bb;
            c4a = ba; c4b = bb;
            c5a = ba; c5b = bb;
            c6a = ba; c6b = bb;
            c7a = ba; c7b = bb;
            c8a = ba; c8b = bb;
            c9a = ba; c9b = bb;
            c10a = ba; c10b = bb;
            c11a = ba; c11b = bb;
        }
        else
        {
            c0a = Vector512.Load(c + 0 * ldc); c0b = Vector512.Load(c + 0 * ldc + 16);
            c1a = Vector512.Load(c + 1 * ldc); c1b = Vector512.Load(c + 1 * ldc + 16);
            c2a = Vector512.Load(c + 2 * ldc); c2b = Vector512.Load(c + 2 * ldc + 16);
            c3a = Vector512.Load(c + 3 * ldc); c3b = Vector512.Load(c + 3 * ldc + 16);
            c4a = Vector512.Load(c + 4 * ldc); c4b = Vector512.Load(c + 4 * ldc + 16);
            c5a = Vector512.Load(c + 5 * ldc); c5b = Vector512.Load(c + 5 * ldc + 16);
            c6a = Vector512.Load(c + 6 * ldc); c6b = Vector512.Load(c + 6 * ldc + 16);
            c7a = Vector512.Load(c + 7 * ldc); c7b = Vector512.Load(c + 7 * ldc + 16);
            c8a = Vector512.Load(c + 8 * ldc); c8b = Vector512.Load(c + 8 * ldc + 16);
            c9a = Vector512.Load(c + 9 * ldc); c9b = Vector512.Load(c + 9 * ldc + 16);
            c10a = Vector512.Load(c + 10 * ldc); c10b = Vector512.Load(c + 10 * ldc + 16);
            c11a = Vector512.Load(c + 11 * ldc); c11b = Vector512.Load(c + 11 * ldc + 16);
            if (mode == InitLoadBias)
            {
                var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);
                c0a += ba; c0b += bb;
                c1a += ba; c1b += bb;
                c2a += ba; c2b += bb;
                c3a += ba; c3b += bb;
                c4a += ba; c4b += bb;
                c5a += ba; c5b += bb;
                c6a += ba; c6b += bb;
                c7a += ba; c7b += bb;
                c8a += ba; c8b += bb;
                c9a += ba; c9b += bb;
                c10a += ba; c10b += bb;
                c11a += ba; c11b += bb;
            }
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector512.Load(b); var b1 = Vector512.Load(b + 16);
            { var x = Vector512.Create(a[0]); c0a = Avx512F.FusedMultiplyAdd(x, b0, c0a); c0b = Avx512F.FusedMultiplyAdd(x, b1, c0b); }
            { var x = Vector512.Create(a[1]); c1a = Avx512F.FusedMultiplyAdd(x, b0, c1a); c1b = Avx512F.FusedMultiplyAdd(x, b1, c1b); }
            { var x = Vector512.Create(a[2]); c2a = Avx512F.FusedMultiplyAdd(x, b0, c2a); c2b = Avx512F.FusedMultiplyAdd(x, b1, c2b); }
            { var x = Vector512.Create(a[3]); c3a = Avx512F.FusedMultiplyAdd(x, b0, c3a); c3b = Avx512F.FusedMultiplyAdd(x, b1, c3b); }
            { var x = Vector512.Create(a[4]); c4a = Avx512F.FusedMultiplyAdd(x, b0, c4a); c4b = Avx512F.FusedMultiplyAdd(x, b1, c4b); }
            { var x = Vector512.Create(a[5]); c5a = Avx512F.FusedMultiplyAdd(x, b0, c5a); c5b = Avx512F.FusedMultiplyAdd(x, b1, c5b); }
            { var x = Vector512.Create(a[6]); c6a = Avx512F.FusedMultiplyAdd(x, b0, c6a); c6b = Avx512F.FusedMultiplyAdd(x, b1, c6b); }
            { var x = Vector512.Create(a[7]); c7a = Avx512F.FusedMultiplyAdd(x, b0, c7a); c7b = Avx512F.FusedMultiplyAdd(x, b1, c7b); }
            { var x = Vector512.Create(a[8]); c8a = Avx512F.FusedMultiplyAdd(x, b0, c8a); c8b = Avx512F.FusedMultiplyAdd(x, b1, c8b); }
            { var x = Vector512.Create(a[9]); c9a = Avx512F.FusedMultiplyAdd(x, b0, c9a); c9b = Avx512F.FusedMultiplyAdd(x, b1, c9b); }
            { var x = Vector512.Create(a[10]); c10a = Avx512F.FusedMultiplyAdd(x, b0, c10a); c10b = Avx512F.FusedMultiplyAdd(x, b1, c10b); }
            { var x = Vector512.Create(a[11]); c11a = Avx512F.FusedMultiplyAdd(x, b0, c11a); c11b = Avx512F.FusedMultiplyAdd(x, b1, c11b); }
            a += 12; b += 32;
        }
        if (act == Activation.Silu)
        {
            c0a = VMath.Silu(c0a); c0b = VMath.Silu(c0b);
            c1a = VMath.Silu(c1a); c1b = VMath.Silu(c1b);
            c2a = VMath.Silu(c2a); c2b = VMath.Silu(c2b);
            c3a = VMath.Silu(c3a); c3b = VMath.Silu(c3b);
            c4a = VMath.Silu(c4a); c4b = VMath.Silu(c4b);
            c5a = VMath.Silu(c5a); c5b = VMath.Silu(c5b);
            c6a = VMath.Silu(c6a); c6b = VMath.Silu(c6b);
            c7a = VMath.Silu(c7a); c7b = VMath.Silu(c7b);
            c8a = VMath.Silu(c8a); c8b = VMath.Silu(c8b);
            c9a = VMath.Silu(c9a); c9b = VMath.Silu(c9b);
            c10a = VMath.Silu(c10a); c10b = VMath.Silu(c10b);
            c11a = VMath.Silu(c11a); c11b = VMath.Silu(c11b);
        }
        else if (act == Activation.Relu)
        {
            var z = Vector512<float>.Zero;
            c0a = Vector512.Max(c0a, z); c0b = Vector512.Max(c0b, z);
            c1a = Vector512.Max(c1a, z); c1b = Vector512.Max(c1b, z);
            c2a = Vector512.Max(c2a, z); c2b = Vector512.Max(c2b, z);
            c3a = Vector512.Max(c3a, z); c3b = Vector512.Max(c3b, z);
            c4a = Vector512.Max(c4a, z); c4b = Vector512.Max(c4b, z);
            c5a = Vector512.Max(c5a, z); c5b = Vector512.Max(c5b, z);
            c6a = Vector512.Max(c6a, z); c6b = Vector512.Max(c6b, z);
            c7a = Vector512.Max(c7a, z); c7b = Vector512.Max(c7b, z);
            c8a = Vector512.Max(c8a, z); c8b = Vector512.Max(c8b, z);
            c9a = Vector512.Max(c9a, z); c9b = Vector512.Max(c9b, z);
            c10a = Vector512.Max(c10a, z); c10b = Vector512.Max(c10b, z);
            c11a = Vector512.Max(c11a, z); c11b = Vector512.Max(c11b, z);
        }
        Vector512.Store(c0a, c + 0 * ldc); Vector512.Store(c0b, c + 0 * ldc + 16);
        Vector512.Store(c1a, c + 1 * ldc); Vector512.Store(c1b, c + 1 * ldc + 16);
        Vector512.Store(c2a, c + 2 * ldc); Vector512.Store(c2b, c + 2 * ldc + 16);
        Vector512.Store(c3a, c + 3 * ldc); Vector512.Store(c3b, c + 3 * ldc + 16);
        Vector512.Store(c4a, c + 4 * ldc); Vector512.Store(c4b, c + 4 * ldc + 16);
        Vector512.Store(c5a, c + 5 * ldc); Vector512.Store(c5b, c + 5 * ldc + 16);
        Vector512.Store(c6a, c + 6 * ldc); Vector512.Store(c6b, c + 6 * ldc + 16);
        Vector512.Store(c7a, c + 7 * ldc); Vector512.Store(c7b, c + 7 * ldc + 16);
        Vector512.Store(c8a, c + 8 * ldc); Vector512.Store(c8b, c + 8 * ldc + 16);
        Vector512.Store(c9a, c + 9 * ldc); Vector512.Store(c9b, c + 9 * ldc + 16);
        Vector512.Store(c10a, c + 10 * ldc); Vector512.Store(c10b, c + 10 * ldc + 16);
        Vector512.Store(c11a, c + 11 * ldc); Vector512.Store(c11b, c + 11 * ldc + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel256(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias, Activation act)
    {
        Vector256<float> c0a, c0b, c0c, c0d, c1a, c1b, c1c, c1d;
        if (mode == InitZero)
        {
            c0a = c0b = c0c = c0d = c1a = c1b = c1c = c1d = Vector256<float>.Zero;
        }
        else if (mode == InitBias)
        {
            c0a = c1a = Vector256.Load(bias + 0);
            c0b = c1b = Vector256.Load(bias + 8);
            c0c = c1c = Vector256.Load(bias + 16);
            c0d = c1d = Vector256.Load(bias + 24);
        }
        else
        {
            c0a = Vector256.Load(c + 0); c0b = Vector256.Load(c + 8); c0c = Vector256.Load(c + 16); c0d = Vector256.Load(c + 24);
            c1a = Vector256.Load(c + ldc + 0); c1b = Vector256.Load(c + ldc + 8); c1c = Vector256.Load(c + ldc + 16); c1d = Vector256.Load(c + ldc + 24);
            if (mode == InitLoadBias)
            {
                var ba = Vector256.Load(bias + 0); var bb = Vector256.Load(bias + 8); var bc = Vector256.Load(bias + 16); var bd = Vector256.Load(bias + 24);
                c0a += ba; c0b += bb; c0c += bc; c0d += bd;
                c1a += ba; c1b += bb; c1c += bc; c1d += bd;
            }
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 8); var b2 = Vector256.Load(b + 16); var b3 = Vector256.Load(b + 24);
            { var x = Vector256.Create(a[0]); c0a = Fma.MultiplyAdd(x, b0, c0a); c0b = Fma.MultiplyAdd(x, b1, c0b); c0c = Fma.MultiplyAdd(x, b2, c0c); c0d = Fma.MultiplyAdd(x, b3, c0d); }
            { var x = Vector256.Create(a[1]); c1a = Fma.MultiplyAdd(x, b0, c1a); c1b = Fma.MultiplyAdd(x, b1, c1b); c1c = Fma.MultiplyAdd(x, b2, c1c); c1d = Fma.MultiplyAdd(x, b3, c1d); }
            a += 2; b += 32;
        }
        if (act == Activation.Silu)
        {
            c0a = VMath.Silu(c0a); c0b = VMath.Silu(c0b); c0c = VMath.Silu(c0c); c0d = VMath.Silu(c0d);
            c1a = VMath.Silu(c1a); c1b = VMath.Silu(c1b); c1c = VMath.Silu(c1c); c1d = VMath.Silu(c1d);
        }
        else if (act == Activation.Relu)
        {
            var z = Vector256<float>.Zero;
            c0a = Vector256.Max(c0a, z); c0b = Vector256.Max(c0b, z); c0c = Vector256.Max(c0c, z); c0d = Vector256.Max(c0d, z);
            c1a = Vector256.Max(c1a, z); c1b = Vector256.Max(c1b, z); c1c = Vector256.Max(c1c, z); c1d = Vector256.Max(c1d, z);
        }
        Vector256.Store(c0a, c + 0); Vector256.Store(c0b, c + 8); Vector256.Store(c0c, c + 16); Vector256.Store(c0d, c + 24);
        Vector256.Store(c1a, c + ldc + 0); Vector256.Store(c1b, c + ldc + 8); Vector256.Store(c1c, c + ldc + 16); Vector256.Store(c1d, c + ldc + 24);
    }

    private static void KernelScalar(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias, Activation act)
    {
        // MR == 1 on this path.
        Span<float> acc = stackalloc float[NR];
        for (int j = 0; j < NR; j++)
            acc[j] = (mode == InitLoad || mode == InitLoadBias ? c[j] : 0f) + (mode == InitBias || mode == InitLoadBias ? bias[j] : 0f);
        for (int p = 0; p < kc; p++)
        {
            float x = a[p];
            float* bp = b + p * NR;
            for (int j = 0; j < NR; j++) acc[j] = MathF.FusedMultiplyAdd(x, bp[j], acc[j]);
        }
        for (int j = 0; j < NR; j++) c[j] = Apply(acc[j], act);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Apply(float x, Activation act) => act switch
    {
        Activation.Silu => VMath.Silu(x),
        Activation.Relu => MathF.Max(x, 0f),
        _ => x,
    };

    // ---------------------------------------------------------------------------------------------
    // m ≤ 4: memory-bound products (prediction network, joint). Every panel is streamed once.
    // ---------------------------------------------------------------------------------------------

    private static void SmallM(float* a, long lda, int m, PackedMatrix b, int nb0, int nb1, float* c, long ldc, Parallelism par,
                               float* bias, bool accumulate, Activation act)
    {
        int blocks = nb1 - nb0;
        long bytes = (long)blocks * b.K * NR * sizeof(float);
        // Below ~256 KB of weights the fork/join costs more than it saves.
        int chunks = par.MaxThreads == 1 || bytes < (256 << 10) ? 1 : Math.Min(par.MaxThreads, blocks);
        if (chunks == 1)
        {
            SmallMRange(a, lda, m, b, nb0, nb1, c, ldc, bias, accumulate, act);
            return;
        }
        par.For(chunks, ch =>
        {
            int s = nb0 + (int)((long)ch * blocks / chunks), e = nb0 + (int)((long)(ch + 1) * blocks / chunks);
            SmallMRange(a, lda, m, b, s, e, c, ldc, bias, accumulate, act);
        });
    }

    private static void SmallMRange(float* a, long lda, int m, PackedMatrix b, int nb0, int nb1, float* c, long ldc,
                                    float* bias, bool accumulate, Activation act)
    {
        int K = b.K, N = b.N;
        float* tmp = stackalloc float[4 * NR];
        for (int nb = nb0; nb < nb1; nb++)
        {
            float* panel = b.Data + (long)nb * K * NR;
            int cols = Math.Min(NR, N - nb * NR);
            if (UseAvx512)
            {
                // Two independent accumulator sets per row hide FMA latency for m = 1.
                Vector512<float> r0a = default, r0b = default, r1a = default, r1b = default;
                Vector512<float> r2a = default, r2b = default, r3a = default, r3b = default;
                Vector512<float> s0a = default, s0b = default;
                int p = 0;
                if (m == 1)
                {
                    for (; p + 1 < K; p += 2)
                    {
                        float* bp = panel + (long)p * NR;
                        var x0 = Vector512.Create(a[p]);
                        var x1 = Vector512.Create(a[p + 1]);
                        r0a = Avx512F.FusedMultiplyAdd(x0, Vector512.Load(bp), r0a);
                        r0b = Avx512F.FusedMultiplyAdd(x0, Vector512.Load(bp + 16), r0b);
                        s0a = Avx512F.FusedMultiplyAdd(x1, Vector512.Load(bp + 32), s0a);
                        s0b = Avx512F.FusedMultiplyAdd(x1, Vector512.Load(bp + 48), s0b);
                    }
                    for (; p < K; p++)
                    {
                        float* bp = panel + (long)p * NR;
                        var x0 = Vector512.Create(a[p]);
                        r0a = Avx512F.FusedMultiplyAdd(x0, Vector512.Load(bp), r0a);
                        r0b = Avx512F.FusedMultiplyAdd(x0, Vector512.Load(bp + 16), r0b);
                    }
                    r0a += s0a;
                    r0b += s0b;
                }
                else
                {
                    for (; p < K; p++)
                    {
                        float* bp = panel + (long)p * NR;
                        var b0 = Vector512.Load(bp);
                        var b1 = Vector512.Load(bp + 16);
                        var x = Vector512.Create(a[p]);
                        r0a = Avx512F.FusedMultiplyAdd(x, b0, r0a); r0b = Avx512F.FusedMultiplyAdd(x, b1, r0b);
                        x = Vector512.Create(a[lda + p]);
                        r1a = Avx512F.FusedMultiplyAdd(x, b0, r1a); r1b = Avx512F.FusedMultiplyAdd(x, b1, r1b);
                        if (m > 2)
                        {
                            x = Vector512.Create(a[2 * lda + p]);
                            r2a = Avx512F.FusedMultiplyAdd(x, b0, r2a); r2b = Avx512F.FusedMultiplyAdd(x, b1, r2b);
                        }
                        if (m > 3)
                        {
                            x = Vector512.Create(a[3 * lda + p]);
                            r3a = Avx512F.FusedMultiplyAdd(x, b0, r3a); r3b = Avx512F.FusedMultiplyAdd(x, b1, r3b);
                        }
                    }
                }
                Vector512.Store(r0a, tmp); Vector512.Store(r0b, tmp + 16);
                Vector512.Store(r1a, tmp + 32); Vector512.Store(r1b, tmp + 48);
                Vector512.Store(r2a, tmp + 64); Vector512.Store(r2b, tmp + 80);
                Vector512.Store(r3a, tmp + 96); Vector512.Store(r3b, tmp + 112);
            }
            else
            {
                new Span<float>(tmp, 4 * NR).Clear();
                for (int r = 0; r < m; r++)
                    for (int p = 0; p < K; p++)
                    {
                        float x = a[r * lda + p];
                        float* bp = panel + (long)p * NR;
                        for (int j = 0; j < NR; j++) tmp[r * NR + j] = MathF.FusedMultiplyAdd(x, bp[j], tmp[r * NR + j]);
                    }
            }
            for (int r = 0; r < m; r++)
            {
                float* cp = c + r * ldc + nb * NR;
                for (int j = 0; j < cols; j++)
                {
                    float v = tmp[r * NR + j];
                    if (bias != null) v += bias[nb * NR + j];
                    if (accumulate) v += cp[j];
                    cp[j] = Apply(v, act);
                }
            }
        }
    }
}
