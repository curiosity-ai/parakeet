using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Parakeet.Numerics;

/// <summary>
/// Dynamically quantized GEMM <c>C[m, N] (+)= act(A[m, K] · B[K, N] + bias)</c> on AVX-VNNI (<c>vpdpbusd</c>, 32 int8
/// multiply-adds per instruction, ≈ 2.2× the FP32 FMA rate of an AVX-512 core). Each call quantizes the rows of A
/// symmetrically to int8 (one scale per row, offset by +128 to the unsigned operand the instruction needs) into 6-row
/// tiles; B is a pre-quantized <see cref="QuantizedMatrix"/> (one scale per column). A 6 × 32 register tile of int32
/// accumulators runs over the whole K, and the epilogue removes the offset (<c>128 · Σ_k q_B</c>), rescales, adds the
/// bias, applies the activation and the residual in place. Weights take a quarter of the FP32 bytes.
/// </summary>
public static unsafe class Int8Gemm
{
    private const int MR = 6;
    private const int NR = QuantizedMatrix.NR;
    private const int PrefetchBytes = 1024;

    /// <summary>True when the CPU has AVX-VNNI (and AVX2/FMA for the epilogue).</summary>
    public static bool IsSupported => AvxVnni.IsSupported && Avx2.IsSupported && Fma.IsSupported;

    [ThreadStatic] private static NativeBuffer<byte>? t_a;
    [ThreadStatic] private static NativeBuffer<float>? t_rowScale;

    /// <summary>Epilogue parameters for one 6 × 32 tile.</summary>
    private readonly struct Epilogue(float* rowScale, float* colScale, int* colCorr, float* bias, bool accumulate, Activation act)
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Store(int row, float* c, Vector256<int> a0, Vector256<int> a1, Vector256<int> a2, Vector256<int> a3)
        {
            var rs = Vector256.Create(rowScale[row]);
            Store8(c, a0, 0, rs);
            Store8(c, a1, 8, rs);
            Store8(c, a2, 16, rs);
            Store8(c, a3, 24, rs);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Store8(float* c, Vector256<int> acc, int o, Vector256<float> rs)
        {
            var v = Avx.ConvertToVector256Single(acc - Vector256.Load(colCorr + o)) * (Vector256.Load(colScale + o) * rs);
            if (bias != null) v += Vector256.Load(bias + o);
            if (accumulate) v += Vector256.Load(c + o);
            if (act == Activation.Silu) v = VMath.Silu(v);
            else if (act == Activation.Relu) v = Vector256.Max(v, Vector256<float>.Zero);
            Vector256.Store(v, c + o);
        }
    }

    /// <summary><c>C = act(A·B + bias)</c>, or <c>C = act(C + A·B + bias)</c> when <paramref name="accumulate"/>.</summary>
    public static void Multiply(float* a, long lda, int m, QuantizedMatrix b, float* c, long ldc, Parallelism par,
                                float* bias = null, bool accumulate = false, Activation act = Activation.None)
    {
        if (m <= 0) return;
        if (!IsSupported) throw new PlatformNotSupportedException("Int8 GEMM needs AVX-VNNI.");
        int K = b.K, groups = K / 4, tiles = (m + MR - 1) / MR;

        long aBytes = (long)tiles * MR * K;
        var aBuf = t_a;
        if (aBuf == null || aBuf.Length < aBytes)
        {
            aBuf?.Dispose();
            aBuf = t_a = new NativeBuffer<byte>(Math.Max(aBytes, 1 << 16), zero: false);
        }
        var sBuf = t_rowScale;
        if (sBuf == null || sBuf.Length < tiles * MR)
        {
            sBuf?.Dispose();
            sBuf = t_rowScale = new NativeBuffer<float>(Math.Max(tiles * MR, 1024), zero: true);
        }
        byte* aq = aBuf.Ptr;
        float* rowScale = sBuf.Ptr;
        par.ForRange(tiles, 2, (t0, t1) =>
        {
            for (int t = t0; t < t1; t++) QuantizeTile(a, lda, m, K, t, aq + (long)t * MR * K, rowScale);
        });

        int panels = b.Panels;
        int items = Math.Min(panels, par.MaxThreads == 1 ? 1 : par.MaxThreads * 2);
        while (items > 1 && panels % items != 0 && items < panels) items++;
        par.For(items, item =>
        {
            int p0 = (int)((long)item * panels / items), p1 = (int)((long)(item + 1) * panels / items);
            for (int p = p0; p < p1; p++)
            {
                sbyte* panel = b.Panel(p);
                float* bp = bias == null ? null : bias + p * NR;
                for (int t = 0; t < tiles; t++)
                {
                    int r0 = t * MR, rows = Math.Min(MR, m - r0);
                    var e = new Epilogue(rowScale + r0, b.Scale + p * NR, b.Correction + p * NR, bp, accumulate, act);
                    KernelRows(rows, aq + (long)t * MR * K, panel, groups, c + (long)r0 * ldc + p * NR, ldc, e);
                }
            }
        });
    }

    /// <summary>Quantizes rows [6t, 6t + 6) of A to unsigned int8 (symmetric int8 + 128) laid out [k / 4][6][4].</summary>
    private static void QuantizeTile(float* a, long lda, int m, int K, int t, byte* dst, float* rowScale)
    {
        int groups = K / 4;
        for (int i = 0; i < MR; i++)
        {
            int r = t * MR + i;
            byte* d = dst + i * 4;
            if (r >= m)
            {
                rowScale[r] = 0;
                for (int g = 0; g < groups; g++) *(uint*)(d + (long)g * MR * 4) = 0x80808080u;
                continue;
            }
            float* row = a + r * lda;
            int k = 0;
            float amax = 0;
            if (Avx512F.IsSupported)
            {
                var vm = Vector512<float>.Zero;
                for (; k + 16 <= K; k += 16) vm = Vector512.Max(vm, Vector512.Abs(Vector512.Load(row + k)));
                for (int l = 0; l < 16; l++) amax = MathF.Max(amax, vm.GetElement(l));
            }
            for (; k < K; k++) amax = MathF.Max(amax, MathF.Abs(row[k]));
            float s = amax > 0 ? amax / 127f : 1f, inv = 1f / s;
            rowScale[r] = s;
            k = 0;
            if (Avx512F.IsSupported)
            {
                var vinv = Vector512.Create(inv);
                var off = Vector512.Create(128);
                for (; k + 16 <= K; k += 16)
                {
                    // Round to nearest even (the default MXCSR mode), offset to [1, 255], narrow to bytes.
                    var qi = Avx512F.ConvertToVector512Int32(Vector512.Load(row + k) * vinv) + off;
                    var bytes = Avx512F.ConvertToVector128ByteWithSaturation(qi.AsUInt32()).AsUInt32();
                    int g = k / 4;
                    *(uint*)(d + (long)g * MR * 4) = bytes.GetElement(0);
                    *(uint*)(d + (long)(g + 1) * MR * 4) = bytes.GetElement(1);
                    *(uint*)(d + (long)(g + 2) * MR * 4) = bytes.GetElement(2);
                    *(uint*)(d + (long)(g + 3) * MR * 4) = bytes.GetElement(3);
                }
            }
            for (; k < K; k++)
            {
                int q = Math.Clamp((int)MathF.Round(row[k] * inv, MidpointRounding.ToEven), -127, 127) + 128;
                d[(long)(k / 4) * MR * 4 + k % 4] = (byte)q;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void KernelRows(int rows, byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        switch (rows)
        {
            case 1: KernelR1(a, b, groups, c, ldc, e); break;
            case 2: KernelR2(a, b, groups, c, ldc, e); break;
            case 3: KernelR3(a, b, groups, c, ldc, e); break;
            case 4: KernelR4(a, b, groups, c, ldc, e); break;
            case 5: KernelR5(a, b, groups, c, ldc, e); break;
            case 6: KernelR6(a, b, groups, c, ldc, e); break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelR1(byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        Vector256<int> c00, c01, c02, c03;
        c00 = Vector256<int>.Zero; c01 = Vector256<int>.Zero; c02 = Vector256<int>.Zero; c03 = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++)
        {
            Sse.Prefetch0(b + PrefetchBytes);
            Sse.Prefetch0(b + PrefetchBytes + 64);
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 32); var b2 = Vector256.Load(b + 64); var b3 = Vector256.Load(b + 96);
            { var x = Vector256.Create(*(int*)(a + 0)).AsByte(); c00 = AvxVnni.MultiplyWideningAndAdd(c00, x, b0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, x, b1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, x, b2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, x, b3); }
            a += 24; b += 128;
        }
        e.Store(0, c + 0 * ldc, c00, c01, c02, c03);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelR2(byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        Vector256<int> c00, c01, c02, c03, c10, c11, c12, c13;
        c00 = Vector256<int>.Zero; c01 = Vector256<int>.Zero; c02 = Vector256<int>.Zero; c03 = Vector256<int>.Zero; c10 = Vector256<int>.Zero; c11 = Vector256<int>.Zero; c12 = Vector256<int>.Zero; c13 = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++)
        {
            Sse.Prefetch0(b + PrefetchBytes);
            Sse.Prefetch0(b + PrefetchBytes + 64);
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 32); var b2 = Vector256.Load(b + 64); var b3 = Vector256.Load(b + 96);
            { var x = Vector256.Create(*(int*)(a + 0)).AsByte(); c00 = AvxVnni.MultiplyWideningAndAdd(c00, x, b0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, x, b1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, x, b2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 4)).AsByte(); c10 = AvxVnni.MultiplyWideningAndAdd(c10, x, b0); c11 = AvxVnni.MultiplyWideningAndAdd(c11, x, b1); c12 = AvxVnni.MultiplyWideningAndAdd(c12, x, b2); c13 = AvxVnni.MultiplyWideningAndAdd(c13, x, b3); }
            a += 24; b += 128;
        }
        e.Store(0, c + 0 * ldc, c00, c01, c02, c03);
        e.Store(1, c + 1 * ldc, c10, c11, c12, c13);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelR3(byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        Vector256<int> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23;
        c00 = Vector256<int>.Zero; c01 = Vector256<int>.Zero; c02 = Vector256<int>.Zero; c03 = Vector256<int>.Zero; c10 = Vector256<int>.Zero; c11 = Vector256<int>.Zero; c12 = Vector256<int>.Zero; c13 = Vector256<int>.Zero; c20 = Vector256<int>.Zero; c21 = Vector256<int>.Zero; c22 = Vector256<int>.Zero; c23 = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++)
        {
            Sse.Prefetch0(b + PrefetchBytes);
            Sse.Prefetch0(b + PrefetchBytes + 64);
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 32); var b2 = Vector256.Load(b + 64); var b3 = Vector256.Load(b + 96);
            { var x = Vector256.Create(*(int*)(a + 0)).AsByte(); c00 = AvxVnni.MultiplyWideningAndAdd(c00, x, b0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, x, b1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, x, b2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 4)).AsByte(); c10 = AvxVnni.MultiplyWideningAndAdd(c10, x, b0); c11 = AvxVnni.MultiplyWideningAndAdd(c11, x, b1); c12 = AvxVnni.MultiplyWideningAndAdd(c12, x, b2); c13 = AvxVnni.MultiplyWideningAndAdd(c13, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 8)).AsByte(); c20 = AvxVnni.MultiplyWideningAndAdd(c20, x, b0); c21 = AvxVnni.MultiplyWideningAndAdd(c21, x, b1); c22 = AvxVnni.MultiplyWideningAndAdd(c22, x, b2); c23 = AvxVnni.MultiplyWideningAndAdd(c23, x, b3); }
            a += 24; b += 128;
        }
        e.Store(0, c + 0 * ldc, c00, c01, c02, c03);
        e.Store(1, c + 1 * ldc, c10, c11, c12, c13);
        e.Store(2, c + 2 * ldc, c20, c21, c22, c23);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelR4(byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        Vector256<int> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
        c00 = Vector256<int>.Zero; c01 = Vector256<int>.Zero; c02 = Vector256<int>.Zero; c03 = Vector256<int>.Zero; c10 = Vector256<int>.Zero; c11 = Vector256<int>.Zero; c12 = Vector256<int>.Zero; c13 = Vector256<int>.Zero; c20 = Vector256<int>.Zero; c21 = Vector256<int>.Zero; c22 = Vector256<int>.Zero; c23 = Vector256<int>.Zero; c30 = Vector256<int>.Zero; c31 = Vector256<int>.Zero; c32 = Vector256<int>.Zero; c33 = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++)
        {
            Sse.Prefetch0(b + PrefetchBytes);
            Sse.Prefetch0(b + PrefetchBytes + 64);
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 32); var b2 = Vector256.Load(b + 64); var b3 = Vector256.Load(b + 96);
            { var x = Vector256.Create(*(int*)(a + 0)).AsByte(); c00 = AvxVnni.MultiplyWideningAndAdd(c00, x, b0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, x, b1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, x, b2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 4)).AsByte(); c10 = AvxVnni.MultiplyWideningAndAdd(c10, x, b0); c11 = AvxVnni.MultiplyWideningAndAdd(c11, x, b1); c12 = AvxVnni.MultiplyWideningAndAdd(c12, x, b2); c13 = AvxVnni.MultiplyWideningAndAdd(c13, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 8)).AsByte(); c20 = AvxVnni.MultiplyWideningAndAdd(c20, x, b0); c21 = AvxVnni.MultiplyWideningAndAdd(c21, x, b1); c22 = AvxVnni.MultiplyWideningAndAdd(c22, x, b2); c23 = AvxVnni.MultiplyWideningAndAdd(c23, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 12)).AsByte(); c30 = AvxVnni.MultiplyWideningAndAdd(c30, x, b0); c31 = AvxVnni.MultiplyWideningAndAdd(c31, x, b1); c32 = AvxVnni.MultiplyWideningAndAdd(c32, x, b2); c33 = AvxVnni.MultiplyWideningAndAdd(c33, x, b3); }
            a += 24; b += 128;
        }
        e.Store(0, c + 0 * ldc, c00, c01, c02, c03);
        e.Store(1, c + 1 * ldc, c10, c11, c12, c13);
        e.Store(2, c + 2 * ldc, c20, c21, c22, c23);
        e.Store(3, c + 3 * ldc, c30, c31, c32, c33);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelR5(byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        Vector256<int> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, c40, c41, c42, c43;
        c00 = Vector256<int>.Zero; c01 = Vector256<int>.Zero; c02 = Vector256<int>.Zero; c03 = Vector256<int>.Zero; c10 = Vector256<int>.Zero; c11 = Vector256<int>.Zero; c12 = Vector256<int>.Zero; c13 = Vector256<int>.Zero; c20 = Vector256<int>.Zero; c21 = Vector256<int>.Zero; c22 = Vector256<int>.Zero; c23 = Vector256<int>.Zero; c30 = Vector256<int>.Zero; c31 = Vector256<int>.Zero; c32 = Vector256<int>.Zero; c33 = Vector256<int>.Zero; c40 = Vector256<int>.Zero; c41 = Vector256<int>.Zero; c42 = Vector256<int>.Zero; c43 = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++)
        {
            Sse.Prefetch0(b + PrefetchBytes);
            Sse.Prefetch0(b + PrefetchBytes + 64);
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 32); var b2 = Vector256.Load(b + 64); var b3 = Vector256.Load(b + 96);
            { var x = Vector256.Create(*(int*)(a + 0)).AsByte(); c00 = AvxVnni.MultiplyWideningAndAdd(c00, x, b0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, x, b1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, x, b2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 4)).AsByte(); c10 = AvxVnni.MultiplyWideningAndAdd(c10, x, b0); c11 = AvxVnni.MultiplyWideningAndAdd(c11, x, b1); c12 = AvxVnni.MultiplyWideningAndAdd(c12, x, b2); c13 = AvxVnni.MultiplyWideningAndAdd(c13, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 8)).AsByte(); c20 = AvxVnni.MultiplyWideningAndAdd(c20, x, b0); c21 = AvxVnni.MultiplyWideningAndAdd(c21, x, b1); c22 = AvxVnni.MultiplyWideningAndAdd(c22, x, b2); c23 = AvxVnni.MultiplyWideningAndAdd(c23, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 12)).AsByte(); c30 = AvxVnni.MultiplyWideningAndAdd(c30, x, b0); c31 = AvxVnni.MultiplyWideningAndAdd(c31, x, b1); c32 = AvxVnni.MultiplyWideningAndAdd(c32, x, b2); c33 = AvxVnni.MultiplyWideningAndAdd(c33, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 16)).AsByte(); c40 = AvxVnni.MultiplyWideningAndAdd(c40, x, b0); c41 = AvxVnni.MultiplyWideningAndAdd(c41, x, b1); c42 = AvxVnni.MultiplyWideningAndAdd(c42, x, b2); c43 = AvxVnni.MultiplyWideningAndAdd(c43, x, b3); }
            a += 24; b += 128;
        }
        e.Store(0, c + 0 * ldc, c00, c01, c02, c03);
        e.Store(1, c + 1 * ldc, c10, c11, c12, c13);
        e.Store(2, c + 2 * ldc, c20, c21, c22, c23);
        e.Store(3, c + 3 * ldc, c30, c31, c32, c33);
        e.Store(4, c + 4 * ldc, c40, c41, c42, c43);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelR6(byte* a, sbyte* b, int groups, float* c, long ldc, in Epilogue e)
    {
        Vector256<int> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, c40, c41, c42, c43, c50, c51, c52, c53;
        c00 = Vector256<int>.Zero; c01 = Vector256<int>.Zero; c02 = Vector256<int>.Zero; c03 = Vector256<int>.Zero; c10 = Vector256<int>.Zero; c11 = Vector256<int>.Zero; c12 = Vector256<int>.Zero; c13 = Vector256<int>.Zero; c20 = Vector256<int>.Zero; c21 = Vector256<int>.Zero; c22 = Vector256<int>.Zero; c23 = Vector256<int>.Zero; c30 = Vector256<int>.Zero; c31 = Vector256<int>.Zero; c32 = Vector256<int>.Zero; c33 = Vector256<int>.Zero; c40 = Vector256<int>.Zero; c41 = Vector256<int>.Zero; c42 = Vector256<int>.Zero; c43 = Vector256<int>.Zero; c50 = Vector256<int>.Zero; c51 = Vector256<int>.Zero; c52 = Vector256<int>.Zero; c53 = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++)
        {
            Sse.Prefetch0(b + PrefetchBytes);
            Sse.Prefetch0(b + PrefetchBytes + 64);
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 32); var b2 = Vector256.Load(b + 64); var b3 = Vector256.Load(b + 96);
            { var x = Vector256.Create(*(int*)(a + 0)).AsByte(); c00 = AvxVnni.MultiplyWideningAndAdd(c00, x, b0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, x, b1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, x, b2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 4)).AsByte(); c10 = AvxVnni.MultiplyWideningAndAdd(c10, x, b0); c11 = AvxVnni.MultiplyWideningAndAdd(c11, x, b1); c12 = AvxVnni.MultiplyWideningAndAdd(c12, x, b2); c13 = AvxVnni.MultiplyWideningAndAdd(c13, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 8)).AsByte(); c20 = AvxVnni.MultiplyWideningAndAdd(c20, x, b0); c21 = AvxVnni.MultiplyWideningAndAdd(c21, x, b1); c22 = AvxVnni.MultiplyWideningAndAdd(c22, x, b2); c23 = AvxVnni.MultiplyWideningAndAdd(c23, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 12)).AsByte(); c30 = AvxVnni.MultiplyWideningAndAdd(c30, x, b0); c31 = AvxVnni.MultiplyWideningAndAdd(c31, x, b1); c32 = AvxVnni.MultiplyWideningAndAdd(c32, x, b2); c33 = AvxVnni.MultiplyWideningAndAdd(c33, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 16)).AsByte(); c40 = AvxVnni.MultiplyWideningAndAdd(c40, x, b0); c41 = AvxVnni.MultiplyWideningAndAdd(c41, x, b1); c42 = AvxVnni.MultiplyWideningAndAdd(c42, x, b2); c43 = AvxVnni.MultiplyWideningAndAdd(c43, x, b3); }
            { var x = Vector256.Create(*(int*)(a + 20)).AsByte(); c50 = AvxVnni.MultiplyWideningAndAdd(c50, x, b0); c51 = AvxVnni.MultiplyWideningAndAdd(c51, x, b1); c52 = AvxVnni.MultiplyWideningAndAdd(c52, x, b2); c53 = AvxVnni.MultiplyWideningAndAdd(c53, x, b3); }
            a += 24; b += 128;
        }
        e.Store(0, c + 0 * ldc, c00, c01, c02, c03);
        e.Store(1, c + 1 * ldc, c10, c11, c12, c13);
        e.Store(2, c + 2 * ldc, c20, c21, c22, c23);
        e.Store(3, c + 3 * ldc, c30, c31, c32, c33);
        e.Store(4, c + 4 * ldc, c40, c41, c42, c43);
        e.Store(5, c + 5 * ldc, c50, c51, c52, c53);
    }

}
