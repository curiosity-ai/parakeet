using Parakeet.IO;
using Parakeet.Numerics;

namespace Parakeet.Model;

/// <summary>Reads named tensors from a checkpoint and converts them into the layouts the kernels use.</summary>
internal sealed unsafe class WeightReader(ITensorSource src, bool hugePages = false)
{
    /// <summary>Whether packed weights are backed by transparent huge pages.</summary>
    public bool HugePages => hugePages;

    public bool Has(string name) => src.Contains(name);

    public Tensor Get(string name, params int[] shape)
    {
        if (!src.Contains(name)) throw new KeyNotFoundException($"Checkpoint has no tensor '{name}'.");
        var t = src.Get(name);
        if (shape.Length > 0 && (t.Length != Tensor.Count(shape)))
            throw new InvalidDataException($"Tensor '{name}' has shape {t.ShapeString}, expected [{string.Join(", ", shape)}].");
        return t;
    }

    /// <summary>Copies a tensor into a new native buffer.</summary>
    public NativeBuffer<float> Vector(string name, int length, float scale = 1f)
    {
        var buf = new NativeBuffer<float>(length);
        var span = Get(name, length).Span;
        for (int i = 0; i < length; i++) buf.Ptr[i] = span[i] * scale;
        return buf;
    }

    public NativeBuffer<float>? OptionalVector(string name, int length, float scale = 1f) => Has(name) ? Vector(name, length, scale) : null;

    /// <summary>Packs a PyTorch linear (or 1×1 conv) weight <c>[out, in(, 1)]</c> as the GEMM operand.</summary>
    public PackedMatrix Linear(string name, int outDim, int inDim, float scale = 1f)
    {
        var t = Get(name, outDim, inDim);
        PackedMatrix? m = null;
        t.With(p => m = PackedMatrix.FromLinear((float*)p, outDim, inDim, inDim, scale, hugePages));
        return m!;
    }

    /// <summary>Packs several linear weights stacked along the output dimension (fused projections).</summary>
    public PackedMatrix FusedLinear(int inDim, params (string Name, int Out)[] parts)
    {
        int n = parts.Sum(p => p.Out);
        using var tmp = new NativeBuffer<float>((long)n * inDim, zero: false);
        long off = 0;
        foreach (var (name, outDim) in parts)
        {
            var span = Get(name, outDim, inDim).Span;
            span.CopyTo(new Span<float>(tmp.Ptr + off, span.Length));
            off += span.Length;
        }
        return PackedMatrix.FromLinear(tmp.Ptr, n, inDim, hugePages: hugePages);
    }

    /// <summary>Concatenates optional biases of fused projections (zeros where absent); null when none exist.</summary>
    public NativeBuffer<float>? FusedBias(params (string Name, int Out)[] parts)
    {
        if (!parts.Any(p => Has(p.Name))) return null;
        var buf = new NativeBuffer<float>(parts.Sum(p => p.Out));
        int off = 0;
        foreach (var (name, outDim) in parts)
        {
            if (Has(name)) Get(name, outDim).Span.CopyTo(new Span<float>(buf.Ptr + off, outDim));
            off += outDim;
        }
        return buf;
    }
}
