namespace Parakeet.IO;

/// <summary>
/// A read-only float32 tensor in contiguous row-major order, either viewing memory owned elsewhere (a memory-mapped
/// checkpoint) or owning a managed array. Shapes follow PyTorch conventions (e.g. linear weights are <c>[out, in]</c>).
/// </summary>
public sealed unsafe class Tensor
{
    private readonly float[]? _array;
    private readonly float* _ptr;
    private readonly object? _owner;

    internal Tensor(int[] shape, float* ptr, object owner)
    {
        Shape = shape;
        _ptr = ptr;
        _owner = owner;
    }

    public Tensor(int[] shape, float[] data)
    {
        if (data.LongLength != Count(shape)) throw new ArgumentException("Data length does not match the shape.");
        Shape = shape;
        _array = data;
    }

    public int[] Shape { get; }
    public long Length => Count(Shape);

    /// <summary>The element span (valid while the source checkpoint is open).</summary>
    public ReadOnlySpan<float> Span => _array != null ? _array : new ReadOnlySpan<float>(_ptr, checked((int)Length));

    public float[] ToArray() => Span.ToArray();

    /// <summary>Runs <paramref name="body"/> with a pointer to the data (pinned when managed).</summary>
    public void With(Action<IntPtr> body)
    {
        if (_array == null)
        {
            body((IntPtr)_ptr);
            GC.KeepAlive(_owner);
            return;
        }
        fixed (float* p = _array) body((IntPtr)p);
    }

    public string ShapeString => "[" + string.Join(", ", Shape) + "]";

    internal static long Count(int[] shape)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        return n;
    }
}

/// <summary>A named collection of tensors (a checkpoint's state dict), using PyTorch parameter names.</summary>
public interface ITensorSource : IDisposable
{
    bool Contains(string name);
    Tensor Get(string name);
    IEnumerable<string> Names { get; }
}
