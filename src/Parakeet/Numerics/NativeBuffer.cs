using System.Runtime.InteropServices;

namespace Parakeet.Numerics;

/// <summary>A 64-byte aligned block of unmanaged memory. Weights and large activations live here so the
/// SIMD kernels can address them with raw pointers (no pinning, no bounds checks, no GC moves).</summary>
public sealed unsafe class NativeBuffer<T> : IDisposable where T : unmanaged
{
    private T* _ptr;

    /// <param name="length">Number of elements.</param>
    /// <param name="zero">Clear the memory (otherwise its contents are undefined).</param>
    /// <param name="hugePages">On Linux, back large buffers with transparent huge pages (fewer TLB misses and page faults,
    /// but the kernel may compact memory to provide them, which can stall the first touch for seconds).</param>
    public NativeBuffer(long length, bool zero = true, bool hugePages = true)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        Length = length;
        nuint bytes = (nuint)Math.Max(1, length * sizeof(T));
        if (hugePages && bytes >= HugePageThreshold && OperatingSystem.IsLinux())
        {
            // Large buffers: 2 MB-aligned and advised as transparent huge pages, so first touch costs one
            // fault per 2 MB instead of per 4 KB (activations at audio rate are hundreds of MB).
            bytes = (bytes + HugePage - 1) & ~(nuint)(HugePage - 1);
            _ptr = (T*)NativeMemory.AlignedAlloc(bytes, HugePage);
            LinuxMemory.AdviseHugePages(_ptr, bytes);
        }
        else
        {
            _ptr = (T*)NativeMemory.AlignedAlloc(bytes, 64);
        }
        if (zero) NativeMemory.Clear(_ptr, bytes);
        GC.AddMemoryPressure((long)bytes);
    }

    public long Length { get; }

    private const nuint HugePage = 2 * 1024 * 1024;
    private const nuint HugePageThreshold = 8 * 1024 * 1024;

    public T* Ptr
    {
        get
        {
            ObjectDisposedException.ThrowIf(_ptr == null, this);
            return _ptr;
        }
    }

    public Span<T> AsSpan() => new(Ptr, checked((int)Length));

    public Span<T> AsSpan(long start, int length)
    {
        if (start < 0 || start + length > Length) throw new ArgumentOutOfRangeException(nameof(start));
        return new Span<T>(Ptr + start, length);
    }

    public void Clear() => NativeMemory.Clear(Ptr, (nuint)(Length * sizeof(T)));

    public void Dispose()
    {
        if (_ptr != null)
        {
            NativeMemory.AlignedFree(_ptr);
            _ptr = null;
            GC.RemoveMemoryPressure(Math.Max(1, Length * sizeof(T)));
        }
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer()
    {
        if (_ptr != null) NativeMemory.AlignedFree(_ptr);
    }
}

internal static unsafe class LinuxMemory
{
    private const int MADV_HUGEPAGE = 14;
    private static bool _unavailable;

    [DllImport("libc", SetLastError = false)]
    private static extern int madvise(void* addr, nuint length, int advice);

    public static void AdviseHugePages(void* p, nuint bytes)
    {
        if (_unavailable) return;
        try
        {
            _ = madvise(p, bytes, MADV_HUGEPAGE);
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            _unavailable = true;
        }
    }
}
