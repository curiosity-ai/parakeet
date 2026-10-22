using System.IO.MemoryMappedFiles;

namespace Parakeet.IO;

/// <summary>A read-only memory-mapped file exposed as a raw pointer. Checkpoint tensors are read straight from the
/// mapping (the page cache), so loading a 2.5 GB model costs no intermediate managed copies.</summary>
internal sealed unsafe class MappedFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private byte* _ptr;

    public MappedFile(string path)
    {
        Path = path;
        // Open the file itself (following symbolic links; FileInfo.Length would report a link's own size).
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Length = fs.Length;
        if (Length == 0)
        {
            fs.Dispose();
            throw new InvalidDataException($"{path} is empty.");
        }
        _mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        _view = _mmf.CreateViewAccessor(0, Length, MemoryMappedFileAccess.Read);
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _ptr);
        _ptr += _view.PointerOffset;
    }

    public string Path { get; }
    public long Length { get; }

    public byte* Ptr
    {
        get
        {
            ObjectDisposedException.ThrowIf(_ptr == null, this);
            return _ptr;
        }
    }

    public ReadOnlySpan<byte> Span(long offset, int length)
    {
        if (offset < 0 || offset + length > Length) throw new InvalidDataException($"{Path}: read past the end of the file.");
        return new ReadOnlySpan<byte>(Ptr + offset, length);
    }

    public void Dispose()
    {
        if (_ptr != null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _ptr = null;
        }
        _view.Dispose();
        _mmf.Dispose();
    }
}
