using System.Buffers.Binary;
using System.Text;

namespace Parakeet.IO;

/// <summary>An uncompressed member of a tar or zip archive: its name and the byte range of its data in the file.</summary>
internal readonly record struct ArchiveEntry(string Name, long Offset, long Length);

/// <summary>Minimal readers for the two containers of a <c>.nemo</c> checkpoint: the outer (uncompressed) tar
/// archive and the PyTorch zip file inside it. Only member offsets are computed; data is accessed in place.</summary>
internal static class ArchiveIndex
{
    /// <summary>Lists the regular files of a POSIX/GNU tar archive stored in <c>[start, start + length)</c>.</summary>
    public static List<ArchiveEntry> ReadTar(MappedFile file, long start = 0, long length = -1)
    {
        if (length < 0) length = file.Length - start;
        var entries = new List<ArchiveEntry>();
        long pos = start, end = start + length;
        string? longName = null;
        while (pos + 512 <= end)
        {
            var h = file.Span(pos, 512);
            if (h.IndexOfAnyExcept((byte)0) < 0) break;   // end-of-archive marker
            string name = CString(h[..100]);
            long size = ParseOctal(h.Slice(124, 12));
            char type = (char)h[156];
            // The 155-byte name prefix exists only in POSIX ustar headers ("ustar\0"), not in GNU ones ("ustar  ").
            string prefix = h.Slice(257, 6).SequenceEqual("ustar\0"u8) ? CString(h.Slice(345, 155)) : "";
            long data = pos + 512;
            if (data + size > end) throw new InvalidDataException($"{file.Path}: truncated tar archive (incomplete download?).");
            if (type == 'L')
            {
                longName = Encoding.UTF8.GetString(file.Span(data, (int)size)).TrimEnd('\0');
            }
            else
            {
                string full = longName ?? (prefix.Length > 0 ? prefix + "/" + name : name);
                longName = null;
                if (type is '0' or '\0') entries.Add(new ArchiveEntry(Normalize(full), data, size));
            }
            pos = data + (size + 511) / 512 * 512;
        }
        return entries;
    }

    /// <summary>Lists the members of a zip archive stored in <c>[start, start + length)</c>; only stored (uncompressed)
    /// members can be read in place, which is what <c>torch.save</c> writes.</summary>
    public static List<ArchiveEntry> ReadZip(MappedFile file, long start, long length)
    {
        long end = start + length;
        // End of central directory record: scan back over a possible comment.
        long eocd = -1;
        for (long p = end - 22; p >= Math.Max(start, end - 22 - 65535); p--)
        {
            if (BinaryPrimitives.ReadUInt32LittleEndian(file.Span(p, 4)) == 0x06054b50) { eocd = p; break; }
        }
        if (eocd < 0) throw new InvalidDataException($"{file.Path}: zip end-of-central-directory record not found.");
        var e = file.Span(eocd, 22);
        long count = BinaryPrimitives.ReadUInt16LittleEndian(e[10..]);
        long cdOffset = BinaryPrimitives.ReadUInt32LittleEndian(e[16..]);
        if (count == 0xFFFF || cdOffset == 0xFFFFFFFF)
        {
            // Zip64: the locator sits right before the EOCD record and points at the zip64 EOCD record.
            long loc = eocd - 20;
            if (BinaryPrimitives.ReadUInt32LittleEndian(file.Span(loc, 4)) != 0x07064b50)
                throw new InvalidDataException($"{file.Path}: zip64 locator not found.");
            long z64 = start + (long)BinaryPrimitives.ReadUInt64LittleEndian(file.Span(loc + 8, 8));
            var r = file.Span(z64, 56);
            if (BinaryPrimitives.ReadUInt32LittleEndian(r) != 0x06064b50) throw new InvalidDataException($"{file.Path}: bad zip64 record.");
            count = (long)BinaryPrimitives.ReadUInt64LittleEndian(r[32..]);
            cdOffset = (long)BinaryPrimitives.ReadUInt64LittleEndian(r[48..]);
        }
        var entries = new List<ArchiveEntry>((int)count);
        long pos = start + cdOffset;
        for (long i = 0; i < count; i++)
        {
            var c = file.Span(pos, 46);
            if (BinaryPrimitives.ReadUInt32LittleEndian(c) != 0x02014b50) throw new InvalidDataException($"{file.Path}: bad zip central directory.");
            int method = BinaryPrimitives.ReadUInt16LittleEndian(c[10..]);
            long csize = BinaryPrimitives.ReadUInt32LittleEndian(c[20..]);
            long usize = BinaryPrimitives.ReadUInt32LittleEndian(c[24..]);
            int nameLen = BinaryPrimitives.ReadUInt16LittleEndian(c[28..]);
            int extraLen = BinaryPrimitives.ReadUInt16LittleEndian(c[30..]);
            int commentLen = BinaryPrimitives.ReadUInt16LittleEndian(c[32..]);
            long local = BinaryPrimitives.ReadUInt32LittleEndian(c[42..]);
            string name = Encoding.UTF8.GetString(file.Span(pos + 46, nameLen));
            // Zip64 extended information: 64-bit values for the fields saturated at 0xFFFFFFFF, in this order.
            var extra = file.Span(pos + 46 + nameLen, extraLen);
            while (extra.Length >= 4)
            {
                int id = BinaryPrimitives.ReadUInt16LittleEndian(extra);
                int sz = BinaryPrimitives.ReadUInt16LittleEndian(extra[2..]);
                if (id == 0x0001)
                {
                    var z = extra.Slice(4, sz);
                    if (usize == 0xFFFFFFFF) { usize = (long)BinaryPrimitives.ReadUInt64LittleEndian(z); z = z[8..]; }
                    if (csize == 0xFFFFFFFF) { csize = (long)BinaryPrimitives.ReadUInt64LittleEndian(z); z = z[8..]; }
                    if (local == 0xFFFFFFFF) local = (long)BinaryPrimitives.ReadUInt64LittleEndian(z);
                }
                extra = extra[Math.Min(extra.Length, 4 + sz)..];
            }
            var lh = file.Span(start + local, 30);
            if (BinaryPrimitives.ReadUInt32LittleEndian(lh) != 0x04034b50) throw new InvalidDataException($"{file.Path}: bad zip local header for {name}.");
            long data = start + local + 30 + BinaryPrimitives.ReadUInt16LittleEndian(lh[26..]) + BinaryPrimitives.ReadUInt16LittleEndian(lh[28..]);
            if (method == 0)
            {
                if (data + usize > end) throw new InvalidDataException($"{file.Path}: zip member {name} runs past the archive.");
                entries.Add(new ArchiveEntry(name, data, usize));
            }
            pos += 46 + nameLen + extraLen + commentLen;
        }
        return entries;
    }

    private static string Normalize(string name) => name.StartsWith("./", StringComparison.Ordinal) ? name[2..] : name;

    private static string CString(ReadOnlySpan<byte> b)
    {
        int n = b.IndexOf((byte)0);
        return Encoding.UTF8.GetString(n < 0 ? b : b[..n]);
    }

    private static long ParseOctal(ReadOnlySpan<byte> b)
    {
        if ((b[0] & 0x80) != 0)
        {
            // GNU base-256 encoding for sizes ≥ 8 GB.
            long v = b[0] & 0x7F;
            for (int i = 1; i < b.Length; i++) v = (v << 8) | b[i];
            return v;
        }
        long r = 0;
        foreach (byte c in b)
        {
            if (c is (byte)' ' or 0) { if (r > 0) break; continue; }
            r = r * 8 + (c - '0');
        }
        return r;
    }
}
