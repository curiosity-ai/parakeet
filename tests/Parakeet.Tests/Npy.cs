using System.Text;
using System.Text.RegularExpressions;

namespace Parakeet.Tests;

/// <summary>Minimal reader for NumPy .npy files (little-endian f4 / f8 / i8 / i4, C order).</summary>
public sealed class Npy
{
    public required int[] Shape { get; init; }
    public required float[] Data { get; init; }
    public long[] Longs => Data.Select(v => (long)v).ToArray();
    public int[] Ints => Data.Select(v => (int)v).ToArray();

    public static Npy Load(string path)
    {
        using var br = new BinaryReader(File.OpenRead(path));
        var magic = br.ReadBytes(6);
        if (magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY") throw new InvalidDataException(path);
        byte major = br.ReadByte();
        br.ReadByte();
        int hlen = major == 1 ? br.ReadUInt16() : (int)br.ReadUInt32();
        string header = Encoding.ASCII.GetString(br.ReadBytes(hlen));
        string descr = Regex.Match(header, @"'descr':\s*'([^']+)'").Groups[1].Value;
        if (header.Contains("'fortran_order': True")) throw new NotSupportedException("fortran order");
        var shapeStr = Regex.Match(header, @"'shape':\s*\(([^)]*)\)").Groups[1].Value;
        int[] shape = shapeStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        long count = shape.Aggregate(1L, (a, b) => a * b);
        var data = new float[count];
        for (long i = 0; i < count; i++)
        {
            data[i] = descr switch
            {
                "<f4" => br.ReadSingle(),
                "<f8" => (float)br.ReadDouble(),
                "<i8" => br.ReadInt64(),
                "<i4" => br.ReadInt32(),
                _ => throw new NotSupportedException(descr),
            };
        }
        return new Npy { Shape = shape, Data = data };
    }
}
