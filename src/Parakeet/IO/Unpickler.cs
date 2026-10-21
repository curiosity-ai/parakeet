using System.Buffers.Binary;
using System.Text;

namespace Parakeet.IO;

/// <summary>A reference to a Python global (<c>module.name</c>) as produced by the GLOBAL opcodes.</summary>
internal sealed record PickleGlobal(string Module, string Name)
{
    public override string ToString() => Module + "." + Name;
}

/// <summary>A call <c>callable(*args)</c> that the unpickler does not evaluate itself (REDUCE / NEWOBJ).</summary>
internal sealed record PickleCall(object Callable, object?[] Args)
{
    public object? State { get; set; }
}

/// <summary>
/// A small pickle (protocols 2-5) interpreter covering what <c>torch.save</c> emits for a state dict. Objects become
/// plain C# values: tuples <c>object?[]</c>, lists <c>List&lt;object?&gt;</c>, dicts <c>Dictionary&lt;object, object?&gt;</c>,
/// strings, longs, doubles, booleans. <c>collections.OrderedDict()</c> is evaluated to a dictionary; any other call
/// is returned as a <see cref="PickleCall"/> for <paramref name="reduce"/> to interpret. Persistent ids (storage
/// references) are passed to <paramref name="persistentLoad"/>.
/// </summary>
internal sealed class Unpickler(ReadOnlyMemory<byte> data, Func<object?, object?> persistentLoad, Func<PickleCall, object?>? reduce = null)
{
    private static readonly object Mark = new();
    private readonly List<object?> _stack = new();
    private readonly Dictionary<long, object?> _memo = new();
    private int _pos;

    public object? Load()
    {
        var d = data.Span;
        while (true)
        {
            byte op = d[_pos++];
            switch (op)
            {
                case 0x80: _pos++; break;                                   // PROTO
                case 0x95: _pos += 8; break;                                // FRAME
                case (byte)'.': return Pop();                               // STOP
                case (byte)'(': _stack.Add(Mark); break;                    // MARK
                case (byte)'}': _stack.Add(new Dictionary<object, object?>()); break;
                case (byte)']': _stack.Add(new List<object?>()); break;
                case (byte)')': _stack.Add(Array.Empty<object?>()); break;
                case (byte)'N': _stack.Add(null); break;
                case 0x88: _stack.Add(true); break;                         // NEWTRUE
                case 0x89: _stack.Add(false); break;                        // NEWFALSE
                case (byte)'K': _stack.Add((long)d[_pos++]); break;         // BININT1
                case (byte)'M': _stack.Add((long)BinaryPrimitives.ReadUInt16LittleEndian(d[_pos..])); _pos += 2; break;
                case (byte)'J': _stack.Add((long)BinaryPrimitives.ReadInt32LittleEndian(d[_pos..])); _pos += 4; break;
                case 0x8a:                                                  // LONG1
                {
                    int n = d[_pos++];
                    long v = 0;
                    for (int i = n - 1; i >= 0; i--) v = (v << 8) | d[_pos + i];
                    if (n > 0 && n < 8 && (d[_pos + n - 1] & 0x80) != 0) v -= 1L << (8 * n);
                    _pos += n;
                    _stack.Add(v);
                    break;
                }
                case (byte)'G': _stack.Add(BinaryPrimitives.ReadDoubleBigEndian(d[_pos..])); _pos += 8; break;
                case (byte)'X': { int n = BinaryPrimitives.ReadInt32LittleEndian(d[_pos..]); _pos += 4; _stack.Add(Encoding.UTF8.GetString(d.Slice(_pos, n))); _pos += n; break; }
                case 0x8c: { int n = d[_pos++]; _stack.Add(Encoding.UTF8.GetString(d.Slice(_pos, n))); _pos += n; break; }
                case (byte)'C': { int n = d[_pos++]; _stack.Add(d.Slice(_pos, n).ToArray()); _pos += n; break; }
                case (byte)'B': { int n = BinaryPrimitives.ReadInt32LittleEndian(d[_pos..]); _pos += 4; _stack.Add(d.Slice(_pos, n).ToArray()); _pos += n; break; }
                case (byte)'c':                                             // GLOBAL module\nname\n
                {
                    string module = ReadLine(d), name = ReadLine(d);
                    _stack.Add(new PickleGlobal(module, name));
                    break;
                }
                case 0x93:                                                  // STACK_GLOBAL
                {
                    var name = (string)Pop()!;
                    var module = (string)Pop()!;
                    _stack.Add(new PickleGlobal(module, name));
                    break;
                }
                case (byte)'q': _memo[d[_pos++]] = Peek(); break;           // BINPUT
                case (byte)'r': _memo[BinaryPrimitives.ReadUInt32LittleEndian(d[_pos..])] = Peek(); _pos += 4; break;
                case 0x94: _memo[_memo.Count] = Peek(); break;              // MEMOIZE
                case (byte)'h': _stack.Add(_memo[d[_pos++]]); break;        // BINGET
                case (byte)'j': _stack.Add(_memo[BinaryPrimitives.ReadUInt32LittleEndian(d[_pos..])]); _pos += 4; break;
                case (byte)'t': _stack.Add(PopMark().ToArray()); break;     // TUPLE
                case 0x85: { var a = Pop(); _stack.Add(new[] { a }); break; }
                case 0x86: { var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b }); break; }
                case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b, c }); break; }
                case (byte)'a': { var v = Pop(); ((List<object?>)Peek()!).Add(v); break; }   // APPEND
                case (byte)'e': { var items = PopMark(); ((List<object?>)Peek()!).AddRange(items); break; }
                case (byte)'s': { var v = Pop(); var k = Pop()!; ((Dictionary<object, object?>)Peek()!)[k] = v; break; }
                case (byte)'u':                                             // SETITEMS
                {
                    var items = PopMark();
                    var dict = (Dictionary<object, object?>)Peek()!;
                    for (int i = 0; i + 1 < items.Count; i += 2) dict[items[i]!] = items[i + 1];
                    break;
                }
                case (byte)'Q': _stack.Add(persistentLoad(Pop())); break;  // BINPERSID
                case (byte)'R':                                             // REDUCE
                case 0x81:                                                  // NEWOBJ
                {
                    var args = (object?[])Pop()!;
                    var callable = Pop()!;
                    _stack.Add(Call(callable, args));
                    break;
                }
                case (byte)'b':                                             // BUILD: state is ignored except on opaque calls
                {
                    var state = Pop();
                    if (Peek() is PickleCall pc) pc.State = state;
                    break;
                }
                default:
                    throw new InvalidDataException($"Unsupported pickle opcode 0x{op:x2} at offset {_pos - 1}.");
            }
        }
    }

    private object? Call(object callable, object?[] args)
    {
        if (callable is PickleGlobal { Module: "collections", Name: "OrderedDict" }) return new Dictionary<object, object?>();
        var call = new PickleCall(callable, args);
        return reduce != null ? reduce(call) : call;
    }

    private string ReadLine(ReadOnlySpan<byte> d)
    {
        int n = d[_pos..].IndexOf((byte)'\n');
        string s = Encoding.UTF8.GetString(d.Slice(_pos, n));
        _pos += n + 1;
        return s;
    }

    private object? Pop()
    {
        var v = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        return v;
    }

    private object? Peek() => _stack[^1];

    private List<object?> PopMark()
    {
        int m = _stack.LastIndexOf(Mark);
        var items = _stack.GetRange(m + 1, _stack.Count - m - 1);
        _stack.RemoveRange(m, _stack.Count - m);
        return items;
    }
}
