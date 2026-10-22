using System.Text;

namespace Parakeet.IO;

/// <summary>
/// Reads a PyTorch state dict saved with <c>torch.save</c> (zip format: <c>&lt;root&gt;/data.pkl</c> plus one raw
/// storage file per tensor under <c>&lt;root&gt;/data/</c>) without Python. The zip may itself be a member of another
/// archive (a <c>.nemo</c> tar): it is addressed by its byte range inside the memory-mapped file, and contiguous
/// float32 tensors are returned as views straight into the mapping.
/// </summary>
internal sealed unsafe class TorchCheckpoint
{
    private sealed record StorageRef(string Key, string DType);
    private sealed record TensorRef(StorageRef Storage, long Offset, int[] Shape, long[] Stride);

    private readonly MappedFile _file;
    private readonly Dictionary<string, ArchiveEntry> _members;
    private readonly Dictionary<string, TensorRef> _tensors = new(StringComparer.Ordinal);
    private readonly string _root;

    public TorchCheckpoint(MappedFile file, long start, long length)
    {
        _file = file;
        var entries = ArchiveIndex.ReadZip(file, start, length);
        _members = entries.ToDictionary(e => e.Name, StringComparer.Ordinal);
        var pkl = entries.FirstOrDefault(e => e.Name.EndsWith("/data.pkl", StringComparison.Ordinal) || e.Name == "data.pkl");
        if (pkl.Name == null) throw new InvalidDataException($"{file.Path}: no data.pkl in the PyTorch checkpoint.");
        _root = pkl.Name.Length > "data.pkl".Length ? pkl.Name[..^"data.pkl".Length] : "";
        if (_members.TryGetValue(_root + "byteorder", out var bo) &&
            Encoding.ASCII.GetString(file.Span(bo.Offset, (int)bo.Length)).Trim() != "little")
            throw new NotSupportedException("Big-endian PyTorch checkpoints are not supported.");

        var bytes = file.Span(pkl.Offset, checked((int)pkl.Length)).ToArray();
        var root = new Unpickler(bytes, PersistentLoad, Reduce).Load();
        if (root is not Dictionary<object, object?> dict) throw new InvalidDataException($"{file.Path}: the checkpoint is not a state dict.");
        // Lightning checkpoints nest the weights under "state_dict".
        if (dict.TryGetValue("state_dict", out var sd) && sd is Dictionary<object, object?> inner) dict = inner;
        foreach (var (k, v) in dict)
            if (k is string name && v is TensorRef t) _tensors[name] = t;
    }

    public IEnumerable<string> Names => _tensors.Keys;

    public bool Contains(string name) => _tensors.ContainsKey(name) && IsFloat(_tensors[name].Storage.DType);

    public int[] ShapeOf(string name) => _tensors[name].Shape;

    public Tensor Get(string name)
    {
        if (!_tensors.TryGetValue(name, out var t)) throw new KeyNotFoundException($"Tensor '{name}' not found in {_file.Path}.");
        if (!_members.TryGetValue(_root + "data/" + t.Storage.Key, out var member))
            throw new InvalidDataException($"{_file.Path}: storage {t.Storage.Key} of '{name}' is missing.");
        int elem = t.Storage.DType == "float32" ? 4 : t.Storage.DType is "float16" or "bfloat16" ? 2
                 : throw new NotSupportedException($"Tensor '{name}' has unsupported dtype {t.Storage.DType}.");
        long count = Tensor.Count(t.Shape);
        long maxIndex = t.Offset;
        for (int i = 0; i < t.Shape.Length; i++) maxIndex += (t.Shape[i] - 1L) * t.Stride[i];
        if (count > 0 && (maxIndex + 1) * elem > member.Length)
            throw new InvalidDataException($"{_file.Path}: tensor '{name}' runs past its storage.");
        byte* basePtr = _file.Ptr + member.Offset;

        if (elem == 4 && IsContiguous(t.Shape, t.Stride) && ((long)(basePtr + t.Offset * 4) & 3) == 0)
            return new Tensor(t.Shape, (float*)basePtr + t.Offset, _file);

        // Strided or reduced-precision tensors are materialized as float32.
        var data = new float[count];
        var idx = new int[t.Shape.Length];
        for (long n = 0; n < count; n++)
        {
            long src = t.Offset;
            for (int i = 0; i < idx.Length; i++) src += idx[i] * t.Stride[i];
            data[n] = t.Storage.DType switch
            {
                "float32" => ((float*)basePtr)[src],
                "float16" => (float)((Half*)basePtr)[src],
                _ => BitConverter.Int32BitsToSingle(((ushort*)basePtr)[src] << 16),
            };
            for (int i = idx.Length - 1; i >= 0; i--)
            {
                if (++idx[i] < t.Shape[i]) break;
                idx[i] = 0;
            }
        }
        return new Tensor(t.Shape, data);
    }

    private static bool IsFloat(string dtype) => dtype is "float32" or "float16" or "bfloat16";

    private static bool IsContiguous(int[] shape, long[] stride)
    {
        long expect = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            if (shape[i] != 1 && stride[i] != expect) return false;
            expect *= shape[i];
        }
        return true;
    }

    private static object? PersistentLoad(object? pid)
    {
        // ('storage', <torch.FloatStorage>, key, location, numel)
        if (pid is object?[] { Length: >= 3 } t && t[0] is "storage")
        {
            string dtype = t[1] switch
            {
                PickleGlobal { Name: "FloatStorage" } => "float32",
                PickleGlobal { Name: "HalfStorage" } => "float16",
                PickleGlobal { Name: "BFloat16Storage" } => "bfloat16",
                PickleGlobal { Name: "LongStorage" } => "int64",
                PickleGlobal { Name: "IntStorage" } => "int32",
                PickleGlobal { Name: "BoolStorage" } => "bool",
                PickleGlobal g => g.Name,
                _ => "unknown",
            };
            return new StorageRef(t[2]?.ToString() ?? "", dtype);
        }
        throw new InvalidDataException("Unsupported persistent id in PyTorch checkpoint.");
    }

    private static object? Reduce(PickleCall call)
    {
        if (call.Callable is PickleGlobal { Module: "torch._utils", Name: "_rebuild_tensor_v2" or "_rebuild_tensor" } &&
            call.Args.Length >= 4 && call.Args[0] is StorageRef storage)
        {
            int[] shape = ((object?[])call.Args[2]!).Select(o => checked((int)(long)o!)).ToArray();
            long[] stride = ((object?[])call.Args[3]!).Select(o => (long)o!).ToArray();
            return new TensorRef(storage, (long)call.Args[1]!, shape, stride);
        }
        if (call.Callable is PickleGlobal { Module: "torch._utils", Name: "_rebuild_parameter" } && call.Args.Length >= 1)
            return call.Args[0];
        return call;
    }
}
