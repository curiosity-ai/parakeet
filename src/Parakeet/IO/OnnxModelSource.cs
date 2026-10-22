using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using Parakeet.Model;

namespace Parakeet.IO;

/// <summary>
/// Weights of the ONNX export of Parakeet (<c>encoder-model.onnx</c> [+ <c>.data</c>], <c>decoder_joint-model.onnx</c>,
/// <c>vocab.txt</c>, as published by istupakov / onnx-asr) presented under the PyTorch parameter names of the NeMo
/// checkpoint, read with a minimal protobuf parser (no ONNX Runtime). The exporter renamed every MatMul weight to a
/// numeric initializer and folded batch norm into the depthwise convolutions; names are recovered from the producing
/// node's output (<c>/layers.3/feed_forward1/linear1/MatMul_output_0</c> → <c>encoder.layers.3.feed_forward1.linear1.weight</c>),
/// MatMul weights are transposed back to <c>[out, in]</c> and LSTM gates reordered from ONNX <c>iofc</c> to PyTorch <c>ifgo</c>.
/// Only fp32 exports are supported (not the int8-quantized files).
/// </summary>
public sealed unsafe class OnnxModelSource : ITensorSource
{
    private readonly List<MappedFile> _files = new();
    private readonly Dictionary<string, Func<Tensor>> _tensors = new(StringComparer.Ordinal);

    private sealed record Init(string Name, int[] Dims, int DataType, MappedFile File, long Offset, long Length);
    private sealed record Node(string OpType, string[] Inputs, string[] Outputs);

    private OnnxModelSource(string dir)
    {
        try
        {
            LoadGraph(Path.Combine(dir, "encoder-model.onnx"), "encoder.");
            LoadGraph(Path.Combine(dir, "decoder_joint-model.onnx"), "");
            var vocabFile = Path.Combine(dir, "vocab.txt");
            Vocabulary = Text.Vocabulary.FromFile(vocabFile) is var v ? Enumerable.Range(0, v.Count).Select(i => v[i]).ToArray() : [];
            Config = InferConfig(dir);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public ParakeetConfig Config { get; }
    public string[] Vocabulary { get; }

    public static OnnxModelSource Open(string dir) => new(dir);

    public IEnumerable<string> Names => _tensors.Keys;
    public bool Contains(string name) => _tensors.ContainsKey(name);
    public Tensor Get(string name) => _tensors.TryGetValue(name, out var f) ? f() : throw new KeyNotFoundException($"ONNX model has no tensor '{name}'.");

    private ParakeetConfig InferConfig(string dir)
    {
        var d = new ParakeetConfig();
        int layers = 0;
        while (Contains($"encoder.layers.{layers}.norm_out.weight")) layers++;
        int[] Shape(string n) => Get(n).Shape;
        int dModel = Shape("encoder.layers.0.norm_out.weight")[0];
        var posU = Shape("encoder.layers.0.self_attn.pos_bias_u");
        int ff = Shape("encoder.layers.0.feed_forward1.linear1.weight")[0];
        int kernel = Shape("encoder.layers.0.conv.depthwise_conv.weight")[^1];
        int subCh = Shape("encoder.pre_encode.conv.0.weight")[0];
        var embed = Shape("decoder.prediction.embed.weight");
        int outDim = Shape("joint.joint_net.2.weight")[0];
        int lstm = 0;
        while (Contains($"decoder.prediction.dec_rnn.lstm.weight_ih_l{lstm}")) lstm++;
        int features = d.Mel.Features, factor = d.SubsamplingFactor;
        var cfgPath = Path.Combine(dir, "config.json");
        if (File.Exists(cfgPath))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(cfgPath));
            if (doc.RootElement.TryGetProperty("features_size", out var f)) features = f.GetInt32();
            if (doc.RootElement.TryGetProperty("subsampling_factor", out var s)) factor = s.GetInt32();
        }
        int vocab = embed[0] - 1;
        return d with
        {
            Mel = d.Mel with { Features = features },
            Layers = layers,
            DModel = dModel,
            Heads = posU[0],
            FfExpansion = ff / dModel,
            ConvKernel = kernel,
            SubsamplingFactor = factor,
            SubsamplingChannels = subCh,
            PredHidden = embed[1],
            PredLayers = lstm,
            JointHidden = Shape("joint.pred.weight")[0],
            VocabSize = vocab,
            Durations = Enumerable.Range(0, outDim - vocab - 1).ToArray(),
        };
    }

    private void LoadGraph(string path, string prefix)
    {
        var file = new MappedFile(path);
        _files.Add(file);
        var model = new ProtoReader(file, 0, file.Length);
        long gOff = -1, gLen = 0;
        while (model.Next(out int field, out int wire))
        {
            if (field == 7 && wire == 2) (gOff, gLen) = model.Bytes();
            else model.Skip(wire);
        }
        if (gOff < 0) throw new InvalidDataException($"{path}: no graph in the ONNX model.");

        var inits = new Dictionary<string, Init>(StringComparer.Ordinal);
        var nodes = new List<Node>();
        var externals = new Dictionary<string, MappedFile>(StringComparer.Ordinal);
        var g = new ProtoReader(file, gOff, gLen);
        while (g.Next(out int field, out int wire))
        {
            if (field == 1 && wire == 2) { var (o, l) = g.Bytes(); nodes.Add(ReadNode(file, o, l)); }
            else if (field == 5 && wire == 2) { var (o, l) = g.Bytes(); var t = ReadTensor(file, o, l, path, externals); inits[t.Name] = t; }
            else g.Skip(wire);
        }

        // Semantically named initializers keep their names.
        foreach (var t in inits.Values)
            if (!t.Name.StartsWith("onnx::", StringComparison.Ordinal) && t.DataType == 1)
                Add(prefix + t.Name, t, transpose: false);

        int lstmIndex = 0;
        foreach (var n in nodes)
        {
            switch (n.OpType)
            {
                case "MatMul" when n.Inputs.Length == 2 && inits.TryGetValue(n.Inputs[1], out var w) && w.Dims.Length == 2:
                {
                    string module = ModulePath(n.Outputs[0], "/MatMul_output_0");
                    if (module.Length > 0) Add(prefix + module + ".weight", w, transpose: true);
                    break;
                }
                case "Conv" when n.Inputs.Length >= 2 && inits.TryGetValue(n.Inputs[1], out var w) && w.Name.StartsWith("onnx::", StringComparison.Ordinal):
                {
                    string module = ModulePath(n.Outputs[0], "/Conv_output_0");
                    if (module.Length == 0) break;
                    Add(prefix + module + ".weight", w, transpose: false);
                    if (n.Inputs.Length >= 3 && inits.TryGetValue(n.Inputs[2], out var b)) Add(prefix + module + ".bias", b, transpose: false);
                    break;
                }
                case "LSTM":
                {
                    var W = inits[n.Inputs[1]]; var R = inits[n.Inputs[2]];
                    string p = $"decoder.prediction.dec_rnn.lstm.";
                    int l = lstmIndex++;
                    _tensors[$"{p}weight_ih_l{l}"] = () => LstmGates(W, 0, 1);
                    _tensors[$"{p}weight_hh_l{l}"] = () => LstmGates(R, 0, 1);
                    if (n.Inputs.Length >= 4 && inits.TryGetValue(n.Inputs[3], out var B))
                    {
                        _tensors[$"{p}bias_ih_l{l}"] = () => LstmGates(B, 0, 2);
                        _tensors[$"{p}bias_hh_l{l}"] = () => LstmGates(B, 1, 2);
                    }
                    break;
                }
            }
        }
    }

    /// <summary>"/layers.0/feed_forward1/linear1/MatMul_output_0" → "layers.0.feed_forward1.linear1". Scopes of children
    /// of containers repeat the container name ("/joint/joint_net/joint_net.2/…" → "joint.joint_net.2").</summary>
    private static string ModulePath(string output, string suffix)
    {
        if (!output.StartsWith('/') || !output.EndsWith(suffix, StringComparison.Ordinal)) return "";
        var parts = new List<string>();
        foreach (var scope in output[1..^suffix.Length].Split('/'))
        {
            if (parts.Count > 0 && scope.StartsWith(parts[^1] + ".", StringComparison.Ordinal)) parts[^1] = scope;
            else parts.Add(scope);
        }
        return string.Join('.', parts);
    }

    private void Add(string name, Init t, bool transpose)
    {
        if (t.DataType != 1) throw new NotSupportedException($"Tensor {t.Name} is not float32 (int8-quantized exports are not supported).");
        _tensors[name] = () =>
        {
            float* src = (float*)(t.File.Ptr + t.Offset);
            if (!transpose) return new Tensor(t.Dims, src, t.File);
            int rows = t.Dims[0], cols = t.Dims[1];
            var data = new float[(long)rows * cols];
            fixed (float* d = data) Numerics.Ops.Transpose(src, d, rows, cols);
            return new Tensor([cols, rows], data);
        };
    }

    /// <summary>Slices part <paramref name="part"/> of <paramref name="parts"/> from an ONNX LSTM tensor (<c>[1, 4H, in]</c> or
    /// <c>[1, 8H]</c>) and reorders its gates from ONNX i, o, f, c to PyTorch i, f, g(c), o.</summary>
    private static Tensor LstmGates(Init t, int part, int parts)
    {
        float* src = (float*)(t.File.Ptr + t.Offset);
        long total = t.Length / 4;
        long per = total / parts;
        long gate = per / 4;
        int[] order = [0, 2, 3, 1];   // PyTorch gate g takes ONNX gate order[g]
        var data = new float[per];
        for (int g = 0; g < 4; g++)
            new ReadOnlySpan<float>(src + part * per + order[g] * gate, (int)gate).CopyTo(data.AsSpan((int)(g * gate)));
        int[] shape = t.Dims.Length == 3 ? [t.Dims[1], t.Dims[2]] : [(int)per];
        return new Tensor(shape, data);
    }

    private static Node ReadNode(MappedFile file, long off, long len)
    {
        var r = new ProtoReader(file, off, len);
        var inputs = new List<string>();
        var outputs = new List<string>();
        string op = "";
        while (r.Next(out int field, out int wire))
        {
            if (field == 1 && wire == 2) inputs.Add(r.String());
            else if (field == 2 && wire == 2) outputs.Add(r.String());
            else if (field == 4 && wire == 2) op = r.String();
            else r.Skip(wire);
        }
        return new Node(op, inputs.ToArray(), outputs.ToArray());
    }

    private Init ReadTensor(MappedFile file, long off, long len, string modelPath, Dictionary<string, MappedFile> externals)
    {
        var r = new ProtoReader(file, off, len);
        var dims = new List<int>();
        int dataType = 0;
        string name = "";
        long rawOff = -1, rawLen = 0, floatOff = -1, floatLen = 0;
        string? location = null;
        long extOffset = 0, extLength = -1;
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 0: dims.Add((int)r.Varint()); break;
                case 1 when wire == 2: { var (o, l) = r.Bytes(); var p = new ProtoReader(file, o, l); while (!p.End) dims.Add((int)p.Varint()); break; }
                case 2: dataType = (int)r.Varint(); break;
                case 4 when wire == 2: (floatOff, floatLen) = r.Bytes(); break;
                case 8: name = r.String(); break;
                case 9: (rawOff, rawLen) = r.Bytes(); break;
                case 13:
                {
                    var (o, l) = r.Bytes();
                    var e = new ProtoReader(file, o, l);
                    string key = "", value = "";
                    while (e.Next(out int f2, out int w2))
                    {
                        if (f2 == 1) key = e.String();
                        else if (f2 == 2) value = e.String();
                        else e.Skip(w2);
                    }
                    if (key == "location") location = value;
                    else if (key == "offset") extOffset = long.Parse(value);
                    else if (key == "length") extLength = long.Parse(value);
                    break;
                }
                default: r.Skip(wire); break;
            }
        }
        long count = 1;
        foreach (int dd in dims) count *= dd;
        if (location != null)
        {
            if (!externals.TryGetValue(location, out var ext))
            {
                ext = new MappedFile(Path.Combine(Path.GetDirectoryName(modelPath)!, location));
                _files.Add(ext);
                externals[location] = ext;
            }
            long length = extLength >= 0 ? extLength : count * 4;
            if (extOffset + length > ext.Length) throw new InvalidDataException($"{location}: tensor {name} runs past the file (incomplete download?).");
            return new Init(name, dims.ToArray(), dataType, ext, extOffset, length);
        }
        if (rawOff >= 0) return new Init(name, dims.ToArray(), dataType, file, rawOff, rawLen);
        return new Init(name, dims.ToArray(), dataType, file, floatOff, floatLen);
    }

    public void Dispose()
    {
        foreach (var f in _files) f.Dispose();
        _files.Clear();
    }

    /// <summary>Protobuf wire-format reader over a byte range of a memory-mapped file.</summary>
    private struct ProtoReader(MappedFile file, long offset, long length)
    {
        private long _pos = offset;
        private readonly long _end = offset + length;

        public readonly bool End => _pos >= _end;

        public bool Next(out int field, out int wire)
        {
            if (_pos >= _end) { field = wire = 0; return false; }
            ulong tag = Varint();
            field = (int)(tag >> 3);
            wire = (int)(tag & 7);
            return true;
        }

        public ulong Varint()
        {
            ulong v = 0;
            int shift = 0;
            byte* p = file.Ptr;
            while (true)
            {
                byte b = p[_pos++];
                v |= (ulong)(b & 0x7F) << shift;
                if (b < 0x80) return v;
                shift += 7;
            }
        }

        public (long Offset, long Length) Bytes()
        {
            long len = (long)Varint();
            long o = _pos;
            _pos += len;
            if (_pos > _end) throw new InvalidDataException($"{file.Path}: malformed protobuf.");
            return (o, len);
        }

        public string String()
        {
            var (o, l) = Bytes();
            return Encoding.UTF8.GetString(file.Span(o, (int)l));
        }

        public void Skip(int wire)
        {
            switch (wire)
            {
                case 0: Varint(); break;
                case 1: _pos += 8; break;
                case 2: Bytes(); break;
                case 5: _pos += 4; break;
                default: throw new InvalidDataException($"{file.Path}: unsupported protobuf wire type {wire}.");
            }
        }
    }
}
