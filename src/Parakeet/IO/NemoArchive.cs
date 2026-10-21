using System.Text;
using Parakeet.Model;

namespace Parakeet.IO;

/// <summary>
/// A NeMo checkpoint: either a <c>.nemo</c> file (an uncompressed tar holding <c>model_config.yaml</c>,
/// <c>model_weights.ckpt</c> and the SentencePiece tokenizer files) or a directory with the same files extracted.
/// The archive is memory-mapped and tensors are read in place.
/// </summary>
public sealed class NemoArchive : ITensorSource
{
    private readonly List<MappedFile> _files = new();
    private readonly TorchCheckpoint _ckpt;

    private NemoArchive(string path)
    {
        Path = path;
        try
        {
            string yaml;
            if (Directory.Exists(path))
            {
                yaml = File.ReadAllText(System.IO.Path.Combine(path, "model_config.yaml"));
                var ckpt = Map(System.IO.Path.Combine(path, "model_weights.ckpt"));
                _ckpt = new TorchCheckpoint(ckpt, 0, ckpt.Length);
                Vocabulary = ReadVocabulary(Directory.GetFiles(path).Select(f => (System.IO.Path.GetFileName(f), File.ReadAllBytes(f) as byte[])).Where(f => IsVocabFile(f.Item1)));
            }
            else
            {
                var tar = Map(path);
                var entries = ArchiveIndex.ReadTar(tar).ToDictionary(e => System.IO.Path.GetFileName(e.Name), StringComparer.Ordinal);
                if (!entries.TryGetValue("model_config.yaml", out var cfg) || !entries.TryGetValue("model_weights.ckpt", out var w))
                    throw new InvalidDataException($"{path} is not a NeMo checkpoint (model_config.yaml / model_weights.ckpt missing).");
                yaml = Encoding.UTF8.GetString(tar.Span(cfg.Offset, (int)cfg.Length));
                _ckpt = new TorchCheckpoint(tar, w.Offset, w.Length);
                Vocabulary = ReadVocabulary(entries.Values.Where(e => IsVocabFile(e.Name))
                    .Select(e => (e.Name, tar.Span(e.Offset, (int)e.Length).ToArray())));
            }
            Yaml = MiniYaml.Parse(yaml);
            Config = ParakeetConfig.FromNemoYaml(Yaml);
            if (Vocabulary.Length == 0)
                Vocabulary = MiniYaml.GetList(Yaml, "joint.vocabulary")?.ToArray() ?? MiniYaml.GetList(Yaml, "labels")?.ToArray() ?? [];
            if (Vocabulary.Length != Config.VocabSize)
                throw new InvalidDataException($"Tokenizer has {Vocabulary.Length} pieces but the model expects {Config.VocabSize}.");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public string Path { get; }
    public ParakeetConfig Config { get; }
    /// <summary>SentencePiece pieces by id (without the blank).</summary>
    public string[] Vocabulary { get; private set; } = [];
    internal Dictionary<string, object?> Yaml { get; }

    public static NemoArchive Open(string path) => new(path);

    private MappedFile Map(string file)
    {
        var f = new MappedFile(file);
        _files.Add(f);
        return f;
    }

    public IEnumerable<string> Names => _ckpt.Names;
    public bool Contains(string name) => _ckpt.Contains(name);
    public Tensor Get(string name) => _ckpt.Get(name);

    private static bool IsVocabFile(string name) => name.EndsWith("tokenizer.vocab", StringComparison.Ordinal);

    /// <summary>SentencePiece <c>.vocab</c>: one <c>piece\tscore</c> line per id.</summary>
    private static string[] ReadVocabulary(IEnumerable<(string Name, byte[] Data)> files)
    {
        foreach (var (_, data) in files)
        {
            var lines = Encoding.UTF8.GetString(data).Split('\n');
            return lines.Where(l => l.Length > 0).Select(l => l.Split('\t')[0]).ToArray();
        }
        return [];
    }

    public void Dispose()
    {
        foreach (var f in _files) f.Dispose();
        _files.Clear();
    }
}
