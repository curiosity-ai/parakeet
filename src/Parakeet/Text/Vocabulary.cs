using System.Text;

namespace Parakeet.Text;

/// <summary>SentencePiece pieces by id; decoding concatenates pieces and turns the word-boundary mark ▁ into spaces.</summary>
public sealed class Vocabulary
{
    public const char WordBoundary = '▁';

    private readonly string[] _pieces;

    public Vocabulary(IReadOnlyList<string> pieces)
    {
        _pieces = pieces.ToArray();
    }

    /// <summary>Number of pieces (the blank symbol is not included).</summary>
    public int Count => _pieces.Length;

    public string this[int id] => _pieces[id];

    /// <summary>True for control pieces such as <c>&lt;unk&gt;</c> that never produce text.</summary>
    public bool IsSpecial(int id) => id < 0 || id >= _pieces.Length || (_pieces[id].StartsWith('<') && _pieces[id].EndsWith('>'));

    /// <summary>Whether the piece starts a new word.</summary>
    public bool StartsWord(int id) => !IsSpecial(id) && _pieces[id].Length > 0 && _pieces[id][0] == WordBoundary;

    /// <summary>The text a piece contributes (▁ → space; empty for special pieces).</summary>
    public string PieceText(int id) => IsSpecial(id) ? "" : _pieces[id].Replace(WordBoundary, ' ');

    /// <summary>Detokenizes ids into text, collapsing runs of whitespace and trimming the ends.</summary>
    public string Decode(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (int id in ids) sb.Append(PieceText(id));
        return Normalize(sb.ToString());
    }

    internal static string Normalize(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool space = false;
        foreach (char c in s)
        {
            if (char.IsWhiteSpace(c)) { space = true; continue; }
            if (space && sb.Length > 0) sb.Append(' ');
            space = false;
            sb.Append(c);
        }
        return sb.ToString();
    }

    /// <summary>Reads <c>vocab.txt</c> of the ONNX export (<c>piece id</c> per line) or a SentencePiece <c>.vocab</c>
    /// file (<c>piece\tscore</c>, id = line number). The blank (<c>&lt;blk&gt;</c>) entry, if present, is dropped.</summary>
    public static Vocabulary FromFile(string path)
    {
        var lines = File.ReadAllLines(path, Encoding.UTF8).Where(l => l.Length > 0).ToArray();
        var pieces = new SortedDictionary<int, string>();
        for (int i = 0; i < lines.Length; i++)
        {
            string l = lines[i];
            int tab = l.IndexOf('\t');
            if (tab >= 0)
            {
                pieces[i] = l[..tab];
                continue;
            }
            int sp = l.LastIndexOf(' ');
            if (sp > 0 && int.TryParse(l[(sp + 1)..], out int id)) pieces[id] = l[..sp];
            else pieces[i] = l;
        }
        var list = pieces.Values.ToList();
        if (list.Count > 0 && list[^1] == "<blk>") list.RemoveAt(list.Count - 1);
        return new Vocabulary(list);
    }
}
