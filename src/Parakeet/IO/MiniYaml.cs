using System.Globalization;

namespace Parakeet.IO;

/// <summary>
/// A small block-style YAML reader, enough for NeMo's <c>model_config.yaml</c>: nested maps, block and flow
/// sequences, plain / quoted scalars, comments and literal blocks. Values are <c>Dictionary&lt;string, object?&gt;</c>,
/// <c>List&lt;object?&gt;</c> or <c>string</c>.
/// </summary>
internal static class MiniYaml
{
    private readonly record struct Line(int Indent, string Text);

    public static Dictionary<string, object?> Parse(string text)
    {
        var lines = new List<Line>();
        foreach (var raw in text.Replace("\r\n", "\n").Split('\n'))
        {
            string s = StripComment(raw).TrimEnd();
            if (s.Trim().Length == 0 || s.TrimStart().StartsWith("---", StringComparison.Ordinal)) continue;
            int ind = s.Length - s.TrimStart(' ').Length;
            lines.Add(new Line(ind, s[ind..]));
        }
        int i = 0;
        return lines.Count == 0 ? new() : ParseBlock(lines, ref i, lines[0].Indent) as Dictionary<string, object?> ?? new();
    }

    /// <summary>Looks up a dotted path (<c>encoder.d_model</c>); returns null when absent.</summary>
    public static object? Get(Dictionary<string, object?> root, string path)
    {
        object? cur = root;
        foreach (var part in path.Split('.'))
        {
            if (cur is Dictionary<string, object?> d && d.TryGetValue(part, out var v)) cur = v;
            else return null;
        }
        return cur;
    }

    public static int GetInt(Dictionary<string, object?> root, string path, int fallback) =>
        Get(root, path) is string s && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : fallback;

    public static double GetDouble(Dictionary<string, object?> root, string path, double fallback) =>
        Get(root, path) is string s && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : fallback;

    public static string? GetString(Dictionary<string, object?> root, string path) => Get(root, path) as string;

    public static bool GetBool(Dictionary<string, object?> root, string path, bool fallback) =>
        Get(root, path) is string s ? s.Equals("true", StringComparison.OrdinalIgnoreCase) : fallback;

    public static List<string>? GetList(Dictionary<string, object?> root, string path) =>
        Get(root, path) is List<object?> l ? l.Select(o => o?.ToString() ?? "").ToList() : null;

    private static object? ParseBlock(List<Line> lines, ref int i, int indent)
    {
        if (i >= lines.Count) return null;
        if (IsListItem(lines[i].Text)) return ParseList(lines, ref i, indent);
        var map = new Dictionary<string, object?>(StringComparer.Ordinal);
        while (i < lines.Count && lines[i].Indent == indent && !IsListItem(lines[i].Text))
        {
            string t = lines[i].Text;
            int colon = FindKeyColon(t);
            if (colon < 0) { i++; continue; }
            string key = Unquote(t[..colon].Trim());
            string rest = t[(colon + 1)..].Trim();
            i++;
            map[key] = ParseValue(lines, ref i, indent, rest);
        }
        return map;
    }

    private static object? ParseValue(List<Line> lines, ref int i, int indent, string rest)
    {
        if (rest is "|" or ">" or "|-" or ">-" or "|+" or ">+")
        {
            var parts = new List<string>();
            while (i < lines.Count && lines[i].Indent > indent) parts.Add(lines[i++].Text);
            return string.Join(rest.StartsWith('|') ? "\n" : " ", parts);
        }
        if (rest.Length > 0) return Scalar(rest);
        if (i < lines.Count && (lines[i].Indent > indent || (lines[i].Indent == indent && IsListItem(lines[i].Text))))
            return ParseBlock(lines, ref i, lines[i].Indent);
        return null;
    }

    private static List<object?> ParseList(List<Line> lines, ref int i, int indent)
    {
        var list = new List<object?>();
        while (i < lines.Count && lines[i].Indent == indent && IsListItem(lines[i].Text))
        {
            string content = lines[i].Text.Length > 1 ? lines[i].Text[2..].Trim() : "";
            if (content.Length == 0)
            {
                i++;
                list.Add(i < lines.Count && lines[i].Indent > indent ? ParseBlock(lines, ref i, lines[i].Indent) : null);
            }
            else if (FindKeyColon(content) >= 0 && !content.StartsWith('[') && !content.StartsWith('{'))
            {
                // "- key: value" starts a map whose keys sit at indent + 2.
                lines[i] = new Line(indent + 2, content);
                list.Add(ParseBlock(lines, ref i, indent + 2));
            }
            else
            {
                i++;
                list.Add(Scalar(content));
            }
        }
        return list;
    }

    private static object? Scalar(string s)
    {
        if (s.StartsWith('[') && s.EndsWith(']'))
        {
            string inner = s[1..^1].Trim();
            return inner.Length == 0 ? new List<object?>() : SplitFlow(inner).Select(x => (object?)Unquote(x.Trim())).ToList();
        }
        if (s is "null" or "~") return null;
        return Unquote(s);
    }

    private static IEnumerable<string> SplitFlow(string s)
    {
        int depth = 0, start = 0;
        char quote = '\0';
        for (int k = 0; k < s.Length; k++)
        {
            char c = s[k];
            if (quote != '\0') { if (c == quote) quote = '\0'; continue; }
            if (c is '\'' or '"') quote = c;
            else if (c is '[' or '{') depth++;
            else if (c is ']' or '}') depth--;
            else if (c == ',' && depth == 0) { yield return s[start..k]; start = k + 1; }
        }
        yield return s[start..];
    }

    private static bool IsListItem(string t) => t == "-" || t.StartsWith("- ", StringComparison.Ordinal);

    private static int FindKeyColon(string t)
    {
        char quote = '\0';
        for (int k = 0; k < t.Length; k++)
        {
            char c = t[k];
            if (quote != '\0') { if (c == quote) quote = '\0'; continue; }
            if ((c is '\'' or '"') && k == 0) { quote = c; continue; }
            if (c == ':' && (k + 1 == t.Length || t[k + 1] == ' ')) return k;
        }
        return -1;
    }

    private static string Unquote(string s)
    {
        if (s.Length >= 2 && s[0] == '\'' && s[^1] == '\'') return s[1..^1].Replace("''", "'");
        if (s.Length >= 2 && s[0] == '"' && s[^1] == '"') return System.Text.RegularExpressions.Regex.Unescape(s[1..^1]);
        return s;
    }

    private static string StripComment(string s)
    {
        char quote = '\0';
        for (int k = 0; k < s.Length; k++)
        {
            char c = s[k];
            if (quote != '\0') { if (c == quote) quote = '\0'; continue; }
            if (c is '\'' or '"') quote = c;
            else if (c == '#' && (k == 0 || s[k - 1] == ' ')) return s[..k];
        }
        return s;
    }
}
