namespace Parakeet;

/// <summary>The published Parakeet TDT checkpoints this library runs.</summary>
public enum ParakeetModel
{
    /// <summary><c>nvidia/parakeet-tdt-0.6b-v2</c>: English (1024 SentencePiece tokens), ~2.5 GB download.</summary>
    TdtV2,

    /// <summary><c>nvidia/parakeet-tdt-0.6b-v3</c>: 25 European languages (8192 tokens), ~2.5 GB download.</summary>
    TdtV3,
}

/// <summary>Download locations, sizes and cache paths of the <see cref="ParakeetModel"/> checkpoints.</summary>
public static class ParakeetModels
{
    /// <summary>Hugging Face repository id the checkpoint is published under.</summary>
    public static string GetRepository(ParakeetModel model) => model switch
    {
        ParakeetModel.TdtV2 => "nvidia/parakeet-tdt-0.6b-v2",
        ParakeetModel.TdtV3 => "nvidia/parakeet-tdt-0.6b-v3",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>File name of the <c>.nemo</c> checkpoint.</summary>
    public static string GetFileName(ParakeetModel model) => model switch
    {
        ParakeetModel.TdtV2 => "parakeet-tdt-0.6b-v2.nemo",
        ParakeetModel.TdtV3 => "parakeet-tdt-0.6b-v3.nemo",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>Exact size in bytes of the published checkpoint (used to validate downloads and cached files).</summary>
    public static long GetFileSize(ParakeetModel model) => model switch
    {
        ParakeetModel.TdtV2 => 2_472_222_720,
        ParakeetModel.TdtV3 => 2_509_332_480,
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>Default download URL: the unmodified <c>.nemo</c> file on Hugging Face. This is the one place to point
    /// at another host that serves the same bytes (the size check in <see cref="GetFileSize"/> still applies).</summary>
    public static string GetDownloadUrl(ParakeetModel model) =>
        $"https://huggingface.co/{GetRepository(model)}/resolve/main/{GetFileName(model)}";

    /// <summary>Directory downloaded checkpoints are cached in: <c>$PARAKEET_CACHE</c>, else <c>~/.cache/parakeet</c>.</summary>
    public static string CacheDirectory =>
        Environment.GetEnvironmentVariable("PARAKEET_CACHE") is { Length: > 0 } c
            ? c
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "parakeet");

    /// <summary>Default local path of a downloaded checkpoint: <c>&lt;CacheDirectory&gt;/&lt;file&gt;</c>.</summary>
    public static string GetDefaultCachePath(ParakeetModel model) => Path.Combine(CacheDirectory, GetFileName(model));

    /// <summary>Parses a model name: <c>v2</c> / <c>v3</c>, the enum name, the checkpoint name or the Hugging Face repository id
    /// (case-insensitive).</summary>
    public static bool TryParse(string? name, out ParakeetModel model)
    {
        model = default;
        if (string.IsNullOrWhiteSpace(name)) return false;
        string n = name.Trim();
        foreach (var m in Enum.GetValues<ParakeetModel>())
        {
            string file = GetFileName(m);
            string[] aliases = [m.ToString(), GetRepository(m), file, Path.GetFileNameWithoutExtension(file),
                                m == ParakeetModel.TdtV2 ? "v2" : "v3"];
            if (aliases.Any(a => string.Equals(a, n, StringComparison.OrdinalIgnoreCase)))
            {
                model = m;
                return true;
            }
        }
        return false;
    }
}
