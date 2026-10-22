using System.Net.Http.Headers;
using System.Text.Json;

namespace Parakeet.IO;

/// <summary>Fetches a Parakeet checkpoint from the Hugging Face Hub (the repository's <c>.nemo</c> file) into a local
/// cache, resuming interrupted downloads and verifying the final size.</summary>
public static class ModelDownloader
{
    public const string DefaultRepo = "nvidia/parakeet-tdt-0.6b-v2";

    /// <summary>Default cache location: <c>$PARAKEET_CACHE</c> or <c>~/.cache/parakeet</c>.</summary>
    public static string DefaultCacheDir =>
        Environment.GetEnvironmentVariable("PARAKEET_CACHE") is { Length: > 0 } c
            ? c
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "parakeet");

    /// <summary>Returns <paramref name="modelOrRepo"/> when it is an existing file or directory, otherwise downloads the
    /// repository's <c>.nemo</c> checkpoint (skipping a complete local copy) and returns its path.</summary>
    public static async Task<string> EnsureAsync(string modelOrRepo, string? cacheDir = null, IProgress<string>? progress = null,
                                                 CancellationToken ct = default)
    {
        if (File.Exists(modelOrRepo) || Directory.Exists(modelOrRepo)) return modelOrRepo;
        string dir = Path.Combine(cacheDir ?? DefaultCacheDir, modelOrRepo.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(dir);
        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        if (Environment.GetEnvironmentVariable("HF_TOKEN") is { Length: > 0 } token)
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);

        var (file, size) = await FindNemoFileAsync(http, modelOrRepo, ct).ConfigureAwait(false);
        string dest = Path.Combine(dir, file);
        if (File.Exists(dest) && (size <= 0 || ResolvedLength(dest) == size)) return dest;

        string partial = dest + ".partial";
        string url = $"https://huggingface.co/{modelOrRepo}/resolve/main/{file}";
        for (int attempt = 1; ; attempt++)
        {
            long have = File.Exists(partial) ? new FileInfo(partial).Length : 0;
            try
            {
                progress?.Report($"Downloading {modelOrRepo}/{file} ({FormatSize(size)}){(have > 0 ? $", resuming at {FormatSize(have)}" : "")} ...");
                using var req = new HttpRequestMessage(HttpMethod.Get, url);
                if (have > 0) req.Headers.Range = new RangeHeaderValue(have, null);
                using var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
                resp.EnsureSuccessStatusCode();
                bool append = have > 0 && resp.StatusCode == System.Net.HttpStatusCode.PartialContent;
                await using (var fs = new FileStream(partial, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
                await using (var body = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
                    await body.CopyToAsync(fs, 1 << 20, ct).ConfigureAwait(false);
                long got = new FileInfo(partial).Length;
                if (size > 0 && got != size) throw new IOException($"download incomplete ({got} of {size} bytes)");
                File.Move(partial, dest, overwrite: true);
                return dest;
            }
            catch (Exception e) when (attempt < 5 && e is IOException or HttpRequestException && !ct.IsCancellationRequested)
            {
                progress?.Report($"Download interrupted ({e.Message}), retrying ...");
                await Task.Delay(TimeSpan.FromSeconds(2 << attempt), ct).ConfigureAwait(false);
            }
        }
    }

    private static async Task<(string File, long Size)> FindNemoFileAsync(HttpClient http, string repo, CancellationToken ct)
    {
        try
        {
            using var resp = await http.GetAsync($"https://huggingface.co/api/models/{repo}/tree/main", ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false));
            foreach (var e in doc.RootElement.EnumerateArray())
            {
                string path = e.GetProperty("path").GetString() ?? "";
                if (path.EndsWith(".nemo", StringComparison.Ordinal))
                    return (path, e.TryGetProperty("size", out var s) ? s.GetInt64() : -1);
            }
            throw new FileNotFoundException($"Repository {repo} has no .nemo file.");
        }
        catch (HttpRequestException)
        {
            // Offline / API unavailable: fall back to the conventional file name.
            return (repo[(repo.LastIndexOf('/') + 1)..] + ".nemo", -1);
        }
    }

    /// <summary>Length of the file, following a symbolic link (FileInfo.Length reports the link itself).</summary>
    private static long ResolvedLength(string path)
    {
        var fi = new FileInfo(path);
        return (fi.LinkTarget != null ? fi.ResolveLinkTarget(true) as FileInfo : null)?.Length ?? fi.Length;
    }

    private static string FormatSize(long bytes) => bytes <= 0 ? "unknown size" : $"{bytes / 1e9:F2} GB";
}
