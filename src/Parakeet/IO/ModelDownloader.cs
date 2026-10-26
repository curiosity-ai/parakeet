using System.Net.Http.Headers;

namespace Parakeet.IO;

/// <summary>Progress of a model download: bytes received so far, the total when known, and their ratio (0 when unknown).</summary>
public sealed record DownloadProgress(long DownloadedBytes, long? TotalBytes, float Fraction, string FileName);

/// <summary>
/// Resumable, single-flight model downloader. Files are streamed to a <c>*.download</c> sibling and only moved to their
/// final name once complete (and, when known, of the expected size), so an interrupted download never leaves a truncated
/// file that a later run would mistake for a model. A dropped connection is retried with HTTP range requests, and a
/// <c>*.download</c> left by an interrupted process is resumed rather than started over.
/// </summary>
public static class ModelDownloader
{
    private static readonly SemaphoreSlim OneDownloadAtATime = new(1, 1);
    private static readonly HttpClient Client = CreateClient();

    private static HttpClient CreateClient()
    {
        var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("ParakeetSharp/1.0");
        return client;
    }

    /// <summary>Returns true while a download is in progress.</summary>
    public static bool IsDownloading() => OneDownloadAtATime.CurrentCount < 1;

    /// <summary>
    /// Downloads <paramref name="url"/> to <paramref name="localPath"/> unless the file already exists (and, when
    /// <paramref name="expectedSize"/> is given, has exactly that size). Retries with HTTP range requests when the
    /// connection drops. Sends <c>$HF_TOKEN</c> as a bearer token to huggingface.co, for gated repositories.
    /// </summary>
    public static async Task DownloadFileAsync(string url, string localPath, Action<DownloadProgress>? reportProgress = null,
                                               long? expectedSize = null, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(localPath)) throw new ArgumentException("Local path must be non-empty.", nameof(localPath));
        if (IsComplete(localPath, expectedSize)) return;
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http"))
            throw new InvalidOperationException($"Invalid model URL: '{url}'. Use a valid http(s) URL.");
        string? directory = Path.GetDirectoryName(Path.GetFullPath(localPath));
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        string fileName = Path.GetFileName(localPath);
        string tempPath = localPath + ".download";
        long? totalBytes = expectedSize;

        await OneDownloadAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (IsComplete(localPath, expectedSize)) return;   // another caller finished it while we waited
            if (File.Exists(localPath)) File.Delete(localPath); // wrong size: a leftover of an older or corrupt download

            await using var fileStream = new FileStream(tempPath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
            long received = fileStream.Length;
            if (expectedSize is long want && received > want) received = 0;   // not a prefix of this file
            fileStream.SetLength(received);
            fileStream.Position = received;

            var buffer = new byte[1 << 20];
            long lastReport = 0;
            int failures = 0;
            Report(received);
            while (true)
            {
                try
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, uri);
                    if (uri.Host.EndsWith("huggingface.co", StringComparison.OrdinalIgnoreCase) &&
                        Environment.GetEnvironmentVariable("HF_TOKEN") is { Length: > 0 } token)
                        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                    if (received > 0) request.Headers.Range = new RangeHeaderValue(received, null);
                    using var response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                    if (received > 0 && response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
                    {
                        if (expectedSize is long full && received == full) break;   // the partial file is already complete
                        received = 0;                                              // stale partial file: start over
                        fileStream.SetLength(0);
                        fileStream.Position = 0;
                        if (++failures > 20) throw new IOException($"Download of '{fileName}' keeps failing (HTTP 416).");
                        continue;
                    }
                    response.EnsureSuccessStatusCode();
                    if (received > 0 && response.StatusCode != System.Net.HttpStatusCode.PartialContent)
                    {
                        // The server ignored the range: start over.
                        received = 0;
                        fileStream.SetLength(0);
                        fileStream.Position = 0;
                    }
                    if (response.Content.Headers.ContentLength is long len) totalBytes = received + len;

                    await using var content = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                    int read;
                    while ((read = await content.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                    {
                        await fileStream.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                        received += read;
                        long now = Environment.TickCount64;
                        if (now - lastReport >= 500)
                        {
                            lastReport = now;
                            Report(received);
                        }
                    }
                    if (totalBytes is long expected && received < expected)
                        throw new IOException($"Download of '{fileName}' ended prematurely: received {received} of {expected} bytes.");
                    break;
                }
                catch (Exception ex) when (ex is IOException or HttpRequestException && !cancellationToken.IsCancellationRequested)
                {
                    if (ex is HttpRequestException { StatusCode: { } code } && (int)code is >= 400 and < 500 and not 408 and not 429)
                        throw;   // not found, unauthorized, ...: retrying will not help
                    if (++failures > 20) throw;
                    await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
                    await Task.Delay(TimeSpan.FromSeconds(Math.Min(30, 2 * failures)), cancellationToken).ConfigureAwait(false);
                }
            }
            Report(received);
            await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
            fileStream.Close();

            long got = new FileInfo(tempPath).Length;
            if (expectedSize is long size && got != size)
            {
                File.Delete(tempPath);
                throw new InvalidDataException($"Downloaded '{fileName}' has {got} bytes, expected {size}.");
            }
            File.Move(tempPath, localPath, overwrite: true);
        }
        finally
        {
            // A failed or cancelled transfer keeps its .download file, so the next attempt resumes it.
            OneDownloadAtATime.Release();
        }

        void Report(long done)
        {
            if (reportProgress is null) return;
            float fraction = totalBytes is long total && total > 0 ? Math.Clamp(done / (float)total, 0f, 1f) : 0f;
            reportProgress(new DownloadProgress(done, totalBytes, fraction, fileName));
        }
    }

    /// <summary>True when <paramref name="path"/> exists and, if a size is expected, has it (following symbolic links:
    /// <c>FileInfo.Length</c> reports a link's own size).</summary>
    private static bool IsComplete(string path, long? expectedSize)
    {
        if (!File.Exists(path)) return false;
        if (expectedSize is not long size) return true;
        var fi = new FileInfo(path);
        long length = (fi.LinkTarget != null ? fi.ResolveLinkTarget(true) as FileInfo : null)?.Length ?? fi.Length;
        return length == size;
    }
}
