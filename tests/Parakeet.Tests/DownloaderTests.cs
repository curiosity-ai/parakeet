using System.Net;
using System.Net.Sockets;
using Parakeet.IO;
using Xunit.Abstractions;

namespace Parakeet.Tests;

/// <summary><see cref="ModelDownloader"/> against a local HTTP server with range support and injected faults, and the
/// <see cref="ParakeetModels"/> catalog.</summary>
public sealed class DownloaderTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"parakeet-dl-{Guid.NewGuid():N}");

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* ignore */ }
    }

    /// <summary>Serves <c>/model.bin</c> (and 404 elsewhere), honouring <c>Range: bytes=N-</c>. The first
    /// <paramref name="dropAfter"/>-byte responses are cut off after that many bytes.</summary>
    private sealed class TestServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly byte[] _data;
        private int _drops;

        public TestServer(byte[] data, int drops = 0, long dropAfter = 0)
        {
            _data = data;
            _drops = drops;
            DropAfter = dropAfter;
            int port;
            using (var probe = new TcpListener(IPAddress.Loopback, 0))
            {
                probe.Start();
                port = ((IPEndPoint)probe.LocalEndpoint).Port;
                probe.Stop();
            }
            Url = $"http://127.0.0.1:{port}/";
            _listener.Prefixes.Add(Url);
            _listener.Start();
            _ = Task.Run(Serve);
        }

        public string Url { get; }
        public long DropAfter { get; }
        public List<long> RangeStarts { get; } = new();
        public int Requests;

        private async Task Serve()
        {
            while (_listener.IsListening)
            {
                HttpListenerContext ctx;
                try { ctx = await _listener.GetContextAsync(); }
                catch { return; }
                Interlocked.Increment(ref Requests);
                var res = ctx.Response;
                try
                {
                    if (ctx.Request.Url!.AbsolutePath != "/model.bin")
                    {
                        res.StatusCode = 404;
                        res.Close();
                        continue;
                    }
                    long start = 0;
                    string? range = ctx.Request.Headers["Range"];
                    if (range != null && range.StartsWith("bytes=", StringComparison.Ordinal))
                        start = long.Parse(range["bytes=".Length..].TrimEnd('-'));
                    lock (RangeStarts) RangeStarts.Add(start);
                    res.Headers["Accept-Ranges"] = "bytes";
                    if (start > 0)
                    {
                        res.StatusCode = 206;
                        res.Headers["Content-Range"] = $"bytes {start}-{_data.Length - 1}/{_data.Length}";
                    }
                    res.ContentLength64 = _data.Length - start;
                    if (Interlocked.Decrement(ref _drops) >= 0)
                    {
                        await res.OutputStream.WriteAsync(_data.AsMemory((int)start, (int)Math.Min(DropAfter, _data.Length - start)));
                        await res.OutputStream.FlushAsync();
                        res.Abort();   // connection reset mid-body
                        continue;
                    }
                    await res.OutputStream.WriteAsync(_data.AsMemory((int)start));
                    res.Close();
                }
                catch
                {
                    try { res.Abort(); } catch { /* ignore */ }
                }
            }
        }

        public void Dispose() => _listener.Close();
    }

    private static byte[] Payload(int n)
    {
        var b = new byte[n];
        new Random(5).NextBytes(b);
        return b;
    }

    [Fact]
    public async Task DownloadsAndReportsProgress()
    {
        var data = Payload(3 << 20);
        using var server = new TestServer(data);
        string path = Path.Combine(_dir, "sub", "model.bin");
        var reports = new List<DownloadProgress>();
        await ModelDownloader.DownloadFileAsync(server.Url + "model.bin", path, reports.Add, data.Length);
        Assert.Equal(data, File.ReadAllBytes(path));
        Assert.False(File.Exists(path + ".download"));
        Assert.Equal(1f, reports[^1].Fraction);
        Assert.Equal("model.bin", reports[^1].FileName);

        // Cached with the right size: no request at all.
        int before = server.Requests;
        await ModelDownloader.DownloadFileAsync(server.Url + "model.bin", path, null, data.Length);
        Assert.Equal(before, server.Requests);
    }

    [Fact]
    public async Task ResumesAfterAConnectionDrop()
    {
        var data = Payload(2 << 20);
        using var server = new TestServer(data, drops: 1, dropAfter: 700_000);
        string path = Path.Combine(_dir, "model.bin");
        await ModelDownloader.DownloadFileAsync(server.Url + "model.bin", path, null, data.Length);
        Assert.Equal(data, File.ReadAllBytes(path));
        output.WriteLine("range starts: " + string.Join(", ", server.RangeStarts));
        Assert.Equal(0, server.RangeStarts[0]);
        Assert.True(server.RangeStarts[^1] > 0, "the retry should resume with a range request");
    }

    [Fact]
    public async Task ResumesAPartialFileFromAnEarlierProcess()
    {
        var data = Payload(1 << 20);
        using var server = new TestServer(data);
        string path = Path.Combine(_dir, "model.bin");
        Directory.CreateDirectory(_dir);
        File.WriteAllBytes(path + ".download", data[..300_000]);
        await ModelDownloader.DownloadFileAsync(server.Url + "model.bin", path, null, data.Length);
        Assert.Equal(data, File.ReadAllBytes(path));
        Assert.Equal([300_000L], server.RangeStarts);
    }

    [Fact]
    public async Task ReplacesACachedFileOfTheWrongSize()
    {
        var data = Payload(100_000);
        using var server = new TestServer(data);
        string path = Path.Combine(_dir, "model.bin");
        Directory.CreateDirectory(_dir);
        File.WriteAllBytes(path, new byte[10]);
        await ModelDownloader.DownloadFileAsync(server.Url + "model.bin", path, null, data.Length);
        Assert.Equal(data, File.ReadAllBytes(path));
    }

    [Fact]
    public async Task RejectsAnUnexpectedSizeAndMissingFiles()
    {
        var data = Payload(100_000);
        using var server = new TestServer(data);
        string path = Path.Combine(_dir, "model.bin");
        await Assert.ThrowsAsync<InvalidDataException>(() => ModelDownloader.DownloadFileAsync(server.Url + "model.bin", path, null, data.Length + 1));
        Assert.False(File.Exists(path));

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() => ModelDownloader.DownloadFileAsync(server.Url + "missing.bin", path));
        Assert.Equal(HttpStatusCode.NotFound, ex.StatusCode);
        Assert.False(File.Exists(path));
    }

    [Fact]
    public void CatalogNamesAndUrls()
    {
        foreach (var alias in new[] { "v2", "TdtV2", "nvidia/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2", "PARAKEET-TDT-0.6B-V2.nemo" })
        {
            Assert.True(ParakeetModels.TryParse(alias, out var m), alias);
            Assert.Equal(ParakeetModel.TdtV2, m);
        }
        Assert.True(ParakeetModels.TryParse("v3", out var v3));
        Assert.Equal(ParakeetModel.TdtV3, v3);
        Assert.False(ParakeetModels.TryParse("whisper", out _));
        Assert.Equal("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo",
                     ParakeetModels.GetDownloadUrl(ParakeetModel.TdtV3));
        Assert.EndsWith("parakeet-tdt-0.6b-v2.nemo", ParakeetModels.GetDefaultCachePath(ParakeetModel.TdtV2));
    }

    /// <summary>The catalog's sizes match the files actually served (network; set <c>PARAKEET_NETWORK_TESTS=1</c>).</summary>
    [Fact]
    public async Task PublishedSizesMatchTheCatalog()
    {
        if (Environment.GetEnvironmentVariable("PARAKEET_NETWORK_TESTS") != "1")
        {
            output.WriteLine("SKIPPED: set PARAKEET_NETWORK_TESTS=1 to check the download URLs.");
            return;
        }
        using var http = new HttpClient();
        foreach (var model in Enum.GetValues<ParakeetModel>())
        {
            using var req = new HttpRequestMessage(HttpMethod.Get, ParakeetModels.GetDownloadUrl(model));
            req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(0, 0);
            using var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
            resp.EnsureSuccessStatusCode();
            long? total = resp.Content.Headers.ContentRange?.Length;
            output.WriteLine($"{model}: {total} bytes");
            Assert.Equal(ParakeetModels.GetFileSize(model), total);
        }
    }
}
