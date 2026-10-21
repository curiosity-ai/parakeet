using System.Text.Json;
using Xunit.Abstractions;

namespace Parakeet.Tests;

/// <summary>
/// Locates the checkpoints and the ONNX Runtime reference dumps written by <c>scripts/onnx_reference.py</c>:
/// <list type="bullet">
/// <item><c>PARAKEET_MODEL</c>: the <c>.nemo</c> checkpoint (or an extracted NeMo directory)</item>
/// <item><c>PARAKEET_ONNX_DIR</c>: the fp32 ONNX export directory (optional, for the ONNX loader tests)</item>
/// <item><c>PARAKEET_REFERENCE_DIR</c>: the reference dump directory (one sub-directory per clip)</item>
/// </list>
/// Tests that need them return early (and say so) when they are missing, so the kernel tests still run anywhere.
/// </summary>
public static class Fixtures
{
    public static string? ModelPath => ExistingPath(Environment.GetEnvironmentVariable("PARAKEET_MODEL"));
    public static string? OnnxDir => ExistingPath(Environment.GetEnvironmentVariable("PARAKEET_ONNX_DIR"));
    public static string? ReferenceDir => ExistingPath(Environment.GetEnvironmentVariable("PARAKEET_REFERENCE_DIR"));

    /// <summary>Test clips shipped in <c>testdata/audio</c> (copied next to the test assembly).</summary>
    public static string AudioDir => Path.Combine(AppContext.BaseDirectory, "testdata", "audio");

    private static string? ExistingPath(string? p) => !string.IsNullOrEmpty(p) && (File.Exists(p) || Directory.Exists(p)) ? p : null;

    public static bool Available(ITestOutputHelper output, bool model = true, bool reference = false, bool onnx = false)
    {
        var missing = new List<string>();
        if (model && ModelPath == null) missing.Add("PARAKEET_MODEL");
        if (reference && ReferenceDir == null) missing.Add("PARAKEET_REFERENCE_DIR");
        if (onnx && OnnxDir == null) missing.Add("PARAKEET_ONNX_DIR");
        if (missing.Count == 0) return true;
        output.WriteLine($"SKIPPED: set {string.Join(", ", missing)} (see README, 'Validation').");
        return false;
    }

    public static IReadOnlyList<Clip> Clips =>
        JsonSerializer.Deserialize<List<Clip>>(File.ReadAllText(Path.Combine(AudioDir, "manifest.json"))) ?? [];

    /// <summary>Clips that have a reference dump.</summary>
    public static IEnumerable<string> ReferenceClips() =>
        ReferenceDir == null ? [] : Directory.GetDirectories(ReferenceDir).Where(d => File.Exists(Path.Combine(d, "result.json")))
                                             .Select(Path.GetFileName).OrderBy(s => s, StringComparer.Ordinal)!;

    public static Npy Ref(string clip, string name) => Npy.Load(Path.Combine(ReferenceDir!, clip, name + ".npy"));

    public static ReferenceResult RefResult(string clip) =>
        JsonSerializer.Deserialize<ReferenceResult>(File.ReadAllText(Path.Combine(ReferenceDir!, clip, "result.json")),
                                                     new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })!;

    /// <summary>Asserts closeness and reports max-abs / relative-L2 error.</summary>
    public static (double RelL2, double MaxAbs) AssertClose(ITestOutputHelper output, string label, ReadOnlySpan<float> expected,
                                                            ReadOnlySpan<float> actual, double maxRelL2, double maxAbs = double.PositiveInfinity)
    {
        Assert.Equal(expected.Length, actual.Length);
        var (rel, mx) = Error(expected, actual);
        output.WriteLine($"{label,-28} n={expected.Length,9} maxAbs={mx:E2} relL2={rel:E2}");
        Assert.True(rel <= maxRelL2, $"{label}: relative L2 error {rel:E3} > {maxRelL2:E1}");
        Assert.True(mx <= maxAbs, $"{label}: max abs error {mx:E3} > {maxAbs:E1}");
        return (rel, mx);
    }

    public static (double RelL2, double MaxAbs) Error(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        double num = 0, den = 0, mx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
            mx = Math.Max(mx, Math.Abs(d));
        }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), mx);
    }

    /// <summary>Transposes <c>[rows, cols]</c> to <c>[cols, rows]</c>.</summary>
    public static float[] Transpose(float[] x, int rows, int cols)
    {
        var y = new float[x.Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) y[(long)c * rows + r] = x[(long)r * cols + c];
        return y;
    }

    /// <summary>Word error rate after lower-casing and stripping punctuation.</summary>
    public static double WordErrorRate(string reference, string hypothesis)
    {
        static string[] Words(string s) => new string(s.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) || c == '\'' ? c : ' ').ToArray())
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var r = Words(reference);
        var h = Words(hypothesis);
        var d = new int[r.Length + 1, h.Length + 1];
        for (int i = 0; i <= r.Length; i++) d[i, 0] = i;
        for (int j = 0; j <= h.Length; j++) d[0, j] = j;
        for (int i = 1; i <= r.Length; i++)
            for (int j = 1; j <= h.Length; j++)
                d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (r[i - 1] == h[j - 1] ? 0 : 1));
        return r.Length == 0 ? h.Length : (double)d[r.Length, h.Length] / r.Length;
    }
}

public sealed record Clip(string File, string Text, string Source, string Voice, string Region, double Seconds)
{
    public string? Written { get; init; }
}

public sealed record ReferenceResult(string Wav, int Samples, int FeatureFrames, int EncodedFrames, string Text, int[] Tokens, int[] Frames, int[] Durations);

/// <summary>Loads the model once for all tests in the "model" collection.</summary>
public sealed class ModelFixture : IDisposable
{
    private readonly Lazy<ParakeetAsr?> _asr = new(() => Fixtures.ModelPath == null ? null : ParakeetAsr.Load(Fixtures.ModelPath));

    public ParakeetAsr? Asr => _asr.Value;

    public void Dispose()
    {
        if (_asr.IsValueCreated) _asr.Value?.Dispose();
    }
}

[CollectionDefinition("model")]
public sealed class ModelCollection : ICollectionFixture<ModelFixture>;
