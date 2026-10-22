using System.Diagnostics;

namespace Parakeet.Numerics;

/// <summary>Opt-in wall-clock breakdown of the forward pass by stage (<c>PARAKEET_PROFILE=1</c>); free when disabled.</summary>
public static class Profiler
{
    public static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("PARAKEET_PROFILE") == "1";

    private static readonly Dictionary<string, (double Ms, long Count)> Totals = new();
    private static long _last;

    /// <summary>Starts timing (call <see cref="Mark"/> after each stage).</summary>
    public static void Start()
    {
        if (Enabled) _last = Stopwatch.GetTimestamp();
    }

    /// <summary>Charges the time since the previous mark to <paramref name="stage"/>.</summary>
    public static void Mark(string stage)
    {
        if (!Enabled) return;
        long now = Stopwatch.GetTimestamp();
        double ms = Stopwatch.GetElapsedTime(_last, now).TotalMilliseconds;
        _last = now;
        lock (Totals)
        {
            Totals.TryGetValue(stage, out var v);
            Totals[stage] = (v.Ms + ms, v.Count + 1);
        }
    }

    public static string Report()
    {
        lock (Totals)
        {
            double total = Totals.Values.Sum(v => v.Ms);
            return string.Join(Environment.NewLine, Totals.OrderByDescending(kv => kv.Value.Ms)
                .Select(kv => $"{kv.Key,-20} {kv.Value.Ms,9:F1} ms {100 * kv.Value.Ms / total,5:F1}%  ({kv.Value.Count} calls)"));
        }
    }

    public static void Reset()
    {
        lock (Totals) Totals.Clear();
    }
}
