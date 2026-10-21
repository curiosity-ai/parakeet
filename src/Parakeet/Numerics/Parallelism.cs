namespace Parakeet.Numerics;

/// <summary>Degree-of-parallelism control shared by every kernel. <c>1</c> runs everything inline on the
/// calling thread (the single-core configuration); larger values split work with <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>.</summary>
public sealed class Parallelism
{
    public static Parallelism Single { get; } = new(1);

    public Parallelism(int maxThreads)
    {
        if (maxThreads <= 0) maxThreads = Environment.ProcessorCount;
        MaxThreads = maxThreads;
        Options = new ParallelOptions { MaxDegreeOfParallelism = maxThreads };
    }

    public int MaxThreads { get; }

    internal ParallelOptions Options { get; }

    /// <summary>Runs <paramref name="body"/> for every index in [0, <paramref name="count"/>).</summary>
    public void For(int count, Action<int> body)
    {
        if (count <= 0) return;
        if (MaxThreads == 1 || count == 1)
        {
            for (int i = 0; i < count; i++) body(i);
            return;
        }
        Parallel.For(0, count, Options, body);
    }

    /// <summary>Splits [0, <paramref name="total"/>) into at most <see cref="MaxThreads"/> contiguous ranges
    /// of at least <paramref name="minChunk"/> items.</summary>
    public void ForRange(int total, int minChunk, Action<int, int> body)
    {
        if (total <= 0) return;
        int chunks = Math.Max(1, Math.Min(MaxThreads, total / Math.Max(1, minChunk)));
        if (chunks == 1)
        {
            body(0, total);
            return;
        }
        int per = (total + chunks - 1) / chunks;
        Parallel.For(0, chunks, Options, c =>
        {
            int s = c * per;
            int e = Math.Min(total, s + per);
            if (s < e) body(s, e);
        });
    }
}
