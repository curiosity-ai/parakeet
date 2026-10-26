namespace Parakeet.Chunking;

/// <summary>
/// One analysis window over the mel feature sequence, in absolute mel frames, half-open <c>[Start, End)</c>. The encoder
/// runs over the whole window; only tokens in <c>[EmitStart, EmitEnd)</c> are kept, so the overlap shared with a
/// neighbour is emitted by exactly one of them. Adjacent emit ranges tile the timeline: <c>EmitEnd</c> of one window
/// equals <c>EmitStart</c> of the next.
/// </summary>
internal readonly record struct ChunkWindow(int Start, int End, int EmitStart, int EmitEnd);

/// <summary>The overlap shared by two neighbouring windows, which an <see cref="IBoundaryOracle"/> splits.</summary>
internal readonly record struct OverlapRegion(int Start, int End, int Midpoint);

/// <summary>Chooses where, inside an overlap, ownership passes from the earlier window to the later one.</summary>
internal interface IBoundaryOracle
{
    /// <summary>The chosen mel frame, or false to let the next layer of a <see cref="ChainBoundaryOracle"/> decide.</summary>
    bool TryGetBoundary(OverlapRegion region, out int frame);
}

/// <summary>The arithmetic midpoint of the overlap. Always decides, so it ends every chain.</summary>
internal sealed class MidpointBoundaryOracle : IBoundaryOracle
{
    public bool TryGetBoundary(OverlapRegion region, out int frame)
    {
        frame = region.Midpoint;
        return true;
    }
}

/// <summary>
/// The quietest point of the overlap. A cut inside a word makes both windows time it slightly differently, so it is
/// emitted twice or lost; moving the cut onto a pause avoids that. Works on the already-computed features: the sum of
/// a frame's normalized log-mel bins is a loudness proxy, smoothed over ~150 ms so a dip inside a word is not mistaken
/// for a pause. Loud, continuous music has no quiet frame, and the result is then no better than the midpoint.
/// </summary>
internal sealed class MelEnergyBoundaryOracle : IBoundaryOracle
{
    /// <summary>Moving-average width in mel frames (10 ms each).</summary>
    public const int SmoothingFrames = 15;

    private readonly double[] _smoothed;

    public MelEnergyBoundaryOracle(ReadOnlySpan<float> features, int frames, int bins)
    {
        var energy = new double[frames];
        for (int t = 0; t < frames; t++)
        {
            double sum = 0;
            foreach (float v in features.Slice(t * bins, bins)) sum += v;
            energy[t] = sum;
        }
        _smoothed = Smooth(energy, SmoothingFrames);
    }

    public bool TryGetBoundary(OverlapRegion region, out int frame)
    {
        frame = 0;
        if (_smoothed.Length == 0 || region.End <= region.Start) return false;
        frame = ArgMin(_smoothed, region.Start, region.End);
        return true;
    }

    /// <summary>Centred moving average of <paramref name="width"/> frames (prefix sums); a width of 1 or less is the identity.</summary>
    internal static double[] Smooth(double[] values, int width)
    {
        if (width <= 1 || values.Length == 0) return values;
        int n = values.Length, half = width / 2;
        var prefix = new double[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + values[i];
        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            int lo = Math.Max(0, i - half), hi = Math.Min(n, i + half + 1);
            result[i] = (prefix[hi] - prefix[lo]) / (hi - lo);
        }
        return result;
    }

    /// <summary>Index of the smallest value in <c>[start, end)</c>, clamped to the array; the earliest wins a tie.</summary>
    internal static int ArgMin(double[] values, int start, int end)
    {
        start = Math.Max(0, start);
        end = Math.Min(values.Length, end);
        if (start >= end) return start;
        int best = start;
        for (int i = start + 1; i < end; i++)
            if (values[i] < values[best]) best = i;
        return best;
    }
}

/// <summary>Asks each oracle in turn and takes the first decision.</summary>
internal sealed class ChainBoundaryOracle(params IBoundaryOracle[] oracles) : IBoundaryOracle
{
    public bool TryGetBoundary(OverlapRegion region, out int frame)
    {
        foreach (var oracle in oracles)
            if (oracle.TryGetBoundary(region, out frame)) return true;
        frame = 0;
        return false;
    }
}

/// <summary>Splits a mel sequence into overlapping windows for audio longer than one pass should cover.</summary>
internal static class ChunkPlanner
{
    /// <summary>
    /// Windows of at most <paramref name="chunkFrames"/> mel frames, each sharing about <paramref name="overlapFrames"/>
    /// with its neighbours. Window starts are multiples of <paramref name="alignment"/> (the encoder's subsampling
    /// factor), so a window's first encoder frame is exactly <c>Start / alignment</c> and token times can be mapped back
    /// without rounding. Ownership of each interior overlap is split where <paramref name="oracle"/> says (the
    /// midpoint when it is null or declines). Whatever it returns is clamped, so the emit ranges always tile
    /// <c>[0, total)</c> with no gap or overlap and stay inside their windows. Audio that fits one window gives one window.
    /// </summary>
    public static List<ChunkWindow> Plan(int total, int chunkFrames, int overlapFrames, int alignment = 1, IBoundaryOracle? oracle = null)
    {
        Validate(chunkFrames, overlapFrames, alignment);
        if (total <= chunkFrames) return [new ChunkWindow(0, total, 0, total)];

        int stride = (chunkFrames - overlapFrames) / alignment * alignment;
        var windows = new List<ChunkWindow>();
        for (int start = 0; ; start += stride)
        {
            int end = Math.Min(start + chunkFrames, total);
            windows.Add(new ChunkWindow(start, end, 0, 0));
            if (end == total) break;
        }

        int prevEmitEnd = 0;
        for (int i = 0; i < windows.Count; i++)
        {
            var w = windows[i];
            if (i == windows.Count - 1)
            {
                windows[i] = w with { EmitStart = prevEmitEnd, EmitEnd = total };
                break;
            }
            int overlapStart = windows[i + 1].Start, overlapEnd = w.End;
            int mid = (overlapStart + overlapEnd) / 2;
            int boundary = mid;
            if (oracle != null && oracle.TryGetBoundary(new OverlapRegion(overlapStart, overlapEnd, mid), out int chosen))
                boundary = chosen;
            boundary = Math.Clamp(boundary, overlapStart, overlapEnd);
            boundary = Math.Max(boundary, prevEmitEnd);
            windows[i] = w with { EmitStart = prevEmitEnd, EmitEnd = boundary };
            prevEmitEnd = boundary;
        }
        return windows;
    }

    internal static void Validate(int chunkFrames, int overlapFrames, int alignment)
    {
        if (alignment < 1) throw new ArgumentOutOfRangeException(nameof(alignment));
        if (chunkFrames <= 0) throw new ArgumentException($"Chunk size must be positive, got {chunkFrames} frames.");
        if (overlapFrames < 0) throw new ArgumentException($"Chunk overlap must not be negative, got {overlapFrames} frames.");
        if (overlapFrames >= chunkFrames)
            throw new ArgumentException($"Chunk overlap ({overlapFrames} frames) must be smaller than the chunk size ({chunkFrames} frames).");
        if (chunkFrames - overlapFrames < alignment)
            throw new ArgumentException($"Chunk size minus overlap must be at least {alignment} frames.");
    }
}
