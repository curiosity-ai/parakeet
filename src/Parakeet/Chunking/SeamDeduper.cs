using Parakeet.Model;

namespace Parakeet.Chunking;

/// <summary>
/// Drops tokens that two windows both emitted around a seam. Even with the cut on a pause, a word near it is timed a
/// little differently by each window, so it can appear twice, or as two different guesses at one position. Tokens carry
/// absolute encoder frames, so the leading tokens of a window can be compared with the tail of the previous one.
/// </summary>
internal static class SeamDeduper
{
    /// <summary>How many leading tokens of a window wait for the seam check; the rest stream as they are decoded.</summary>
    public const int HeadTokens = 3;

    /// <summary>How close (encoder frames, 80 ms each) two tokens must be to count as the same position: ~240 ms.</summary>
    public const int ToleranceFrames = 3;

    /// <summary>
    /// The tokens of <paramref name="head"/> (the start of window i+1) that survive against <paramref name="previous"/>
    /// (the tokens window i emitted). A head token within the tolerance of one of the last <see cref="HeadTokens"/>
    /// previous tokens is dropped: a duplicate if the text matches, otherwise a collision that the earlier window wins,
    /// because its decoder reaches the seam warmed up while the later one is still warming up.
    /// </summary>
    public static List<DecodedToken> Resolve(IReadOnlyList<DecodedToken> previous, IReadOnlyList<DecodedToken> head)
    {
        if (previous.Count == 0) return [.. head];
        int from = Math.Max(0, previous.Count - HeadTokens);
        var survivors = new List<DecodedToken>(head.Count);
        foreach (var h in head)
        {
            bool drop = false;
            for (int i = from; i < previous.Count && !drop; i++)
                drop = Math.Abs(h.Frame - previous[i].Frame) <= ToleranceFrames;
            if (!drop) survivors.Add(h);
        }
        return survivors;
    }
}
