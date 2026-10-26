using Parakeet.Audio;
using Parakeet.Chunking;
using Parakeet.Model;
using Xunit.Abstractions;

namespace Parakeet.Tests;

public sealed class ChunkPlannerTests
{
    private static void AssertTiles(IReadOnlyList<ChunkWindow> plan, int total)
    {
        Assert.Equal(0, plan[0].EmitStart);
        Assert.Equal(total, plan[^1].EmitEnd);
        for (int i = 0; i < plan.Count; i++)
        {
            var w = plan[i];
            Assert.True(w.Start <= w.EmitStart && w.EmitStart <= w.EmitEnd && w.EmitEnd <= w.End, $"window {i}: {w}");
            if (i > 0) Assert.Equal(plan[i - 1].EmitEnd, w.EmitStart);
        }
    }

    [Fact]
    public void ShortAudioIsOneWindow()
    {
        var plan = ChunkPlanner.Plan(500, 1000, 100);
        Assert.Equal([new ChunkWindow(0, 500, 0, 500)], plan);
    }

    [Fact]
    public void WindowsOverlapAndTileTheTimeline()
    {
        var plan = ChunkPlanner.Plan(2500, 1000, 200, alignment: 8);
        Assert.True(plan.Count > 2);
        Assert.All(plan, w => Assert.True(w.End - w.Start <= 1000 && w.Start % 8 == 0));
        Assert.Equal(2500, plan[^1].End);
        for (int i = 1; i < plan.Count; i++) Assert.True(plan[i].Start < plan[i - 1].End, "neighbours must overlap");
        AssertTiles(plan, 2500);
    }

    [Fact]
    public void DefaultsToTheMidpointOfTheOverlap()
    {
        var plan = ChunkPlanner.Plan(1800, 1000, 200);
        Assert.Equal(2, plan.Count);
        Assert.Equal((plan[1].Start + plan[0].End) / 2, plan[0].EmitEnd);
    }

    private sealed class FixedOracle(int frame) : IBoundaryOracle
    {
        public bool TryGetBoundary(OverlapRegion region, out int f) { f = frame; return true; }
    }

    [Theory]
    [InlineData(-5000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(900)]
    [InlineData(int.MaxValue)]
    public void AnyOracleAnswerStillTiles(int answer)
    {
        var plan = ChunkPlanner.Plan(4000, 1000, 200, alignment: 8, new FixedOracle(answer));
        AssertTiles(plan, 4000);
    }

    [Fact]
    public void DecliningOracleFallsThroughTheChain()
    {
        var plan = ChunkPlanner.Plan(1800, 1000, 200, 1, new ChainBoundaryOracle(new MidpointBoundaryOracle()));
        Assert.Equal(ChunkPlanner.Plan(1800, 1000, 200)[0].EmitEnd, plan[0].EmitEnd);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1000, 1000)]
    [InlineData(1000, 1500)]
    [InlineData(1000, -1)]
    public void RejectsBadSizes(int chunk, int overlap) =>
        Assert.Throws<ArgumentException>(() => ChunkPlanner.Plan(5000, chunk, overlap));

    [Fact]
    public void MelEnergyOracleCutsAtTheQuietestFrame()
    {
        const int frames = 600, bins = 4;
        var features = new float[frames * bins];
        Array.Fill(features, 1f);
        for (int t = 340; t < 360; t++)
            for (int m = 0; m < bins; m++) features[t * bins + m] = -3f;
        var oracle = new MelEnergyBoundaryOracle(features, frames, bins);
        Assert.True(oracle.TryGetBoundary(new OverlapRegion(300, 400, 350), out int frame));
        Assert.InRange(frame, 340, 359);
        Assert.False(oracle.TryGetBoundary(new OverlapRegion(400, 400, 400), out _));
    }

    [Fact]
    public void ArgMinKeepsTheEarliestTieAndClamps()
    {
        double[] v = [3, 1, 1, 2];
        Assert.Equal(1, MelEnergyBoundaryOracle.ArgMin(v, 0, 4));
        Assert.Equal(3, MelEnergyBoundaryOracle.ArgMin(v, 3, 100));
        Assert.Equal(2, MelEnergyBoundaryOracle.ArgMin(v, 2, 2));
    }

    [Fact]
    public void SmoothingAveragesWithATruncatedEdge()
    {
        var r = MelEnergyBoundaryOracle.Smooth([0, 0, 9, 0, 0], 3);
        Assert.Equal([0, 3, 3, 3, 0], r);
        var same = new double[] { 1, 2 };
        Assert.Same(same, MelEnergyBoundaryOracle.Smooth(same, 1));
    }
}

public sealed class SeamDeduperTests
{
    private static DecodedToken T(int id, int frame) => new(id, frame, 1, 0f);

    [Fact]
    public void DropsADuplicateAtTheSamePosition()
    {
        var r = SeamDeduper.Resolve([T(1, 100), T(2, 102)], [T(2, 103), T(3, 110)]);
        Assert.Equal([T(3, 110)], r);
    }

    [Fact]
    public void TheEarlierWindowWinsACollision()
    {
        var r = SeamDeduper.Resolve([T(1, 100)], [T(9, 101)]);
        Assert.Empty(r);
    }

    [Fact]
    public void KeepsTokensAwayFromTheSeamAndRepeatsFurtherApart()
    {
        var r = SeamDeduper.Resolve([T(1, 100)], [T(1, 104), T(5, 120)]);
        Assert.Equal([T(1, 104), T(5, 120)], r);
    }

    [Fact]
    public void OnlyTheLastTokensOfThePreviousWindowCount()
    {
        var previous = new[] { T(1, 10), T(2, 90), T(3, 92), T(4, 94) };
        Assert.Equal([T(7, 11)], SeamDeduper.Resolve(previous, [T(7, 11)]));
    }

    [Fact]
    public void NothingToCompareKeepsEverything()
    {
        Assert.Equal([T(1, 5)], SeamDeduper.Resolve([], [T(1, 5)]));
        Assert.Empty(SeamDeduper.Resolve([T(1, 5)], []));
    }
}

/// <summary>Long-audio mode against the single-pass result, on the Voxwerk clips.</summary>
[Collection("model")]
public sealed class ChunkedTranscriptionTests(ModelFixture model, ITestOutputHelper output)
{
    private static ParakeetAsr Load(double chunk, double overlap) =>
        ParakeetAsr.Load(Fixtures.ModelPath!, new ParakeetOptions { ChunkSeconds = chunk, ChunkOverlapSeconds = overlap });

    [Fact]
    public void AudioThatFitsOneWindowIsUnchanged()
    {
        if (!Fixtures.Available(output)) return;
        var audio = WaveFile.ReadMono(Path.Combine(Fixtures.AudioDir, "synth-long.wav"), 16000);
        var reference = model.Asr!.Transcribe(audio);
        using var chunked = Load(300, 15);
        var r = chunked.Transcribe(audio);
        Assert.Equal(reference.Text, r.Text);
        Assert.Equal(reference.Tokens, r.Tokens);
    }

    [Fact]
    public void SplitAudioReadsLikeTheSinglePass()
    {
        if (!Fixtures.Available(output)) return;
        var audio = WaveFile.ReadMono(Path.Combine(Fixtures.AudioDir, "synth-long.wav"), 16000);
        var reference = model.Asr!.Transcribe(audio);

        using var chunked = Load(16, 4);
        var deltas = new List<string>();
        var r = chunked.Transcribe(audio, onText: deltas.Add);

        double wer = Fixtures.WordErrorRate(reference.Text, r.Text);
        output.WriteLine($"{audio.Length / 16000.0:F1}s in windows of 16s/4s, WER vs single pass {wer:P1}\n  single:  {reference.Text}\n  chunked: {r.Text}");
        Assert.True(wer < 0.05, $"WER {wer:P1} against the single pass");
        Assert.Equal(r.Text, Text.Vocabulary.Normalize(string.Concat(deltas)));
        Assert.True(r.Tokens.Zip(r.Tokens.Skip(1)).All(p => p.First.Frame <= p.Second.Frame), "token frames must not go backwards");
        Assert.True(r.Words[^1].End <= r.AudioSeconds + 0.2);
    }

    [Fact]
    public void RejectsAnOverlapThatIsNotSmallerThanTheChunk()
    {
        if (!Fixtures.Available(output)) return;
        Assert.Throws<ArgumentException>(() => Load(10, 10));
    }
}
