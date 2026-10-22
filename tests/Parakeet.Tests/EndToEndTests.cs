using Parakeet.Audio;
using Parakeet.IO;
using Xunit.Abstractions;

namespace Parakeet.Tests;

/// <summary>Transcribes the Voxwerk-generated clips in <c>testdata/audio</c> and scores them against the text they were
/// generated from.</summary>
[Collection("model")]
public sealed class EndToEndTests(ModelFixture model, ITestOutputHelper output)
{
    [Fact]
    public void TranscribesGeneratedSpeech()
    {
        if (!Fixtures.Available(output)) return;
        var asr = model.Asr!;
        double errors = 0, words = 0, audio = 0, time = 0;
        // parakeet-tdt-0.6b-v2 is English-only (1024 pieces); v3 is multilingual (8192 pieces).
        bool multilingual = asr.Config.VocabSize > 1024;
        foreach (var clip in Fixtures.Clips.Where(c => multilingual || c.Language == "english"))
        {
            var r = asr.TranscribeFile(Path.Combine(Fixtures.AudioDir, clip.File));
            string expected = clip.Written ?? clip.Text;
            double wer = Fixtures.WordErrorRate(expected, r.Text);
            int n = expected.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
            errors += wer * n;
            words += n;
            audio += r.AudioSeconds;
            time += r.TotalTime.TotalSeconds;
            output.WriteLine($"{clip.File,-44} {clip.Language,-10} WER {wer,6:P1}  {r.Text}");
            Assert.True(wer < 0.25, $"{clip.File}: WER {wer:P1}\n  expected: {expected}\n  got:      {r.Text}");
            Assert.NotEmpty(r.Words);
            Assert.True(r.Words.Zip(r.Words.Skip(1)).All(p => p.First.Start <= p.Second.Start), "word times must be ordered");
            Assert.True(r.Words[^1].End <= r.AudioSeconds + 0.2);
        }
        output.WriteLine($"overall WER {errors / words:P2} over {words} words, {audio:F0}s audio in {time:F1}s (RTF {time / audio:F3})");
        Assert.True(errors / words < 0.05);
    }

    [Fact]
    public void StreamingDeltasConcatenateToTheTranscript()
    {
        if (!Fixtures.Available(output)) return;
        var asr = model.Asr!;
        var audio = WaveFile.ReadMono(Path.Combine(Fixtures.AudioDir, "synth-weather.wav"));
        var deltas = new List<string>();
        var r = asr.Transcribe(audio, onText: deltas.Add);
        Assert.Equal(r.Text, Text.Vocabulary.Normalize(string.Concat(deltas)));
    }

    [Fact]
    public void SilenceAndShortInputProduceNoText()
    {
        if (!Fixtures.Available(output)) return;
        var asr = model.Asr!;
        Assert.Equal("", asr.Transcribe(new float[16000]).Text);
        Assert.Equal("", asr.Transcribe(new float[100]).Text);
    }

    [Fact]
    public void ResampledInputStillTranscribes()
    {
        if (!Fixtures.Available(output)) return;
        var asr = model.Asr!;
        string path = Path.Combine(Fixtures.AudioDir, "synth-science.wav");
        var audio16 = WaveFile.ReadMono(path);
        var audio44 = Resampler.Resample(audio16, 16000, 44100);
        string tmp = Path.Combine(Path.GetTempPath(), $"parakeet-test-{Guid.NewGuid():N}.wav");
        try
        {
            WaveFile.WriteMono16(tmp, audio44, 44100);
            var a = asr.TranscribeFile(path);
            var b = asr.TranscribeFile(tmp);
            output.WriteLine(a.Text);
            output.WriteLine(b.Text);
            Assert.True(Fixtures.WordErrorRate(a.Text, b.Text) < 0.1);
        }
        finally
        {
            File.Delete(tmp);
        }
    }
}
