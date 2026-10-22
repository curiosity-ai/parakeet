using Parakeet.Audio;
using Parakeet.Numerics;
using Xunit.Abstractions;

namespace Parakeet.Tests;

/// <summary>The opt-in int8 encoder projections: accuracy against the spoken text and agreement with the FP32 reference.</summary>
public sealed class Int8Tests(ITestOutputHelper output)
{
    [Fact]
    public void Int8EncoderKeepsTranscripts()
    {
        if (!Fixtures.Available(output) || !Int8Gemm.IsSupported) return;
        using var asr = ParakeetAsr.Load(Fixtures.ModelPath!, new ParakeetOptions { Precision = WeightPrecision.Int8 });
        Assert.Equal(WeightPrecision.Int8, asr.Precision);
        bool multilingual = asr.Config.VocabSize > 1024;
        double errors = 0, words = 0;
        int clips = 0, identical = 0, compared = 0;
        foreach (var clip in Fixtures.Clips.Where(c => multilingual || c.Language == "english"))
        {
            var r = asr.TranscribeFile(Path.Combine(Fixtures.AudioDir, clip.File));
            string expected = clip.Written ?? clip.Text;
            double wer = Fixtures.WordErrorRate(expected, r.Text);
            int n = expected.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
            errors += wer * n;
            words += n;
            clips++;
            string name = Path.GetFileNameWithoutExtension(clip.File);
            string note = "";
            if (Fixtures.ReferenceDir != null && File.Exists(Path.Combine(Fixtures.ReferenceDir, name, "result.json")))
            {
                compared++;
                string fp32 = Fixtures.RefResult(name).Text;
                if (fp32 == r.Text) identical++;
                else note = $"   (FP32: {fp32})";
            }
            output.WriteLine($"{clip.File,-44} WER {wer,6:P1}  {r.Text}{note}");
        }
        output.WriteLine($"int8: overall WER {errors / words:P2} over {words} words in {clips} clips; " +
                         $"{identical}/{compared} transcripts identical to the FP32 ONNX Runtime reference");
        Assert.True(errors / words < 0.05, $"int8 WER {errors / words:P2}");
        if (compared > 0) Assert.True(identical >= 0.75 * compared, $"only {identical}/{compared} transcripts match FP32");
    }
}
