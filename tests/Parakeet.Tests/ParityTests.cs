using Parakeet.Audio;
using Parakeet.Model;
using Parakeet.Numerics;
using Xunit.Abstractions;

namespace Parakeet.Tests;

/// <summary>
/// Layer-by-layer parity with ONNX Runtime running the ONNX export (dumps from <c>scripts/onnx_reference.py</c>).
/// Each stage is fed the reference's own input (ORT features into the encoder, ORT encoder output into the decoder)
/// so errors are measured per stage rather than accumulated.
/// </summary>
[Collection("model")]
public sealed unsafe class ParityTests(ModelFixture model, ITestOutputHelper output)
{
    private const string Clip = "sample-en-us-aaliyah-renwick";

    private bool Ready(string clip = Clip) =>
        Fixtures.Available(output, reference: true) && File.Exists(Path.Combine(Fixtures.ReferenceDir!, clip, "result.json"));

    [Fact]
    public void FrontendMatchesNemoAndOnnx()
    {
        if (!Ready()) return;
        var asr = model.Asr!;
        // The analytic Slaney filterbank and Hann window equal the ones stored in the checkpoint.
        var fromCkpt = asr.Frontend.Filterbank.ToArray();
        var analytic = MelSpectrogram.SlaneyFilterbank(16000, 512, 128);
        Fixtures.AssertClose(output, "slaney filterbank", fromCkpt, analytic, 1e-6);

        var r = Fixtures.RefResult(Clip);
        var audio = WaveFile.ReadMono(r.Wav);
        Assert.Equal(r.Samples, audio.Length);
        var feats = asr.Frontend.Compute(audio);
        Assert.Equal(r.FeatureFrames, asr.Frontend.FrameCount(audio.Length));
        var reference = Fixtures.Ref(Clip, "features");          // [128, T]
        var expected = Fixtures.Transpose(reference.Data, reference.Shape[0], reference.Shape[1]);
        Fixtures.AssertClose(output, "log-mel features", expected, feats, 1e-5);
    }

    [Fact]
    public void EncoderLayersMatchOnnxRuntime()
    {
        if (!Ready()) return;
        var asr = model.Asr!;
        var r = Fixtures.RefResult(Clip);
        var f = Fixtures.Ref(Clip, "features");
        var feats = Fixtures.Transpose(f.Data, f.Shape[0], f.Shape[1]);   // ORT's features, so only the encoder is measured
        int t = r.EncodedFrames, d = asr.Config.DModel;
        Assert.Equal(t, asr.Config.EncoderFrames(r.FeatureFrames));

        var traced = new Dictionary<string, float[]>();
        TraceCallback trace = (name, data, rows, cols) => traced[name] = new ReadOnlySpan<float>(data, rows * cols).ToArray();
        var enc = asr.Encoder.Forward(feats, r.FeatureFrames, asr.Parallelism, trace);

        Fixtures.AssertClose(output, "pre_encode", Fixtures.Ref(Clip, "pre_encode").Data, traced["pre_encode"], 1e-5);
        Fixtures.AssertClose(output, "pos_emb", Fixtures.Ref(Clip, "pos_emb").Data, traced["pos_emb"], 1e-6);
        Fixtures.AssertClose(output, "layer0.norm_ff1", Fixtures.Ref(Clip, "layer0.norm_ff1").Data, traced["layer0.norm_ff1"], 1e-5);
        Fixtures.AssertClose(output, "layer0.ff1", Fixtures.Ref(Clip, "layer0.ff1").Data, traced["layer0.ff1"], 1e-5);
        Fixtures.AssertClose(output, "layer0.q", Fixtures.Ref(Clip, "layer0.q").Data, traced["layer0.q"], 1e-5);
        Fixtures.AssertClose(output, "layer0.p", Fixtures.Ref(Clip, "layer0.p").Data, traced["layer0.p"], 1e-5);
        Fixtures.AssertClose(output, "layer0.attn", Fixtures.Ref(Clip, "layer0.attn").Data, traced["layer0.attn"], 1e-5);
        var glu = Fixtures.Ref(Clip, "layer0.glu");                // [d, T] (ONNX Conv layout)
        Fixtures.AssertClose(output, "layer0.glu", Fixtures.Transpose(glu.Data, glu.Shape[0], glu.Shape[1]), traced["layer0.glu"], 1e-5);
        Fixtures.AssertClose(output, "layer0.conv", Fixtures.Ref(Clip, "layer0.conv").Data, traced["layer0.conv"], 1e-5);
        Fixtures.AssertClose(output, "layer0.ff2", Fixtures.Ref(Clip, "layer0.ff2").Data, traced["layer0.ff2"], 1e-5);
        for (int i = 0; i < asr.Config.Layers; i++)
            Fixtures.AssertClose(output, $"layer{i}", Fixtures.Ref(Clip, $"layer{i}").Data, traced[$"layer{i}"], 1e-4);
        Fixtures.AssertClose(output, "encoder output", Fixtures.Ref(Clip, "encoder").Data, enc, 1e-4);
        Assert.Equal(t * d, enc.Length);
    }

    [Fact]
    public void DecoderMatchesOnnxRuntime()
    {
        if (!Ready()) return;
        var asr = model.Asr!;
        var r = Fixtures.RefResult(Clip);
        var enc = Fixtures.Ref(Clip, "encoder");                 // ORT encoder output [T, d]
        var refLogits = Fixtures.Ref(Clip, "joint_logits");      // [steps, V + 1 + durations]
        var steps = new List<float[]>();
        StepCallback onStep = (frame, logits, n) => steps.Add(new ReadOnlySpan<float>(logits, n).ToArray());
        var tokens = asr.Decoder.Decode(enc.Data, enc.Shape[0], asr.Parallelism, onStep: onStep);

        Assert.Equal(r.Tokens, tokens.Select(x => x.Id).ToArray());
        Assert.Equal(r.Frames, tokens.Select(x => x.Frame).ToArray());
        Assert.Equal(r.Durations, tokens.Select(x => x.Duration).ToArray());
        Assert.Equal(refLogits.Shape[0], steps.Count);
        Fixtures.AssertClose(output, $"joint logits ({steps.Count} steps)", refLogits.Data, steps.SelectMany(s => s).ToArray(), 1e-5);
        output.WriteLine(asr.Vocabulary.Decode(tokens.Select(x => x.Id)));
    }

    [Fact]
    public void EveryClipMatchesOnnxRuntimeEndToEnd()
    {
        if (!Fixtures.Available(output, reference: true)) return;
        var asr = model.Asr!;
        int clips = 0, tokenMismatches = 0;
        foreach (var clip in Fixtures.ReferenceClips())
        {
            var r = Fixtures.RefResult(clip);
            var res = asr.Transcribe(WaveFile.ReadMono(r.Wav));
            var ids = res.Tokens.Select(x => x.Id).ToArray();
            bool same = ids.SequenceEqual(r.Tokens);
            if (!same) tokenMismatches++;
            var e = Fixtures.Ref(clip, "encoder");
            float[] enc = asr.Encoder.Forward(asr.Frontend.Compute(WaveFile.ReadMono(r.Wav)), r.FeatureFrames, asr.Parallelism);
            var (rel, _) = Fixtures.Error(e.Data, enc);
            output.WriteLine($"{clip,-40} tokens {(same ? "identical" : "DIFFER")}  encoder relL2 {rel:E2}  {res.Text}");
            if (!same) output.WriteLine($"{"",-40} ORT: {r.Text}");
            Assert.True(rel < 1e-3, $"{clip}: encoder relL2 {rel}");
            clips++;
        }
        output.WriteLine($"{clips} clips, {tokenMismatches} with different tokens");
        Assert.Equal(0, tokenMismatches);
    }

    [Fact]
    public void OnnxExportLoadsTheSameWeights()
    {
        if (!Fixtures.Available(output, onnx: true, reference: true)) return;
        var asr = model.Asr!;
        using var onnx = ParakeetAsr.Load(Fixtures.OnnxDir!);
        Assert.Equal(asr.Config.Layers, onnx.Config.Layers);
        Assert.Equal(asr.Config.VocabSize, onnx.Config.VocabSize);
        Assert.Equal(asr.Config.Durations, onnx.Config.Durations);
        var r = Fixtures.RefResult(Clip);
        var audio = WaveFile.ReadMono(r.Wav);
        var a = asr.Transcribe(audio);
        var b = onnx.Transcribe(audio);
        Assert.Equal(r.Tokens, b.Tokens.Select(x => x.Id).ToArray());
        Assert.Equal(a.Text, b.Text);
        var fa = asr.Frontend.Compute(audio);
        var ea = asr.Encoder.Forward(fa, r.FeatureFrames, asr.Parallelism);
        var eb = onnx.Encoder.Forward(fa, r.FeatureFrames, onnx.Parallelism);
        Fixtures.AssertClose(output, "encoder (.nemo vs ONNX weights)", ea, eb, 1e-5);
    }
}
