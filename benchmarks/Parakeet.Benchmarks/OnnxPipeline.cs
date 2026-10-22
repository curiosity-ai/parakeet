using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Text;

namespace Parakeet.Benchmarks;

/// <summary>
/// The ONNX Runtime pipeline the original Go server ran (.reference/internal/asr/transcriber.go), in-process:
/// <c>nemo128.onnx</c> features → <c>encoder-model[.int8].onnx</c> → greedy TDT with one <c>decoder_joint-model</c> run per
/// step (reusing pre-allocated tensors, like the Go decoder workers).
/// </summary>
public sealed class OnnxPipeline : IDisposable
{
    private readonly InferenceSession _pre, _enc, _dec;
    private readonly Vocabulary _vocab;
    private readonly int _blank, _vocabSize;
    private readonly int[] _durations = [0, 1, 2, 3, 4];

    public OnnxPipeline(string dir, int threads, bool int8)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
        if (threads > 0)
        {
            opts.IntraOpNumThreads = threads;
            opts.InterOpNumThreads = 1;
        }
        string Pick(string name) => int8 && File.Exists(Path.Combine(dir, name + ".int8.onnx")) ? Path.Combine(dir, name + ".int8.onnx") : Path.Combine(dir, name + ".onnx");
        EncoderFile = Path.GetFileName(Pick("encoder-model"));
        _pre = new InferenceSession(Path.Combine(dir, "nemo128.onnx"), opts);
        _enc = new InferenceSession(Pick("encoder-model"), opts);
        _dec = new InferenceSession(Pick("decoder_joint-model"), opts);
        _vocab = Vocabulary.FromFile(Path.Combine(dir, "vocab.txt"));
        _vocabSize = _vocab.Count + 1;
        _blank = _vocab.Count;
    }

    public string EncoderFile { get; }

    public sealed record Result(string Text, int[] Tokens, TimeSpan Features, TimeSpan Encoder, TimeSpan Decoder)
    {
        public TimeSpan Total => Features + Encoder + Decoder;
    }

    public Result Transcribe(float[] audio)
    {
        var sw = Stopwatch.StartNew();
        using var pre = _pre.Run([
            NamedOnnxValue.CreateFromTensor("waveforms", new DenseTensor<float>(audio, [1, audio.Length])),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", new DenseTensor<long>(new[] { (long)audio.Length }, [1])),
        ]);
        var feats = pre.First(v => v.Name == "features").AsTensor<float>().ToDenseTensor();
        var lens = pre.First(v => v.Name == "features_lens").AsTensor<long>().ToDenseTensor();
        var tFeat = sw.Elapsed;

        sw.Restart();
        using var enc = _enc.Run([
            NamedOnnxValue.CreateFromTensor("audio_signal", feats),
            NamedOnnxValue.CreateFromTensor("length", lens),
        ]);
        var encOut = enc.First(v => v.Name == "outputs").AsTensor<float>().ToDenseTensor();   // [1, d, T]
        int frames = (int)enc.First(v => v.Name == "encoded_lengths").AsTensor<long>()[0];
        var tEnc = sw.Elapsed;

        sw.Restart();
        var tokens = Decode(encOut, frames);
        var tDec = sw.Elapsed;
        return new Result(_vocab.Decode(tokens), tokens.ToArray(), tFeat, tEnc, tDec);
    }

    private List<int> Decode(DenseTensor<float> encOut, int frames)
    {
        int d = encOut.Dimensions[1], total = encOut.Dimensions[2];
        var src = encOut.Buffer.Span;
        // Pre-allocated, pinned inputs/outputs reused for every step (as the Go decoder workers do).
        var frame = new float[d];
        var target = new int[1];
        var s1 = new float[2 * 640]; var s2 = new float[2 * 640];
        var o1 = new float[2 * 640]; var o2 = new float[2 * 640];
        var logits = new float[_vocabSize + _durations.Length];
        using var frameV = OrtValue.CreateTensorValueFromMemory(frame, [1, d, 1]);
        using var targetV = OrtValue.CreateTensorValueFromMemory(target, [1, 1]);
        using var lenV = OrtValue.CreateTensorValueFromMemory(new[] { 1 }, [1]);
        using var s1V = OrtValue.CreateTensorValueFromMemory(s1, [2, 1, 640]);
        using var s2V = OrtValue.CreateTensorValueFromMemory(s2, [2, 1, 640]);
        using var outV = OrtValue.CreateTensorValueFromMemory(logits, [1, 1, 1, logits.Length]);
        using var lenOutV = OrtValue.CreateTensorValueFromMemory(new[] { 0 }, [1]);
        using var o1V = OrtValue.CreateTensorValueFromMemory(o1, [2, 1, 640]);
        using var o2V = OrtValue.CreateTensorValueFromMemory(o2, [2, 1, 640]);
        using var runOpts = new RunOptions();
        string[] inNames = ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"];
        string[] outNames = ["outputs", "prednet_lengths", "output_states_1", "output_states_2"];
        OrtValue[] inputs = [frameV, targetV, lenV, s1V, s2V];
        OrtValue[] outputs = [outV, lenOutV, o1V, o2V];

        var tokens = new List<int>();
        int last = _blank, t = 0;
        while (t < frames)
        {
            for (int i = 0; i < d; i++) frame[i] = src[i * total + t];
            int symbols = 0, skip;
            while (true)
            {
                target[0] = last;
                _dec.Run(runOpts, inNames, inputs, outNames, outputs);
                int k = ArgMax(logits, 0, _vocabSize);
                skip = _durations[ArgMax(logits, _vocabSize, _durations.Length)];
                symbols++;
                if (k != _blank)
                {
                    tokens.Add(k);
                    last = k;
                    o1.CopyTo(s1, 0);
                    o2.CopyTo(s2, 0);
                }
                t += skip;
                if (skip != 0 || k == _blank || symbols >= 10) break;
            }
            if (skip == 0) t++;
        }
        return tokens;
    }

    private static int ArgMax(float[] x, int start, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (x[start + i] > x[start + best]) best = i;
        return best;
    }

    public void Dispose()
    {
        _pre.Dispose();
        _enc.Dispose();
        _dec.Dispose();
    }
}
