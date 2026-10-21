using Parakeet.Audio;
using Parakeet.IO;

namespace Parakeet.Model;

/// <summary>Architecture hyper-parameters of a Parakeet TDT checkpoint (NeMo <c>EncDecRNNTBPEModel</c> with a
/// FastConformer encoder and a TDT decoder). Defaults are those of <c>parakeet-tdt-0.6b-v2</c> / <c>-v3</c>.</summary>
public sealed record ParakeetConfig
{
    public MelOptions Mel { get; init; } = new();

    // FastConformer encoder
    public int Layers { get; init; } = 24;
    public int DModel { get; init; } = 1024;
    public int Heads { get; init; } = 8;
    public int FfExpansion { get; init; } = 4;
    public int ConvKernel { get; init; } = 9;
    public int SubsamplingFactor { get; init; } = 8;
    public int SubsamplingChannels { get; init; } = 256;
    public bool XScaling { get; init; }
    public bool UseBias { get; init; }
    public float LayerNormEps { get; init; } = 1e-5f;
    public float BatchNormEps { get; init; } = 1e-5f;

    // TDT prediction network + joint
    public int PredHidden { get; init; } = 640;
    public int PredLayers { get; init; } = 2;
    public int JointHidden { get; init; } = 640;
    /// <summary>Number of text tokens; the blank symbol is the extra id <c>VocabSize</c>.</summary>
    public int VocabSize { get; init; } = 1024;
    public int[] Durations { get; init; } = [0, 1, 2, 3, 4];
    public int MaxSymbolsPerStep { get; init; } = 10;

    public int BlankId => VocabSize;
    public int FfHidden => DModel * FfExpansion;
    public int HeadDim => DModel / Heads;

    /// <summary>Encoder frames for <paramref name="melFrames"/> feature frames (three stride-2, kernel-3, pad-1 convolutions).</summary>
    public int EncoderFrames(int melFrames)
    {
        int t = melFrames;
        for (int s = SubsamplingFactor; s > 1; s >>= 1) t = (t - 1) / 2 + 1;
        return t;
    }

    /// <summary>Seconds of audio per encoder frame (80 ms).</summary>
    public double FrameSeconds => (double)Mel.HopLength * SubsamplingFactor / Mel.SampleRate;

    /// <summary>Reads the hyper-parameters from a NeMo <c>model_config.yaml</c>.</summary>
    internal static ParakeetConfig FromNemoYaml(Dictionary<string, object?> y)
    {
        var d = new ParakeetConfig();
        var durations = MiniYaml.GetList(y, "model_defaults.tdt_durations") ?? MiniYaml.GetList(y, "decoding.durations");
        string? subsampling = MiniYaml.GetString(y, "encoder.subsampling");
        if (subsampling != null && subsampling != "dw_striding")
            throw new NotSupportedException($"Subsampling '{subsampling}' is not supported (only dw_striding).");
        string? attn = MiniYaml.GetString(y, "encoder.self_attention_model");
        if (attn != null && attn != "rel_pos") throw new NotSupportedException($"Attention '{attn}' is not supported (only rel_pos).");
        var ctx = MiniYaml.GetList(y, "encoder.att_context_size");
        if (ctx != null && ctx.Any(c => c != "-1"))
            throw new NotSupportedException("Limited attention context is not supported (only full context [-1, -1]).");
        return d with
        {
            Mel = d.Mel with
            {
                SampleRate = MiniYaml.GetInt(y, "preprocessor.sample_rate", d.Mel.SampleRate),
                Features = MiniYaml.GetInt(y, "preprocessor.features", d.Mel.Features),
                NFft = MiniYaml.GetInt(y, "preprocessor.n_fft", d.Mel.NFft),
                WindowLength = (int)Math.Round(MiniYaml.GetDouble(y, "preprocessor.window_size", 0.025) * MiniYaml.GetInt(y, "preprocessor.sample_rate", 16000)),
                HopLength = (int)Math.Round(MiniYaml.GetDouble(y, "preprocessor.window_stride", 0.01) * MiniYaml.GetInt(y, "preprocessor.sample_rate", 16000)),
                NormalizePerFeature = (MiniYaml.GetString(y, "preprocessor.normalize") ?? "per_feature") == "per_feature",
            },
            Layers = MiniYaml.GetInt(y, "encoder.n_layers", d.Layers),
            DModel = MiniYaml.GetInt(y, "encoder.d_model", d.DModel),
            Heads = MiniYaml.GetInt(y, "encoder.n_heads", d.Heads),
            FfExpansion = MiniYaml.GetInt(y, "encoder.ff_expansion_factor", d.FfExpansion),
            ConvKernel = MiniYaml.GetInt(y, "encoder.conv_kernel_size", d.ConvKernel),
            SubsamplingFactor = MiniYaml.GetInt(y, "encoder.subsampling_factor", d.SubsamplingFactor),
            SubsamplingChannels = MiniYaml.GetInt(y, "encoder.subsampling_conv_channels", d.SubsamplingChannels),
            XScaling = MiniYaml.GetBool(y, "encoder.xscaling", d.XScaling),
            UseBias = MiniYaml.GetBool(y, "encoder.use_bias", d.UseBias),
            PredHidden = MiniYaml.GetInt(y, "decoder.prednet.pred_hidden", d.PredHidden),
            PredLayers = MiniYaml.GetInt(y, "decoder.prednet.pred_rnn_layers", d.PredLayers),
            JointHidden = MiniYaml.GetInt(y, "joint.jointnet.joint_hidden", d.JointHidden),
            VocabSize = MiniYaml.GetInt(y, "decoder.vocab_size", d.VocabSize),
            Durations = durations?.Select(int.Parse).ToArray() ?? d.Durations,
            MaxSymbolsPerStep = MiniYaml.GetInt(y, "decoding.greedy.max_symbols", d.MaxSymbolsPerStep),
        };
    }
}
