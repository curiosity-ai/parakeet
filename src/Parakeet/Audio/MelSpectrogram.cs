using System.Numerics;
using Parakeet.Numerics;

namespace Parakeet.Audio;

/// <summary>How the waveform is padded before the centered STFT.</summary>
public enum StftPadMode
{
    /// <summary>Reflection padding, as in the <c>nemo128.onnx</c> preprocessor of the ONNX export (default).</summary>
    Reflect,
    /// <summary>Zero padding, as in NeMo's <c>FilterbankFeatures</c> (<c>torch.stft(pad_mode="constant")</c>).</summary>
    Constant,
}

/// <summary>Settings of NeMo's <c>AudioToMelSpectrogramPreprocessor</c> used by Parakeet.</summary>
public sealed record MelOptions
{
    public int SampleRate { get; init; } = 16000;
    public int Features { get; init; } = 128;
    public int NFft { get; init; } = 512;
    public int WindowLength { get; init; } = 400;
    public int HopLength { get; init; } = 160;
    public float Preemphasis { get; init; } = 0.97f;
    /// <summary>Added before the log (NeMo <c>log_zero_guard_value</c> = 2⁻²⁴).</summary>
    public float LogZeroGuard { get; init; } = 5.9604645e-08f;
    public StftPadMode PadMode { get; init; } = StftPadMode.Reflect;
    /// <summary>Per-feature mean / std normalization over time (NeMo <c>normalize: per_feature</c>).</summary>
    public bool NormalizePerFeature { get; init; } = true;
}

/// <summary>
/// Log-mel filterbank features exactly as NeMo computes them for Parakeet: pre-emphasis, centered STFT (512-point,
/// 25 ms symmetric Hann window, 10 ms hop) in double precision, power spectrum, Slaney-normalized mel filterbank,
/// <c>log(x + 2⁻²⁴)</c>, then per-feature normalization with the unbiased standard deviation (+1e-5).
/// Output is time-major <c>[frames, features]</c>.
/// </summary>
public sealed unsafe class MelSpectrogram
{
    private readonly MelOptions _o;
    private readonly double[] _window;      // n_fft long, the win_length Hann window centered with zero padding
    private readonly float[] _filters;      // [features, bins] Slaney mel filterbank
    private readonly int[] _filterStart, _filterEnd;
    private readonly Fft _fft;

    public MelSpectrogram(MelOptions? options = null, float[]? filterbank = null, float[]? window = null)
    {
        _o = options ?? new MelOptions();
        int bins = _o.NFft / 2 + 1;
        _window = new double[_o.NFft];
        var win = window != null ? Array.ConvertAll(window, w => (double)w) : HannWindow(_o.WindowLength);
        int off = (_o.NFft - _o.WindowLength) / 2;
        for (int i = 0; i < win.Length; i++) _window[off + i] = win[i];
        _filters = filterbank ?? SlaneyFilterbank(_o.SampleRate, _o.NFft, _o.Features);
        if (_filters.Length != _o.Features * bins) throw new ArgumentException("Filterbank must be [features, n_fft/2 + 1].");
        _filterStart = new int[_o.Features];
        _filterEnd = new int[_o.Features];
        for (int m = 0; m < _o.Features; m++)
        {
            int s = 0, e = bins;
            while (s < bins && _filters[m * bins + s] == 0) s++;
            while (e > s && _filters[m * bins + e - 1] == 0) e--;
            _filterStart[m] = s;
            _filterEnd[m] = e;
        }
        _fft = new Fft(_o.NFft);
    }

    public MelOptions Options => _o;
    public ReadOnlySpan<float> Filterbank => _filters;

    /// <summary>Number of feature frames for <paramref name="samples"/> samples (<c>samples / hop + 1</c>).</summary>
    public int FrameCount(int samples) => samples / _o.HopLength + 1;

    /// <summary>Computes <c>[frames, features]</c> features of 16 kHz mono audio.</summary>
    public float[] Compute(ReadOnlySpan<float> audio, Parallelism? par = null)
    {
        par ??= Parallelism.Single;
        int n = audio.Length;
        if (n < 2) throw new ArgumentException("Audio is too short.");
        int pad = _o.NFft / 2;
        int frames = FrameCount(n);
        int F = _o.Features, bins = _o.NFft / 2 + 1;

        // Pre-emphasis (float32, like the reference), then center padding.
        var padded = new float[n + 2 * pad];
        padded[pad] = audio[0];
        for (int i = 1; i < n; i++) padded[pad + i] = audio[i] - _o.Preemphasis * audio[i - 1];
        if (_o.PadMode == StftPadMode.Reflect)
        {
            if (n <= pad) throw new ArgumentException($"Reflect padding needs more than {pad} samples.");
            for (int i = 0; i < pad; i++)
            {
                padded[pad - 1 - i] = padded[pad + 1 + i];
                padded[pad + n + i] = padded[pad + n - 2 - i];
            }
        }

        var feats = new float[(long)frames * F];
        fixed (float* src = padded)
        fixed (float* dst = feats)
        {
            float* s = src;
            float* d = dst;
            par.ForRange(frames, 16, (f0, f1) =>
            {
                var re = new double[_o.NFft];
                var im = new double[_o.NFft];
                var power = new float[bins];
                for (int f = f0; f < f1; f++)
                {
                    float* x = s + (long)f * _o.HopLength;
                    for (int i = 0; i < _o.NFft; i++) { re[i] = x[i] * _window[i]; im[i] = 0; }
                    _fft.Forward(re, im);
                    for (int k = 0; k < bins; k++) power[k] = (float)(re[k] * re[k] + im[k] * im[k]);
                    float* row = d + (long)f * F;
                    for (int m = 0; m < F; m++)
                    {
                        float acc = 0;
                        int b0 = _filterStart[m], b1 = _filterEnd[m];
                        for (int k = b0; k < b1; k++) acc += power[k] * _filters[m * bins + k];
                        row[m] = MathF.Log(acc + _o.LogZeroGuard);
                    }
                }
            });
        }
        if (_o.NormalizePerFeature) NormalizePerFeature(feats, frames, F);
        return feats;
    }

    private static void NormalizePerFeature(float[] feats, int frames, int F)
    {
        var mean = new double[F];
        var sq = new double[F];
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < F; m++) mean[m] += feats[(long)t * F + m];
        for (int m = 0; m < F; m++) mean[m] /= frames;
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < F; m++)
            {
                double dlt = feats[(long)t * F + m] - mean[m];
                sq[m] += dlt * dlt;
            }
        var mf = new float[F];
        var inv = new float[F];
        for (int m = 0; m < F; m++)
        {
            double std = Math.Sqrt(sq[m] / Math.Max(1, frames - 1)) + 1e-5;
            mf[m] = (float)mean[m];
            inv[m] = (float)std;
        }
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < F; m++)
                feats[(long)t * F + m] = (feats[(long)t * F + m] - mf[m]) / inv[m];
    }

    /// <summary>Symmetric Hann window (<c>torch.hann_window(n, periodic=False)</c>).</summary>
    public static double[] HannWindow(int n)
    {
        var w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (n - 1));
        return w;
    }

    /// <summary>librosa's <c>filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm="slaney")</c> as float32
    /// <c>[n_mels, n_fft/2 + 1]</c>, which is what NeMo stores in the checkpoint.</summary>
    public static float[] SlaneyFilterbank(int sampleRate, int nFft, int nMels, double fMin = 0, double? fMax = null)
    {
        double fmax = fMax ?? sampleRate / 2.0;
        int bins = nFft / 2 + 1;
        var fftFreqs = new double[bins];
        for (int k = 0; k < bins; k++) fftFreqs[k] = k * (sampleRate / 2.0) / (bins - 1);
        double melMin = HzToMel(fMin), melMax = HzToMel(fmax);
        var melF = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) melF[i] = MelToHz(melMin + (melMax - melMin) * i / (nMels + 1));
        var w = new float[nMels * bins];
        for (int i = 0; i < nMels; i++)
        {
            double fd0 = melF[i + 1] - melF[i], fd1 = melF[i + 2] - melF[i + 1];
            double enorm = 2.0 / (melF[i + 2] - melF[i]);
            for (int k = 0; k < bins; k++)
            {
                double lower = (fftFreqs[k] - melF[i]) / fd0;      // -ramps[i] / fdiff[i]
                double upper = (melF[i + 2] - fftFreqs[k]) / fd1;  // ramps[i + 2] / fdiff[i + 1]
                double v = Math.Max(0, Math.Min(lower, upper));
                w[i * bins + k] = (float)(v * enorm);
            }
        }
        return w;
    }

    private const double FSp = 200.0 / 3, MinLogHz = 1000.0, MinLogMel = MinLogHz / FSp;
    private static readonly double LogStep = Math.Log(6.4) / 27.0;

    private static double HzToMel(double hz) => hz >= MinLogHz ? MinLogMel + Math.Log(hz / MinLogHz) / LogStep : hz / FSp;

    private static double MelToHz(double mel) => mel >= MinLogMel ? MinLogHz * Math.Exp(LogStep * (mel - MinLogMel)) : FSp * mel;
}

/// <summary>Iterative radix-2 complex FFT in double precision with precomputed twiddles.</summary>
internal sealed class Fft
{
    private readonly int _n;
    private readonly int[] _rev;
    private readonly double[] _cos, _sin;

    public Fft(int n)
    {
        if (n < 2 || !BitOperations.IsPow2(n)) throw new ArgumentException("FFT size must be a power of two.");
        _n = n;
        int bits = BitOperations.Log2((uint)n);
        _rev = new int[n];
        for (int i = 0; i < n; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
            _rev[i] = r;
        }
        _cos = new double[n / 2];
        _sin = new double[n / 2];
        for (int i = 0; i < n / 2; i++)
        {
            _cos[i] = Math.Cos(2 * Math.PI * i / n);
            _sin[i] = -Math.Sin(2 * Math.PI * i / n);
        }
    }

    public void Forward(double[] re, double[] im)
    {
        int n = _n;
        for (int i = 0; i < n; i++)
        {
            int j = _rev[i];
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (int size = 2; size <= n; size <<= 1)
        {
            int half = size >> 1, step = n / size;
            for (int start = 0; start < n; start += size)
            {
                for (int j = 0; j < half; j++)
                {
                    double wr = _cos[j * step], wi = _sin[j * step];
                    int a = start + j, b = a + half;
                    double tr = wr * re[b] - wi * im[b];
                    double ti = wr * im[b] + wi * re[b];
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] += tr; im[a] += ti;
                }
            }
        }
    }
}
