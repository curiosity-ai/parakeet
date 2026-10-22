using System.Buffers.Binary;

namespace Parakeet.Audio;

/// <summary>Minimal RIFF/WAVE reading and writing: PCM 8/16/24/32-bit and IEEE float 32/64-bit, any channel count
/// (mixed down to mono), with band-limited resampling to the model's 16 kHz.</summary>
public static class WaveFile
{
    /// <summary>Reads a WAV file as mono float samples at <paramref name="targetRate"/> Hz.</summary>
    public static float[] ReadMono(string path, int targetRate = 16000) => ReadMono(File.ReadAllBytes(path), targetRate);

    /// <summary>Reads WAV bytes as mono float samples at <paramref name="targetRate"/> Hz.</summary>
    public static float[] ReadMono(ReadOnlySpan<byte> wav, int targetRate = 16000)
    {
        var (samples, rate) = Decode(wav);
        return Resampler.Resample(samples, rate, targetRate);
    }

    /// <summary>True when the bytes start with a RIFF/WAVE (or RF64) header.</summary>
    public static bool IsWave(ReadOnlySpan<byte> data) =>
        data.Length >= 12 && (data[..4].SequenceEqual("RIFF"u8) || data[..4].SequenceEqual("RF64"u8)) && data.Slice(8, 4).SequenceEqual("WAVE"u8);

    /// <summary>Decodes WAV bytes to mono float samples at their native rate.</summary>
    public static (float[] Samples, int SampleRate) Decode(ReadOnlySpan<byte> wav)
    {
        if (!IsWave(wav)) throw new InvalidDataException("Not a RIFF/WAVE file.");
        int format = 0, channels = 0, rate = 0, bits = 0;
        ReadOnlySpan<byte> data = default;
        bool haveData = false;
        int pos = 12;
        while (pos + 8 <= wav.Length)
        {
            var id = wav.Slice(pos, 4);
            long size = BinaryPrimitives.ReadUInt32LittleEndian(wav[(pos + 4)..]);
            int body = pos + 8;
            if (id.SequenceEqual("fmt "u8))
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(wav[body..]);
                channels = BinaryPrimitives.ReadUInt16LittleEndian(wav[(body + 2)..]);
                rate = BinaryPrimitives.ReadInt32LittleEndian(wav[(body + 4)..]);
                bits = BinaryPrimitives.ReadUInt16LittleEndian(wav[(body + 14)..]);
                if (format == 0xFFFE && size >= 26) format = BinaryPrimitives.ReadUInt16LittleEndian(wav[(body + 24)..]);
            }
            else if (id.SequenceEqual("data"u8))
            {
                // Streams written without a final size (0 or 0xFFFFFFFF) run to the end of the file.
                long avail = wav.Length - body;
                long len = size == 0 || size == 0xFFFFFFFF || size > avail ? avail : size;
                data = wav.Slice(body, (int)len);
                haveData = true;
                break;
            }
            pos = body + (int)Math.Min(size + (size & 1), wav.Length - body);
        }
        if (!haveData || channels <= 0 || rate <= 0) throw new InvalidDataException("WAV file has no fmt or data chunk.");
        int bytesPer = bits / 8;
        if (bytesPer <= 0) throw new InvalidDataException($"Unsupported WAV sample size {bits}.");
        int frames = data.Length / (bytesPer * channels);
        var mono = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                var s = data.Slice((f * channels + c) * bytesPer, bytesPer);
                sum += (format, bits) switch
                {
                    (1, 8) => (s[0] - 128) / 128f,
                    (1, 16) => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
                    (1, 24) => ((s[0] << 8 | s[1] << 16 | s[2] << 24) >> 8) / 8388608f,
                    (1, 32) => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
                    (3, 32) => BinaryPrimitives.ReadSingleLittleEndian(s),
                    (3, 64) => (float)BinaryPrimitives.ReadDoubleLittleEndian(s),
                    _ => throw new InvalidDataException($"Unsupported WAV encoding (format {format}, {bits} bits)."),
                };
            }
            mono[f] = sum / channels;
        }
        return (mono, rate);
    }

    /// <summary>Writes mono samples as 16-bit PCM WAV.</summary>
    public static void WriteMono16(string path, ReadOnlySpan<float> samples, int sampleRate = 16000)
    {
        var bytes = new byte[44 + samples.Length * 2];
        "RIFF"u8.CopyTo(bytes);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), 36 + samples.Length * 2);
        "WAVEfmt "u8.CopyTo(bytes.AsSpan(8));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16), 16);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(20), 1);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(22), 1);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28), sampleRate * 2);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(32), 2);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(34), 16);
        "data"u8.CopyTo(bytes.AsSpan(36));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40), samples.Length * 2);
        for (int i = 0; i < samples.Length; i++)
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(44 + 2 * i), (short)Math.Round(Math.Clamp(samples[i], -1f, 1f) * 32767f));
        File.WriteAllBytes(path, bytes);
    }
}

/// <summary>Band-limited resampling with a Kaiser-windowed sinc kernel (no-op when the rates match).</summary>
public static class Resampler
{
    public static float[] Resample(float[] input, int fromRate, int toRate, int zeroCrossings = 32)
    {
        if (fromRate == toRate || input.Length == 0) return input;
        if (fromRate <= 0 || toRate <= 0) throw new ArgumentOutOfRangeException(nameof(fromRate));
        double ratio = (double)toRate / fromRate;
        double cutoff = Math.Min(1.0, ratio) * 0.97;           // fraction of the input Nyquist kept
        double halfWidth = zeroCrossings / cutoff;             // kernel half width in input samples
        const double beta = 8.6;
        double i0Beta = BesselI0(beta);
        long outLen = (long)Math.Ceiling(input.Length * ratio);
        var output = new float[outLen];
        Parallel.For(0, (int)((outLen + 4095) / 4096), chunk =>
        {
            long o0 = chunk * 4096L, o1 = Math.Min(outLen, o0 + 4096);
            for (long o = o0; o < o1; o++)
            {
                double center = o / ratio;
                int i0 = (int)Math.Ceiling(center - halfWidth), i1 = (int)Math.Floor(center + halfWidth);
                double acc = 0, wsum = 0;
                for (int i = Math.Max(0, i0); i <= Math.Min(input.Length - 1, i1); i++)
                {
                    double x = i - center;
                    double r = x / halfWidth;
                    double win = BesselI0(beta * Math.Sqrt(Math.Max(0, 1 - r * r))) / i0Beta;
                    double arg = Math.PI * x * cutoff;
                    double sinc = Math.Abs(arg) < 1e-12 ? 1 : Math.Sin(arg) / arg;
                    double w = cutoff * sinc * win;
                    acc += w * input[i];
                    wsum += w;
                }
                // Normalizing by the kernel sum keeps the DC gain exact near the edges.
                output[o] = wsum != 0 ? (float)(acc / wsum) : 0f;
            }
        });
        return output;
    }

    private static double BesselI0(double x)
    {
        double sum = 1, term = 1, q = x * x / 4;
        for (int k = 1; k < 50; k++)
        {
            term *= q / (k * k);
            sum += term;
            if (term < 1e-12 * sum) break;
        }
        return sum;
    }
}
