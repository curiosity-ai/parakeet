using System.Text.Json;
using Voxwerk;
using Voxwerk.Audio;
using Voxwerk.Talker;
using Voxwerk.Voices;
using Voxwerk.Voices.English.Samples;

// Generates the speech fixtures used by the Parakeet tests and benchmarks with Voxwerk (pure C# Qwen3-TTS):
//
//   samples  decode the reference recordings shipped in the Voxwerk.Voices.English.* NuGet packages
//            (one voice per accent region; each comes with its exact transcript)
//   synth    synthesize new sentences with a Qwen3-TTS CustomVoice checkpoint (downloaded on first use)
//
// Every clip is written as 16 kHz mono 16-bit WAV next to a manifest.json holding its transcript.

const int TargetRate = 16000;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("""
        usage: Parakeet.TestAudio samples [--out dir] [--per-region n]
               Parakeet.TestAudio synth   [--out dir] [--model repo|dir] [--threads n] [--seed n]
        """);
    return 1;
}

string command = args[0];
string outDir = Option("--out") ?? Path.Combine("testdata", "audio");
Directory.CreateDirectory(outDir);
string manifestPath = Path.Combine(outDir, "manifest.json");
var manifest = File.Exists(manifestPath)
    ? JsonSerializer.Deserialize<List<Clip>>(File.ReadAllText(manifestPath)) ?? []
    : [];

switch (command)
{
    case "samples":
        Samples(int.Parse(Option("--per-region") ?? "1"));
        break;
    case "synth":
        await Synth(Option("--model") ?? "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", int.Parse(Option("--threads") ?? "0"),
                    int.Parse(Option("--seed") ?? "1234"));
        break;
    default:
        Console.Error.WriteLine($"unknown command '{command}'");
        return 1;
}

manifest.Sort((a, b) => string.CompareOrdinal(a.File, b.File));
File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"{manifest.Count} clips in {manifestPath}");
return 0;

void Samples(int perRegion)
{
    EnglishVoiceSamples.RegisterAll();
    var picked = VoiceRegistry.All
        .Where(v => VoiceRegistry.HasSample(v.Id))
        .GroupBy(v => v.Country + "/" + v.Category + "/" + v.Style + "/" + v.Gender)
        .OrderBy(g => g.Key, StringComparer.Ordinal)
        .SelectMany(g => g.OrderBy(v => v.Id, StringComparer.Ordinal).Take(perRegion));
    foreach (var info in picked)
    {
        var sample = VoiceRegistry.GetVoiceSample(info.Id)!;
        string file = $"sample-{info.Id}.wav";
        float[] audio = AudioFile.Resample(sample.Audio, sample.SampleRate, TargetRate);
        AudioFile.WriteWav(Path.Combine(outDir, file), audio, TargetRate);
        Add(new Clip(file, info.SampleText, "voxwerk-sample", info.Id, info.Country, Math.Round(audio.Length / (double)TargetRate, 2)));
        Console.WriteLine($"{file}: {audio.Length / (double)TargetRate:F1}s  {info.SampleText}");
    }
}

async Task Synth(string model, int threads, int seed)
{
    // Written: the expected transcript where it differs from the spoken text (the model writes numbers as digits).
    (string Id, string Speaker, string Language, string Text, string? Written)[] sentences =
    [
        ("synth-weather", "ryan", "english", "Good morning! The weather today is sunny with a light breeze, perfect for a walk in the park.", null),
        ("synth-numbers", "serena", "english", "The meeting was moved to three thirty on Tuesday, and we expect about two hundred and fifty people to attend.",
            "The meeting was moved to 3:30 on Tuesday, and we expect about 250 people to attend."),
        ("synth-science", "aiden", "english", "Photosynthesis converts light energy into chemical energy, which plants store as sugar in their leaves.", null),
        ("synth-story", "vivian", "english", "Once upon a time, a little fox lived at the edge of a quiet forest, and every evening she watched the stars appear one by one.", null),
        ("synth-tech", "eric", "english", "To install the package, open a terminal, change into the project directory, and run the build command.", null),
        ("synth-long", "ryan", "english",
            "Speech recognition has come a long way over the last decade. Early systems needed careful dictation in a quiet room, " +
            "and they still made many mistakes. Modern neural models listen to ordinary conversation, handle different accents, " +
            "and produce punctuated text in a fraction of real time. This progress came from large amounts of training data, " +
            "better architectures such as the conformer, and decoding methods that predict both tokens and their durations.", null),
        // Other languages Qwen3-TTS speaks and Parakeet TDT 0.6B v3 transcribes (multilingual tests).
        ("synth-de", "ryan", "german", "Guten Morgen! Das Wetter ist heute sonnig und perfekt für einen Spaziergang im Park.", null),
        ("synth-fr", "serena", "french", "Bonjour à tous, la réunion commencera dans dix minutes dans la grande salle.", null),
        ("synth-es", "aiden", "spanish", "El tren de la mañana llegó tarde, pero todos los pasajeros estaban tranquilos.", null),
        ("synth-it", "vivian", "italian", "La biblioteca resterà aperta fino alle otto di sera durante tutta la settimana.", null),
        ("synth-pt", "ryan", "portuguese", "O mercado da cidade vende frutas frescas e pão quente todas as manhãs.", null),
        ("synth-ru", "serena", "russian", "Сегодня вечером мы будем смотреть фильм вместе с друзьями.", null),
    ];

    using var tts = await Qwen3Tts.LoadAsync(model, threads);
    var speakers = new HashSet<string>(tts.Speakers, StringComparer.OrdinalIgnoreCase);
    foreach (var (id, speaker, language, text, written) in sentences)
    {
        string file = id + ".wav";
        if (manifest.FirstOrDefault(c => c.File == file) is { } existing && File.Exists(Path.Combine(outDir, file)))
        {
            Add(existing with { Text = text, Written = written, Language = language });
            Console.WriteLine($"{file}: exists, skipped");
            continue;
        }
        string voice = speakers.Contains(speaker) ? speaker : tts.Speakers.First();
        var result = tts.SynthesizeCustomVoice(text, voice, language, options: new GenerationOptions { Seed = seed });
        float[] audio = AudioFile.Resample(result.Audio, result.SampleRate, TargetRate);
        AudioFile.WriteWav(Path.Combine(outDir, file), audio, TargetRate);
        Add(new Clip(file, text, "voxwerk-synth", voice, "", Math.Round(audio.Length / (double)TargetRate, 2)) { Written = written, Language = language });
        Console.WriteLine($"{file}: {audio.Length / (double)TargetRate:F1}s (RTF {result.RealTimeFactor:F2})  {text}");
    }
}

void Add(Clip clip)
{
    manifest.RemoveAll(c => c.File == clip.File);
    manifest.Add(clip);
}

string? Option(string name)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}

internal sealed record Clip(string File, string Text, string Source, string Voice, string Region, double Seconds)
{
    /// <summary>Expected transcript in written form, when it differs from <see cref="Text"/> (e.g. numbers as digits).</summary>
    public string? Written { get; init; }

    /// <summary>Spoken language (english unless stated).</summary>
    public string Language { get; init; } = "english";
}
