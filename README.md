# Parakeet (pure C#)

Pure C# (.NET 10) inference for NVIDIA's **[Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)**
speech recognition models, `v2` (English) and `v3` (25 European languages), as a library plus a command-line tool.

- **No native dependencies**: no ONNX Runtime, no Python, no BLAS. The log-mel front end, the FastConformer
  encoder, the Token-and-Duration Transducer decoder, the checkpoint readers and the SIMD kernels are all managed
  code (`System.Runtime.Intrinsics`, `Span<T>`, `System.Numerics.Tensors`).
- **Loads the official checkpoint**: the `.nemo` file NVIDIA publishes (a tar archive holding a PyTorch zip and a
  pickle, all parsed in C# and memory-mapped), or the ONNX export used by the original Go server.
- **Validated layer by layer against ONNX Runtime**: every encoder layer matches to ~1e-6 relative error, and every
  test clip gives a **token-identical** transcript (23/23 for v2, 29/29 for v3 in 7 languages). See [Validation](#validation).
- **Faster than ONNX Runtime** on the same CPU and thread count: **1.22× end to end** in exact FP32 (encoder 1.12×,
  decoder 2.8×), also single-threaded (1.14×). The opt-in int8 mode on AVX-VNNI is **2.2×** faster than ONNX Runtime
  running either its FP32 or its int8 export, with transcripts identical to FP32 on v2. See [Performance](#performance).

The original Go/ONNX Runtime server this repository started from is kept in [`.reference/`](.reference) (its CI
workflows are disabled); it was used as the reference implementation for the port.

## Quick start

```bash
# Transcribe (downloads nvidia/parakeet-tdt-0.6b-v2 to ~/.cache/parakeet on first use, 2.5 GB)
dotnet run -c Release --project src/Parakeet.Cli -- transcribe audio.wav

# Multilingual model, word timestamps
dotnet run -c Release --project src/Parakeet.Cli -- transcribe --model nvidia/parakeet-tdt-0.6b-v3 --words audio.wav

# A local .nemo file, an extracted .nemo directory, or the ONNX export directory work too
dotnet run -c Release --project src/Parakeet.Cli -- transcribe --model ./parakeet-tdt-0.6b-v2.nemo --json a.wav b.wav
```

| Option | Meaning |
|---|---|
| `--model <path\|repo>` | `.nemo` file, extracted NeMo directory, ONNX export directory, or Hugging Face repo (default `nvidia/parakeet-tdt-0.6b-v2`) |
| `--threads <n>` | Worker threads (default: all cores; `1` = single core) |
| `--int8` | Int8 encoder projections on AVX-VNNI: about twice as fast, approximate (see [Int8](#int8-mode)) |
| `--zero-pad` | Zero-pad the STFT like NeMo itself (default: reflection, like the ONNX export) |
| `--words` / `--json` | Word timestamps / JSON output with token and word timestamps |
| `bench --runs <n>` | `parakeet bench` times repeated transcriptions (set `PARAKEET_PROFILE=1` for a per-stage breakdown) |

### Library

```csharp
using Parakeet;

using var asr = await ParakeetAsr.LoadAsync();                  // or ParakeetAsr.Load("/path/to/model.nemo")
var result = asr.TranscribeFile("audio.wav");                   // any WAV: resampled to 16 kHz mono
Console.WriteLine(result.Text);
foreach (var w in result.Words) Console.WriteLine($"{w.Start:F2}-{w.End:F2} {w.Text}");
Console.WriteLine($"RTF {result.RealTimeFactor:F3}");

float[] samples = /* 16 kHz mono */;
var r = asr.Transcribe(samples, onText: delta => Console.Write(delta));   // streams text as tokens are decoded
```

`ParakeetOptions` sets `Threads`, `Precision` (`Float32` / `Int8`), `PadMode`, `PositionCacheFrames` and `HugePages`.
One `ParakeetAsr` instance is thread-safe and can serve concurrent `Transcribe` calls. The stages are public as well:
`MelSpectrogram` (features), `ConformerEncoder` (with a per-layer `TraceCallback`), `TdtDecoder` (with a per-step
`StepCallback`), `NemoArchive` / `OnnxModelSource` (checkpoint readers), `WaveFile` / `Resampler`.

## How it works

```
16 kHz audio ─► pre-emphasis ─► centered STFT (512, 25 ms Hann, 10 ms hop, double precision) ─► |X|² ─► Slaney mel (128)
             ─► log(x + 2⁻²⁴) ─► per-feature normalization                                  [T, 128], 100 frames/s
             ─► dw-striding subsampling ×8: Conv2d 3×3/2 ─► 2 × (depthwise 3×3/2 ─► pointwise) ─► Linear 4096→1024
             ─► 24 × FastConformer block (d = 1024, 8 heads, FF 4096, conv kernel 9):              [T/8, 1024], 12.5 frames/s
                  x += ½ FF(LN x);  x += RelPosMHA(LN x);  x += Conv(LN x);  x += ½ FF(LN x);  x = LN x
             ─► TDT greedy decoding: 2-layer LSTM prediction network (640) over the last token,
                joint = W_out · relu(W_enc · enc_t + W_pred · g) ─► argmax token (1024 or 8192 + blank)
                                                                   and argmax duration (0-4 frames to skip)
```

| Source | Contents |
|---|---|
| `src/Parakeet/ParakeetAsr.cs` | Public API: load (with download), transcribe, timestamps |
| `src/Parakeet/Audio/` | NeMo-compatible log-mel front end with a double-precision FFT, WAV reading, resampling |
| `src/Parakeet/Model/` | Config, subsampling, conformer encoder, relative-position attention, TDT decoder |
| `src/Parakeet/Numerics/` | GEMM (FP32 AVX-512, int8 AVX-VNNI), packed / quantized weights, worker pool, vector math, row ops |
| `src/Parakeet/IO/` | `.nemo` reader (tar → zip → pickle, memory-mapped), ONNX protobuf reader, YAML, Hugging Face download |
| `src/Parakeet/Text/` | SentencePiece vocabulary / detokenization |
| `src/Parakeet.Cli/` | `parakeet` command-line tool |
| `tests/Parakeet.Tests/` | Kernel unit tests, layer-by-layer parity, end-to-end, int8 accuracy |
| `benchmarks/Parakeet.Benchmarks/` | In-process comparison with ONNX Runtime (the Go server's pipeline), GEMM micro-benchmarks |
| `tools/Parakeet.TestAudio/` | Test-audio generator built on the [Voxwerk](https://github.com/curiosity-ai/voxwerk) NuGet packages |
| `scripts/onnx_reference.py` | ONNX Runtime reference: dumps per-layer tensors, joint logits and transcripts |
| `scripts/gen_kernels.py` | Generates the unrolled FP32 / int8 micro-kernels |
| `testdata/audio/` | 29 generated test clips (+ `manifest.json` with their transcripts and languages) |

### Performance design

The encoder is ~15 GFLOP per second of audio, nearly all of it in GEMMs against 2.4 GB of FP32 weights that are
each used once per forward pass, so the work is to keep the FMA units busy while the weights stream from DRAM.

- **GEMM** (`Numerics/Gemm.cs`): weights are packed once at load into 32-column panels, stored k-block-major so a
  thread sweeping its panels reads one contiguous stream; the next block is prefetched into L2 during the current
  block's tile passes. A is packed once per call into 12-row micro-panels and a 12 × 32 tile of 24 zmm accumulators
  sweeps each panel (generated kernels for 1–12 rows, so short inputs waste nothing). Bias, SiLU / ReLU and the
  residual add are fused into the epilogue, and the macaron ½ is folded into the weights. ~150 GFLOP/s per core with
  hot weights (≈ 87% of the measured FMA peak), on par with ONNX Runtime's MLAS with cold weights on 4 threads.
- **Worker pool** (`Numerics/Parallelism.cs`): a forward pass has a few hundred parallel regions of a few ms each.
  `Parallel.For` lost ~30% to wake-up latency and range-partitioning stragglers; dedicated workers that spin
  briefly between jobs and claim items one at a time scale 2.4× → 3.4× on 4 cores.
- **Attention** (`Model/RelPositionAttention.cs`): instead of materializing `(q + v)·Pᵀ` over all 2T-1 distances and
  rel-shifting it (as NeMo and the ONNX graph do), each block of 64 query rows multiplies only the band of distances
  it needs, then softmax and the value GEMM run while the block is cache-hot. Memory is O(T · 64).
- **Positional projections** (`Model/ConformerEncoder.cs`): `PE · W_posᵀ` depends only on the length, and the rows a
  shorter input needs are an exact slice of a longer input's table, so they are cached across calls (bounded).
- **Subsampling** (`Model/Subsampling.cs`): the 2-D convolutions run in independent bands of output frames with
  small per-thread buffers, so the full-resolution `[T/2, 64, 256]` feature maps never stream through memory.
- **Decoder** (`Model/TdtDecoder.cs`): the encoder half of the joint is one GEMM over all frames; the embedding is
  folded into the first LSTM layer's input projection (a lookup table); the prediction network runs only after a
  non-blank token; per-step GEMVs run on the worker pool. 2.8× faster than ONNX Runtime's per-step decoder graph.
- **Memory**: the checkpoint is memory-mapped and packed in parallel (loads in ~1.5 s from the page cache);
  activation buffers are pooled across calls (allocating them per call cost a page fault + zeroing per page).

### Int8 mode

`ParakeetOptions.Precision = WeightPrecision.Int8` (CLI `--int8`) quantizes the encoder's large projections to int8
per output channel and multiplies them on AVX-VNNI (`vpdpbusd`, 32 int8 multiply-adds per instruction, measured at
2.2× the FP32 FMA rate per core) against activations quantized per row on the fly. Attention scores, normalization,
the residual stream and the decoder stay FP32. Weights take a quarter of the memory. On the test set the v2
transcripts are identical to FP32 (23/23, WER 1.31%) and v3 differs on 3 of 29 clips by a word or two (WER 2.09% vs
1.96%). It falls back to FP32 on CPUs without AVX-VNNI. (.NET 10 exposes neither AVX-512 VNNI, AVX-512 BF16 nor AMX,
so the int8 kernel uses 256-bit vectors.)

## Validation

`scripts/onnx_reference.py` runs the ONNX export (`nemo128.onnx` + `encoder-model.onnx` + `decoder_joint-model.onnx`)
on ONNX Runtime exactly as the Go server did, exposes intermediate encoder tensors as extra graph outputs and saves
them with the per-step joint logits and transcript. The xUnit parity tests feed each C# stage the reference's own
input (ORT features into the encoder, ORT encoder output into the decoder), so errors are measured per stage:

| Stage (v2, 10.8 s clip) | Relative L2 error vs ONNX Runtime |
|---|---|
| Slaney filterbank vs the checkpoint's | 5.6e-8 |
| Log-mel features | 1.9e-6 |
| Subsampling output | 6.6e-7 |
| Relative positional encodings | 1.1e-7 |
| Layer 0 internals (LN, FF1, q, positional projection, attention, GLU, conv, FF2) | 8e-7 – 2.0e-6 |
| Conformer layers 0–23 | 1.6e-6 – 6.2e-6 |
| Encoder output | 6.1e-6 (v3: 5.6e-6) |
| Joint logits, all 67 decode steps | 1.9e-7 (v3: 2.3e-7) |
| Encoder with weights read from the ONNX export instead of the `.nemo` | 2.6e-6 vs `.nemo` weights |

End to end:

| | Clips | Token-identical to ONNX Runtime | WER vs the spoken text |
|---|---|---|---|
| v2 (English) | 23 | **23 / 23** | 1.31% (687 words) |
| v3 (en, de, fr, es, it, pt, ru) | 29 | **29 / 29** | 1.96% (762 words) |
| v2, int8 mode | 23 | 23 / 23 | 1.31% |
| v3, int8 mode | 29 | 26 / 29 | 2.09% |

The remaining WER is the clips' own content (TTS pronunciations of invented names, "three thirty" written as "3.30"),
identical in both implementations.

### Test audio

The clips in `testdata/audio` were generated with the [Voxwerk](https://github.com/curiosity-ai/voxwerk) NuGet packages
(pure C# Qwen3-TTS) by `tools/Parakeet.TestAudio`:

- `samples`: the reference recordings shipped in the `Voxwerk.Voices.English.*` packages (17 voices: US, British and
  African accents, narration, characters, conversational styles), each with its exact transcript;
- `synth`: sentences synthesized with `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, including numbers, a 37 s passage and
  German, French, Spanish, Italian, Portuguese and Russian sentences for the multilingual model.

```bash
dotnet run -c Release --project tools/Parakeet.TestAudio -- samples --out testdata/audio
dotnet run -c Release --project tools/Parakeet.TestAudio -- synth --out testdata/audio
```

### Running the tests

```bash
dotnet test -c Release        # kernel tests; model tests skip (and say so) without the environment below

# Reference dumps from ONNX Runtime (Python: numpy onnx onnxruntime)
python scripts/onnx_reference.py --models /path/to/parakeet-tdt-0.6b-v2-onnx --out /tmp/ref testdata/audio/*.wav

PARAKEET_MODEL=/path/to/parakeet-tdt-0.6b-v2.nemo \
PARAKEET_ONNX_DIR=/path/to/parakeet-tdt-0.6b-v2-onnx \
PARAKEET_REFERENCE_DIR=/tmp/ref \
dotnet test -c Release
```

The ONNX exports are [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx)
and [-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) (the fp32 files; the int8 ones are used only
by the benchmark).

## Performance

Intel Xeon (Sapphire Rapids) cloud VM, 4 cores @ 2.1 GHz (AVX-512, AVX-VNNI), 16 GB; .NET 10.0.13; ONNX Runtime
1.31.0 (Microsoft.ML.OnnxRuntime, CPU, `ORT_ENABLE_ALL`, intra-op threads = 4). The ONNX Runtime side is the Go
server's pipeline run in-process (`benchmarks/Parakeet.Benchmarks/OnnxPipeline.cs`): `nemo128.onnx`, the encoder, and one
`decoder_joint-model.onnx` run per decoding step with pre-allocated tensors. The two engines are run alternately on
each clip and the medians of 5 runs (after a warm-up) are summed; the VM's timings vary by ±15% between runs, which
the interleaving cancels for the comparison. Model v2, all 23 English clips (287 s of audio):

**FP32, C# vs ONNX Runtime** (both exact; transcripts identical on 23 / 23 clips):

| Total over 23 clips | Features | Encoder | Decoder | **Total** | RTF |
|---|---|---|---|---|---|
| ONNX Runtime (`encoder-model.onnx`) | 412 ms | 13 868 ms | 1 886 ms | 16 165 ms | 0.056 |
| Parakeet C# | 252 ms | 12 354 ms | 677 ms | **13 283 ms** | **0.046** |
| Speed-up | 1.64× | 1.12× | 2.78× | **1.22×** | |

**Int8, C# vs ONNX Runtime's int8 export** (the Go server's recommended models):

| Total over 23 clips | Features | Encoder | Decoder | **Total** | RTF |
|---|---|---|---|---|---|
| ONNX Runtime (`encoder-model.int8.onnx`) | 389 ms | 15 246 ms | 998 ms | 16 633 ms | 0.058 |
| Parakeet C# (`--int8`) | 289 ms | 6 637 ms | 695 ms | **7 621 ms** | **0.027** |
| Speed-up | 1.34× | 2.30× | 1.44× | **2.18×** | |

The C# int8 transcripts equal the FP32 ones on all 23 clips; ONNX Runtime's int8 export differs from FP32 on 3
(punctuation, a name, "three thirty" spelled out). ONNX Runtime's int8 encoder is no faster than its FP32 one on this
CPU, so the C# int8 mode is also 2.2× faster than ONNX Runtime FP32 (encoder 2.3×).

Per clip (median, total time):

| Clip | Audio | ORT FP32 | C# FP32 | ORT int8 | C# int8 |
|---|---|---|---|---|---|
| synth-weather | 6.9 s | 422 ms | 330 ms | 470 ms | 199 ms |
| sample-en-us-aaliyah-renwick | 10.8 s | 579 ms | 472 ms | 552 ms | 294 ms |
| sample-en-gb-mei-lin-ashcombe | 15.2 s | 804 ms | 642 ms | 842 ms | 348 ms |
| synth-long | 36.9 s | 1 956 ms | 1 557 ms | 2 256 ms | 909 ms |

Single thread (`--threads 1`, 3 clips, 59 s of audio): C# FP32 7.27 s vs ONNX Runtime 8.30 s (**1.14×**; encoder 1.13×,
decoder 1.37×).

Loading from the page cache: **~1.5 s** for the C# engine (memory-mapped `.nemo`, layers packed in parallel) vs 4–9 s
for ONNX Runtime session creation.

```bash
dotnet run -c Release --project benchmarks/Parakeet.Benchmarks -- \
    --model parakeet-tdt-0.6b-v2.nemo --onnx parakeet-tdt-0.6b-v2-onnx --threads 4 --runs 5 testdata/audio/*.wav
#   --cs-int8: C# int8 mode; --int8: ONNX Runtime runs the int8 export; --only csharp|onnx
dotnet run -c Release --project benchmarks/Parakeet.Benchmarks -- --gemm --threads 4   # GEMM throughput
```

## Building

```bash
dotnet build Parakeet.slnx -c Release
```

Requires the .NET 10 SDK. AVX-512 (or AVX2 + FMA) is used when available, with portable fallbacks; the int8 mode
needs AVX-VNNI.

## License

Apache-2.0 (see [LICENSE](LICENSE)), like the original server. The low-level numerics (native buffers, worker pool
pattern, packed-panel GEMM structure) are adapted from [Voxwerk](https://github.com/curiosity-ai/voxwerk) (MIT, see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)). The Parakeet models are released by NVIDIA under
[CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and are downloaded separately.
