#!/usr/bin/env python3
"""Reference pipeline for Parakeet TDT on ONNX Runtime, used to validate the pure C# port.

Runs the istupakov ONNX export (nemo128.onnx preprocessor, encoder-model.onnx, decoder_joint-model.onnx) exactly
like the Go server in .reference/ does: log-mel features -> Conformer encoder -> greedy TDT decoding with the
decoder/joint network evaluated one step at a time.

For each input WAV it writes a directory of .npy tensors plus result.json:

  features.npy               [128, T]       normalized log-mel features (preprocessor output)
  pre_encode.npy             [T', 1024]     subsampling output (dw_striding conv stack + linear)
  pos_emb.npy                [2T'-1, 1024]  relative positional encodings
  layer0.<step>.npy          [T', 1024]     residual stream inside layer 0 after ff1 / attn / conv / ff2
  layer0.<op>.npy                           selected layer 0 internals (q projection, attention probs, depthwise conv)
  layer{i}.npy               [T', 1024]     output of conformer layer i (after norm_out)
  encoder.npy                [T', 1024]     encoder output (time-major)
  joint_logits.npy           [steps, V+5]   joint output of every decode step
  result.json                text, token ids, frame indices, durations, timings

Usage:
  python scripts/onnx_reference.py --models /path/to/parakeet-tdt-0.6b-v2-onnx --out /tmp/ref testdata/audio/*.wav
  python scripts/onnx_reference.py --models ... --bench 3 testdata/audio/synth-long.wav     # timing only
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np
import onnx
import onnxruntime as ort

LAYER0_STEPS = {
    "ff1": "/layers.0/Add_output_0",
    "attn": "/layers.0/Add_1_output_0",
    "conv": "/layers.0/Add_2_output_0",
    "ff2": "/layers.0/Add_3_output_0",
}
LAYER0_OPS = {
    "norm_ff1": "/layers.0/norm_feed_forward1/LayerNormalization_output_0",
    "ff1_linear1": "/layers.0/feed_forward1/linear1/MatMul_output_0",
    "q": "/layers.0/self_attn/linear_q/MatMul_output_0",
    "p": "/layers.0/self_attn/linear_pos/MatMul_output_0",
    "scores": "/layers.0/self_attn/Div_output_0",
    "probs": "/layers.0/self_attn/Softmax_output_0",
    "glu": "/layers.0/conv/Mul_output_0",
    "dwconv": "/layers.0/conv/depthwise_conv/Conv_output_0",
}


def read_wav(path):
    with wave.open(path, "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        raise ValueError(f"{path}: expected 16 kHz audio, got {rate}")
    return x.astype(np.float32)


def debug_encoder(models, n_layers=24):
    """Encoder graph with intermediate tensors exposed as extra outputs (shares the original external data file)."""
    src = os.path.join(models, "encoder-model.onnx")
    dst = os.path.join(models, "encoder-model.debug.onnx")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    m = onnx.load(src, load_external_data=False)
    names = ["/pre_encode/out/Add_output_0", "/pos_enc/Slice_output_0"]
    names += list(LAYER0_STEPS.values()) + list(LAYER0_OPS.values())
    names += [f"/layers.{i}/norm_out/LayerNormalization_output_0" for i in range(n_layers)]
    produced = {o for n in m.graph.node for o in n.output}
    for name in names:
        if name not in produced:
            raise KeyError(f"tensor {name} not found in encoder graph")
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    onnx.save(m, dst)
    return dst


class Reference:
    def __init__(self, models, debug=False, threads=0):
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.vocab = {}
        with open(os.path.join(models, "vocab.txt"), encoding="utf-8") as f:
            for line in f:
                tok, idx = line.rstrip("\n").rsplit(" ", 1)
                self.vocab[int(idx)] = tok
        self.blank = max(self.vocab)  # <blk> is the last id
        self.pre = ort.InferenceSession(os.path.join(models, "nemo128.onnx"), opts, providers=["CPUExecutionProvider"])
        enc_path = debug_encoder(models) if debug else os.path.join(models, "encoder-model.onnx")
        self.enc = ort.InferenceSession(enc_path, opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(models, "decoder_joint-model.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        self.debug = debug
        self.max_symbols = 10
        self.durations = [0, 1, 2, 3, 4]

    def features(self, wav):
        feats, lens = self.pre.run(None, {"waveforms": wav[None, :], "waveforms_lens": np.array([len(wav)], np.int64)})
        return feats, lens

    def encode(self, feats, lens):
        names = [o.name for o in self.enc.get_outputs()]
        outs = self.enc.run(None, {"audio_signal": feats, "length": lens})
        return dict(zip(names, outs))

    def decode(self, enc, enc_len):
        """Greedy TDT decoding, one decoder/joint evaluation per step (NeMo GreedyTDTInfer semantics)."""
        state1 = np.zeros((2, 1, 640), np.float32)
        state2 = np.zeros((2, 1, 640), np.float32)
        last = self.blank
        tokens, frames, durs, logits_log = [], [], [], []
        t = 0
        v = len(self.vocab)
        while t < enc_len:
            frame = enc[:, :, t:t + 1]
            symbols = 0
            while True:
                out, _, s1, s2 = self.dec.run(None, {
                    "encoder_outputs": frame,
                    "targets": np.array([[last]], np.int32),
                    "target_length": np.array([1], np.int32),
                    "input_states_1": state1,
                    "input_states_2": state2,
                })
                logits = out[0, 0, 0]
                logits_log.append(logits.copy())
                k = int(np.argmax(logits[:v]))
                skip = self.durations[int(np.argmax(logits[v:]))]
                if k != self.blank:
                    tokens.append(k)
                    frames.append(t)
                    durs.append(skip)
                    state1, state2, last = s1, s2, k
                symbols += 1
                t += skip
                # A blank with duration 0 would repeat the same evaluation until max_symbols: stop right away.
                if skip != 0 or k == self.blank or symbols >= self.max_symbols:
                    break
            if skip == 0:
                t += 1  # blank with duration 0, or max_symbols reached: advance one frame
        return tokens, frames, durs, np.stack(logits_log) if logits_log else np.zeros((0, v + 5), np.float32)

    def text(self, tokens):
        return "".join(self.vocab[k] for k in tokens).replace("▁", " ").strip()

    def transcribe(self, wav):
        t0 = time.perf_counter()
        feats, lens = self.features(wav)
        t1 = time.perf_counter()
        outs = self.encode(feats, lens)
        t2 = time.perf_counter()
        enc_len = int(outs["encoded_lengths"][0])
        tokens, frames, durs, logits = self.decode(outs["outputs"], enc_len)
        t3 = time.perf_counter()
        timing = {"preprocess": t1 - t0, "encoder": t2 - t1, "decoder": t3 - t2, "total": t3 - t0}
        return feats, lens, outs, tokens, frames, durs, logits, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="directory with the istupakov ONNX export")
    ap.add_argument("--out", help="output directory for tensors (one subdirectory per input)")
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = ORT default)")
    ap.add_argument("--bench", type=int, default=0, help="time N transcriptions per file instead of dumping")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    ref = Reference(args.models, debug=bool(args.out) and not args.bench, threads=args.threads)
    for path in args.wavs:
        wav = read_wav(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if args.bench:
            ref.transcribe(wav)  # warm-up
            runs = [ref.transcribe(wav)[-1] for _ in range(args.bench)]
            best = {k: min(r[k] for r in runs) for k in runs[0]}
            secs = len(wav) / 16000
            print(f"{name}: {secs:.1f}s audio  " + "  ".join(f"{k} {v * 1000:.0f} ms" for k, v in best.items())
                  + f"  RTF {best['total'] / secs:.4f}")
            continue

        feats, lens, outs, tokens, frames, durs, logits, timing = ref.transcribe(wav)
        text = ref.text(tokens)
        print(f"{name}: {text}")
        if not args.out:
            continue
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "features.npy"), feats[0])
        np.save(os.path.join(d, "encoder.npy"), outs["outputs"][0].T.copy())
        np.save(os.path.join(d, "pre_encode.npy"), outs["/pre_encode/out/Add_output_0"][0])
        np.save(os.path.join(d, "pos_emb.npy"), outs["/pos_enc/Slice_output_0"][0])
        for step, tensor in LAYER0_STEPS.items():
            np.save(os.path.join(d, f"layer0.{step}.npy"), outs[tensor][0])
        for op, tensor in LAYER0_OPS.items():
            np.save(os.path.join(d, f"layer0.{op}.npy"), outs[tensor][0])
        for i in range(24):
            np.save(os.path.join(d, f"layer{i}.npy"), outs[f"/layers.{i}/norm_out/LayerNormalization_output_0"][0])
        np.save(os.path.join(d, "joint_logits.npy"), logits)
        with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"wav": os.path.abspath(path), "samples": len(wav), "feature_frames": int(lens[0]),
                       "encoded_frames": int(outs["encoded_lengths"][0]), "text": text, "tokens": tokens,
                       "frames": frames, "durations": durs, "timing": timing}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
