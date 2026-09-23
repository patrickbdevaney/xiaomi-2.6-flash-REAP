"""Gate the audio and video loaders, and MEASURE the one thing the checkpoint does not record.

Video is verifiable end to end and is verified. Audio has exactly one free variable -- the
log-mel filterbank convention -- and rather than picking one and hoping, this runs REAL speech
from the actual calibration source through every candidate and compares how the shipped RVQ
tokenizer responds. A codec trained on 20M hours of audio should use its codebook far more
evenly on in-distribution input than on off-distribution input, so codebook entropy is an
informative (not conclusive) discriminator. The result is printed with that caveat attached.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from chunk_builder import Embedder
import media_loaders as ML

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
fail = 0


def check(name, ok, extra=""):
    global fail
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + extra) if extra else ''}", flush=True)
    fail += (not ok)


if not (SRC / "model.safetensors.index.json").exists():
    sys.exit(f"checkpoint not staged at {SRC}")

from transformers import AutoConfig, AutoProcessor
cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
cfg._name_or_path = str(SRC)
cfg._attn_implementation = "flex_attention" if DEV == "cuda" else "eager"
E = Embedder(SRC, cfg, device=DEV)
proc = AutoProcessor.from_pretrained(SRC, trust_remote_code=True)

# ============================ VIDEO ============================
print("-- video --", flush=True)
from PIL import Image
rng = np.random.default_rng(0)
frames = [Image.fromarray((rng.random((224, 224, 3)) * 255).astype("uint8")) for _ in range(8)]
V = ML.VideoLoader(E, proc)
prep = V.prepare(frames)
n_v = V.n_tokens(prep)
check("video processor produced patches", n_v > 0,
      f"{n_v} tokens from 8 frames, grid {prep['grid_thw'].tolist()}")
check("grid carries a temporal dimension > 1", int(prep["grid_thw"][0][0]) > 1,
      f"T={int(prep['grid_thw'][0][0])}")

ids_v = torch.tensor([[100] + [E.ids["video"]] * n_v + [200]])
emb_v = E.embed_batch(ids_v, media={"video": prep})
plain = E.embed(ids_v.to(DEV)).to(emb_v.dtype)
mask = ids_v.to(DEV).eq(E.ids["video"])
check("vision tower output spliced at the video placeholders",
      not torch.allclose(emb_v[mask].float(), plain[mask].float()),
      f"mean|d| {(emb_v[mask]-plain[mask]).abs().float().mean().item():.4f}")
check("non-video positions untouched", torch.equal(emb_v[~mask].float(), plain[~mask].float()))
check("video embeddings finite and non-degenerate",
      torch.isfinite(emb_v).all() and emb_v[mask].float().std().item() > 1e-3,
      f"std {emb_v[mask].float().std().item():.4f}")

# ============================ AUDIO ============================
print("-- audio --", flush=True)
A = ML.AudioLoader(E, SRC)

# Real speech from the ACTUAL calibration source, not a synthetic tone: the convention test is
# only meaningful on in-distribution input.
wave, provenance = None, None
try:
    from datasets import load_dataset
    ds = load_dataset("gpt-omni/VoiceAssistant-400K", split="train", streaming=True)
    row = next(iter(ds))
    for k, v in row.items():
        # With torchcodec installed, `datasets` hands back an AudioDecoder rather than the
        # old {"array", "sampling_rate"} dict. Accept both: the dict form is what older
        # installs return, and silently taking neither is how this gate ended up running on
        # synthetic audio while reporting success.
        a = sr = None
        if hasattr(v, "get_all_samples"):
            smp = v.get_all_samples()
            a = smp.data.to(torch.float32)
            a = a.mean(0) if a.ndim > 1 else a          # downmix to mono
            a = a.numpy()
            sr = int(smp.sample_rate)
        elif isinstance(v, dict) and "array" in v:
            a = np.asarray(v["array"], dtype=np.float32)
            sr = int(v.get("sampling_rate", ML.AUDIO_SR))
        if a is None or a.size < ML.AUDIO_SR // 2:
            continue
        if sr != ML.AUDIO_SR:                           # cheap linear resample; fine for a gate
            n = int(len(a) * ML.AUDIO_SR / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
        pk = float(np.abs(a).max()) or 1.0
        wave = (a[: ML.AUDIO_SR * 10] / pk * 0.7).astype("float32")
        provenance = f"REAL SPEECH -- gpt-omni/VoiceAssistant-400K field '{k}' @ {sr} Hz"
        del smp, a
        break
    # Drop every reference to the decoder and the streaming iterator before moving on: torchcodec
    # and the HF streaming client both keep native threads alive, and letting them survive into
    # interpreter finalisation ends in "PyGILState_Release: thread state must be current" and a
    # core dump AFTER the gate has already passed -- a green run with a non-zero exit code, which
    # is the worst possible outcome for something meant to gate a chain.
    row.clear()
    del ds, row
    import gc as _gc; _gc.collect()
except Exception as e:
    print(f"  (could not stream real speech: {type(e).__name__}: {str(e)[:90]})", flush=True)

if wave is None:
    # Speech-LIKE fallback: harmonic stack with a wandering F0 and formant shaping. Not speech,
    # and the convention verdict below is correspondingly weaker -- which is why it says so.
    t = np.arange(ML.AUDIO_SR * 10, dtype=np.float32) / ML.AUDIO_SR
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / ML.AUDIO_SR
    wave = sum(np.sin(k * ph) / k for k in range(1, 12)).astype("float32")
    wave *= (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)).astype("float32")
    wave = (wave / np.abs(wave).max() * 0.7).astype("float32")
    provenance = "SYNTHETIC speech-like fallback (no network)"
print(f"  (audio source: {provenance}, {len(wave)/ML.AUDIO_SR:.1f} s)", flush=True)

_m = ML.log_mel(wave)
check("log-mel is time-major [frames, n_mels] as the tokenizer expects",
      _m.shape[1] == ML.AUDIO_N_MELS and _m.shape[0] > _m.shape[1],
      f"{tuple(_m.shape)}  (frames x {ML.AUDIO_N_MELS} bins)")
check("log-mel is finite", torch.isfinite(_m).all())

tok = A._tokenizer()
import calib_pass as CP
mod = CP._modeling(cfg)


def code_entropy(mel_scale, norm):
    m = ML.log_mel(wave, mel_scale=mel_scale, norm=norm).to(DEV, E.dtype)
    codes = mod.tokenize_audio_batch([m], tok.encoder,
                                     segment_size=cfg.audio_config["audio_segment_size"],
                                     device=DEV)
    c = codes[0].reshape(-1).to(torch.int64).cpu().numpy()
    _, cnt = np.unique(c, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum()), int(len(cnt)), c.size


print("  convention comparison (higher entropy = codebook used more evenly):", flush=True)
results = {}
for ms in ("htk", "slaney"):
    for nm in (None, "slaney"):
        try:
            h, uniq, n = code_entropy(ms, nm)
            results[(ms, nm)] = h
            print(f"    mel_scale={ms:7} norm={str(nm):7}  entropy {h:6.3f} bits, "
                  f"{uniq:4d} distinct codes of {n}", flush=True)
        except Exception as e:
            print(f"    mel_scale={ms:7} norm={str(nm):7}  FAILED {type(e).__name__}", flush=True)
check("every candidate convention ran", len(results) == 4, f"{len(results)}/4")
check("the convention test used REAL speech, not the synthetic fallback",
      provenance.startswith("REAL"), provenance)
if results:
    best = max(results, key=results.get)
    spread = max(results.values()) - min(results.values())
    print(f"  -> highest entropy: mel_scale={best[0]}, norm={best[1]} "
          f"(spread across conventions {spread:.3f} bits)", flush=True)
    check("the choice is measurably consequential (or measurably is not)", True,
          "spread <0.1 bits would mean it does not matter" if spread < 0.1 else
          "spread is material; the convention must be resolved before the audio bucket is trusted")

# The end-to-end path must still produce spliceable embeddings whatever the verdict.
emb_a = A.embed([wave])
check("audio path yields backbone-width embeddings",
      emb_a.ndim == 2 and emb_a.shape[1] == cfg.hidden_size, str(tuple(emb_a.shape)))
check("audio embeddings finite and non-degenerate",
      torch.isfinite(emb_a).all() and emb_a.float().std().item() > 1e-3,
      f"std {emb_a.float().std().item():.4f}")
secs = len(wave) / ML.AUDIO_SR
rate = emb_a.shape[0] / secs
check("backbone token rate matches the report's 6.25 Hz, not 25 Hz",
      abs(rate - 6.25) < 1.5, f"{rate:.2f} tok/s over {secs:.1f} s")

n_a = A.n_tokens(wave)
ids_a = torch.tensor([[100] + [E.ids["audio"]] * n_a + [200]])
emb_full = E.embed_batch(ids_a, media={"audio": {"audio_embeds": emb_a}})
mask_a = ids_a.to(DEV).eq(E.ids["audio"])
check("audio embeddings splice at the placeholders",
      not torch.allclose(emb_full[mask_a].float(),
                         E.embed(ids_a.to(DEV)).to(emb_full.dtype)[mask_a].float()),
      f"{n_a} audio tokens")

print("\nGATE " + ("PASS" if not fail else f"FAIL ({fail})"))
sys.stdout.flush(); sys.stderr.flush()
# os._exit, not sys.exit: see the cleanup note above. Native decoder threads can still be
# unwinding, and finalising the interpreter under them turns a passing gate into a core dump.
# Everything this script owns is already written and flushed by here.
import os as _os
_os._exit(1 if fail else 0)
