"""Audio and video loaders: real media into the shape the towers expect.

VIDEO is fully supported by shipped code. `AutoProcessor` resolves a `Qwen2VLVideoProcessor`,
frames go through the SAME vision tower as images, and the only difference is `grid_thw` carrying
T > 1. Nothing here is guessed.

AUDIO IS NOT, AND THIS FILE IS EXPLICIT ABOUT WHICH PART. The repo ships no audio feature
extractor -- `AutoProcessor` returns image and video processors and no `feature_extractor` -- so
the log-mel front end has to be reconstructed from `processor_config`:

    sampling_rate 24000, n_mels 128, n_fft 960, hop 240, window 960, fmin 0, fmax None

Those numbers are the checkpoint's own. What the checkpoint does NOT record is the FILTERBANK
CONVENTION -- htk vs slaney mel scale, and whether the filters are area-normalised. The
technical report says only "128 mel bins" and that the tokenizer "follows the training recipe of
MiMo-Audio (Xiaomi, 2025)". Pick the wrong convention and nothing raises: the mels are finite,
the tokenizer emits codes, the encoder emits embeddings, and we would calibrate the audio
experts on off-distribution input -- which is worse than not calibrating them, because it
retains the WRONG experts.

`gate_media_loaders.py` therefore does not assume. It streams REAL speech from the actual
calibration source and runs every convention through the shipped RVQ tokenizer, comparing
codebook entropy. MEASURED RESULT: all four land within 0.011 bits of each other, so the
convention is not load-bearing. See AUDIO_MEL_CONVENTION below for the numbers and the caveat.

TOKEN COUNT IS ASKED FOR, NEVER DERIVED. `processor_config.audio_input_id_per_second` is 25.0,
but the technical report says four consecutive 25 Hz frames are grouped into one patch, giving a
backbone rate of 6.25 Hz -- a 4x difference, and exactly the kind of number that is wrong when
assumed. So the count comes from the tokenizer's own `get_output_length` and the encoder's own
grouping.
"""
from __future__ import annotations

import numpy as np
import torch

AUDIO_SR = 24000
AUDIO_N_MELS = 128
AUDIO_N_FFT = 960
AUDIO_HOP = 240
AUDIO_WINDOW = 960
# Clip lengths from mimo_corpus_spec.CLIP_SECONDS, mirrored here so the loaders do not import
# the spec (the spec imports the corpus, and the loaders must stay usable on their own).
# ---------------------------------------------------------------- the patch-row budget
#
# THE ALLOCATION THAT KILLED THREE RUNS. MiMoVisionAttention materialises a DENSE
# [1, num_heads, L, L] tensor for every attention chunk -- not as a fallback, but explicitly,
# purely to place a sink bias in column 0:
#
#     sink_bias = torch.zeros(1, self.num_heads, q_c.shape[2], k_c.shape[2], ...)
#     sink_bias[..., 0] = self.sinks.view(1, self.num_heads, 1)
#     attn_mask = sink_bias if attn_mask is None else attn_mask + sink_bias
#
# `attn_mask + sink_bias` makes a second copy, and passing an explicit float mask to
# scaled_dot_product_attention forces the math path, which materialises the scores as a third.
# The 24 windowed blocks additionally build a dense [L, L] mask. With num_heads=32 the cost is
# ~64 L^2 bytes PER COPY.
#
# MEASURED: a WebSight screenshot is 2560x2176 -> L = 21,760 patch rows -> 30.3 GB per copy.
# The run lost 95 GB in under five seconds and was OOM-killed; `cached` never moved, because
# this is a live allocation, not page cache. The checkpoint's own preprocessor_config allows
# max_pixels = 12,845,056, i.e. L = 50,176, which would be a 161 TB sink_bias -- the configured
# cap is effectively absent, so it must be supplied here.
#
# L_MAX is therefore chosen from the memory it costs, not from image aesthetics:
#     bytes ~= 4 copies * 32 heads * L^2 * 2 B = 256 * L^2
#     L=4096 -> 4.3 GB      L=3072 -> 2.4 GB      L=2048 -> 1.07 GB
# 4096 patch rows = 1024 merged tokens, so a capped image still occupies a quarter of a
# 4096-token sequence -- ample for calibration, where COVERAGE matters and resolution does not.
MAX_PATCH_ROWS = 4096

# Video packs the WHOLE CLIP into a single attention chunk: grid is (T, h, w) with
# T = frames / temporal_patch_size, and L = T*h*w. Ten 1080p frames would be T=5, h*w=8160,
# L=40,800 -- four times worse than the screenshot that killed the run. The per-frame budget is
# therefore the total budget divided by the temporal extent.
CLIP_SECONDS_AUDIO = 30
CLIP_FRAMES_VIDEO = 10

# MEASURED 2026-09-23 on REAL SPEECH, not chosen. gate_media_loaders streams a clip from
# gpt-omni/VoiceAssistant-400K -- the actual audio calibration source -- and runs all four
# candidate conventions through the shipped RVQ tokenizer, comparing how evenly each uses the
# codebook. A codec trained on 20M hours should use its codebook more evenly on in-distribution
# input, so entropy discriminates:
#
#     mel_scale=htk     norm=None      7.401 bits, 260 distinct codes
#     mel_scale=htk     norm=slaney    7.393 bits, 253
#     mel_scale=slaney  norm=None      7.404 bits, 253
#     mel_scale=slaney  norm=slaney    7.402 bits, 255
#
# SPREAD 0.011 BITS on real speech -- 0.15% of the entropy, and TIGHTER than the 0.063 measured
# on synthetic speech-like audio, which is the direction that should reassure: the closer the
# input is to what the codec was trained on, the less the convention matters. The one thing the
# checkpoint declines to record turns out not to be load-bearing.
#
# This is a heuristic, not a proof -- there is no waveform decoder in MiMoAudioTokenizer, so no
# reconstruction oracle exists. But a convention that mattered would have shown a large spread,
# and none did.
AUDIO_MEL_CONVENTION = {"mel_scale": "slaney", "norm": None, "verified": "measured",
                        "note": "all 4 conventions within 0.011 bits of codebook entropy on real "
                                "speech from gpt-omni/VoiceAssistant-400K; the convention is not "
                                "load-bearing"}


def log_mel(wave: np.ndarray, sr: int = AUDIO_SR, mel_scale: str | None = None,
            norm: str | None = "__default__") -> torch.Tensor:
    """waveform [T] float32 in [-1, 1] -> log-mel [frames, n_mels], float32.

    TIME-MAJOR, which is not the usual orientation. `tokenize_audio_batch` reads the frame count
    as `m.size(0)` and concatenates several clips along dim 0, so a [n_mels, frames] tensor is
    not merely transposed -- it would be segmented along the MEL axis and the convolution would
    see 401 "channels" instead of 128. Caught by exactly that error.

    Uses transformers' own `mel_filter_bank`/`spectrogram` rather than a local implementation, so
    the only free variables are the two convention flags.
    """
    from transformers.audio_utils import mel_filter_bank, spectrogram
    ms = mel_scale or AUDIO_MEL_CONVENTION["mel_scale"]
    nm = AUDIO_MEL_CONVENTION["norm"] if norm == "__default__" else norm
    fb = mel_filter_bank(num_frequency_bins=1 + AUDIO_N_FFT // 2, num_mel_filters=AUDIO_N_MELS,
                         min_frequency=0.0, max_frequency=sr / 2.0, sampling_rate=sr,
                         norm=nm, mel_scale=ms)
    spec = spectrogram(np.asarray(wave, dtype=np.float32),
                       window=np.hanning(AUDIO_WINDOW + 1)[:-1].astype(np.float32),
                       frame_length=AUDIO_WINDOW, hop_length=AUDIO_HOP, fft_length=AUDIO_N_FFT,
                       power=2.0, mel_filters=fb, log_mel="log10")
    return torch.from_numpy(np.asarray(spec, dtype=np.float32)).transpose(0, 1).contiguous()


class AudioLoader:
    """Waveform -> audio-patch embeddings, via the shipped tokenizer and encoder."""

    def __init__(self, embedder, src):
        self.E = embedder
        self.src = src
        self._tok = None

    def _tokenizer(self):
        """The shipped 1.87 GB audio tokenizer. Built through the model's own loader so its
        config parsing and partial-load behaviour are the vendor's, not a reimplementation."""
        if self._tok is None:
            import calib_pass as CP
            from safetensors.torch import load_file
            import json as _json
            from pathlib import Path
            mod = CP._modeling(self.E.cfg)
            d = Path(self.src) / "audio_tokenizer"
            cfg = mod.MiMoAudioTokenizerConfig(**_json.loads((d / "config.json").read_text()))
            t = mod.MiMoAudioTokenizer(cfg)
            t.load_state_dict(load_file(str(d / "model.safetensors"), device="cpu"), strict=False)
            self._tok = t.to(device=self.E.device, dtype=self.E.dtype).eval().requires_grad_(False)
        return self._tok

    @torch.no_grad()
    def embed(self, waves: list[np.ndarray]) -> torch.Tensor:
        """-> [n_backbone_tokens, hidden]. One entry per audio patch, ready to splice."""
        assert waves, "no audio supplied"
        enc = self.E._lazy("audio")
        mels = [log_mel(w).to(self.E.device, self.E.dtype) for w in waves]
        return enc.get_audio_feature(mels, self.E._speech, self._tokenizer().encoder)

    @torch.no_grad()
    def n_tokens(self, wave: np.ndarray) -> int:
        """ASK the model how many backbone tokens this waveform becomes. Never assume."""
        return int(self.embed([wave]).shape[0])


class VideoLoader:
    """Frames -> vision-tower embeddings. Same tower as images; grid_thw carries T > 1."""

    def __init__(self, embedder, processor):
        self.E = embedder
        self.proc = processor

    def prepare(self, frames) -> dict:
        """frames: list of PIL images or an array [T, H, W, 3] -> processor tensors."""
        out = self.proc.video_processor(videos=[frames], return_tensors="pt")
        key = "pixel_values_videos" if "pixel_values_videos" in out else "pixel_values"
        grid = "video_grid_thw" if "video_grid_thw" in out else "image_grid_thw"
        return {"pixel_values": out[key], "grid_thw": out[grid]}

    def n_tokens(self, prepared: dict) -> int:
        merge = self.E.cfg.vision_config["spatial_merge_size"]
        return int(prepared["grid_thw"].prod(-1).sum().item()) // (merge ** 2)


def patch_rows(grid_thw) -> int:
    """Attention length L the vision tower will actually run at, summed over the chunk."""
    return int(grid_thw.prod(-1).sum().item())


def fit_image(img, cfg, max_patch_rows: int = MAX_PATCH_ROWS):
    """Downscale a PIL image so the tower's attention length stays inside the budget.

    We resize the image OURSELVES rather than asking the processor to. Setting `max_pixels`
    (and `size["longest_edge"]`) on Qwen2VLImageProcessor was MEASURED to change nothing: a
    WebSight screenshot still came back as 21,760 patch rows, the exact value that killed the
    run. A cap that the downstream component is free to ignore is not a cap, and the failure
    mode is a machine-wide SIGKILL, so this is enforced where it cannot be overridden.

    Aspect ratio is preserved; only images already over budget are touched.
    """
    vc = cfg.vision_config
    patch = int(vc["patch_size"] if isinstance(vc, dict) else vc.patch_size)
    budget_px = max_patch_rows * patch * patch
    w, h = img.size
    if w * h <= budget_px:
        return img
    scale = (budget_px / float(w * h)) ** 0.5
    # floor to whole patches, and leave a patch of slack so rounding inside the processor
    # cannot push it back over the budget
    nw = max(patch, int(w * scale) // patch * patch - patch)
    nh = max(patch, int(h * scale) // patch * patch - patch)
    from PIL import Image
    return img.resize((nw, nh), Image.BICUBIC)


def fit_frames(frames, cfg, max_patch_rows: int = MAX_PATCH_ROWS,
               frames_per_clip: int = CLIP_FRAMES_VIDEO):
    """Same, for a video clip: the WHOLE clip is one attention chunk, so the budget is shared
    across the temporal extent rather than applied per frame."""
    vc = cfg.vision_config
    tpatch = int((vc["temporal_patch_size"] if isinstance(vc, dict) else vc.temporal_patch_size)
                 or 1)
    t_grid = max(1, len(frames) // max(1, tpatch))
    per_frame = max(64, max_patch_rows // t_grid)
    return [fit_image(f, cfg, per_frame) for f in frames]


def configure_media_limits(proc, cfg, max_patch_rows: int = MAX_PATCH_ROWS,
                           frames_per_clip: int = CLIP_FRAMES_VIDEO) -> dict:
    """Cap the processors so no sample can hand the tower an L that will not fit.

    Returns the limits applied, for the log -- a silent cap is how the original 12.8M-pixel
    default went unnoticed.
    """
    vc = cfg.vision_config
    patch = int(vc["patch_size"] if isinstance(vc, dict) else vc.patch_size)
    tpatch = int((vc["temporal_patch_size"] if isinstance(vc, dict) else vc.temporal_patch_size)
                 or 1)
    img_px = max_patch_rows * patch * patch
    t_grid = max(1, frames_per_clip // tpatch)
    vid_px = max(patch * patch, (max_patch_rows // t_grid) * patch * patch)
    applied = {"max_patch_rows": max_patch_rows, "image_max_pixels": img_px,
               "video_max_pixels_per_frame": vid_px, "t_grid": t_grid}
    for name, px in (("image_processor", img_px), ("video_processor", vid_px)):
        sub = getattr(proc, name, None)
        if sub is None:
            continue
        for attr in ("max_pixels",):
            if hasattr(sub, attr):
                setattr(sub, attr, px)
        # Newer processors carry the bound inside `size`, and ignore the flat attribute.
        sz = getattr(sub, "size", None)
        if isinstance(sz, dict) and "longest_edge" in sz:
            sz["longest_edge"] = px
    return applied
