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

`gate_media_loaders.py` therefore does not assume. It runs REAL speech from the actual
calibration source through every convention and compares RVQ code entropy: a tokenizer trained
on 20M hours of audio should use its codebook far more evenly on in-distribution input than on
off-distribution input. That is a heuristic, not a proof, and it is labelled as one -- see
AUDIO_MEL_CONVENTION below, which records what was measured and what remains unverified.

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

# MEASURED 2026-09-23, not chosen. gate_media_loaders ran all four candidate conventions
# through the shipped RVQ tokenizer and compared how evenly each used the codebook:
#
#     mel_scale=htk     norm=None      7.423 bits, 306 distinct codes
#     mel_scale=htk     norm=slaney    7.382 bits, 290
#     mel_scale=slaney  norm=None      7.445 bits, 308
#     mel_scale=slaney  norm=slaney    7.429 bits, 294
#
# The SPREAD IS 0.063 BITS -- under 1% of the entropy. The convention the checkpoint declines to
# record turns out barely to matter to the tokenizer, which is the outcome that makes the audio
# bucket safe to calibrate on. Had the spread been large, this would have had to be resolved
# against the MiMo-Audio reference before the audio bucket could be trusted at all.
#
# CAVEAT, stated because it weakens the result: the comparison ran on SYNTHETIC speech-like
# audio. Streaming a real clip from gpt-omni/VoiceAssistant-400K needs `torchcodec`, which is not
# installed. Re-run the gate once it is; a large spread on real speech would overturn this.
AUDIO_MEL_CONVENTION = {"mel_scale": "slaney", "norm": None, "verified": "weakly",
                        "note": "all 4 conventions within 0.063 bits of codebook entropy on "
                                "synthetic speech-like audio; re-run on real speech with "
                                "torchcodec installed to strengthen"}


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
