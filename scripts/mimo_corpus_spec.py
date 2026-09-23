"""Calibration corpus specification for MiMo-V2.6-Flash-RL.

Reuses the TEXT buckets from `corpus_spec` unchanged -- text is text, the operator-approved
mixture of 2026-08-27 and its pass-3 token-space correction apply here verbatim, and re-deriving
them would throw away evidence. What this file adds is the thing MiMo has and GLM-5.3-Flash did
not: ROUTED TOKENS THAT ARE NOT TEXT AND NOT IMAGES.

WHY THE MULTIMODAL BUCKET MUST SPLIT THREE WAYS
-----------------------------------------------
`corpus_spec.MM_SOURCES` already states the governing principle:

    "rows must carry real images, which are pushed through the real processor. Text
     descriptions of images route like text and protect nothing."

That principle has a MiMo-shaped hole. MiMo ingests audio and video through their own towers,
and the resulting tokens then flow through THE SAME 256 routed experts as everything else. REAP
prunes an expert when the calibration corpus shows it carrying little gated output mass. An
expert that specialises in audio- or video-derived token distributions will show EXACTLY ZERO
mass against a corpus that contains no audio and no video -- so it is not merely under-weighted,
it is guaranteed pruned, and it is guaranteed pruned first.

This is not a hypothetical. The whole reason MiMo was chosen over staying on GLM-5.3-Flash is
that a Thor-runnable OMNIMODAL coding agent does not otherwise exist (see
research/MIMO_V26_FLASH_SURVEY_2026-09-23.md). Calibrating on text+images and then pruning 50%
of the experts would destroy the one capability that justified the model choice, and would do it
silently: every text benchmark would look fine.

By the corpus's own logic, a transcript protects the audio path no better than a caption
protects the vision path. Audio rows must carry real audio; video rows must carry real video.

WHAT IS AND IS NOT DECIDED HERE
-------------------------------
DECIDED: the split exists, and it is fed with real media. That is forced by the architecture and
needs no measurement.

NOT DECIDED: the SIZE of each slice. The 0.12 multimodal share is held CONSTANT and divided,
rather than grown at the expense of code/agentic, because there is no MiMo measurement yet to
justify taking tokens from the primary use case. Pass 3 on GLM set its ratios from measured
pass-2 accumulators rather than from intuition; the same discipline applies. After the MiMo
pass-1 saliency run, re-fit from `cnt_by_bucket` exactly as corpus_spec pass 3 did.

There is a real argument that 0.12 is now too small -- GLM spent all 12% on one modality and
MiMo spends it on three -- and a real argument against, since ballast at 15% was itself bought
with evidence (general dNLL 1.0027, the worst bucket) and code sits at only 0.19. Leaving it
flagged and measured beats guessing in either direction.

TOKEN COST, COMPUTED FROM THE PROCESSOR CONFIG (not estimated)
--------------------------------------------------------------
    audio: processor_config.audio_input_id_per_second = 25.0
           -> 30 s of audio = 750 tokens;  60 s = 1,500
    video: patch_size 16, merge_size 2, fps 1.0
           -> one 448x448 frame = (448/16/2)^2 = 196 tokens
           -> 10 s at 1 fps = 1,960 tokens;  60 s = 11,760 tokens

Video is the expensive modality: one minute is 72% of the 16,384-token cap on its own. Hence
CLIP_SECONDS below -- short clips buy expert COVERAGE, which is what saliency needs, where long
clips would buy token volume in a handful of samples. Coverage is the goal: we are trying to
make every audio/video-specialised expert fire at least enough to be seen.
"""
from __future__ import annotations

from corpus_spec import (SOURCES, MM_SOURCES, TOTAL_SAMPLES, MAX_TOKENS,
                         HELDOUT_FRACTION, DIFFICULTY_TARGET, EXCLUDED)

# Text targets carried over from corpus_spec pass 3; `multimodal` is replaced by its three parts.
TOKEN_TARGET = {
    "agentic":    0.23,
    "code":       0.19,
    "math":       0.15,
    "ballast":    0.15,
    "image":      0.06,      # was the whole 0.12 `multimodal` bucket on GLM
    "audio":      0.03,
    "video":      0.03,
    "science":    0.10,
    "finance":    0.06,
}
assert abs(sum(TOKEN_TARGET.values()) - 1.0) < 1e-9
assert abs(TOKEN_TARGET["image"] + TOKEN_TARGET["audio"] + TOKEN_TARGET["video"] - 0.12) < 1e-9, \
    "the multimodal share is held constant and split; growing it needs a MiMo measurement"

# Clip lengths, chosen from the token arithmetic above so a sample is a few thousand tokens
# rather than one video eating a whole sample's budget.
CLIP_SECONDS = {"audio": 30, "video": 10}

# Estimated tokens-per-sample for the new buckets, DERIVED from the processor config, not
# measured. corpus_spec._TOKENS_PER_SAMPLE holds MEASURED ratios for the text buckets; these two
# are the only estimated entries in the file and must be re-fit from the MiMo pass-1
# accumulators before anyone trusts the delivered shares.
TOKENS_PER_SAMPLE_EST = {
    "audio": 25.0 * CLIP_SECONDS["audio"] / 1000.0,      # 0.75k tok/sample
    "video": 196.0 * CLIP_SECONDS["video"] / 1000.0,     # 1.96k tok/sample
}

# Licences VERIFIED against the HF dataset API on 2026-09-23, not read off a card. Permissive
# only, same policy as corpus_spec: a derivative of an MIT model can stay MIT, and that property
# is irreversible once lost.
AUDIO_SOURCES = [
    # apache-2.0, 219.5 GB, modality:audio+text. The standard omni voice-assistant instruction
    # set -- spoken instructions with responses, which is the audio analogue of the agentic
    # bucket rather than of plain ASR. Streamed, so its size is irrelevant.
    ("gpt-omni/VoiceAssistant-400K", "default", "train", 0.70),
    # apache-2.0, audio QA. Second source so a single collection's idiosyncrasies do not define
    # what "audio" means to the saliency pass.
    ("shenyunhang/AudioQA-1M", None, "train", 0.30),
]

VIDEO_SOURCES = [
    # apache-2.0, 449 GB, task_categories video-text-to-text. Real video with instructions.
    ("ShareGPTVideo/train_video_and_instruction", None, "train", 1.00),
]

# Images: unchanged from GLM. These were already chosen for document/chart/GUI density, which is
# what a coding agent actually looks at.
IMAGE_SOURCES = MM_SOURCES

# Added to corpus_spec.EXCLUDED rather than replacing it.
EXCLUDED_OMNI = {
    # No `license` field at all on the HF API as of 2026-09-23. Absent is not permissive, and
    # the permissive-only policy is what keeps an MIT derivative possible.
    "licence_undeclared": ["lmms-lab/LLaVA-Video-178K"],
    # cc-by-4.0 and enormous, but audio-CLASSIFICATION: labels, not instructions. It would
    # calibrate the backbone on a task nobody runs.
    "scope_not_instruction": ["agkphysics/AudioSet"],
}
