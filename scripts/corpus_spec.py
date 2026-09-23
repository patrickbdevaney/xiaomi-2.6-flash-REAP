"""VENDORED from ~/glm-5.3-reap/scripts/corpus_spec.py on 2026-09-23, so this repo runs standalone.

THE GLM REPO'S COPY REMAINS AUTHORITATIVE FOR THE SHIPPED GLM-5.3-Flash-REAP50 ARTIFACT. The
MiMo work overrides TOKEN_TARGET in `mimo_corpus_spec`; change numbers there, never here, so the
provenance of the GLM REAP stays intact and the two copies cannot silently diverge.

`mimo_corpus_spec` imports the TEXT buckets below unchanged -- text is text, the mixture is
operator-approved (2026-08-27) and pass-3 corrected (2026-09-09) from measured accumulators, and
re-deriving it would throw that evidence away.

Original docstring follows.

Calibration corpus specification.

Mixture and licence policy are operator-approved (2026-08-27) and reasoned about in
wiki/80-calibration.md. Summary of the two decisions encoded here:

  * Mixture is weighted for "a strong coder/agent that stays empirically knowledgeable",
    so code-adjacent (agentic + code + math) is 60%, world knowledge sits at a sufficiency
    floor of 18%, and multimodal holds at 15% because R3 is the one risk where failure is
    certain rather than probable.
  * Permissive licences only. GLM-5.3-Flash is MIT and a derivative can be MIT; that property
    is irreversible if lost. Every NC source had a permissive replacement, the main one 4x
    larger. Sources excluded on licence/scope are listed at the bottom so the exclusion stays
    visibly deliberate.

`text_fn` maps a raw row to a training string. Returning None drops the row.
"""
from __future__ import annotations

import json

TOTAL_SAMPLES = 12_288
MAX_TOKENS = 16_384
HELDOUT_FRACTION = 0.08          # stratified, per bucket, for the section-8 proxies

# PASS 3 (2026-09-09). Two changes, both evidence-driven; pass-2 values kept in
# corpus_spec.py.pass2 and the derivation below so the change stays auditable.
#
# (1) TOKEN SPACE, NOT SAMPLE SPACE. MIXTURE is consumed as a SAMPLE quota
#     (s02_corpus: TOTAL_SAMPLES * f) but saliency accumulates per ROUTED TOKEN, and the
#     two differ by up to 4x because bucket sequence lengths differ wildly. Measured from
#     the pass-2 accumulators (cnt_by_bucket summed over all 42 layers):
#
#         bucket      sample%   token%   ratio
#         agentic      24.0%    28.4%    1.18x
#         code         21.0%     9.3%    0.44x     <-- specified 21%, DELIVERED 9.3%
#         math         15.0%    26.0%    1.73x
#         multimodal   15.0%     8.8%    0.58x
#         science      10.0%    17.8%    1.78x     <-- specified 10%, DELIVERED 17.8%
#         finance       8.0%     4.9%    0.62x
#         ballast       7.0%     4.8%    0.68x
#
#     So the 2026-08-27 approved mixture never actually happened: code, the headline use
#     case, got under half its intended weight and science got nearly double. TOKEN_TARGET
#     below is the intent; MIXTURE is derived as target/ratio, renormalised. s02 reports
#     achieved token shares so the ratios can be re-fit if the source pool shifts.
#
# (2) BALLAST 4.8% -> 15% of tokens. This DOES revise the 2026-08-27 decision, which set
#     ballast to a 7% sufficiency floor. That was made before the cost was measurable.
#     It now is: s09_eval by_domain has ballast dNLL 1.0027 / top-1 0.579 against the FP8
#     teacher, versus code 0.057/0.916 and math 0.024/0.916 -- and the shipped mask retains
#     only 48.7% of general saliency mass versus 71-75% everywhere else. A bucket-balanced
#     RE-MASK of the pass-2 accumulators recovers just +0.028, so this cannot be fixed
#     downstream of calibration: re-masking cannot create information the corpus never
#     collected. Code-adjacent still leads at 57% of tokens.

TOKEN_TARGET = {                  # bucket -> share of ROUTED CALIBRATION TOKENS
    "agentic":    0.23,
    "code":       0.19,
    "math":       0.15,
    "ballast":    0.15,
    "multimodal": 0.12,
    "science":    0.10,
    "finance":    0.06,
}
assert abs(sum(TOKEN_TARGET.values()) - 1.0) < 1e-9

# token% / sample% measured on the pass-2 run (see table above).
_TOKENS_PER_SAMPLE = {
    "agentic": 1.18, "code": 0.44, "math": 1.73, "multimodal": 0.58,
    "science": 1.78, "finance": 0.62, "ballast": 0.68,
}
_raw = {b: TOKEN_TARGET[b] / _TOKENS_PER_SAMPLE[b] for b in TOKEN_TARGET}
_z = sum(_raw.values())
MIXTURE = {b: v / _z for b, v in _raw.items()}       # bucket -> SAMPLE share
assert abs(sum(MIXTURE.values()) - 1.0) < 1e-9

# Difficulty policy: medium-hard, NOT maximal. Hard-only calibration degrades general
# perplexity 6.2-12.1% vs 1.5-4.2% mixed (arXiv 2510.10618), which would damage exactly the
# connective tissue the ballast slice exists to protect.
DIFFICULTY_TARGET = {"medium": 0.60, "hard": 0.30, "easy": 0.10}
# Token-length bands used as the difficulty proxy (cheap, and correlates with reasoning depth).
BAND_EASY = (0, 700)
BAND_MEDIUM = (700, 4_000)
BAND_HARD = (4_000, MAX_TOKENS)


def _first(row, *keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _messages(row, key="messages"):
    """Handle {role,content} and ShareGPT {from,value}, and columns that hold a JSON *string*.

    CoderForge stores its whole trajectory as a JSON string in `messages`; treating that as a
    list silently yields nothing, which is how 30,000 rows produced zero samples.
    """
    msgs = row.get(key)
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return None
    if not isinstance(msgs, list):
        return None
    parts = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content", m.get("value"))
        if isinstance(c, list):
            c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
        if c:
            parts.append(f"{m.get('role', m.get('from', 'user'))}: {c}")
    return "\n\n".join(parts) or None


def _any_messages(row):
    for k in ("messages", "conversations", "conversation", "turns"):
        v = _messages(row, k)
        if v:
            return v
    return None


def _qa(row, q_keys, a_keys):
    q = _first(row, *q_keys)
    a = _first(row, *a_keys)
    if not q:
        return None
    return f"{q}\n\n{a}" if a else q


# Each entry: (hf_id, config, split, weight_within_bucket, text_fn)
SOURCES: dict[str, list[tuple]] = {
    "agentic": [
        ("togethercomputer/CoderForge-Preview", "trajectories", "filtered_reward1", 0.30, lambda r: _any_messages(r) or _first(r, "text", "trajectory")),
        ("nebius/SWE-rebench", None, "test", 0.20, lambda r: _qa(r, ["problem_statement", "text"], ["patch", "solution"])),
        ("SWE-bench/SWE-smith-trajectories", None, "tool", 0.20, lambda r: _any_messages(r) or _first(r, "text")),
        ("SWE-Gym/SWE-Gym", None, "train", 0.10, lambda r: _qa(r, ["problem_statement"], ["patch"])),
        ("arcee-ai/agent-data", None, "train", 0.10, lambda r: _any_messages(r) or _qa(r, ["query", "instruction"], ["answers", "output"])),
        ("open-thoughts/AgentTrove", None, "train", 0.10, lambda r: _any_messages(r) or _first(r, "text")),
    ],
    "code": [
        ("nvidia/OpenCodeReasoning-2", "train", "python", 0.30, lambda r: _qa(r, ["input", "question", "problem"], ["output", "solution", "r1_generation"])),
        ("nvidia/OpenCodeReasoning-2", "train", "cpp", 0.15, lambda r: _qa(r, ["input", "question", "problem"], ["output", "solution", "r1_generation"])),
        ("nvidia/OpenCodeInstruct", None, "train", 0.25, lambda r: _qa(r, ["input", "instruction"], ["output", "response"])),
        ("GPUMODE/KernelBook", None, "train", 0.15, lambda r: _first(r, "python_code", "triton_code", "code", "text")),
        ("SakanaAI/AI-CUDA-Engineer-Archive", None, "level_1", 0.10, lambda r: _first(r, "CUDA_Code", "Kernel_Code", "cuda_code")),
        
    ],
    "math": [
        ("nvidia/OpenMathReasoning", "default", "cot", 0.35, lambda r: _qa(r, ["problem", "question"], ["generated_solution", "solution", "answer"])),
        ("zwhe99/DeepMath-103K", None, "train", 0.25, lambda r: _qa(r, ["question", "problem"], ["r1_solution_1", "final_answer", "solution"])),
        ("open-r1/OpenR1-Math-220k", None, "train", 0.20, lambda r: _qa(r, ["problem", "question"], ["solution", "answer"])),
        ("AI-MO/NuminaMath-1.5", None, "train", 0.10, lambda r: _qa(r, ["problem"], ["solution"])),
        ("internlm/Lean-Workbook", None, "train", 0.10, lambda r: _qa(r, ["natural_language_statement", "problem"], ["formal_statement", "answer"])),
    ],
    "science": [
        ("open-thoughts/OpenThoughts3-1.2M", None, "train", 0.45, lambda r: _any_messages(r) or _qa(r, ["problem", "question"], ["solution", "answer"])),
        ("nvidia/sft_datablend_v1", None, "train", 0.25, lambda r: _any_messages(r)),
        ("TIGER-Lab/MMLU-Pro", None, "test", 0.15, lambda r: _qa(r, ["question"], ["cot_content", "answer"])),
        ("jablonkagroup/ChemBench", "organic_chemistry", "train", 0.15, lambda r: _qa(r, ["question", "input"], ["answer", "target"])),
        
    ],
    "finance": [
        # DocFinQA is dropped, reluctantly: its ~123k-word contexts overflow Arrow's int32
        # string offsets on every load attempt. t2-ragbench inherits its long-context role -
        # its FinQA/ConvFinQA/TAT-DQA configs carry document-level context+table+pre_text.
        # eloukas/edgar-corpus would have been the other long-context fallback but is
        # script-based, which datasets 5.x no longer supports.
        ("G4KMU/t2-ragbench", "FinQA", "train", 0.22,
         lambda r: _qa(r, ["pre_text", "context", "question"], ["original_answer", "program_answer"])),
        ("G4KMU/t2-ragbench", "ConvFinQA", "train", 0.15,
         lambda r: _qa(r, ["pre_text", "context", "question"], ["original_answer", "program_answer"])),
        ("G4KMU/t2-ragbench", "TAT-DQA", "train", 0.13,
         lambda r: _qa(r, ["pre_text", "context", "question"], ["original_answer", "program_answer"])),
        ("kensho/bizbench", "default", "train", 0.20,
         lambda r: _qa(r, ["question", "context"], ["answer", "program"])),
        ("ChanceFocus/flare-finqa", None, "train", 0.15,
         lambda r: _qa(r, ["text", "query"], ["answer"])),
        ("next-tat/tat-llm-instructions", None, "train", 0.15,
         lambda r: _qa(r, ["instruction", "input"], ["output", "response"])),
        ("TheFinAI/Fino1_Reasoning_Path_FinQA", None, "train", 0.10,
         lambda r: _qa(r, ["Open-ended Verifiable Question", "question"],
                       ["Complex_CoT", "Response", "answer"])),
        ("TheFinAI/flare-convfinqa", None, "train", 0.10,
         lambda r: _qa(r, ["text", "query"], ["answer"])),
        ("sujet-ai/Sujet-Financial-RAG-EN-Dataset", None, "train", 0.08,
         lambda r: _qa(r, ["context", "question"], [])),
    ],
    "ballast": [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.50, lambda r: _first(r, "text")),
        ("allenai/tulu-3-sft-mixture", None, "train", 0.30, lambda r: _messages(r)),
        ("HuggingFaceTB/finemath", "finemath-3plus", "train", 0.20, lambda r: _first(r, "text")),
    ],
}

# Multimodal is handled by a separate loader: rows must carry real images, which are pushed
# through the real processor. Text descriptions of images route like text and protect nothing.
MM_SOURCES = [
    # Configs are EXPLICIT here. Nemotron-VLM-Dataset-v2 has 46 configs of which the first
    # alphabetically is `wiki_de` - German Wikipedia text. Letting the loader auto-correct a
    # missing config would silently calibrate the R3-critical bucket on text.
    ("nvidia/Nemotron-VLM-Dataset-v2", "chartqa_cot", "train", 0.10),
    ("nvidia/Nemotron-VLM-Dataset-v2", "docvqa_cot", "train", 0.10),
    ("nvidia/Nemotron-VLM-Dataset-v2", "llava_cot_100k", "train", 0.10),
    ("nvidia/Nemotron-VLM-Dataset-v2", "infographicsvqa_cot", "train", 0.06),
    ("nvidia/Nemotron-VLM-Dataset-v2", "plotqa_cot", "train", 0.06),
    ("nvidia/Nemotron-VLM-Dataset-v2", "fintabnet_cot", "train", 0.05),
    ("nvidia/Nemotron-VLM-Dataset-v2", "visual_web_instruct_cot", "train", 0.05),
    ("HuggingFaceM4/the_cauldron", "chartqa", "train", 0.08),
    ("HuggingFaceM4/the_cauldron", "ai2d", "train", 0.06),
    ("HuggingFaceM4/the_cauldron", "docvqa", "train", 0.06),
    ("HuggingFaceM4/Docmatix", "images", "train", 0.08),
    ("lmms-lab/LLaVA-OneVision-Data", "CLEVR-Math(MathV360K)", "train", 0.05),
    ("allenai/pixmo-docs", "charts", "train", 0.05),
    ("xlangai/aguvis-stage2", "default", "train", 0.05),
    ("ServiceNow/BigDocs-Bench", "GUI-VQA", "train", 0.05),
]

EXCLUDED = {
    "licence_nc": ["EricLu/SCP-116K", "camel-ai/physics", "camel-ai/chemistry",
                   "camel-ai/biology", "osunlp/UGround-V1-Data"],
    "licence_sharealike": ["tattabio/OG"],
    "scope_clinical": ["qiaojin/PubMedQA", "bigbio/pubmed_qa"],
    "gated_401": ["nvidia/Nemotron-CC-Math", "TheFinAI/MultiFinBen", "mlfoundations/MINT-1T"],
}
