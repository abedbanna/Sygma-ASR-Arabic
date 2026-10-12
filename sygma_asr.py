"""Sygma ASR Arabic: robust N-best decoding for CohereLabs/cohere-transcribe-arabic-07-2026.

The acoustic model and its weights are Cohere's, unchanged (Apache-2.0). What Sygma adds is
decoding:
  1. 8-best beam search (length_penalty = 0) with a word-count bonus (beta = 1.0) that corrects
     beam search's bias toward short hypotheses;
  2. a reference-free detector that discards degenerate hypotheses (runaway loops, symbol runs);
  3. optional (repair=True, NOT used for the reported benchmark numbers): a fallback re-decode
     (beam 4 + no-repeat n-grams + repetition penalty) when all 8 hypotheses are degenerate;
  4. clips over 35 s use the model's own chunked transcription.
beta was tuned only on non-test Saudi audio; no leaderboard test data was used.

    from sygma_asr import SygmaASR
    asr = SygmaASR(device="cuda")
    texts = asr.transcribe(["clip.wav"])            # or audio_arrays=[np.ndarray], sample_rates=[16000]
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
BASE_REVISION = "c3e911b42149bf7a1e53d5cef9878aee87515a23"  # the base-model revision used for all paper results
SR = 16000
BETA = 1.0
BEAMS = 8
MAX_CLIP_S = 35.0
FALLBACKS = [
    {"num_beams": 4, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2},
    {"num_beams": 4, "no_repeat_ngram_size": 2, "repetition_penalty": 1.5},
]
_ARABIC = re.compile(r"[ء-ي]")


def is_degenerate(text):
    """True for decoding breakdowns: symbol-only output, long character runs, or word loops."""
    t = text.strip()
    if not t:
        return False
    if not _ARABIC.search(t) and re.fullmatch(r"[\W_]+", t):
        return True
    if re.search(r"([^\sء-ي])\1{4,}", t) or re.search(r"(.)\1{9,}", t):
        return True
    toks = re.findall(r"[ء-ي]+", t)
    if len(toks) >= 6:
        _, c = Counter(toks).most_common(1)[0]
        if c >= 6 and c / len(toks) >= 0.5:
            return True
    for n in (2, 3, 4):
        for i in range(len(toks) - 4 * n + 1):
            g = toks[i : i + n]
            if all(toks[i + k * n : i + (k + 1) * n] == g for k in range(1, 4)):
                return True
    return False


def _words(text):
    """Deployment-mode word count: punctuation replaced by spaces on the raw text."""
    t = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]", " ", text)
    return len(t.split())


def normalize_arabic_text(text):
    """The Open Universal Arabic ASR Leaderboard's eval.py normalization (verbatim)."""
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]', '', text)
    text = re.sub(r'[ً-ْ]', '', text)
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)
    for e, w in zip('٠١٢٣٤٥٦٧٨٩', '0123456789'):
        text = text.replace(e, w)
    text = re.sub(r'(^|\s)و\s+', r'\1و', text)
    return re.sub(r'\s+', ' ', text).strip()


def select_hypothesis(hyps, beta=BETA, mode="benchmark", use_filter=True):
    """Pick one hypothesis from an N-best list of (text, summed log-prob) pairs.

    mode="benchmark" reproduces the paper exactly: words are counted after the leaderboard
    normalization, and if every hypothesis is degenerate all of them are re-ranked.
    mode="deployment" counts words on punctuation-stripped raw text and returns None when every
    hypothesis is degenerate (the caller then repairs or keeps the top beam hypothesis).
    use_filter=False disables the breakdown filter (the paper's reward-only configuration).

        select_hypothesis(hyps, 1.0, "benchmark", use_filter=False)  # reward only (primary result)
        select_hypothesis(hyps, 1.0, "benchmark", use_filter=True)   # Sygma (reward + filter)
        select_hypothesis(hyps, 0.0, "benchmark", use_filter=False)  # Beam-8 top
    """
    good = [h for h in hyps if not (use_filter and is_degenerate(h[0]))]
    if mode == "benchmark":
        return max(good or hyps, key=lambda h: h[1] + beta * len(normalize_arabic_text(h[0]).split()))[0]
    if mode == "deployment":
        return max(good, key=lambda h: h[1] + beta * _words(h[0]))[0] if good else None
    raise ValueError(f"unknown mode {mode!r}")


def _load_cohere(device):
    """Load the Cohere processor + model the way its repo code expects (transformers 4.x)."""
    import sys

    import transformers
    from huggingface_hub import snapshot_download

    local = snapshot_download(BASE_MODEL, revision=BASE_REVISION)  # weights, tokenizer, mel buffers (Cohere, unchanged)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # cohere_asr/ = Cohere's Apache-2.0 model code, vendored; one change: GenerationMixin added (see NOTICE)
    from cohere_asr.modeling_cohere_asr import CohereAsrForConditionalGeneration
    from cohere_asr.processing_cohere_asr import (CohereAsrFeatureExtractor, CohereAsrProcessor,
                                                  _maybe_load_preprocessor_buffers_from_checkpoint)
    from cohere_asr.tokenization_cohere_asr import CohereAsrTokenizer

    transformers.CohereAsrFeatureExtractor = CohereAsrFeatureExtractor
    transformers.CohereAsrTokenizer = CohereAsrTokenizer
    pp = json.load(open(f"{local}/config.json"))["preprocessor"]
    fe = CohereAsrFeatureExtractor(
        feature_size=pp["features"], sampling_rate=pp["sample_rate"],
        n_window_size=int(pp["window_size"] * pp["sample_rate"]), n_window_stride=int(pp["window_stride"] * pp["sample_rate"]),
        window=pp["window"], normalize=pp["normalize"], n_fft=pp["n_fft"], log=pp["log"],
        frame_splicing=pp["frame_splicing"], dither=pp["dither"], pad_to=pp["pad_to"], padding_value=pp["pad_value"])
    _maybe_load_preprocessor_buffers_from_checkpoint(feature_extractor=fe, model_dir=Path(local))
    processor = CohereAsrProcessor(feature_extractor=fe, tokenizer=CohereAsrTokenizer.from_pretrained(local))
    model = CohereAsrForConditionalGeneration.from_pretrained(local, torch_dtype=torch.float32).to(device).eval()
    return processor, model


class SygmaASR:
    def __init__(self, device=None, beta=BETA, beams=BEAMS, punctuation=True, repair=False, mode="benchmark",
                 use_filter=True):
        """mode="benchmark" (default) is the configuration evaluated in the paper; mode="deployment"
        is the earlier released behaviour and is required for repair=True."""
        if mode not in ("benchmark", "deployment"):
            raise ValueError(f"unknown mode {mode!r}")
        if repair and mode != "deployment":
            raise ValueError("repair=True is only available with mode='deployment'")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor, self.model = _load_cohere(self.device)
        self.beta, self.beams, self.repair, self.mode, self.use_filter = beta, beams, repair, mode, use_filter
        self.prompt = self.model.build_prompt(language="ar", punctuation=punctuation)
        self.amp = self.device.startswith("cuda")

    def _generate(self, audio, n_return, **gen):
        inputs = self.processor(audio=audio, text=[self.prompt] * len(audio), sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        dec = inputs.pop("input_ids")
        inputs.pop("attention_mask", None)
        max_new = int(max(len(a) for a in audio) / SR * 12) + 20
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model.generate(**inputs, decoder_input_ids=dec, decoder_attention_mask=torch.ones_like(dec),
                                      max_new_tokens=max_new, do_sample=False, num_return_sequences=n_return,
                                      output_scores=True, return_dict_in_generate=True,
                                      decoder_start_token_id=int(dec[0, 0]), **gen)
        texts = self.processor.tokenizer.batch_decode(out.sequences[:, dec.shape[1]:], skip_special_tokens=True)
        scores = out.sequences_scores.float().tolist() if getattr(out, "sequences_scores", None) is not None else [0.0] * len(texts)
        return [t.strip() for t in texts], scores

    def _decode_short(self, batch):
        texts, scores = self._generate(batch, self.beams, num_beams=self.beams, length_penalty=0.0)
        results = []
        for i in range(len(batch)):
            hyps = list(zip(texts[i * self.beams:(i + 1) * self.beams], scores[i * self.beams:(i + 1) * self.beams]))
            pick = select_hypothesis(hyps, self.beta, self.mode, self.use_filter)
            if pick is not None:
                results.append(pick)
                continue
            fixed = None
            for fb in (FALLBACKS if self.repair else []):
                t = self._generate([batch[i]], 1, **fb)[0][0]
                if not is_degenerate(t):
                    fixed = t
                    break
            results.append(fixed if fixed is not None else hyps[0][0])
        return results

    def transcribe(self, audio_files=None, audio_arrays=None, sample_rates=None, batch_size=16):
        if audio_files is not None:
            audio_arrays, sample_rates = [], []
            for f in audio_files:
                a, sr = sf.read(f, dtype="float32")
                audio_arrays.append(a.mean(axis=1) if a.ndim > 1 else a)
                sample_rates.append(sr)
        if any(sr != SR for sr in sample_rates):
            import librosa
            audio_arrays = [librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a for a, sr in zip(audio_arrays, sample_rates)]
        audio_arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in audio_arrays]
        out = [None] * len(audio_arrays)
        short = sorted([i for i, a in enumerate(audio_arrays) if len(a) / SR <= MAX_CLIP_S], key=lambda i: len(audio_arrays[i]))
        long_ = [i for i in range(len(audio_arrays)) if i not in set(short)]
        for s in range(0, len(short), batch_size):
            idx = short[s:s + batch_size]
            for i, t in zip(idx, self._decode_short([audio_arrays[i] for i in idx])):
                out[i] = t
        if long_:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                texts = self.model.transcribe(self.processor, language="ar", audio_arrays=[audio_arrays[i] for i in long_],
                                              sample_rates=[SR] * len(long_), batch_size=4)
            for i, t in zip(long_, texts):
                out[i] = t.strip()
        return out
