"""Offline analyses requested in review, computed from saved artifacts (no new decoding).

Inputs (downloaded from gdrive:ASR/...):
  nbest/<set>.jsonl        8-best lists per clip: text, am (summed log-prob, length penalty 0),
                           degenerate flag, qwen (Qwen3-1.7B-Base log-prob, nats)
                           [gdrive:ASR/paper_artifacts/qwen_rescoring]
  base/base_<set>.jsonl    greedy outputs of the base model   [gdrive:ASR/benchmarks/leaderboard_base]
  audar/audar[fix]_<set>   Audar raw / with repair             [gdrive:ASR/paper_artifacts/pod_final_snapshot/bench_outputs]
  lb/<set>.json            leaderboard manifests (GitHub)

    .venv/bin/python paper/analysis/reviewer_analyses.py --art <dir> --out paper/analysis/results.json
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The decoder (sygma_asr.py + cohere_asr/) sits at the repository root in the Hugging Face release
# (analysis/ is one level down) and in hf_release/Sygma-ASR-Arabic/ in the development tree.
_here = Path(__file__).resolve().parent
DECODER = next(c for c in (_here.parent, _here.parents[1] / "hf_release" / "Sygma-ASR-Arabic")
               if (c / "sygma_asr.py").exists())
sys.path.insert(0, str(DECODER))
from sygma_asr import is_degenerate  # noqa: E402  (the released detector)

SETS = ["sada_test", "commonvoice_test", "masc_test_clean", "masc_test_noisy", "mgb2_test", "casablanca_test"]
BASE_FILES = {"sada_test": "base_sada_test", "commonvoice_test": "base_cv_commonvoice_test",
              "masc_test_clean": "base_masc_test_clean", "masc_test_noisy": "base_masc_test_noisy",
              "mgb2_test": "base_mgb2_test", "casablanca_test": "base_casablanca_test"}
BETA, Q_ALPHA, Q_BETA = 1.0, 0.01, 1.5


def normalize_arabic_text(text):
    """Verbatim copy of the leaderboard's eval.py normalization (= finetune/bench.py)."""
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


def sdi(ref, hyp):
    """Word-level (substitutions, deletions, insertions, reference words), scored with jiwer
    exactly as in finetune/bench.py."""
    import jiwer
    n = len(ref.split())
    if not ref.strip():
        return 0, 0, len(hyp.split()), 0
    if not hyp.strip():
        return 0, n, 0, n
    o = jiwer.process_words(ref, hyp)
    return o.substitutions, o.deletions, o.insertions, n


def _words(text):
    """Word count used for the benchmarked results: words after the leaderboard normalization
    (finetune/rescore.py: len(N(t).split()))."""
    return len(normalize_arabic_text(text).split())


def cluster_of(set_name, cid):
    """Recording-level cluster for the bootstrap (clips of one recording are resampled together)."""
    if set_name == "sada_test":  # 6k_v_SBA2_2011_1-seg_... -> episode 6k_v_SBA2_2011
        return re.sub(r"_\d+$", "", cid.split("-seg")[0])
    if set_name.startswith("masc"):  # <video_id>_<start>
        return cid.rsplit("_", 1)[0]
    if set_name == "mgb2_test":  # <program-guid>_spk-..._seg-...
        return cid.split("_spk")[0]
    if set_name == "casablanca_test":  # <recording>_<segment>
        return cid.split("_")[0]
    return cid  # Common Voice: no speaker id survives in the manifest (its `id` is unique per clip) -> clip level


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stem(name):
    return Path(name).name.rsplit(".wav", 1)[0]


def pick(hyps, key, filt):
    """Benchmarked selection (finetune/rescore.py): if every hypothesis is degenerate, all are re-scored."""
    pool = [h for h in hyps if not (filt and h["degenerate"])] or hyps
    return max(pool, key=key)["text"]


def pick_release(hyps):
    """Released sygma_asr.py: word count on punctuation-stripped raw text; top beam if all are degenerate."""
    from sygma_asr import _words as raw_words
    good = [h for h in hyps if not h["degenerate"]]
    return max(good, key=lambda h: h["am"] + BETA * raw_words(h["text"]))["text"] if good else hyps[0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    art = Path(args.art)
    res = {}

    # token lengths for the length-normalization baseline (the model's own tokenizer)
    from huggingface_hub import snapshot_download
    from cohere_asr.tokenization_cohere_asr import CohereAsrTokenizer
    tok = CohereAsrTokenizer.from_pretrained(snapshot_download("CohereLabs/cohere-transcribe-arabic-07-2026"))
    tl_cache = {}

    def ntok(t):
        if t not in tl_cache:
            tl_cache[t] = len(tok(t, add_special_tokens=False)["input_ids"]) + 1  # + EOS
        return tl_cache[t]

    # ---------- dev set: tune the length-normalization exponent (pooled 608 clips) ----------
    dev = load_jsonl(art / "nbest/dev_youtube_najdi.jsonl") + load_jsonl(art / "nbest/dev_saudi_podcasts.jsonl")

    def corpus_wer(pairs):
        e = w = 0
        for ref, hyp in pairs:
            s, d, i, n = sdi(normalize_arabic_text(ref), normalize_arabic_text(hyp))
            e += s + d + i
            w += n
        return 100 * e / w

    lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    dev_ln = {}
    for filt in (False, True):
        for lam in lam_grid:
            dev_ln[(filt, lam)] = corpus_wer([(r["ref"], pick(r["hyps"], lambda h: h["am"] / ntok(h["text"]) ** lam, filt)) for r in dev])
    best_ln = {filt: min(lam_grid, key=lambda l: dev_ln[(filt, l)]) for filt in (False, True)}
    res["dev_lennorm"] = {f"{'filter' if f else 'nofilter'}_lambda{l}": round(v, 2) for (f, l), v in dev_ln.items()}
    res["dev_lennorm_best"] = {("filter" if f else "nofilter"): l for f, l in best_ln.items()}
    dev_sys = {
        "beam_top": lambda r: r["hyps"][0]["text"],
        "beam_filter": lambda r: pick(r["hyps"], lambda h: h["am"], True),
        "beam_bonus": lambda r: pick(r["hyps"], lambda h: h["am"] + BETA * _words(h["text"]), False),
        "sygma": lambda r: pick(r["hyps"], lambda h: h["am"] + BETA * _words(h["text"]), True),
    }
    res["dev"] = {k: round(corpus_wer([(r["ref"], f(r)) for r in dev]), 2) for k, f in dev_sys.items()}
    res["dev_beta_curve"] = {("filter" if filt else "nofilter"): {str(b): round(corpus_wer(
        [(r["ref"], pick(r["hyps"], lambda h, b=b: h["am"] + b * _words(h["text"]), filt)) for r in dev]), 2)
        for b in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)} for filt in (False, True)}
    res["dev_youtube_only_beam_top"] = round(corpus_wer([(r["ref"], r["hyps"][0]["text"]) for r in dev[:600]]), 2)

    # ---------- test sets ----------
    systems = ["greedy", "beam_top", "beam_filter", "beam_bonus", "sygma", "sygma_release", "lennorm", "lennorm_filter",
               "qwen", "nolm_b15", "audar_raw", "audar_fix"]
    per = {}  # set -> system -> list of (S, D, I, N) aligned by clip
    meta = {}
    for s in SETS:
        nb = {stem(r["file"]): r for r in load_jsonl(art / f"nbest/{s}.jsonl")}
        greedy = {stem(r["audio_filepath"]): r for r in load_jsonl(art / f"base/{BASE_FILES[s]}.jsonl")}
        araw = {stem(r["audio_filepath"]): r for r in load_jsonl(art / f"audar/audar_{s}.jsonl")}
        afix = {stem(r["audio_filepath"]): r for r in load_jsonl(art / f"audar/audarfix_{s}.jsonl")}
        ids = [i for i in greedy if i in nb and i in araw and i in afix]
        outs = defaultdict(list)
        for cid in ids:
            hy = nb[cid]["hyps"]
            outs["greedy"].append(greedy[cid]["pred_text"])
            outs["beam_top"].append(hy[0]["text"])
            outs["beam_filter"].append(pick(hy, lambda h: h["am"], True))
            outs["beam_bonus"].append(pick(hy, lambda h: h["am"] + BETA * _words(h["text"]), False))
            outs["sygma"].append(pick(hy, lambda h: h["am"] + BETA * _words(h["text"]), True))
            outs["sygma_release"].append(pick_release(hy))
            outs["lennorm"].append(pick(hy, lambda h: h["am"] / ntok(h["text"]) ** best_ln[False], False))
            outs["lennorm_filter"].append(pick(hy, lambda h: h["am"] / ntok(h["text"]) ** best_ln[True], True))
            outs["qwen"].append(pick(hy, lambda h: h["am"] + Q_ALPHA * h["qwen"] + Q_BETA * _words(h["text"]), True))
            outs["nolm_b15"].append(pick(hy, lambda h: h["am"] + Q_BETA * _words(h["text"]), True))
            outs["audar_raw"].append(araw[cid]["pred_text"])
            outs["audar_fix"].append(afix[cid]["pred_text"])
        refs = [greedy[c]["text"] for c in ids]
        per[s] = {k: [sdi(normalize_arabic_text(r), normalize_arabic_text(h)) for r, h in zip(refs, outs[k])] for k in systems}
        meta[s] = {"ids": ids, "dur": [greedy[c]["duration"] for c in ids], "refs": refs,
                   "outs": outs,
                   "clusters": [greedy[c]["id"] if s == "commonvoice_test" else cluster_of(s, c) for c in ids]}
        print(f"[{s}] {len(ids)} clips", flush=True)

    def wer_of(rows, idx=None):
        it = rows if idx is None else (rows[i] for i in idx)
        e = w = 0
        for a, b, c, n in it:
            e += a + b + c
            w += n
        return 100 * e / w if w else float("nan")

    table = {}
    for s in SETS:
        table[s] = {}
        for k in systems:
            S_ = sum(x[0] for x in per[s][k]); D = sum(x[1] for x in per[s][k]); I = sum(x[2] for x in per[s][k]); N = sum(x[3] for x in per[s][k])
            table[s][k] = {"wer": round(wer_of(per[s][k]), 2), "S": S_, "D": D, "I": I, "N": N,
                           "degenerate": sum(is_degenerate(o) for o in meta[s]["outs"][k])}
        table[s]["clips"] = len(meta[s]["ids"])
    table["average"] = {k: round(sum(wer_of(per[s][k]) for s in SETS) / len(SETS), 2) for k in systems}
    res["test"] = table

    # ---------- paired cluster bootstrap (resample recordings; macro-average recomputed per replicate) ----------
    import numpy as np
    nrng = np.random.default_rng(0)
    cl_idx, cl_err, cl_n = {}, {}, {}
    for s in SETS:
        names = sorted(set(meta[s]["clusters"]))
        pos = {c: j for j, c in enumerate(names)}
        cl_idx[s] = np.array([pos[c] for c in meta[s]["clusters"]])
        cl_err[s], cl_n[s] = {}, {}
        for k in systems:
            e = np.array([a + b_ + c for a, b_, c, _ in per[s][k]], dtype=float)
            n = np.array([x[3] for x in per[s][k]], dtype=float)
            cl_err[s][k] = np.bincount(cl_idx[s], weights=e, minlength=len(names))
            cl_n[s][k] = np.bincount(cl_idx[s], weights=n, minlength=len(names))
    res["n_clusters"] = {s: int(cl_idx[s].max() + 1) for s in SETS}
    # per-set, per-system WER for every replicate; replicates drawn in chunks to bound memory.
    # Each replicate draws n_clusters recordings with replacement (multinomial counts), so a
    # recording drawn twice contributes its errors and words twice.
    boot = {s: {k: [] for k in systems} for s in SETS}
    chunk = 1000
    for start in range(0, args.boot, chunk):
        m = min(chunk, args.boot - start)
        for s in SETS:
            n_c = res["n_clusters"][s]
            cnt = nrng.multinomial(n_c, np.full(n_c, 1 / n_c), size=m).astype(np.float32)
            for k in systems:
                boot[s][k].append(100 * (cnt @ cl_err[s][k]) / (cnt @ cl_n[s][k]))
    boot = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in boot.items()}

    def boot_wer(s, k):
        return boot[s][k]

    def ci(v):
        lo, hi = np.percentile(np.asarray(v), [2.5, 97.5])  # numpy default (linear interpolation)
        return [round(float(lo), 2), round(float(hi), 2)]

    pairs = [("sygma", "greedy"), ("beam_bonus", "greedy"), ("beam_filter", "greedy"), ("beam_top", "greedy"),
             ("sygma", "beam_bonus"), ("sygma", "beam_filter"), ("sygma", "lennorm_filter"), ("lennorm_filter", "greedy"),
             ("qwen", "sygma"), ("qwen", "nolm_b15"), ("sygma", "audar_fix"), ("beam_bonus", "audar_fix"),
             ("beam_bonus", "lennorm")]
    res["bootstrap"] = {}
    for a, bb in pairs:
        d = {s: boot_wer(s, a) - boot_wer(s, bb) for s in SETS}
        avg = sum(d[s] for s in SETS) / len(SETS)
        point = {s: round(wer_of(per[s][a]) - wer_of(per[s][bb]), 2) for s in SETS}
        point["average"] = round(sum(wer_of(per[s][a]) - wer_of(per[s][bb]) for s in SETS) / len(SETS), 2)
        res["bootstrap"][f"{a}-{bb}"] = {**{s: {"diff": point[s], "ci95": ci(d[s])} for s in SETS},
                                         "average": {"diff": point["average"], "ci95": ci(avg)}}

    # ---------- same non-degenerate intersection ----------
    inter = {}
    for s in SETS:
        o = meta[s]["outs"]
        keep = [i for i in range(len(meta[s]["ids"]))
                if not any(is_degenerate(o[k][i]) for k in ("greedy", "sygma", "audar_raw"))]
        inter[s] = {"clips": len(keep), **{k: round(wer_of(per[s][k], keep), 2) for k in ("greedy", "sygma", "audar_raw", "audar_fix")}}
    res["nondegenerate_intersection"] = inter

    # ---------- SADA by duration (all clips) ----------
    buckets = [(0, 5), (5, 10), (10, 20), (20, 31)]
    dd = {}
    for lo, hi in buckets:
        idx = [i for i, d in enumerate(meta["sada_test"]["dur"]) if lo <= d < hi]
        dd[f"{lo}-{hi}s"] = {"clips": len(idx), **{k: round(wer_of(per["sada_test"][k], idx), 2)
                                                  for k in ("greedy", "beam_top", "beam_bonus", "sygma", "audar_fix")}}
    res["sada_by_duration"] = dd

    # ---------- detector audit ----------
    aud = {}
    for s in SETS:
        refs = meta[s]["refs"]
        o = meta[s]["outs"]
        flagged_refs = [r for r in refs if is_degenerate(r)]
        fl = [i for i, h in enumerate(o["greedy"]) if is_degenerate(h)]
        low = [i for i in fl if wer_of([per[s]["greedy"][i]]) <= 50]
        aud[s] = {"references_flagged": len(flagged_refs), "references": len(refs),
                  "greedy_flagged": len(fl), "greedy_flagged_with_wer_le_50": len(low),
                  "examples_flagged_refs": flagged_refs[:5]}
        # clips where the filter changed Sygma's choice vs. bonus-only, and the effect
        ch = [i for i in range(len(refs)) if o["sygma"][i] != o["beam_bonus"][i]]
        aud[s]["filter_changed_choice"] = len(ch)
        aud[s]["filter_changed_wer_bonus_only"] = round(wer_of(per[s]["beam_bonus"], ch), 2) if ch else None
        aud[s]["filter_changed_wer_sygma"] = round(wer_of(per[s]["sygma"], ch), 2) if ch else None
    res["detector_audit"] = aud

    # ---------- benchmark reconstruction audit ----------
    rec = {}
    for s in ("sada_test", "casablanca_test"):
        lb = [json.loads(l) for l in open(art / f"lb/{s}.json", encoding="utf-8")]
        texts = Counter(normalize_arabic_text(r["text"]) for r in lb)
        lid = lambda r: r.get("id") or stem(r["audio_filepath"])
        have = set(meta[s]["ids"])
        missing = [r for r in lb if lid(r) not in have]
        amb = [r for r in lb if texts[normalize_arabic_text(r["text"])] > 1]
        amb_dur = [r for r in amb if sum(1 for q in amb if normalize_arabic_text(q["text"]) == normalize_arabic_text(r["text"])
                                         and abs(q["duration"] - r["duration"]) <= 0.1) > 1]
        rec[s] = {"manifest_rows": len(lb), "matched": len(lb) - len(missing), "missing": len(missing),
                  "rows_with_shared_text": len(amb), "rows_with_shared_text_and_duration_within_0.1s": len(amb_dur),
                  "missing_mean_duration": round(sum(r["duration"] for r in missing) / len(missing), 2) if missing else None,
                  "all_mean_duration": round(sum(r["duration"] for r in lb) / len(lb), 2),
                  "missing_mean_words": round(sum(len(r["text"].split()) for r in missing) / len(missing), 2) if missing else None,
                  "all_mean_words": round(sum(len(r["text"].split()) for r in lb) / len(lb), 2)}
    res["reconstruction"] = rec

    # ---------- sensitivity: drop leaderboard rows whose text+duration match is ambiguous ----------
    sens = {}
    amb_ids = {}
    for s in ("sada_test", "casablanca_test"):
        lb = [json.loads(l) for l in open(art / f"lb/{s}.json", encoding="utf-8")]
        lid = lambda r: r.get("id") or stem(r["audio_filepath"])
        by_text = defaultdict(list)
        for r in lb:
            by_text[normalize_arabic_text(r["text"])].append(r)
        amb_ids[s] = {lid(r) for grp in by_text.values() if len(grp) > 1 for r in grp
                      if sum(1 for q in grp if abs(q["duration"] - r["duration"]) <= 0.1) > 1}
    for s in SETS:
        keep = [i for i, c in enumerate(meta[s]["ids"]) if c not in amb_ids.get(s, set())]
        sens[s] = {"clips": len(keep), **{k: round(wer_of(per[s][k], keep), 2)
                                          for k in ("greedy", "beam_bonus", "sygma", "lennorm_filter", "audar_fix")}}
    sens["average"] = {k: round(sum(sens[s][k] for s in SETS) / len(SETS), 2)
                       for k in ("greedy", "beam_bonus", "sygma", "lennorm_filter", "audar_fix")}
    sens["removed"] = {s: len(amb_ids[s]) for s in amb_ids}
    keepmask = {s: [c not in amb_ids.get(s, set()) for c in meta[s]["ids"]] for s in SETS}
    sb = {s: {} for s in SETS}
    for s in SETS:
        idx = [i for i, k in enumerate(keepmask[s]) if k]
        names = sorted({meta[s]["clusters"][i] for i in idx}); pos = {c: j for j, c in enumerate(names)}
        ci_ = np.array([pos[meta[s]["clusters"][i]] for i in idx])
        for k in ("greedy", "beam_bonus", "sygma", "audar_fix"):
            e = np.array([sum(per[s][k][i][:3]) for i in idx], float); n = np.array([per[s][k][i][3] for i in idx], float)
            sb[s][k] = (np.bincount(ci_, weights=e, minlength=len(names)), np.bincount(ci_, weights=n, minlength=len(names)))
        sb[s]["_n"] = len(names)
    srng = np.random.default_rng(1)
    acc = {pr: {s: [] for s in SETS} for pr in [("beam_bonus", "greedy"), ("sygma", "audar_fix"), ("beam_bonus", "audar_fix")]}
    for start in range(0, args.boot, 1000):
        m = min(1000, args.boot - start)
        for s in SETS:
            n_c = sb[s]["_n"]; cnt = srng.multinomial(n_c, np.full(n_c, 1 / n_c), size=m).astype(np.float32)
            w = {k: 100 * (cnt @ sb[s][k][0]) / (cnt @ sb[s][k][1]) for k in ("greedy", "beam_bonus", "sygma", "audar_fix")}
            for a_, b_ in acc:
                acc[(a_, b_)][s].append(w[a_] - w[b_])
    sens["bootstrap"] = {}
    for (a_, b_), d in acc.items():
        d = {s: np.concatenate(v) for s, v in d.items()}
        avg = sum(d[s] for s in SETS) / len(SETS)
        kept = {s: [i for i, k in enumerate(keepmask[s]) if k] for s in SETS}
        pt = {s: wer_of(per[s][a_], kept[s]) - wer_of(per[s][b_], kept[s]) for s in SETS}
        sens["bootstrap"][f"{a_}-{b_}"] = {**{s: {"diff": round(pt[s], 2), "ci95": ci(d[s])} for s in SETS},
                                           "average": {"diff": round(sum(pt.values()) / len(SETS), 2), "ci95": ci(avg)}}
    res["sensitivity_no_ambiguous"] = sens

    # ---------- WER of the example outputs (Table: real outputs) ----------
    ex_ids = ["6k_v_SBA_187_1-seg_287_573-290_051", "6k_SBA_116_0-seg_400_170-403_210",
              "6k_v_SBA_32_3-seg_284_515-290_800", "6k_v_SBA_41_1-seg_133_300-152_850"]
    pos = {c: i for i, c in enumerate(meta["sada_test"]["ids"])}
    res["examples_wer"] = {c: {k: round(wer_of([per["sada_test"][k][pos[c]]]), 1)
                               for k in ("greedy", "beam_bonus", "sygma", "audar_raw", "audar_fix")} for c in ex_ids}
    res["examples_reward_only_same_as_sygma"] = {c: meta["sada_test"]["outs"]["beam_bonus"][pos[c]] == meta["sada_test"]["outs"]["sygma"][pos[c]] for c in ex_ids}

    # ---------- Qwen truncation: hypotheses longer than 400 Qwen tokens ----------
    try:
        from transformers import AutoTokenizer
        qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
        texts = {h["text"] for s in SETS + ["dev_youtube_najdi", "dev_saudi_podcasts"]
                 for r in load_jsonl(art / f"nbest/{s}.jsonl") for h in r["hyps"]}
        long_ = [t for t in texts if len(qtok(t, add_special_tokens=False)["input_ids"]) > 400]
        res["qwen_truncation"] = {"unique_hypotheses": len(texts), "over_400_tokens": len(long_),
                                  "over_400_degenerate": sum(is_degenerate(t) for t in long_)}
    except Exception as e:  # tokenizer download unavailable
        res["qwen_truncation"] = {"error": str(e)}

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("dev", "dev_lennorm_best")}, indent=1))
    print(json.dumps(res["test"]["average"], indent=1))


if __name__ == "__main__":
    main()
