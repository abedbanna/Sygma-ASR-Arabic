"""CPU cost of greedy vs. Sygma decoding (fp32, batch 1, one clip at a time) on 30 random Arabic clips.

Timing boundary: wall-clock (time.perf_counter) around one call of the decoder on an in-memory float32
array, i.e. feature extraction + encoder + search + detokenization; model loading and audio reading
excluded. Greedy = model.transcribe(...) (the model's own routine); Sygma = SygmaASR.transcribe(...).
With --mode greedy|sygma only that decoder runs, so the process's peak RSS is attributable to it.

    .venv/bin/python paper/analysis/cpu_timing.py --clips kaggle_ser/output/test --out paper/analysis/cpu_timing.json
    .venv/bin/python paper/analysis/cpu_timing.py --clips kaggle_ser/output/test --mode greedy --out .../cpu_greedy.json
"""
import argparse, json, random, resource, sys, time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_release" / "Sygma-ASR-Arabic"))
from sygma_asr import SygmaASR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["both", "greedy", "sygma"], default="both")
a = ap.parse_args()
files = sorted(Path(a.clips).glob("*.wav"))
random.Random(0).shuffle(files)
files = files[: a.n]
asr = SygmaASR(device="cpu")
rows = []
for f in files:
    x, sr = sf.read(f, dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    dur = len(x) / sr
    tg = ts = float("nan")
    if a.mode in ("both", "greedy"):
        t0 = time.perf_counter()
        with torch.inference_mode():
            asr.model.transcribe(asr.processor, language="ar", audio_arrays=[x], sample_rates=[sr])
        tg = time.perf_counter() - t0
    if a.mode in ("both", "sygma"):
        t0 = time.perf_counter()
        asr.transcribe(audio_arrays=[x], sample_rates=[sr])
        ts = time.perf_counter() - t0
    rows.append({"file": f.name, "dur": dur, "greedy_s": tg, "sygma_s": ts})
    print(f"{f.name} {dur:.1f}s greedy {tg:.1f}s sygma {ts:.1f}s", flush=True)
tot = sum(r["dur"] for r in rows)
res = {"clips": len(rows), "audio_s": round(tot, 1), "threads": torch.get_num_threads(),
       "rtf_greedy": round(sum(r["greedy_s"] for r in rows) / tot, 3),
       "rtf_sygma": round(sum(r["sygma_s"] for r in rows) / tot, 3),
       "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),  # bytes on macOS
       "rows": rows}
res["mode"] = a.mode
res["torch"] = torch.__version__
Path(a.out).write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "rows"})
