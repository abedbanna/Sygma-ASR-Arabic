"""Re-attach reference transcripts to the released predictions, producing the input directory that
reviewer_analyses.py expects. The release contains no reference text; references come from:

  test sets  the Open Universal Arabic ASR Leaderboard manifests (GitHub, pinned commit)
  dev sets   the Hugging Face datasets ahmedsamirtarjama/youtube_najdi and
             HuggingPanda/Saudi_Podcasts_ASR (pinned revisions), by row index: a dev clip named
             <set>_<i:06d> is row i of the listed files read in sorted path order (finetune/dev_prep.py)

    python paper/analysis/attach_references.py --release RELEASE_DIR --out ART_DIR
    python paper/analysis/reviewer_analyses.py --art ART_DIR --out results.json
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LB_REPO = "Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard"
LB_COMMIT = "8fce8595b49057c98bafc2ba41b69c331cb5176e"
TEST = ["sada_test", "commonvoice_test", "masc_test_clean", "masc_test_noisy", "mgb2_test", "casablanca_test"]
BASE = {"sada_test": "base_sada_test", "commonvoice_test": "base_cv_commonvoice_test",
        "masc_test_clean": "base_masc_test_clean", "masc_test_noisy": "base_masc_test_noisy",
        "mgb2_test": "base_mgb2_test", "casablanca_test": "base_casablanca_test"}
# (repo, revision, files read by dev_prep.py): the youtube_najdi tuning set was built from the
# dataset's save_to_disk .arrow files only (its data/ parquet copy holds the same rows in another order)
DEV = {"dev_youtube_najdi": ("ahmedsamirtarjama/youtube_najdi", "d308545faa94f0f11bfa4d3b180ed661373f4a69", "data-*.arrow"),
       "dev_saudi_podcasts": ("HuggingPanda/Saudi_Podcasts_ASR", "0f9b9b26ced10e26b4883948c517f70e14b01ab4", "data/*.parquet")}
TEXT_COLS = ("text", "transcription", "transcript", "sentence")


def stem(p):
    return Path(p).name.rsplit(".wav", 1)[0]


def rows(src, pattern):
    """Same row order as finetune/dev_prep.py: the dataset's files in sorted path order."""
    for p in sorted(Path(src).glob(pattern)):
        if p.suffix == ".parquet":
            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                yield from pf.read_row_group(i, columns=[c for c in pf.schema_arrow.names if c in TEXT_COLS]).to_pylist()
        elif p.suffix == ".arrow":
            with pa.memory_map(str(p)) as f:
                try:
                    reader = pa.ipc.open_stream(f)
                except pa.ArrowInvalid:
                    reader = pa.ipc.open_file(f)
                for batch in reader:
                    yield from batch.select([c for c in batch.schema.names if c in TEXT_COLS]).to_pylist()


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def dump(rows_, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows_:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel, out = Path(a.release), Path(a.out)
    if out.exists():
        shutil.rmtree(out)

    for s in TEST:
        url = f"https://raw.githubusercontent.com/{LB_REPO}/{LB_COMMIT}/datasets/{s}.json"
        raw = urllib.request.urlopen(url).read().decode("utf-8")
        (out / "lb").mkdir(parents=True, exist_ok=True)
        (out / f"lb/{s}.json").write_text(raw, encoding="utf-8")
        ref = {stem(r["audio_filepath"]): r["text"] for r in map(json.loads, raw.splitlines())}
        nb = load(rel / f"nbest/{s}.jsonl")
        for r in nb:
            r["ref"] = ref[stem(r["file"])]
        dump(nb, out / f"nbest/{s}.jsonl")
        for sub, name in [("base", BASE[s]), ("audar", f"audar_{s}"), ("audar", f"audarfix_{s}")]:
            rs = load(rel / f"{sub}/{name}.jsonl")
            for r in rs:
                r["text"] = ref[stem(r["audio_filepath"])]
            dump(rs, out / f"{sub}/{name}.jsonl")
        print(f"[test] {s}: {len(nb)} clips", flush=True)

    from huggingface_hub import snapshot_download
    for s, (repo, rev, pattern) in DEV.items():
        local = snapshot_download(repo, repo_type="dataset", revision=rev)
        texts = {}
        for i, r in enumerate(rows(local, pattern)):
            col = next((c for c in TEXT_COLS if r.get(c)), None)
            if col:
                texts[i] = r[col].strip()
        nb = load(rel / f"nbest/{s}.jsonl")
        for r in nb:
            r["ref"] = texts[int(stem(r["file"]).rsplit("_", 1)[1])]
        dump(nb, out / f"nbest/{s}.jsonl")
        print(f"[dev] {s}: {len(nb)} clips", flush=True)


if __name__ == "__main__":
    main()
