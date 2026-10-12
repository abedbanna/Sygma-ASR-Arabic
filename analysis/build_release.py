"""Build the public artifact release from the private analysis inputs: identical files with every
reference transcript removed (test and development references are re-attached from their public
sources by attach_references.py).

    python paper/analysis/build_release.py --art ART_DIR --out RELEASE_DIR
"""

import argparse
import json
import shutil
from pathlib import Path

TEST = ["sada_test", "commonvoice_test", "masc_test_clean", "masc_test_noisy", "mgb2_test", "casablanca_test"]
DEV = ["dev_youtube_najdi", "dev_saudi_podcasts"]
BASE = {"sada_test": "base_sada_test", "commonvoice_test": "base_cv_commonvoice_test",
        "masc_test_clean": "base_masc_test_clean", "masc_test_noisy": "base_masc_test_noisy",
        "mgb2_test": "base_mgb2_test", "casablanca_test": "base_casablanca_test"}


def rewrite(src, dst, drop):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            for k in drop:
                r.pop(k, None)
            g.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    art, out = Path(a.art), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for s in TEST + DEV:
        rewrite(art / f"nbest/{s}.jsonl", out / f"nbest/{s}.jsonl", ["ref"])
    for s in TEST:
        rewrite(art / f"base/{BASE[s]}.jsonl", out / f"base/{BASE[s]}.jsonl", ["text"])
        for tag in ("audar", "audarfix"):
            rewrite(art / f"audar/{tag}_{s}.jsonl", out / f"audar/{tag}_{s}.jsonl", ["text"])
    leaks = [p for p in out.rglob("*.jsonl") for l in open(p, encoding="utf-8")
             if '"ref"' in l or '"text": ' in l.split('"hyps"')[0]]
    assert not leaks, f"reference text left in {sorted({str(p) for p in leaks})[:3]}"
    print(f"wrote {sum(1 for _ in out.rglob('*.jsonl'))} files to {out}")


if __name__ == "__main__":
    main()
