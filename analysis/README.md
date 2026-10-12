# Reproducing the paper's tables

The analysis environment is:

- Python 3.12.8
- jiwer 4.0.0
- NumPy 1.26.4
- PyTorch 2.2.2
- transformers 4.57.6
- huggingface_hub 0.36.2
- pyarrow 15.0.2

**Prerequisite:** the analysis loads the pretrained base model's tokenizer (`CohereLabs/cohere-transcribe-arabic-07-2026`), which is a gated model. First accept its terms on Hugging Face, then authenticate with `huggingface-cli login` or the `HF_TOKEN` environment variable. Without access the tokenizer download fails and the script stops. All other inputs, including the leaderboard manifests and the development datasets, are public.

On GitHub, first download the released artifacts:

```bash
huggingface-cli download abedbanna/Sygma-ASR-Arabic --revision paper-v1 --include "paper_artifacts/*" --local-dir .
```

```bash
pip install "jiwer==4.0.0" "numpy==1.26.4" "torch==2.2.2" "transformers==4.57.6" "huggingface_hub==0.36.2" \
            sentencepiece soundfile librosa "pyarrow==15.0.2"

# 1. re-attach reference transcripts from their public sources (leaderboard manifests, dev datasets)
python analysis/attach_references.py --release paper_artifacts --out art

# 2. all tables, bootstrap intervals (10,000 replicates, seed 0), sensitivity analysis, detector checks
python analysis/reviewer_analyses.py --art art --out my_results.json
```

`my_results.json` should equal `analysis/results.json`. We verified this in a newly created environment with the versions above. Run both commands from the repository root: the analysis imports `sygma_asr.py` and `cohere_asr/` from there.

## Configurations

| Paper name | Selection rule |
|---|---|
| Beam-8 top | `select_hypothesis(hyps, beta=0.0, mode="benchmark", use_filter=False)` |
| **Sygma-reward** (primary) | `select_hypothesis(hyps, beta=1.0, mode="benchmark", use_filter=False)` |
| Sygma-filtered | `select_hypothesis(hyps, beta=1.0, mode="benchmark", use_filter=True)` |
| LN re-ranking (± filter) | `pick(hyps, lambda h: h["am"] / ntok(h["text"]) ** 2.5, filt)` in `reviewer_analyses.py` (Cohere tokens + EOS) |
| Neural LM | `pick(hyps, lambda h: h["am"] + 0.01 * h["qwen"] + 1.5 * words(h), True)` |

Words are counted after the leaderboard normalization. Search spaces are given in the paper's appendix. `cpu_timing.py` reproduces the CPU timing, which needs audio clips.
