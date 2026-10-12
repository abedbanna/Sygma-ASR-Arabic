# Sygma ASR Arabic

**Sygma ASR Arabic** is a decoding framework for dialectal Arabic speech recognition, built on the fixed pretrained model [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026). The model weights are not changed. Instead, Sygma re-ranks the model's 8 best beam-search hypotheses with a per-word reward. This corrects beam search's preference for short outputs that drop words. Optionally, Sygma also filters out degenerate hypotheses, such as words repeated in a loop.

Paper: *Sygma ASR Arabic: Improving Dialectal Speech Recognition through Length-Corrected N-Best Decoding*, by Abedal-kareem Al-banna (Sygma Global) and Basel A. Al-banna (University of Petra). The arXiv link will be added after announcement.

| Resource | Link |
|---|---|
| Model card, code and paper artifacts | https://huggingface.co/abedbanna/Sygma-ASR-Arabic (tag `paper-v1`) |
| Interactive demo | https://huggingface.co/spaces/abedbanna/Sygma-ASR-Arabic-demo |
| Test audio | [`test_audio/`](test_audio/) |

## Evidence: does Sygma make a difference?

The evidence comes from the full evaluation in the paper: **46,654 clips** from six public test sets of the Open Universal Arabic ASR Leaderboard. The pretrained base model and the audio are identical for every decoding configuration, so any difference is caused by decoding alone. All numbers are reproducible from the released predictions (see *Reproducing the paper's tables* below).

### 1. Word error rate goes down on every test set

| Test set | Greedy (base model as published) | **Sygma-reward** | Difference (95% CI) |
|---|---|---|---|
| SADA (Saudi TV) | 36.57% | **33.68%** | −2.89 [−3.96, −1.98] |
| MGB-2 (broadcast news) | 15.19% | **13.56%** | −1.64 [−2.31, −1.04] |
| Casablanca (multi-dialect) | 45.70% | **44.75%** | −0.95 [−1.45, −0.52] |
| MASC noisy | 25.78% | **25.13%** | −0.66 [−0.94, −0.38] |
| MASC clean | 16.33% | **15.95%** | −0.37 [−0.63, −0.15] |
| Common Voice 18 | 5.41% | **5.26%** | −0.15 [−0.23, −0.07] |
| **Mean of the six sets** | **24.16%** | **23.05%** | **−1.11 [−1.34, −0.90]** |

The intervals are paired cluster-bootstrap intervals with 10,000 replicates, resampling whole recordings. Every interval excludes zero, so the improvement is statistically significant on every test set. The gain is largest on the hardest, most dialectal and noisiest audio, and smallest on clean read speech.

### 2. Fewer dropped words

Unnormalized beam search prefers short hypotheses that drop words. Sygma's per-word reward counteracts this. Summed over all six test sets (404,449 reference words):

| Decoding | Deletions | Insertions | Substitutions |
|---|---|---|---|
| Greedy | 20,885 | 9,796 | 72,628 |
| Plain 8-best beam search | 28,333 | 6,901 | 68,410 |
| **Sygma-reward** | **16,290** | 10,358 | 71,791 |

Sygma-reward deletes **22% fewer words than greedy decoding** and **43% fewer than plain beam search**, at the cost of some extra insertions.

### 3. Far fewer decoding breakdowns

A *breakdown* is an output stuck in a loop or consisting only of symbols, for example a word repeated 50 times. A single breakdown can add hundreds of errors to a short clip.

| Decoding | Breakdowns over six test sets |
|---|---|
| Greedy | 155 |
| Sygma-reward | 39 (−75%) |
| Sygma-filtered (+ breakdown filter) | 18 (−88%) |

### What the evidence does *not* show

- **Sygma is not a better acoustic model.** On clean, clearly spoken audio, Sygma and greedy decoding usually give the same transcript. The six random clips in `test_audio/` all tie.
- **The average gain is modest:** about 1.1 WER points.
- **Very short clips (under 1 s) can get worse,** because the per-word reward can add a word.
- **Comparison with other models:** in the same pipeline, the leaderboard's top system, Audar-ASR-V1-Turbo, is clearly better on Saudi speech.
- **Not official leaderboard results:** these come from our rebuilt evaluation.

### Illustrations

[`test_audio/`](test_audio/) contains real clips that **illustrate** these effects. They do not prove them, because they were selected to show large differences in both directions. Examples include a MASC clip where greedy decoding loops (1425% WER; see [why WER can exceed 100%](test_audio/README.md#why-can-wer-exceed-100)) and Sygma-reward gives 25%, and a 2-second clip where Sygma-reward adds words (333% vs 0%).

## Usage

```python
# pip install -r requirements.txt   (transformers==4.57.6; the base model is gated: accept its terms and log in to Hugging Face)
from sygma_asr import SygmaASR

asr = SygmaASR(device="cuda", use_filter=False)   # Sygma-reward (primary configuration)
print(asr.transcribe(["test_audio/2.wav"]))
```

`mode="benchmark"` (the default) is the exact configuration evaluated in the paper. `mode="deployment"` is the earlier release behaviour. See `sygma_asr.select_hypothesis`.

## Reproducing the paper's tables

The per-clip predictions and 8-best lists (about 100 MB) are released on Hugging Face and are not stored here.

```bash
huggingface-cli download abedbanna/Sygma-ASR-Arabic --revision paper-v1 --include "paper_artifacts/*" --local-dir .
python analysis/attach_references.py --release paper_artifacts --out art
python analysis/reviewer_analyses.py --art art --out my_results.json    # should equal analysis/results.json
```

Details are in [`analysis/README.md`](analysis/README.md).

## Licence

Code and own recordings are released under Apache-2.0. The vendored model code in `cohere_asr/` is Cohere's Apache-2.0 code (see `NOTICE`). The test-set clips in `test_audio/benchmark_samples/` keep their original licences: Common Voice is CC0 and MASC is CC BY 4.0. See [`test_audio/README.md`](test_audio/README.md).
