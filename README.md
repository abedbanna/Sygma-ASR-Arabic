# Sygma ASR Arabic

**Sygma ASR Arabic** is a decoding framework for dialectal Arabic speech recognition, built on the fixed pretrained model [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026). The model weights are not changed. Instead, Sygma re-ranks the model's 8 best beam-search hypotheses with a per-word reward. This corrects beam search's preference for short outputs that drop words. Optionally, Sygma also filters out degenerate hypotheses, such as words repeated in a loop.

Paper: *Sygma ASR Arabic: Improving Dialectal Speech Recognition through Length-Corrected N-Best Decoding*, by Abedal-kareem Al-banna (Sygma Global) and Basel A. Al-banna (University of Petra). The arXiv link will be added after announcement.

| Resource | Link |
|---|---|
| Model card, code and paper artifacts | https://huggingface.co/abedbanna/Sygma-ASR-Arabic (tag `paper-v1`) |
| Interactive demo | https://huggingface.co/spaces/abedbanna/Sygma-ASR-Arabic-demo |
| Test audio | [`test_audio/`](test_audio/) |

## Main result

These results cover six public test sets from the Open Universal Arabic ASR Leaderboard (SADA, Common Voice 18, MASC clean and noisy, MGB-2, Casablanca). The audio was rebuilt from public mirrors, the scoring uses the leaderboard's own normalization, and "mean" is the unweighted mean of the six corpus-level WERs.

| Decoding | Mean WER |
|---|---|
| Greedy, the pretrained base model as published | 24.16% |
| **Sygma-reward** (8-best + per-word reward, β = 1.0) | **23.05%** (95% CI of the difference: −1.34 to −0.90) |
| Sygma-filtered (+ breakdown filter) | 23.04% |

The gain is modest on average and comes mostly from hard audio, such as TV broadcasts, noise, music and several speakers. On clean, clearly read speech, Sygma and greedy decoding usually give the same transcript. See the paper for ablations, uncertainty estimates and limitations.

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
