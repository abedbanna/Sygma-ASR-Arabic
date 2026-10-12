# Test audio for Sygma ASR Arabic

These are audio samples for trying [Sygma ASR Arabic](https://huggingface.co/abedbanna/Sygma-ASR-Arabic) and its [demo](https://huggingface.co/spaces/abedbanna/Sygma-ASR-Arabic-demo). The decoder is described in the paper *Sygma ASR Arabic: Improving Dialectal Speech Recognition through Length-Corrected N-Best Decoding*.

The folder has two parts:

1. `benchmark_samples/`: 18 clips from the public test sets used in the paper. Only Common Voice and MASC clips are included, because their licences allow redistribution.
2. Own recordings (`1.wav`, `2.wav`, `3.wav`, `6.wav`): Jordanian Arabic read by one of the authors. Reference transcripts are in `own_recordings.csv`.

## Benchmark samples (illustrations, not evidence)

For the statistical evidence, see the *Evidence* section of the [main README](../README.md). It is based on 46,654 clips, with confidence intervals.

### How the clips were selected

The clips were chosen by fixed rules and not hand-picked. The rules were applied to the saved results of the paper, per test set:

- **Largest Sygma gain:** the 3 clips where Sygma-reward reduces errors most compared with greedy decoding.
- **Largest Sygma loss:** the 1 clip where Sygma-reward increases errors most.
- **Random (seed 0):** 2 random clips with at least 4 reference words.

Only clips with at least 3 reference words were considered.

This sample deliberately over-represents large differences, in both directions, so that they are easy to see. It is **not** representative of average performance. For the overall effect, see the paper: the mean WER over six test sets falls from 24.16% to 23.05%.

### Results on the live demo

WER (%) is measured by the demo's light normalization, so it can differ from the paper's leaderboard scoring. Full outputs are in `benchmark_samples/results.csv`.

| File | Test set | Selection | Seconds | Greedy WER | Sygma WER |
|---|---|---|---|---|---|
| `common_voice/common_voice_ar_26081242.mp3` | Common Voice | largest Sygma gain | 5.8 | 54.5 | 0.0 |
| `common_voice/common_voice_ar_19261809.mp3` | Common Voice | largest Sygma gain | 8.2 | 50.0 | 35.7 |
| `common_voice/common_voice_ar_25245399.mp3` | Common Voice | largest Sygma gain | 3.8 | 66.7 | 0.0 |
| `common_voice/common_voice_ar_19644854.mp3` | Common Voice | largest Sygma loss | 2.9 | 100.0 | 300.0 |
| `common_voice/common_voice_ar_27141077.mp3` | Common Voice | random (seed 0) | 3.5 | 0.0 | 0.0 |
| `common_voice/common_voice_ar_24037639.mp3` | Common Voice | random (seed 0) | 9.3 | 0.0 | 0.0 |
| `masc/lIPSYpbkgbk_374.57.wav` | MASC clean | largest Sygma gain | 7.2 | 1425.0 | 25.0 |
| `masc/F4p1b4eYsj8_70.40100000000002.wav` | MASC clean | largest Sygma gain | 19.5 | 433.3 | 8.3 |
| `masc/HdwSajQa1oU_300.73.wav` | MASC clean | largest Sygma gain | 4.4 | 50.0 | 0.0 |
| `masc/GxPNyy9nieA_0.01.wav` | MASC clean | largest Sygma loss | 11.0 | 3.6 | 3.6 |
| `masc/u64JlA2eB1s_544.681.wav` | MASC clean | random (seed 0) | 3.0 | 0.0 | 0.0 |
| `masc/OM3-aj6M7Yc_662.7560000000002.wav` | MASC clean | random (seed 0) | 3.0 | 0.0 | 0.0 |
| `masc/3tnBqMKR9fk_878.727.wav` | MASC noisy | largest Sygma gain | 18.0 | 73.0 | 5.4 |
| `masc/9jPkY1gO0M8_85.36.wav` | MASC noisy | largest Sygma gain | 3.9 | 100.0 | 23.5 |
| `masc/8ZBOgj12JJ8_209.83.wav` | MASC noisy | largest Sygma gain | 6.1 | 100.0 | 16.7 |
| `masc/kAUEmStJbxk_103.957.wav` | MASC noisy | largest Sygma loss | 2.1 | 0.0 | 333.3 |
| `masc/v_Ma1Vf_wDs_76.52.wav` | MASC noisy | random (seed 0) | 2.3 | 20.0 | 20.0 |
| `masc/N08RYJvOPqU_25.252.wav` | MASC noisy | random (seed 0) | 9.5 | 23.5 | 23.5 |

### Why can WER exceed 100%?

WER is calculated as

> WER = (substitutions + deletions + insertions) ÷ number of words in the reference

The divisor is the length of the reference, but the number of insertions has no upper limit. When a system gets stuck in a repetition loop, every extra word counts as an error.

For example, in `masc/lIPSYpbkgbk_374.57.wav`:

| | Text | Words | WER |
|---|---|---|---|
| Reference | القران ثم قالت اقرؤوا | 4 | |
| Greedy | وَلَقَدْ قَدْ قَدْ قَدْ قَدْ … ("قد" repeated) | 57 | **1425%** (about 57 errors ÷ 4 words) |
| Sygma | القرآن، ثم قالت | 3 | **25%** (1 missing word ÷ 4 words) |

Below 100%, WER is roughly the share of words that are wrong. Above 100%, the output contains more erroneous words than the speech itself, usually because of a loop. One such loop can outweigh dozens of clips with small errors. This is why reducing breakdowns matters for the average: across the six test sets, greedy decoding produced 155 of them and Sygma-reward 39.

Most of Sygma's large gains come from greedy decoding getting stuck in a repetition loop, for example a word repeated dozens of times. Sygma's losses are mostly on very short clips (2–3 s), where the per-word reward adds extra words.

### Sources and licences

| Folder | Source | Licence | Attribution |
|---|---|---|---|
| `benchmark_samples/common_voice/` | Mozilla Common Voice 18.0, Arabic test split (via `fsicoli/common_voice_18_0`) | CC0 1.0 | Ardila et al., *Common Voice: A Massively-Multilingual Speech Corpus*, LREC 2020 |
| `benchmark_samples/masc/` | MASC, Massive Arabic Speech Corpus, test split (via `pain/MASC`) | CC BY 4.0 | Al-Fetyani et al., *MASC: Massive Arabic Speech Corpus*, IEEE SLT 2022 |
| `1.wav`, `2.wav`, `3.wav`, `6.wav` | Own recordings | Apache-2.0 (this repository) | Abedal-kareem Al-banna |

The clips are unmodified copies of the original files: Common Voice MP3s, and MASC WAV segments named `<video_id>_<start>` as in the Open Universal Arabic ASR Leaderboard. Reference transcripts come from the datasets.

Clips from SADA (CC BY-NC-SA), Casablanca (CC BY-NC-ND) and MGB-2 (restricted licence) are **not** included because their licences restrict redistribution.

## Try it

1. Open the [demo](https://huggingface.co/spaces/abedbanna/Sygma-ASR-Arabic-demo).
2. Upload a clip.
3. Paste its reference from `results.csv` into the reference box.
4. Click Transcribe.
