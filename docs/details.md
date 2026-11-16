# Pnyx in detail

This file keeps the detail from the earlier README. The [README](../README.md) gives the short version.

## The four layers

- **SEE**: Badges on posts show which claims a post makes and whether it gives reasons. The Habermas model (DeBERTa-v3-small) and an 85-signal heuristic tier for AI-generated text make these badges.
- **PAUSE**: When you click Reply, an overlay shows the extracted claims as checkboxes. You select the claims that you respond to before you write.
- **SHOW**: Your reply carries a visible tag: "Responded to: [claim]". Other readers can see what you heard.
- **EXPLORE**: Branching deliberation on a claim. It has three beats (steel-man, perspective-take, consequence-trace). It uses DeepSeek V3 through OpenRouter.

## Theoretical framework

The technical choices come from three thinkers:

- **Habermas**: The scoring model maps two validity claims from *Theory of Communicative Action*: *Wahrheit* (truth, or claim risk) and *Richtigkeit* (rightness, or argument quality). The goal is discourse legibility, not judgment.
- **Arendt**: The Erscheinung model detects genuine human presence (*Erscheinungsraum*). This is the third validity claim: *Wahrhaftigkeit* (sincerity). Is there a "who" behind the text?
- **Mouffe**: Explore mode treats disagreement as a democratic resource (agonistic pluralism). Escalation is a valid deliberation move, not a failure state.

## Architecture

All scoring runs on-device through ONNX Runtime Web (WASM). Inference makes no API calls.

| Component | Tech | Size |
|-----------|------|------|
| Habermas model | DeBERTa-v3-small, dual-head (claim risk + argument quality) | 270.3 MiB ONNX, FP16 |
| Erscheinung model | DeBERTa-v3-small, pruned vocab (70K tokens) + 85 heuristic signals | 125.7 MiB ONNX, FP16 |
| Inference | ONNX Runtime Web, WASM, single-thread SIMD | ~500-900ms per post |
| Extension | Chrome MV3, offscreen document, MutationObserver feed scanning | |
| Explore | Client-side OpenRouter API (user's own key, localStorage) | |

### How inference works

The demo sends posts to `inference-worker.js`. It loads Habermas with a 256-token limit and uses heuristics for Erscheinung. Its `loadErscheinung()` function does not load an ONNX model.

The extension loads both models through `extension/offscreen.js`. Habermas scores claim risk and argument quality. Erscheinung checks the heuristic score first and skips its 128-token ML inference when that score is at least 4.0. The two scoring functions run in parallel.

The scores go to `claim-extractor.js`. The page renders badges, the Pause Layer and Reply Tags from the extracted claims.

### Model file sizes

Hugging Face file metadata lists these uncompressed ONNX sizes:

| Model | Revision | Bytes | MiB (bytes / 1,048,576) |
|---|---|---:|---:|
| Habermas v3 | [b30f713](https://huggingface.co/onblueroses/pnyx-habermas/tree/b30f713d2d371afee7221edeadf485b5d0f258a8) | 283,475,079 | 270.3 |
| Erscheinung v0.7 | [1cea3a5](https://huggingface.co/onblueroses/pnyx-erscheinung/tree/1cea3a5df75f02872072c4b56b67063c2f692dcc) | 131,778,832 | 125.7 |
| Total | Both ONNX files | 415,253,911 | 396.0 |

The retained model files match the sizes and SHA-256 hashes in that metadata. With both files installed, reproduce the byte count with:

```bash
wc -c scaffold/frontend/model/model.onnx scaffold/frontend/model-slop/model.onnx
```

These counts exclude tokenizers, configs, token remapping and JavaScript or WASM libraries. They do not measure transferred bytes or browser memory use. Transfer size depends on the loaded components, HTTP compression and cache state. Browser memory use is unmeasured.

## Training pipeline

The `scripts/` directory contains the data generation and training pipeline:

1. `01_source_texts.py`: collects source texts from disinformation datasets.
2. `02_label_texts.py`: labels the texts automatically with a discourse quality rubric.
3. `03_balance_and_report.py`: balances the dataset (2,500 samples per cell).
4. `04_generate_boundary_data.py`: makes targeted examples for failure modes.
5. `train_v3_modal.py`: trains the model on Modal (T4 GPU, focal loss, ~27 min/epoch).

The final dataset has 10K balanced samples and 453 boundary examples. Habermas v3 has a reported F1 of 0.974 (0.977 claim risk, 0.972 argument quality).

### What the F1 number measures

Read it as a reported internal macro-F1 on a random row split of model-labeled data. It is not a measure of discourse-detection accuracy.

The labels come from `google/gemini-2.0-flash-001` (`scripts/02_label_texts.py`). The boundary cases come from `scripts/04_generate_boundary_data.py`. Thus the score measures how closely the distilled DeBERTa model copies the labeling model. This repository has no human-annotated evaluation set.

There are two caveats on the split:

- `scripts/train_v3_modal.py` shuffles rows and takes a percentage validation slice. It does not split on unique text before it oversamples the boundary examples. Thus it is not verified that duplicate texts stay on one side of the train/validation boundary.
- `scripts/eval_v3.py` samples from the full labeled file, not from a reserved split.

This repository does not publish the dataset, checkpoint and split manifest behind 0.974. Thus you cannot reproduce the figure from this repository alone.

## Design

The palette is Espresso. The visual language does not use traffic-light colors (no red/green quality judgments). It uses saturation and density to show discourse complexity. [DESIGN-LANGUAGE.md](../DESIGN-LANGUAGE.md) gives the full design system.
