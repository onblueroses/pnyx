# Model Contract: Erscheinung - Authenticity Detection (v0.6)

> Named after Hannah Arendt's *Erscheinungsraum* (space of appearance). The model detects whether genuine human *Erscheinung* exists behind a text - whether there is someone real to listen to here. Maps to Habermas's third validity claim: *Wahrhaftigkeit* (sincerity). In V2, this answers a prerequisite question for the listening infrastructure: is there a "who" to listen to, or merely strategic action masquerading as communicative action?

> **Status (2026-04-11):** ML tier disabled in `inference-worker.js`. v0.6 overfit to RAID dataset (formal articles) - classifies all social media text as AI (90-100% ai_prob). Heuristic tier (85 patterns) active and working correctly. Retrain on custom social media data in progress (v0.7).

## Model

| Version | Base | Params | Size | Status |
|---------|------|--------|------|--------|
| v0.6 | DeBERTa-v3-small (`microsoft/deberta-v3-small`) | 141M | 186 MB FP16 pruned | trained, 97.7% TPR@5%FPR (val). Dropout 0.3/0.2, 99.9% vocab coverage |
| v0.5 | DeBERTa-v3-small (`microsoft/deberta-v3-small`) | 141M | 154 MB FP16 pruned | superseded by v0.6. 86.1% TPR@5%FPR (test), 94.6% (val) |

## Inputs

| Name | Shape | Dtype | Description |
|------|-------|-------|-------------|
| input_ids | [1, 128] | int64 | Tokenized text, max 128 tokens |
| attention_mask | [1, 128] | int64 | 1 for real tokens, 0 for padding |
| features | [1, 8] | float32 | Hand-crafted features (see order below) |

### Tokenizer

| Tokenizer | Type | Original Vocab | Pruned Vocab | Notes |
|-----------|------|----------------|-------------|-------|
| `microsoft/deberta-v3-small` | SentencePiece | 128,100 | 70,198 (v0.6) | Requires `token_remap.json` - remap tokenizer output IDs before model input |

Load from HuggingFace CDN via `AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')`.

### Token Remapping

The model uses a pruned vocabulary (70,198 tokens from 128,100 in v0.6; was 43,648 in v0.5). After tokenization:
1. Load `token_remap.json` (keys are string IDs, values are new integer IDs)
2. For each token ID from the tokenizer, look up `tokenRemap[Number(oldId)]`
3. Unknown IDs (not in remap) map to 0

## Feature Order (index 0-7)

Must match `detection/features.js` `featuresToArray()`:

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | ttr | 0-1 | Type-Token Ratio (unique words / total words) |
| 1 | hapaxRate | 0-1 | Words appearing exactly once / unique words |
| 2 | sentenceLengthVariance | 0-inf | Population variance of sentence word counts |
| 3 | avgSentenceLength | 0-inf | Mean sentence length in words |
| 4 | bigramUniqueness | 0-1 | Unique bigrams / total bigrams |
| 5 | stopWordDensity | 0-1 | Stop words / total words |
| 6 | contractionPresence | 0 or 1 | Binary: text contains English contractions |
| 7 | lowercaseRatio | 0-1 | Lowercase ASCII letters / all ASCII letters |

## Output

| Name | Shape | Dtype | Description |
|------|-------|-------|-------------|
| output | [1, 2] | float32 | Softmax probabilities: [human_prob, ai_prob] |

## Inference

- `ai_prob >= 0.5` -> verdict: `slop`
- `ai_prob < 0.5` -> verdict: `clean`
- Score reported to user: `ai_prob` (0-1 range)

## Architecture (v0.6)

```
DeBERTa CLS (768-dim) + features (8-dim)
  -> LayerNorm(776)
  -> Linear(776, 256) -> GELU -> Dropout(0.3)
  -> Linear(256, 128) -> GELU -> Dropout(0.2)
  -> Linear(128, 2)
```

Vocab pruned to 70,198 tokens (99.9% coverage). FP16 quantized.

v0.6 increases dropout from 0.2/0.1 to 0.3/0.2 to delay overfitting (from epoch 2 to epoch 3). Same architecture otherwise. See `HANDOFF-W5-MODEL-COMPLETE.md` for full experiment results.

## Preprocessing

Before tokenization AND feature extraction:
1. NFKC Unicode normalization
2. Strip zero-width characters (U+200B, U+200C, U+200D, U+FEFF)

This matches `normalizeText()` in `detection/features.js`.

## Execution Config

- Provider: WASM only (WebGPU unreliable across contexts)
- Threads: 1
- SIMD: enabled
- Sequence length: 128 (not 256 - the discourse model uses 256)

## Heuristic Tier (D10)

Before ML inference, run `scoreText()` from `detection/heuristics.js`. If heuristic score >= 4.0, skip ML inference entirely. This saves ~500ms on obvious AI-generated text.
