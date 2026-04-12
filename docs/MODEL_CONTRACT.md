# Model Contract: Habermas - Discourse Legibility (v3)

> Named after Jürgen Habermas's *Theory of Communicative Action* (1981). The model makes discourse structure legible by extracting two dimensions: *Wahrheit* (truth - what claims are being made?) and *Richtigkeit* (rightness - is reasoning present?). Used as a lens for legibility, not a standard for judgment. The third validity claim, *Wahrhaftigkeit* (sincerity), is checked by the Erscheinung model. In V2, model output feeds into the claim extraction pipeline (sentence-level heuristics) and the Pause Layer.

## Model

| Version | Base | Params | Size | Status |
|---------|------|--------|------|--------|
| v3 | DeBERTa-v3-small (`cross-encoder/nli-deberta-v3-small`) | 141M | 271 MB FP16 | trained, F1 0.974 avg (0.977 CR, 0.972 AQ) |
| v2 | DeBERTa-v3-small (`cross-encoder/nli-deberta-v3-small`) | 141M | 271 MB FP16 | superseded by v3 |

## Inputs

| Name | Shape | Dtype | Description |
|------|-------|-------|-------------|
| input_ids | [1, 256] | int64 | Tokenized text, max 256 tokens |
| attention_mask | [1, 256] | int64 | 1 for real tokens, 0 for padding |

### Tokenizer

| Tokenizer | Type | Vocab | Notes |
|-----------|------|-------|-------|
| `cross-encoder/nli-deberta-v3-small` | SentencePiece | 128,100 (full, unpruned) | No token remapping needed - uses full vocabulary |

Load from local directory: `AutoTokenizer.from_pretrained('./model/')`.

Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `config.json`) are bundled in the `model/` directory alongside the ONNX file.

## Outputs

| Name | Shape | Dtype | Validity Claim | Description |
|------|-------|-------|----------------|-------------|
| claim_risk | [1, 2] | float32 | *Wahrheit* (Truth) | Logits: [low_risk, high_risk] |
| argument_quality | [1, 2] | float32 | *Richtigkeit* (Rightness) | Logits: [no_reasoning, has_reasoning] |

Apply softmax to get probabilities. The `[1]` index gives the positive class probability.

## Score Interpretation

**V2 reframe**: Scores indicate discourse structure, not quality judgment. "High claim risk" means "unsupported assertions present" - the claims might be true, they're just not supported yet.

| Probability | Claim Risk (*Wahrheit*) | Argument Quality (*Richtigkeit*) |
|-------------|------------------------|----------------------------------|
| > 0.65 | high - unsupported assertions present | strong - reasoning/evidence present |
| 0.35 - 0.65 | medium - borderline | mixed - partial reasoning |
| < 0.35 | low - qualified/grounded | weak - assertion without justification |

### Color mapping

- Claim Risk: high = danger (red), medium = warn (amber), low = good (green)
- Argument Quality: strong = good (green), mixed = warn (amber), weak = danger (red)

Note the inversion: high claim risk is bad, high argument quality is good.

### V2 Integration

Model output feeds into two downstream systems:
1. **Claim extraction** (`claim-extractor.js`): Model probabilities provide context for sentence-level claim scoring. High claim risk + low argument quality = prioritize extracting assertion sentences.
2. **Pause Layer** (`pause-layer.js`): Extracted claims are surfaced as checkboxes when user clicks Reply. The model doesn't power the Pause directly - the claim extractor mediates.

### Combined interpretation

| claim_risk | argument_quality | Validity Claim Status | Suggested CTA |
|-----------|-----------------|----------------------|---------------|
| low | strong | Both claims upheld | "Keep the exchange open" |
| low | weak | *Richtigkeit* violated | "Add a reason" |
| high | strong | *Wahrheit* violated, but argued | -- |
| high | weak | Both claims violated | "Ask for evidence" |

## Architecture (v3)

```
DeBERTa-v3-small [CLS] (768-dim)
  -> LayerNorm(768) -> Dropout(0.2)
  -> claim_risk head:      Linear(768, 256) -> GELU -> Dropout(0.2) -> Linear(256, 2)
  -> argument_quality head: Linear(768, 256) -> GELU -> Dropout(0.2) -> Linear(256, 2)
```

NLI-pretrained checkpoint (`cross-encoder/nli-deberta-v3-small`) provides entailment/contradiction reasoning as a foundation. Fine-tuned on 10K purpose-built samples (balanced 2,500/cell), then 81 targeted correction examples (3x oversampled).

v3 adds 453 boundary examples (5x oversampled) targeting 4 failure modes: implicit claims, factual-but-not-argumentative, compressed reasoning, and AI slop. Trained with focal loss (gamma=2) + label smoothing (0.05) to reduce overconfidence. All layers unfrozen, lr=5e-6, 5 epochs resumed from v2 checkpoint.

FP16 quantized. INT8 not possible (onnxruntime shape inference bug on DeBERTa attention architecture).

## Inference

```js
import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';

const tokenizer = await AutoTokenizer.from_pretrained('./model/');
const session = await ort.InferenceSession.create('./model/model.onnx');

async function scoreHabermas(text) {
  const { input_ids, attention_mask } = tokenizer(text, {
    padding: true, truncation: true, max_length: 256, return_tensors: 'np',
  });

  const output = await session.run({
    input_ids: new ort.Tensor('int64', input_ids.data, input_ids.dims),
    attention_mask: new ort.Tensor('int64', attention_mask.data, attention_mask.dims),
  });

  const softmax = (a, b) => {
    const m = Math.max(a, b);
    return Math.exp(b - m) / (Math.exp(a - m) + Math.exp(b - m));
  };

  return {
    claimRiskProb: softmax(output.claim_risk.data[0], output.claim_risk.data[1]),
    argQualityProb: softmax(output.argument_quality.data[0], output.argument_quality.data[1]),
  };
}
```

## Message Protocol

```js
// Habermas output (sent from offscreen/worker to content script)
{
  habermas: { claimRiskProb: 0.83, argQualityProb: 0.21 },
  inferenceMs: { habermas: 180 }
}
```

## Execution Config

- Provider: WASM only (WebGPU unreliable across contexts)
- Threads: 1
- SIMD: enabled
- Sequence length: 256 (not 128 - Erscheinung uses 128)

```js
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
```

## Loading Order

Load Erscheinung first, then Habermas. Erscheinung's heuristic tier can score immediately (<1ms) while Habermas downloads and initializes.

## Known Limitations

- Mid-range calibration (0.3-0.7) still has gaps, but much improved over v2 (uncertainty band 22% vs 5%)
- Short colloquial claims with no keywords may still be missed (6 false negatives in 200-sample eval)
- INT8 quantization not possible due to DeBERTa/onnxruntime incompatibility
- 271 MB FP16 is the minimum achievable size for this architecture

## Differences from Erscheinung

| Property | Habermas | Erscheinung |
|----------|----------|-------------|
| Inputs | 2 (tokens + mask) | 3 (tokens + mask + 8 features) |
| Outputs | 2 heads (logits) | 1 head (softmax probs) |
| Sequence length | 256 | 128 |
| Vocab | Full (128,100) | Pruned (70,198 in v0.6) - needs token_remap.json |
| Heuristic tier | None | 70+ patterns, skips ML if score >= 4.0 |
| Size | 271 MB FP16 | 186 MB FP16 pruned (v0.6) |
| Latency | ~500-900ms (Web Worker) | ~500ms |
