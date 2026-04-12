# AI Slop Detection Model (v0.6)

DeBERTa-v3-small with pruned vocabulary, FP16 quantized. 186 MB.

## Setup

Copy these files from the unslop training artifacts into this directory:

```
model-slop/
  model.onnx          # 186 MB - the ONNX model (DO NOT commit to git)
  token_remap.json    # 1.1 MB - Token ID remapping for pruned vocabulary (70K tokens)
```

Source: Modal volume `unslop-training`, path `runs/exp-dropout-03/export/`

## Usage

```js
import { loadSlopModel, scoreForSlop } from '../detection/index.js';

await loadSlopModel('./model-slop/model.onnx', './model-slop/token_remap.json');
const result = await scoreForSlop('some text to check');
// result: { aiProb, verdict, tier, heuristicScore, signals, inferenceMs }
```

## Files expected by code

| File | Required | Size | Purpose |
|------|----------|------|---------|
| model.onnx | Yes | ~186 MB | ONNX inference session (v0.6, FP16) |
| token_remap.json | Yes | ~1.1 MB | Maps original tokenizer IDs to pruned vocab IDs (70K tokens) |

The tokenizer (microsoft/deberta-v3-small) is loaded from HuggingFace CDN at runtime.
