# Pnyx

Listening infrastructure for public discourse. A Chrome extension and demo that makes listening visible on social media feeds.

Built for the Agora Hackathon x TUM.ai E-Lab (April 2026), Challenge Cluster 02: *"Shift digital interaction from performative reaction to visible meaningful listening."*

## What it does

Pnyx runs AI scoring entirely on-device (no data leaves the browser) and adds four behavioral layers to social media:

**SEE** - Badges on posts showing what claims are being made and whether reasoning is present. Powered by the Habermas model (DeBERTa-v3-small, 141M params, F1 0.974) and an 85-signal heuristic tier for AI-generated text detection.

**PAUSE** - When you click Reply, an overlay appears showing the extracted claims as checkboxes. Pick what you're actually responding to before writing.

**SHOW** - Your reply carries a visible tag: "Responded to: [claim]". Other readers can see what was heard.

**EXPLORE** - Branching deliberation on any claim. Three beats (steel-man, perspective-take, consequence-trace) via DeepSeek V3 through OpenRouter.

## Theoretical framework

Three thinkers encode directly into the technical choices:

- **Habermas** - The scoring model maps two validity claims from *Theory of Communicative Action*: *Wahrheit* (truth/claim risk) and *Richtigkeit* (rightness/argument quality). Discourse legibility, not judgment.
- **Arendt** - The Erscheinung model detects genuine human presence (*Erscheinungsraum*). The third validity claim: *Wahrhaftigkeit* (sincerity). Is there a "who" behind the text?
- **Mouffe** - Explore mode treats disagreement as a democratic resource (agonistic pluralism). Escalation is a valid deliberation move, not a failure state.

## Quick start: Demo page

```bash
cd scaffold/frontend
python -m http.server 8080
# Open http://localhost:8080/index.html
```

Models load on first visit (~460 MB, cached in browser after first load). Seven pre-scored posts demonstrate all four layers.

### Model files

ONNX model files are not included in this repo (too large). Download from Hugging Face:

| Model | File | Size | Hugging Face |
|-------|------|------|--------------|
| Habermas v3 | `scaffold/frontend/model/model.onnx` | 271 MB | [onblueroses/pnyx-habermas](https://huggingface.co/onblueroses/pnyx-habermas) |
| Erscheinung v0.7 | `scaffold/frontend/model-slop/model.onnx` | 126 MB | [onblueroses/pnyx-erscheinung](https://huggingface.co/onblueroses/pnyx-erscheinung) |

Tokenizer files and configs are included in the repo. If model files are missing, the demo falls back to heuristic-only scoring.

## Quick start: Chrome extension

```bash
cd scaffold/frontend/extension
bash setup.sh
```

Then in Chrome:
1. Open `chrome://extensions`
2. Enable Developer mode
3. Click "Load unpacked" and select the `scaffold/frontend/extension/` directory

The extension injects Pnyx into LinkedIn feeds. `setup.sh` copies model files, detection modules, and vendor libraries into the extension directory.

## Architecture

All scoring runs on-device via ONNX Runtime Web (WASM). Zero API calls for inference.

| Component | Tech | Size |
|-----------|------|------|
| Habermas model | DeBERTa-v3-small, dual-head (claim risk + argument quality) | 271 MB FP16 |
| Erscheinung model | DeBERTa-v3-small, pruned vocab (70K tokens) + 85 heuristic signals | 126 MB FP16 |
| Inference | ONNX Runtime Web, WASM, single-thread SIMD | ~500-900ms per post |
| Extension | Chrome MV3, offscreen document, MutationObserver feed scanning | |
| Explore | Client-side OpenRouter API (user's own key, localStorage) | |

### How inference works

1. Post text enters a Web Worker (`inference-worker.js`)
2. Erscheinung heuristic tier runs first (<1ms) - if score >= 4.0, ML inference is skipped
3. Both ONNX models run in parallel (Habermas: 256 tokens, Erscheinung: 128 tokens)
4. Scores feed into `claim-extractor.js` for sentence-level claim extraction
5. Badges, Pause Layer, and Reply Tags render from the extracted claims

### Training pipeline

The `scripts/` directory contains the full data generation and training pipeline:

1. `01_source_texts.py` - Source text collection from disinformation datasets
2. `02_label_texts.py` - Automated labeling with discourse quality rubric
3. `03_balance_and_report.py` - Dataset balancing (2,500 samples per cell)
4. `04_generate_boundary_data.py` - Targeted examples for failure modes
5. `train_v3_modal.py` - Model training on Modal (T4 GPU, focal loss, ~27 min/epoch)

Final dataset: 10K balanced samples + 453 boundary examples. Habermas v3 achieves F1 0.974 (0.977 claim risk, 0.972 argument quality).

## Project structure

```
scaffold/frontend/
  index.html             Demo page (all four layers)
  inference-worker.js    Web Worker for ONNX inference
  claim-extractor.js     Sentence-level claim extraction
  pause-layer.js         Pause overlay before reply
  reply-tags.js          Visible listening tags
  explore.html/js/css    Branching deliberation
  detection/             Erscheinung heuristic pipeline
  model/                 Habermas ONNX + tokenizer
  model-slop/            Erscheinung ONNX + tokenizer
  extension/             Chrome MV3 extension

scripts/                 Training pipeline
data/                    Demo datasets + source credibility
docs/                    Model contracts (architecture, I/O specs)
kb/                      Research notes (problem space, prior art)
```

## Design

Espresso palette. The visual language avoids traffic-light color coding (no red/green quality judgments) in favor of saturation and density to signal discourse complexity. See [DESIGN-LANGUAGE.md](DESIGN-LANGUAGE.md) for the full design system.

## License

MIT
