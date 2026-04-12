# Idea 4: On-Device Deliberation Quality Coach

## The Insight

Nobody coaches you to argue better *while you're typing*. Kialo scores aftermath. Perspective API detects toxicity. AQuA analyzes transcripts. But nothing intercepts low-quality discourse *during drafting* and says: "add a reason for your claim" or "acknowledge their point before disagreeing."

Platforms optimize for engagement. This optimizes for quality. And it runs on your phone - no data leaves the device.

## Quick Nav

| Section | Content |
|---------|---------|
| [Product](#product) | What the user sees |
| [Architecture](#architecture) | Model, pipeline, delivery |
| [Training Strategy](#training-strategy) | Data, approach, compute |
| [Build Plan](#build-plan) | Execution order |
| [At the Hackathon](#at-the-hackathon) | 48h execution plan |
| [Risks and Fallbacks](#risks-and-fallbacks) | What breaks and how to recover |

---

## Product

**One sentence**: A real-time deliberation quality coach that runs entirely on your phone.

**User experience**: You're drafting a response in an online discussion. As you type, a small score bar updates:

```
┌──────────────────────────────────────┐
│  Your response                        │
│  ──────────────────                   │
│  "That's completely wrong and you     │
│  know it. Anyone who believes this    │
│  is delusional."                      │
│                                       │
│  ┌─ Deliberation Quality ──────────┐  │
│  │ Justification  ■□□□  weak       │  │
│  │ Respect        ■□□□  dismissive │  │
│  │ Constructive   □□□□  none       │  │
│  │                                 │  │
│  │ 💡 Try: "I disagree because..." │  │
│  │    instead of "that's wrong"    │  │
│  └─────────────────────────────────┘  │
└──────────────────────────────────────┘
```

The user edits their response:

```
│  "I see where you're coming from,     │
│  but the evidence suggests otherwise  │
│  because [study X] found that..."     │
│                                       │
│  ┌─ Deliberation Quality ──────────┐  │
│  │ Justification  ■■■□  qualified  │  │
│  │ Respect        ■■■■  explicit   │  │
│  │ Constructive   ■■□□  mediating  │  │
│  │                                 │  │
│  │ ✓ Strong improvement            │  │
│  └─────────────────────────────────┘  │
└──────────────────────────────────────┘
```

**Key properties**:
- Runs in the browser as a PWA (installable on phone home screen)
- All inference on-device via ONNX Runtime Web (WASM backend)
- No API calls, no data exfiltration, works offline
- Multilingual: German + English minimum (Munich audience)
- Sub-second feedback as you type (debounced, not per-keystroke)

---

## Architecture

```
User types text (debounced 500ms)
  │
  ▼
Tokenizer (DeBERTa tokenizer, runs in JS)
  │
  ▼
ONNX Runtime Web (WASM backend)
  ┌─────────────────────────────────┐
  │ DeBERTa-v3-small (44M params)   │
  │ FP16 quantized (~90MB)          │
  │ Vocab-pruned (~60MB target)     │
  │                                 │
  │ Multi-head output:              │
  │   → justification (0-3 ordinal) │
  │   → respect (0-2 ordinal)       │
  │   → constructiveness (0-2)      │
  └─────────────────────────────────┘
  │
  ▼
Score → Feedback Mapper (rule-based JS)
  │ Maps score combinations to specific suggestions:
  │   low justification → "Add a reason for your claim"
  │   low respect → "Acknowledge their point before disagreeing"
  │   low constructive → "Can you suggest a compromise?"
  │
  ▼
UI update (score bars + suggestion text)
```

**Model specification**:
- Backbone: DeBERTa-v3-small (44M params)
- Input: tokenized text, max 256 tokens
- Output: 3 classification heads (justification 4-class, respect 3-class, constructiveness 3-class)
- Inference: ~150-300ms on phone browser (WASM), ~80-150ms on desktop browser
- Size: ~60-90MB after FP16 + vocab pruning (same technique as unslop)

**Delivery**: PWA
- `index.html` + `manifest.json` + service worker
- ONNX model file served as static asset (cached by service worker for offline)
- Installable on iOS/Android home screen
- No app store, no native code, deployable in minutes via Vercel/Render

---

## Training Strategy

### The Data Problem

AQuA has only 1,953 labeled samples (910 Europolis + 1,043 SFU). That's very small. We solve this with a 3-stage approach:

### Stage 1: Pre-train on argument quality (30K samples)

**Dataset**: IBM Debater `argument_quality_ranking_30k` on HuggingFace
- 30,497 arguments across 71 topics
- Quality labels: binary (convince/not convince) from 10 annotators
- This teaches the model "what a good argument sounds like" generically

```python
from datasets import load_dataset
ds = load_dataset("ibm-research/argument_quality_ranking_30k")
```

### Stage 2: Generate synthetic DQI labels (5K-10K samples)

Use Claude to label ChangeMyView Reddit comments with DQI scores. Claude is good at this because DQI has clear, codifiable rubrics.

```python
LABELING_PROMPT = """Score this comment on 3 deliberation quality dimensions.

Comment: "{text}"

Score each:
- justification: 0 (none) | 1 (vague assertion) | 2 (qualified reason) | 3 (sophisticated, links to evidence/common good)
- respect: 0 (explicit disrespect/insult) | 1 (dismissive/ignoring) | 2 (neutral/implicit acknowledgment) | 3 (explicit respect for opposing view)
- constructiveness: 0 (purely positional) | 1 (acknowledges complexity) | 2 (proposes compromise or alternative)

Return JSON: {"justification": N, "respect": N, "constructiveness": N}"""
```

Cost: ~$5-10 for 10K labels via Claude Haiku.

### Stage 3: Fine-tune on AQuA (1,953 samples)

Final specialization on human-annotated deliberation data. The Europolis dataset has expert DQI annotations - these are the gold labels.

### Adaptation from unslop pipeline

The unslop-training pipeline (`train_v05_compress.py`) needs these changes:

| Component | unslop (current) | deliberation (new) |
|-----------|-------------------|-------------------|
| Backbone | DeBERTa-v3-small | DeBERTa-v3-small (same) |
| Data format | RAID dataset, binary label | Custom dataset, 3 ordinal labels |
| Loss | CrossEntropyLoss (2-class) | CrossEntropyLoss per head (4+3+3 class) |
| Metric | TPR@5%FPR | Macro F1 per dimension |
| Features | 8 linguistic features | Remove or redesign (TBD) |
| Export | ONNX FP16 + vocab prune | ONNX FP16 + vocab prune (same) |

**Handcrafted features**: The 8 unslop features (TTR, hapax rate, sentence variance, etc.) were designed for AI detection. For deliberation quality, different features might help:
- Presence of causal connectives ("because", "therefore", "since")
- Question marks (engagement signal)
- First-person vs. second-person pronouns (self-focused vs. other-acknowledging)
- Hedge words ("perhaps", "I think", "it seems") - correlate with respect
- Negation density (correlates with dismissiveness)

Or: skip handcrafted features entirely and let the model learn from text alone. Simpler, faster, less to debug in 48h.

### Compute

DeBERTa-v3-small fine-tuning on ~32K samples:
- ~30-60 minutes on a single T4 GPU
- Kaggle free tier: 2x T4, 30h/week - more than enough
- Colab free: 1x T4, limited but sufficient

---

## Build Plan

### Phase 1: Data + Model

- [ ] Download AQuA data from `github.com/mabehrendt/AQuA`
- [ ] Download IBM Debater from HuggingFace
- [ ] Download ChangeMyView sample (5K comments) for synthetic labeling
- [ ] Run Claude labeling on 5K-10K ChangeMyView comments
- [ ] Write data preprocessing script: unify IBM + AQuA + synthetic into one format
- [ ] Adapt `train_v05_compress.py` for 3-head multi-label classification
- [ ] Run training on Kaggle (T4): pre-train on IBM, then fine-tune on AQuA + synthetic
- [ ] Evaluate: macro F1 per dimension, spot-check predictions
- [ ] ONNX export + FP16 quantize + vocab prune
- [ ] Test in browser with ONNX Runtime Web

### Phase 2: Frontend

- [ ] Build minimal PWA shell (text input -> model inference -> score display)
- [ ] Test on phone browser (Chrome Android / Safari iOS)

---

## At the Hackathon

### Friday evening (18:30-23:00)
- Listen to kickoff brief
- Confirm this direction fits the theme (it should - "saving public discourse" is exactly this)
- Set up dev environment, confirm model loads in browser

### Saturday (09:00-20:00)
- 09:00-12:00: Feedback UI - score bars, suggestion text, smooth transitions
- 12:00-14:00: Rule engine mapping scores to actionable suggestions
- **14:00: SCOPE LOCK** - what works, works. Polish from here.
- 14:00-16:00: Demo data preparation - pre-written example conversations showing before/after
- 16:00-18:00: Pitch draft + practice
- 18:00-20:00: End-to-end runs, record fallback video

### Sunday (09:00-19:30)
- 09:00-11:00: Fix issues, final polish
- 11:00-13:00: Pitch slides
- 13:00-15:00: Practice pitch 3+ times
- 15:00-17:00: Buffer
- 17:00-19:30: Presentations

---

## Risks and Fallbacks

### Model doesn't converge well
**Fallback**: Use AQuA's existing pre-trained adapters directly (mBERT + 20 adapters). Run 3-5 key adapters server-side via FastAPI. Pitch as "cloud version with on-device roadmap."

### ONNX export breaks for multi-head model
**Fallback**: Export 3 separate small models (one per dimension). Run sequentially. 3x90MB = 270MB total, heavier but functional.

### Model is too slow on phone
**Fallback**: Run on desktop browser only for demo. Pitch the phone angle as roadmap. "We validated it runs in-browser at 150ms; mobile optimization is next."

### Training data isn't enough
**Fallback**: Use Perspective API for the "respect" dimension (it already has toxicity + constructiveness scores). Only fine-tune the "justification" dimension on AQuA/IBM. Hybrid: 1 on-device head + 1 API call.

### Judges don't care about on-device
**Pivot pitch**: Emphasize the product, not the tech. "A coach that makes everyone argue better." The on-device angle is the differentiation, but the value prop is the coaching.

---

## The Pitch

**Hook** (20s): "Every comment section, every forum, every political debate online - the quality of discourse is collapsing. Not because people are stupid, but because nothing helps them argue better. Platforms reward outrage. We built something that rewards quality."

**Problem** (30s): "There's no feedback loop for argument quality. Spell-checkers fixed our grammar. Grammarly fixed our style. Nothing fixes our reasoning. A response that says 'you're wrong and delusional' gets the same input experience as one that says 'I disagree because the evidence shows X.'"

**Demo** (90s): Type a dismissive response → low scores, specific suggestions. Edit it following the suggestions → scores improve in real time. "Watch what happens when I add a reason... justification jumps to 'qualified.' When I acknowledge their point... respect goes to 'explicit.'"

**How** (20s): "DeBERTa model fine-tuned on the Discourse Quality Index, running in your browser via ONNX Runtime. 60 megabytes, sub-second inference, no data leaves your device. Works offline."

**Impact** (20s): "If every discussion platform embedded this, the quality of public discourse measurably improves. Not by censoring bad arguments - by coaching better ones. We start with a browser tool, then a browser extension, then platform integrations."

---

## Why This Wins

1. **Novel**: literally nobody does real-time deliberation coaching during composition
2. **On-device**: privacy-first, no cloud dependency - judges love this
3. **Demo-friendly**: live typing with real-time score changes is visually compelling
4. **Academically grounded**: built on DQI (Steenbergen 2003), not vibes
5. **Directly on-theme**: "Saving Public Discourse" = improving discourse quality at the individual level
6. **Startup potential**: browser extension → platform API → B2B for moderation teams
7. **Benefit for humanity**: this is genuinely good for the world, and that matters at this hackathon
