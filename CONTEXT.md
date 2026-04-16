# Hackathon Context

**Event**: Agora Hackathon x TUM.ai E-Lab - Saving Public Discourse  
**Date**: April 10-12, 2026  
**Format**: 48h hackathon, 6-minute pitch
**Deadline**: April 12, 15:30 (pitches start)

## Quick Nav

| Need | Go to |
|------|-------|
| Problem framing | [kb/problem-space.md](kb/problem-space.md) |
| Prior art | [kb/prior-art.md](kb/prior-art.md) |

## Locked Direction

Challenge Cluster 02 - Resilient Public Discourse: **"Shift digital interaction from performative reaction to visible meaningful listening."**

## Product Context

The product is **Pnyx** - a browser extension that makes listening visible.

**Four layers**:
1. **SEE** - On-device AI extracts what a post claims (existing models, reframed)
2. **PAUSE** - Before you reply, you see what they said and choose what you're engaging with
3. **SHOW** - Your reply carries visible evidence of what you listened to
4. **EXPLORE** - Branching deliberation on any claim through 3 beats of stance/concession/escalation (DeepSeek V3 via OpenRouter)

This is NOT the V1 "content quality scanner." The models are the enabler, not the product. The product is the behavioral shift from reaction to listening.

**What this is NOT**:
- not moderation
- not a fact-checker
- not a consensus engine
- not surveillance (scoring on-device; Explore uses external API with user's own key)
- not a separate platform

## Target User

Anyone who uses LinkedIn or X/Twitter. The Chrome extension works inside existing platforms - no new app adoption required.

## What Judges Need To See

1. A clear behavioral shift (from reaction to listening)
2. A live demo showing SEE -> PAUSE -> SHOW -> EXPLORE
3. Theoretical grounding (Habermas + Arendt + agonistic awareness) - encoded directly in Explore mode prompts
4. Technical depth (dual on-device models for scoring, LLM-powered deliberation)
5. A realistic product path (Chrome extension, no server costs for core features)

## V1 Foundation (Built, Working)

The V1 infrastructure is complete and proven:
- Habermas model (DeBERTa-v3-small, F1 0.974, 271 MB FP16)
- Erscheinung heuristic tier (85 patterns, <1ms). ML model (v0.6) disabled - overfit to RAID dataset, retrain in progress (v0.7)
- Demo page with LinkedIn-styled feed, 7 scored posts
- Chrome MV3 extension with offscreen ONNX inference
- Web Worker inference pipeline
- AI blur + click-to-reveal

V2 adds four layers on top without touching the model infrastructure.

## Repo Structure

| Path | Purpose | V2 status |
|------|---------|-----------|
| scaffold/frontend/index.html | Demo page | DONE (V2 layers, 35855c4) |
| scaffold/frontend/inference-worker.js | ONNX inference | UPDATED (ML tier disabled) |
| scaffold/frontend/model/ | Habermas ONNX + tokenizer | KEEP AS-IS |
| scaffold/frontend/model-slop/ | Erscheinung ONNX + tokenizer | UPDATED (v0.6) |
| scaffold/frontend/detection/ | Erscheinung heuristics + features | KEEP AS-IS |
| scaffold/frontend/extension/ | Chrome MV3 extension | DONE (uncommitted, needs commit+push) |
| scaffold/frontend/claim-extractor.js | Claim extraction | DONE (35855c4) |
| scaffold/frontend/pause-layer.js | Pause overlay | DONE (35855c4) |
| scaffold/frontend/reply-tags.js | Reply tags | DONE (35855c4) |
| scaffold/frontend/explore.html | Explore mode page | DONE |
| scaffold/frontend/explore.js | Explore deliberation engine | DONE |
| scaffold/frontend/explore.css | Explore mode styles | DONE |
| pitch/ | Pitch slides (NEW) | NOT STARTED |
| scripts/ | Training scripts | DONE |
| data/ | Training data | DONE |
| docs/ | Model contracts | UPDATED (v0.6) |
| kb/ | Research + prior art | KEEP |
