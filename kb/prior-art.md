# Prior Art: What Exists and What's Been Tried

## Quick Nav

| Section | Content |
|---------|---------|
| [Hackathon Winners](#hackathon-winners) | Specific projects that placed well and why |
| [Existing Products](#existing-products) | Deployed tools in this space |
| [Research Systems](#research-systems) | Academic / open-source pipelines |
| [Failure Modes](#failure-modes) | What doesn't land with judges or users |

---

## Hackathon Winners

These placed at comparable hackathons. Note the patterns.

**ReadProbe** - 1st place, Canadian AI Misinformation Hackathon 2023 (University of Waterloo)
- What: ChatGPT-powered lateral reading tool - the SIFT method at scale. Paste a URL, get "what others say about this source" surfaced immediately.
- Why it won: Applied an established, validated media literacy technique rather than inventing new theory. Judges could verify the underlying method was sound.
- Lesson: Ground your project in a named, existing framework (lateral reading, prebunking, bridging) - don't position it as novel theory.

**IsItBot** - 2nd place, GMF AI & Democracy Hackathon 2024 (Mexico City)
- What: Paste a social media profile → credibility score + content integrity signals
- Why it won: One input, one output, demo runs in 60 seconds. Narrow scope executed cleanly.
- Lesson: Single clear verb. Scope is a competitive advantage.

**(F)actually** - Democracy's Firewall Hackathon winner
- What: Gamified fact-checking with real news/social claims, role-play as fact-checkers, guided hints, real-time trend dashboard
- Why it won: Working demo + pedagogical framing + legible in 60 seconds. The trend dashboard added credibility.
- Lesson: A second output surface (the dashboard, the graph) gives judges something to look at during Q&A.

**dubio** - Hacking the Fake News (Media Development Foundation)
- What: Community-driven claim-debunking with distributed moderation
- Why it won: Addressed the speed asymmetry by distributing the fact-checking labor, not relying on a central authority.
- Lesson: If your insight is structural ("the problem is speed/scale, not accuracy"), name it explicitly.

**AICrossTheOcean** - TUM.ai Makeathon 2023 (placed well)
- What: AI Risk Inspector for EU AI Act compliance assessment
- Why it placed: Governance-adjacent AI; directly applicable to a real regulatory need; business case was obvious.
- Lesson: At TUM.ai specifically, regulatory/governance tools land well. "Public discourse + policy" is a natural fit.

---

## Existing Products

**Community Notes (X/Twitter)**
- What: Bridging-based crowd-sourced context notes on posts
- Strength: Adversarially robust by design; open data; proven at scale
- Gap: Reactive (hours to days), English-heavy, social-media-specific
- Open data: communitynotes.twitter.com/guide/en/under-the-hood/download-data

**ClaimBuster (UT Arlington)**
- What: API that scores claims by check-worthiness
- Strength: Simple, fast, free API key
- Gap: English only; scores claims but does no verification
- Use as: Pipeline filter, not end product

**FullFact (UK)**
- What: Automated fact-checking with LLM pipeline
- Strength: Production quality, UK-deployed
- Gap: API not publicly open (journalist access only)

**Perspective API (Google/Jigsaw)**
- What: Scores text for toxicity, civility, and (new in 2025) "bridging attributes" - constructive discourse signals
- Strength: Free, fast, directly relevant. Note: shutting down Dec 2026, still active for hackathon.
- Use as: Signal layer in a larger pipeline

**Ground News**
- What: Shows media bias and coverage comparison for stories
- Strength: Consumer product with proven traction
- Gap: No API; no geopolitical depth

---

## Research Systems

**Loki / OpenFactVerification** (Libr-AI)
- Full end-to-end LLM pipeline: text → claim extraction → check-worthiness scoring → evidence retrieval → verdict
- MIT license, web demo at loki.librai.tech
- Requires: OpenAI key + Serper key. Can swap to Claude.
- Realistic to get running in 2-3 hours
- Repo: github.com/Libr-AI/OpenFactVerification

**OpenFactCheck** (modular)
- Swap components in/out of a factuality pipeline
- Good if you want to benchmark approaches or demonstrate modularity to judges
- Repo: github.com/yuxiaw/OpenFactCheck

**Habermas Machine**
- Not open-source, but the paper is short and the method is clear enough to re-implement in simplified form
- Paper: arxiv.org/abs/2311.14105
- Core technique: generate candidate statements, score by cross-partisan agreement, iterate

---

## Failure Modes

Things that routinely fail at this kind of hackathon:

- **"Platform for everything"**: misinformation detection + explanation + correction + user education simultaneously. Descope ruthlessly.
- **No prototype**: a proposal with dashboards-that-require-data-they-don't-have. Live demo required.
- **Binary fake/real classifier**: too blunt, legally risky, not useful to professionals. Judges at a think-tank-adjacent event will push back.
- **US-only social media angle**: Community Notes is US-heavy; Twitter API is unreliable. For Pnyx's audience, multilingual and multi-source matters.
- **Consumer framing at a B2B event**: "anyone can use this" raises the question of who pays and how it grows.
- **Demo crashes**: always record a fallback video. Always.
