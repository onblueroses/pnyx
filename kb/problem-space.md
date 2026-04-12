# Problem Space: Threats to Public Discourse

## Quick Nav

| Section | Content |
|---------|---------|
| [Core Threats](#core-threats) | The main attack vectors on discourse quality |
| [What Actually Works](#what-actually-works) | Validated approaches with evidence |
| [Key Framings](#key-framings) | How to talk about the problem convincingly |
| [What Doesn't Work](#what-doesnt-work) | Common approaches that fail or are played out |

---

## Core Threats

**Misinformation / disinformation**
- AI-generated synthetic media has dropped production cost to near zero
- Coordinated inauthentic behavior (bot networks, sockpuppets) operates at machine scale
- Micro-targeted emotional manipulation exploits platform engagement algorithms

**Polarization**
- Engagement-maximizing algorithms amplify outrage; divisive content has higher dwell time
- Filter bubbles and echo chambers reduce exposure to disconfirming information
- Affective polarization (disliking the other side) has grown faster than issue polarization

**Epistemic erosion**
- Declining trust in institutions (media, science, government) lowers the base rate of belief in corrections
- "Firehose of falsehood" strategy: flood the zone so fast that corrections can't keep up
- Scale asymmetry: one actor can produce more false claims than all fact-checkers can address

**Discourse quality degradation**
- High-volume low-quality participation crowds out substantive deliberation
- Platforms optimize for engagement, not argument quality
- No feedback mechanism for rhetorical quality or epistemic contribution

---

## What Actually Works

**Community Notes model (Twitter/X)**
- Crowd-sourced, bridging-based: notes require agreement from users with different viewpoints to appear
- Adversarially robust by design - can't be gamed by one ideological faction
- Open data published daily; strong research base
- Limitation: slow (median hours to label), reactive not proactive

**Lateral reading (SIFT method)**
- Instead of evaluating a source top-to-bottom, immediately search what others say about it
- Operationalized by professional fact-checkers; outperforms "reading carefully"
- Scalable with LLMs - ReadProbe (2023 hackathon winner) proved this works as a product

**Habermas Machine (Google DeepMind, 2023)**
- LLM-mediated deliberation that finds group statements participants rate more highly than human-mediated
- Paper: arxiv.org/abs/2311.14105 - short, worth reading
- Key insight: AI as mediator, not arbiter - it surfaces common ground, doesn't impose conclusions

**Prebunking (inoculation theory)**
- Expose people to weakened misinformation + refutation before they encounter it at scale
- More durable than debunking after the fact
- Google/Jigsaw ran prebunking campaigns with measurable effect on YouTube

**Bridging-based ranking**
- Promote content that gets positive engagement from people who usually disagree
- Implemented in Community Notes; theorized for broader application
- The "bridging" score is a product signal, not just a research concept

---

## Key Framings

**Deliberation, not detection**
The strongest recent projects move beyond binary fake/real classification toward tools that help humans reason better. Detection is a solved-enough problem (ClaimBuster, Google Fact Check); improving the quality of deliberation is not.

**Speed is the actual problem**
False claims spread faster than corrections. Tools that operate at the speed of consumption (at read time, not after) are more valuable than post-hoc debunkers.

**Professional users > consumers**
Journalists, analysts, researchers, and policy advisors have concrete, high-stakes needs and are willing to use tools that require some sophistication. Consumer tools face adoption barriers that can't be solved in 48h.

**Trust infrastructure, not content moderation**
The framing that resonates most with policy audiences: building the infrastructure that lets institutions be trusted again, not policing what people say.

---

## What Doesn't Work

- Binary classifiers ("fake" / "not fake") - too blunt, legally risky, easily gamed
- Platform-dependent solutions - any demo relying on live Twitter/Meta API will break
- Correction campaigns at scale - documented backfire effect, low uptake
- "Platform for everything" - detection + explanation + correction + education simultaneously
- Consumer apps without a clear acquisition path - judges ask "who pays for this?"
