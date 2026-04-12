---
title: Pnyx - Listening Infrastructure for Public Discourse
---

"Appearance constitutes reality."
{.big}

Hannah Arendt, The Human Condition

"So let's make it visible."

---

# Pnyx
{.big}

Listening infrastructure for public discourse.

---

## Three commitments

**SOVEREIGNTY**
Democratic infrastructure that depends on nobody's permission.

**AGONISM**
The other position has a right to exist.

**DISCOURSE**
The structure of public speech should be legible to everyone.

---

## Four layers

**DETECT** — Is there a real person behind this text?

**SEE** — What claims are being made, and are they grounded?

**PAUSE** — Which claim are you responding to?

**EXPLORE** — Three rounds of structured disagreement

---

## The Erscheinung Model

_Who is speaking?_

**Theory:** Arendt: the Erscheinungsraum requires a _who_ behind the _what_. AI-generated text has no who. So we detect the absence.

**Therefore:** Fine-tuned DeBERTa + 85 heuristic signals. Sub-millisecond. On-device via ONNX. When there's no who, the content fades — not blocks.

---

## The Habermas Model

_What is being said?_

**Theory:** Habermas: every speech act raises validity claims. **Wahrheit** — is a truth claim being made? **Richtigkeit** — is there argument structure? We built a head for each.

**Therefore:** Two binary classification heads on DeBERTa-v3-small. One head per validity dimension. 10K samples. F1 0.974. ONNX, on-device, sub-second.

---

## Explore Mode

_How should we disagree?_

**Theory:** Mouffe: consensus is not always the goal. Disagreement is a democratic resource. Transform antagonism into agonism.

**Therefore:** Three rounds on any claim. Three moves: stance, concession, escalation. No move is ranked above another.

---

## Demo

[Video demo]

---

# Pnyx
{.big}

You own your data.
You own your infrastructure.
Rejoin the discourse.
