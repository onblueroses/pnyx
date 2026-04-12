# Demo Script - Pnyx (3 minutes total)

## Part 0: Real LinkedIn (5s)
- Show extension running on your actual LinkedIn feed
- Claims already visible on real posts
- "This is Pnyx running live on LinkedIn. Let me walk you through what it does."

---

## Part 1: Demo Page - Feed Load (10s)

**Screen:** Demo page loads, model loading overlay briefly visible, feed appears with claims on posts

**Say:**
- Pnyx is a browser extension that lives inside LinkedIn and X
- When a page loads, two models run entirely in your browser
- The **Habermas model** extracts validity claims from every post - grounded in discourse ethics
- Habermas says every speech act raises three validity claims: **Wahrheit** (truth), **Richtigkeit** (rightness), **Wahrhaftigkeit** (sincerity)
- We operationalize the first two: is someone making factual claims? Is there argument structure?
- DeBERTa-v3-small, 22 million parameters, trained on 10,000 samples
- Two binary classification heads - one per validity dimension. F1 0.974
- 271MB ONNX model running entirely in-browser via ONNX Runtime Web
- Zero API calls. The model is yours.

---

## Part 2: Scroll + Badges (10s)

**Screen:** Scroll through posts, badges visible (Wahrheit: high 84%, Richtigkeit: weak 22%)

**Say:**
- Every post gets scored along both dimensions
- This first post: Wahrheit high (84%) - strong factual claims being made. Richtigkeit weak (22%) - no argument structure, no evidence
- The system doesn't judge the content. It makes the **structure of the speech act visible**
- Click the badge row to see the extracted claims: "Officials are hiding the real numbers" - unsupported. "The transit outage was deliberate" - unsupported
- This is Habermas made operational: which validity claims are being raised, and are they being redeemed?

---

## Part 3: Comment + PAUSE (10s)

**Screen:** Click Comment on Post 1, PAUSE overlay appears

**Say:**
- I want to respond to this post. I click Comment -
- And Pnyx interrupts me: **"Before you reply - which claim are you responding to?"**
- This is the PAUSE layer. Three seconds of friction between impulse and reaction
- It forces you to articulate what you're actually engaging with
- Not "I disagree" - but "I disagree with *this specific claim*"
- This transforms reply from **reaction into response**

**Action:** Check a claim, click Continue

---

## Part 4: Engagement Traces (10s)

**Screen:** Scroll to see replies under Post 1 - Jonas tagged, Niko skipped

**Say:**
- Now look at the replies
- Jonas responded to a specific claim: "transit outage was deliberate" - **tagged**. He engaged with what was actually said
- Niko: "Exactly, wake up people!" - **skipped claims**. Pure reaction, no engagement
- That difference is now visible. **Listening has a trace**
- 1 of 2 replies engaged with a specific claim - this is the metric
- Esau (2025): 71% of online comments receive zero deliberative replies. Pnyx makes that visible

---

## Part 5: Erscheinung / Blur (10s)

**Screen:** Scroll to the blurred post (Alex Neumann)

**Say:**
- This post is blurred
- The **Erscheinung model** - from Hannah Arendt - couldn't find genuine human presence behind the text
- Arendt's concept of the **Erscheinungsraum** (space of appearance): the public sphere requires a **who** behind the **what**
- AI-generated text simulates speech without a speaker. It discloses a *what* but never a *who*
- 85 heuristic patterns across 3 confidence tiers, sourced from PubMed vocabulary studies, WikiProject AI Cleanup, and the Pangram Labs DAMAGE paper
- Plus 8 hand-crafted linguistic features: type-token ratio, hapax rate, sentence variance, bigram uniqueness, contraction presence...
- Sub-millisecond inference. On-device. Anti-evasion via NFKC normalization
- It doesn't block the content - it **fades** it. Like fog on glass, not a wall
- You can hover to peek, click to reveal. The choice is yours

---

## Part 6: Network Tab (10s)

**Screen:** Open DevTools, Network tab. Zero outgoing requests.

**Say:**
- Everything you've seen so far - claim extraction, validity scoring, AI detection - ran entirely in your browser
- Open the network tab: **zero outgoing requests**
- 271MB of models loaded into your browser. Zero bytes sent out.
- No data left the browser. No server processed your feed. No API saw what you're reading
- This is what **digital sovereignty** looks like for democratic infrastructure
- The models are yours. The analysis is yours. Nobody can shut it down, change the rules, or revoke access
- Even if we disappeared tomorrow, Pnyx would keep working

---

## Part 7: Explore - Opening (15s)

**Screen:** Click "Explore" on a claim (e.g., "The infrastructure budget was cut by 15%"), full-page takeover, click "Begin deliberation"

**Say:**
- But sometimes you want to go deeper than scoring. You want to **think through a disagreement**
- Click Explore on any claim - full-screen deliberation mode
- This is grounded in **Chantal Mouffe's agonistic pluralism**
- Mouffe argues consensus is not always the goal. **Disagreement is a democratic resource**
- The goal is to transform antagonism (enemy-logic) into **agonism** (adversary-logic): the other position deserves to exist

---

## Part 8: Explore - Deliberation Rounds (20s)

**Screen:** Click through 2-3 rounds, choosing stance/concession/escalation

**Say (Round 1):**
- Three moves: **stance, concession, escalation**. No move is ranked above another
- Stance deepens your position: what else does this commit you to?
- Concession is the agonistic move: acknowledge the opposing position has legitimate ground. **It should cost something**
- Escalation follows the logic further than most people go. What happens if we take this seriously?

**Say (Round 2):**
- Notice: the deliberation **shifts validity dimensions**
- Round 1 was about truth claims (Wahrheit). Now it shifts to **whose perspective is centered** (Arendt)
- The system prompts encode all three theorists directly: Habermas for structure, Arendt for perspective, Mouffe for productive disagreement
- **Theory as code**

**Say (Round 3):**
- Final round surfaces the **fundamental tension** your choices reveal
- Not a resolution. Not a consensus. Clarity about where you stand and why
- This is Mouffe's insight made operational: the point of deliberation isn't agreement, it's **mutual intelligibility**

---

## Part 9: Close (10s)

**Screen:** Deliberation summary visible, or back to feed

**Say:**
- Pnyx. Four layers: **Detect** if there's a real person. **See** what's being claimed. **Pause** before you react. **Explore** the disagreement
- Three philosophers operationalized: **Habermas** for validity claims. **Arendt** for human presence. **Mouffe** for productive conflict
- No data leaves the browser. No server to shut down. No permission to ask
- This is **listening infrastructure for public discourse**

---

## Hard Numbers (drop these naturally)

### Habermas Model
- **Architecture:** DeBERTa-v3-small (~22M parameters)
- **Training data:** 10,000 samples (30K pre-train on argument quality + 10K synthetic DQI labels + 1,953 AQuA)
- **Heads:** 2 binary classification heads (Wahrheit + Richtigkeit)
- **Performance:** F1 0.974
- **Export:** ONNX format, 271MB model file
- **Inference:** Sub-second, entirely in-browser via ONNX Runtime Web
- **API calls:** Zero

### Erscheinung Model
- **Architecture:** DeBERTa + heuristic ensemble
- **Heuristic signals:** 85 regex patterns across 3 confidence tiers (high/medium/low)
- **Pattern sources:** WikiProject AI Cleanup, PubMed excess vocabulary study (280 words), Pangram Labs DAMAGE paper (19 humanizer tools), PNAS 2024 Biber feature analysis
- **Hand-crafted features:** 8 (TTR, hapax rate, sentence length variance, avg sentence length, bigram uniqueness, stop word density, contraction presence, lowercase ratio)
- **Inference:** Sub-millisecond for heuristics, on-device for DeBERTa
- **Defense:** NFKC normalization + zero-width character stripping (anti-evasion)

### Explore Mode
- **Framework:** Mouffe's agonistic pluralism
- **Structure:** 3 rounds, 3 moves per round (stance/concession/escalation)
- **Validity dimension progression:** Round 1 = Wahrheit (truth), Round 2 = Arendt (perspective), Round 3 = fundamental agonistic tension
- **System prompts:** Encode Habermas, Arendt, Mouffe directly
- **Backend:** DeepSeek via OpenRouter (only feature requiring API)

### Platform Stats
- **Extension size:** 453MB (includes ONNX models)
- **Data sent to any server:** 0 bytes (except Explore mode)
- **Supported platforms:** LinkedIn, X (via content script injection)
- **Esau 2025 stat:** 71% of online comments receive zero deliberative replies

### The Arendt Opening
- "Appearance constitutes reality." — Hannah Arendt, The Human Condition
- "So let's make it visible."
- Arendt's space of appearance: reality is constituted through plurality, through being seen
- Online, the seeing is invisible. Reading leaves no trace. The space of appearance became a space of disappearance
- Pnyx restores it: detects presence, makes claims visible, creates friction, structures disagreement

---

## Key phrases to hit (memorize these)

- "Discourse ethics made operational"
- "Wahrheit, Richtigkeit - what is being said, and is there argument structure?"
- "A who behind the what"
- "Listening has a trace"
- "71% of online comments receive zero deliberative replies"
- "DeBERTa, 10K samples, F1 0.974, entirely in-browser"
- "85 heuristic signals, sub-millisecond"
- "Disagreement is a democratic resource"
- "Stance, concession, escalation - no move is better than another"
- "Theory as code"
- "Zero outgoing requests"
- "No data leaves the browser. No server to shut down. No permission to ask."

---

## Theory-to-feature mapping (for Q&A)

| Theorist | Concept | Feature | Technical |
|----------|---------|---------|-----------|
| Habermas | Discourse ethics / validity claims | Claim extraction + scoring | DeBERTa-v3-small, 22M params, 2 binary heads, F1 0.974, ONNX |
| Habermas | Wahrheit (truth) | "Are factual claims being made?" | Binary classification head 1, 10K training samples |
| Habermas | Richtigkeit (rightness) | "Is there argument structure?" | Binary classification head 2 |
| Arendt | Erscheinungsraum | AI content detection / blur | DeBERTa + 85 heuristic patterns, 8 hand-crafted features, sub-ms |
| Arendt | "Who behind the what" | Content fades, doesn't block | 3-tier confidence (high/medium/low), NFKC + zero-width defense |
| Mouffe | Agonistic pluralism | Explore mode / deliberation | 3 rounds x 3 moves, dimension progression per round |
| Mouffe | Adversary vs enemy | "No move is better than another" | System prompts encode all 3 theorists directly |
| All | Digital sovereignty | On-device inference | ONNX Runtime Web, 271MB model, 0 bytes sent |

---

## Timing check

| Section | Duration | Cumulative |
|---------|----------|------------|
| Real LinkedIn | 5s | 0:05 |
| Feed load | 10s | 0:15 |
| Scroll + badges | 10s | 0:25 |
| PAUSE | 10s | 0:35 |
| Engagement traces | 10s | 0:45 |
| Erscheinung | 10s | 0:55 |
| Network tab | 10s | 1:05 |
| Explore opening | 15s | 1:20 |
| Deliberation rounds | 20s | 1:40 |
| Close | 10s | 1:50 |
| **Buffer for pace** | 10s | **2:00** |
| **Slides before/after** | 60s | **3:00** |
