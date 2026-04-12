# Pnyx Design Language

## The Core Question

What kind of presence should Pnyx have in someone's life?

Pnyx is **infrastructure, not destination**. Like a good pair of reading glasses: it changes how you see, not what you see. You don't open Pnyx - Pnyx is already there when you're about to reply. It doesn't ask for your attention. It asks for your attention to shift - from *what you want to say* to *what was actually said*.

This is the central design principle: **Pnyx is a lens, not a platform.**

---

## Role in a Person's Life

Pnyx sits at the moment between reading and reacting. It occupies 3-5 seconds of a person's day per interaction. It does not need to be remembered, liked, or opened. It needs to be *trusted* - trusted enough that when it surfaces a claim, the person looks at it rather than dismissing it.

This means:
- **No personality.** Pnyx doesn't have opinions, moods, or a mascot. It has precision.
- **No brand presence beyond identification.** The word "Pnyx" appears once in the nav badge and once in the loading screen. Nowhere else.
- **No reward loops.** No streaks, no "you listened to 5 posts today!", no gamification. Listening is not a metric to optimize.
- **Subordinate visual weight.** At rest, Pnyx is lighter than the host platform. Only during PAUSE does it become the heaviest thing on screen - because that's the moment it's earning its existence.

**The hierarchy of visual weight across layers:**

| Layer | Visual Weight | Why |
|-------|--------------|-----|
| SEE (badges, claims) | Whisper | Ambient information. Like footnotes - there if you look, invisible if you don't. |
| PAUSE (overlay) | Command | The one moment Pnyx steps forward. Brief, warm, clear. Not a wall - a hand on the shoulder. |
| SHOW (reply tags) | Murmur | Evidence of listening. Quieter than the reply itself. Like a citation, not a headline. |
| EXPLORE (deliberation) | Conversation | A separate space (new page). Can afford full voice. But still restrained - the ideas should be louder than the UI. |

---

## How Theory Shapes the Visual

### Habermas: Legibility Without Judgment

The Habermas model extracts discourse structure. The design must show *what* without implying *whether*.

**Current problem:** The traffic-light system (red/amber/green for risk levels) implies judgment. Red = bad. Green = good. This contradicts the theoretical position: Pnyx makes structure visible, not quality verdicts.

**Design principle:** Use **saturation and density** to signal discourse complexity, not color-coded quality.

- A post with many unsupported claims is *complex*, not *bad*
- A post with strong argument structure is *structured*, not *good*
- The visual signal should be "there is something to look at here" vs "this is fine, move along"
- Use a single hue at varying intensities rather than a traffic-light palette

### Arendt: Genuine Presence

The Erscheinung model detects whether there's a genuine human presence behind the text. The design must signal *absence of a who* without making a certainty claim.

**Current approach (good):** AI-detected content gets blurred. The blur is the right metaphor - it says "we can't see a person here" without saying "this is fake." The warm yellow overlay (`rgba(255,252,240,0.7)`) with "Wahrhaftigkeit concern" is theoretically precise.

**Refinement:** The blur should feel like fog, not censorship. Soft edges, gradual. The reveal-on-click should feel like wiping condensation from glass, not lifting a curtain.

### Mouffe: Conflict as Resource

Agonistic pluralism means disagreement is not failure. The design must never color-code opposing positions as right/wrong.

**Current problem:** Explore mode uses blue for stance, amber for concession, red for escalation. This implies stance = neutral, concession = caution, escalation = danger. But escalation can be the most productive move - pressing a point that needs pressing.

**Design principle:** The three deliberation moves need **distinct but non-hierarchical** visual treatment. No move is better than another. Consider:
- Three colors of equal visual weight, none of which carry inherent positive/negative connotation
- Or: a single color family (teal/slate) with different markers (line pattern, icon, position) rather than color for type differentiation

---

## Color Palette

### Philosophy

Move away from the traffic-light model entirely. The new palette has:
1. **One identity color** - Espresso Dark. The color of Pnyx itself.
2. **One attention color** - Aged Gold. Used only for the PAUSE friction moment.
3. **Neutral territory** - everything else uses the espresso family or achromatic.

Coffee houses were the birthplace of public discourse in Europe. The Viennese Kaffeehaus, the London coffeehouses, the Parisian cafes - all were Pnyx-like spaces where strangers argued. The color carries this lineage.

### Locked Palette

**Primary: Espresso Dark** `#3E2C20`
- Deep, concentrated, bitter-warm. Not corporate, not startup, not governmental.
- Distinct from every major social platform's palette.
- Used for: dots, buttons, text in Pnyx elements, borders, identity marks.

**Attention: Aged Gold** `#7C5E10`
- Tarnished, ancient. The Pnyx hill in sunset. The gold of old civic institutions.
- Used ONLY for the PAUSE overlay and Erscheinung warnings. Nowhere else.
- Light variant for PAUSE backgrounds: `#FFFBEB`.

**Full token system:**

| Token | Value | Use |
|-------|-------|-----|
| `--pnyx-primary` | `#3E2C20` | Identity color. Buttons, dots, borders, headings. |
| `--pnyx-primary-deep` | `#2E1F15` | Hover/emphasis. Darkest espresso. |
| `--pnyx-primary-mid` | `#6B5442` | Secondary text, icons. |
| `--pnyx-primary-light` | `#8B7355` | Tertiary, muted labels. |
| `--pnyx-surface` | `#F8F4F0` | Tag backgrounds, surfaces. "Crema." |
| `--pnyx-attention` | `#7C5E10` | PAUSE only. Aged gold. |
| `--pnyx-attention-light` | `#FFFBEB` | PAUSE overlay background. |
| `--pnyx-border` | `rgba(0,0,0,0.08)` | Borders - opacity, not solid. Adapts to host. |
| `--pnyx-shadow` | `0 1px 3px rgba(0,0,0,0.06)` | PAUSE card only. Slight lift. |

**Discourse indicators (replacing traffic light):**

No color-coded risk levels. The badge shows factual structure, not quality judgment:
- Claim count (number)
- Structure presence ("structured" / no label)
- Human presence ("human" / Erscheinung concern)

No color variation for these states. All use `--pnyx-primary` on `--pnyx-surface`.
After replies exist, "heard/unheard" counts appear - making the listening gap visible.

**Explore Deliberation Moves:**

Three colors of equal weight, none carrying inherent judgment. All in the espresso/earth family:

| Move | Color | Rationale |
|------|-------|-----------|
| Stance | `#3E2C20` (espresso) | Holding your ground. Same as base - your position. |
| Concession | `#7C5E10` (aged gold) | Opening. The warmest move. |
| Escalation | `#4A5043` (dark sage) | Pressing deeper. Green-earth, not red-danger. |

---

## Typography

### System

| Element | Font | Size | Weight | Why |
|---------|------|------|--------|-----|
| Badges/tags | System stack | 11-12px | 500 | Must blend with host platform. System fonts match the surrounding UI. |
| Extracted claims | `'JetBrains Mono', 'SF Mono', 'Consolas', monospace` | 13px | 400 | Monospace signals "this was parsed/extracted" - a discrete unit, not flowing text. Creates visual separation from the post's own words. |
| PAUSE header | Inter | 15px | 600 | The one moment of authority. Slightly larger, heavier. |
| PAUSE claim text | Inter | 14px | 400 | Readable, unhurried. Generous line-height (1.6). |
| Reply tags | System stack | 11px | 400 | Quieter than the reply text. Like a citation marker. |
| Explore body | Inter | 15px | 400 | Full reading context. Line-height 1.65. Max-width 65ch. |
| Explore labels | Inter | 11px | 600 | UPPERCASE, letter-spacing 0.5px. Structural labels. |

### Key rule: monospace for extracted content

When Pnyx shows you what *it* extracted from someone's post, it uses monospace. This is the visual signal that says "this is what the machine saw" - distinct from the human's original words. It's the difference between quoting someone (their voice) and indexing them (our parsing).

---

## Space and Density

### Injection footprint

Pnyx injects into existing platforms. The footprint must be minimal at rest:

- **Badge row**: 28px total height. Flush with post card bottom. No additional margin.
- **Claim expansion**: Slides down, max 120px. Dashed top border (not solid - signals "supplementary").
- **PAUSE overlay**: Appears WITHIN the post card's reply area, not as a modal over the page. Maximum 200px height. Contained.
- **Reply tags**: Single line, 18px height. Left border accent (2px), not full background color.

### Whitespace philosophy

- **Inside Pnyx elements**: Generous. 16px padding minimum. Claims get breathing room.
- **Between Pnyx and host**: Tight. Pnyx should feel attached to the post it annotates, not floating separately.
- **In Explore mode**: Very generous. 32px padding in cards. 1.65 line-height. This is a thinking space.

---

## Shadows and Borders

- **No drop shadows** on injected elements. Shadows signal "I'm above the page" - Pnyx is *part of* the page.
- **Borders**: Use `rgba(0,0,0,0.08)` universally. Adapts to any host background without creating hard edges.
- **PAUSE card only**: `box-shadow: 0 1px 3px rgba(0,0,0,0.06)` - the slightest lift, because PAUSE is the one moment of assertion.
- **Explore cards**: `box-shadow: 0 1px 2px rgba(0,0,0,0.04)` - barely there. Ideas should be heavier than containers.

---

## Animation and Motion

- **Badge row**: No animation. Static. Information at rest.
- **Claim expansion**: `200ms ease-out`, slide down. Quick, not bouncy.
- **PAUSE overlay**: `250ms ease-out`, slide down from reply button position. Feels like it unfolds from the action you took.
- **PAUSE dismiss**: `150ms ease-in`, fade + slide up. Faster out than in.
- **Explore beats**: `400ms ease`, fade + translateY(8px). Thoughtful arrival - things take a moment to appear because the AI is composing.
- **No pulse animations on persistent elements.** Nav badge is steady state, no animation.

---

## What Pnyx Is Not (Visual)

- **Not Material Design.** No FABs, no ripple effects, no elevated surfaces. Too playful.
- **Not Notion-minimal.** Not a blank canvas. Pnyx has opinions (about structure, about listening) and they should be legible.
- **Not governmental/institutional.** No crests, no official-looking badges. Pnyx is civic but grassroots.
- **Not dark mode.** For the hackathon, light only. Dark mode is a polish feature.
- **Not illustration-heavy.** No hero images, no mascots, no decorative SVGs. Typography and structure carry the identity.

---

## The Loading Screen

The loading screen is the only moment Pnyx gets to establish identity (users see it for 2-5 seconds while models load).

Current: LinkedIn-styled card with progress bars. Functional but generic.

Proposed:
- Center-screen, minimal card. White, thin border, slight shadow.
- "Pnyx" in 20px Inter weight 700. Nothing else in the header - no tagline, no icon.
- Model loading with German theoretical terms (already done - Erscheinung, Habermas).
- The loading description lines are good: "Detecting genuine human presence in discourse" / "Mapping claims in public discourse". These do the identity work.
- Footer: "On-device inference - no data leaves your browser" is the trust statement.
- Consider: subtle teal accent on progress bars instead of LinkedIn blue.

---

## The Explore Page

Explore is the one part of Pnyx that is a destination (separate page). Here Pnyx can have full voice.

- **Background**: `#F7FAFC` (slightly blue-grey off-white) instead of LinkedIn's warm beige. Signals: different context, thinking space.
- **Cards**: White, generous padding (32px), max-width 65ch for the reading column.
- **The deliberation choices**: Cards with left border accent (3px) in the move's color. Not full background tinting - that's too heavy for a choice that hasn't been made yet. On hover, the lightest background tint.
- **Position chain sidebar**: Minimal. Just colored dots (6px) + short text. A breadcrumb trail, not a dashboard.
- **Summary/reflection**: The most spacious element. This is where the user sits with what they've explored.

---

## Implementation Priority (Hackathon)

For the pitch demo, focus on:

1. **Replace traffic-light colors** with teal density system (biggest philosophical win, visual coherence)
2. **Teal accent** on Pnyx elements instead of LinkedIn blue (brand differentiation)
3. **Monospace for extracted claims** (instant visual signal of "parsed structure")
4. **PAUSE overlay warmth** - amber attention color for the friction moment
5. **Remove nav badge pulse** - infrastructure doesn't pulse

Lower priority:
- Explore page color refinement
- Shadow/border opacity normalization
- Typography refinement beyond monospace claims

---

## CSS Variable Migration

From current (`--ag-*` and `--li-*` mixed) to locked (`--pnyx-*` own identity):

```
/* Pnyx identity - Espresso Dark */
--pnyx-primary: #3E2C20;
--pnyx-primary-deep: #2E1F15;
--pnyx-primary-mid: #6B5442;
--pnyx-primary-light: #8B7355;
--pnyx-surface: #F8F4F0;        /* "Crema" */
--pnyx-surface-hover: rgba(62,44,32,0.04);

/* Attention - Aged Gold (PAUSE only) */
--pnyx-attention: #7C5E10;
--pnyx-attention-light: #FFFBEB;

/* Structure */
--pnyx-border: rgba(0,0,0,0.08);
--pnyx-shadow: 0 1px 3px rgba(0,0,0,0.06);

/* Deliberation moves */
--pnyx-stance: #3E2C20;         /* espresso - holding ground */
--pnyx-stance-light: #F8F4F0;
--pnyx-concession: #7C5E10;     /* aged gold - opening */
--pnyx-concession-light: #FFFBEB;
--pnyx-escalation: #4A5043;     /* dark sage - pressing deeper */
--pnyx-escalation-light: #f0f4ed;

/* Host platform (keep separate - adapts per platform) */
--host-bg: #f4f2ee;
--host-card: #ffffff;
--host-border: #e0dfdb;
--host-text: #191919;
--host-secondary: #666666;
--host-link: #0a66c2;
```

The `--host-*` variables remain for mimicking the host platform's structure. The `--pnyx-*` variables are for everything Pnyx adds.
