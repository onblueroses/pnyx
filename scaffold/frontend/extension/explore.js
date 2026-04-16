/**
 * Pnyx Explore Mode
 * Branching deliberation through Chosen Path-inspired beats.
 * Three beats of stance/concession/escalation choices, then synthesis.
 */

const MAX_BEATS = 3;

const state = {
	claim: "",
	risk: "",
	beats: [],
	positionChain: [],
	coherence: { tensions: 0 },
	currentBeat: 0,
	phase: "opening", // opening | beat | interlude | summary
	apiConfig: {
		key: "",
		baseUrl: "https://openrouter.ai/api/v1",
		model: "deepseek/deepseek-chat-v3-0324",
	},
};

// --- Config ---

function loadConfig() {
	const saved = localStorage.getItem("pnyx-explore-config");
	if (saved) {
		try {
			const parsed = JSON.parse(saved);
			state.apiConfig = { ...state.apiConfig, ...parsed };
		} catch {
			/* ignore corrupt config */
		}
	}
	document.getElementById("config-key").value = state.apiConfig.key;
	document.getElementById("config-base-url").value = state.apiConfig.baseUrl;
	document.getElementById("config-model").value = state.apiConfig.model;
}

function saveConfig() {
	state.apiConfig.key = document.getElementById("config-key").value.trim();
	state.apiConfig.baseUrl = document
		.getElementById("config-base-url")
		.value.trim();
	state.apiConfig.model = document.getElementById("config-model").value.trim();
	localStorage.setItem("pnyx-explore-config", JSON.stringify(state.apiConfig));
	document.getElementById("config-panel").hidden = true;
	// Re-check if begin button should be enabled
	const beginBtn = document.querySelector(".explore-begin-btn");
	if (beginBtn) {
		beginBtn.disabled = !state.apiConfig.key;
	}
}

// --- URL Params ---

const VALID_RISK_LEVELS = ["low", "medium", "high"];

function initFromParams() {
	const params = new URLSearchParams(window.location.search);
	state.claim = params.get("claim") || "";
	const rawRisk = params.get("risk") || "medium";
	state.risk = VALID_RISK_LEVELS.includes(rawRisk) ? rawRisk : "medium";
	renderOpening();
}

// --- Rendering ---

function renderOpening() {
	state.phase = "opening";
	const root = document.getElementById("explore-root");
	const hasKey = !!state.apiConfig.key;

	const claimDisplay = state.claim
		? `<div class="explore-opening-claim">"${escapeHtml(state.claim)}"</div>
		   <div class="explore-opening-risk ${state.risk}">${state.risk} risk</div>`
		: `<textarea class="explore-opening-input" id="claim-input" placeholder="Paste a claim to explore..." rows="3"></textarea>`;

	root.innerHTML = `
    <div class="explore-opening">
      ${claimDisplay}
      <p class="explore-opening-desc">
        Explore this claim through three rounds of deliberation.
        Steel-man the argument, consider concessions, or escalate the stakes.
        Each choice shapes the next.
      </p>
      <button class="explore-begin-btn" ${hasKey && state.claim ? "" : "disabled"}>
        Begin Exploration
      </button>
      ${hasKey ? "" : '<p style="margin-top:12px;font-size:12px;color:var(--pnyx-primary-mid)">Configure your API key in Settings to begin.</p>'}
    </div>
  `;

	const beginBtn = root.querySelector(".explore-begin-btn");
	if (beginBtn) {
		beginBtn.addEventListener("click", () => beginExploration());
	}

	const claimInput = document.getElementById("claim-input");
	if (claimInput) {
		claimInput.addEventListener("input", () => {
			state.claim = claimInput.value.trim();
			const btn = root.querySelector(".explore-begin-btn");
			if (btn) btn.disabled = !(!!state.apiConfig.key && state.claim);
		});
	}

	renderChainSidebar();
}

function renderBeat(beatIndex) {
	state.phase = "beat";
	state.currentBeat = beatIndex;
	const beat = state.beats[beatIndex];
	if (!beat) return;

	const root = document.getElementById("explore-root");
	const isFirstBeat = beatIndex === 0;

	// Show all previous beats (collapsed) plus current
	let html = "";
	for (let i = 0; i < beatIndex; i++) {
		html += renderPastBeat(i);
	}

	html += `
    <div class="explore-beat" id="beat-${beatIndex}">
      <div class="explore-beat-label">Beat ${beatIndex + 1} of ${MAX_BEATS}</div>
      ${!isFirstBeat && beat.resolution ? `<div class="explore-beat-resolution">${escapeHtml(beat.resolution)}</div>` : ""}
      <div class="explore-beat-situation">${escapeHtml(beat.situation)}</div>
      <div class="explore-options">
        ${beat.options
					.map(
						(opt, i) => `
          <button class="explore-option option-${opt.type}" data-beat="${beatIndex}" data-option="${i}">
            <span class="explore-option-type">${opt.type}</span>
            <span class="explore-option-text">${escapeHtml(opt.text)}</span>
            <span class="explore-option-preview">${escapeHtml(opt.preview)}</span>
          </button>
        `,
					)
					.join("")}
      </div>
    </div>
  `;

	root.innerHTML = html;

	// Bind option click handlers
	root
		.querySelectorAll(".explore-option[data-beat][data-option]")
		.forEach((btn) => {
			btn.addEventListener("click", () => {
				selectOption(parseInt(btn.dataset.beat), parseInt(btn.dataset.option));
			});
		});

	// Scroll to the new beat
	document
		.getElementById(`beat-${beatIndex}`)
		?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPastBeat(beatIndex) {
	const beat = state.beats[beatIndex];
	if (!beat) return "";

	return `
    <div class="explore-beat" style="opacity:0.7">
      <div class="explore-beat-label">Beat ${beatIndex + 1} of ${MAX_BEATS}</div>
      ${beatIndex > 0 && beat.resolution ? `<div class="explore-beat-resolution">${escapeHtml(beat.resolution)}</div>` : ""}
      <div class="explore-beat-situation">${escapeHtml(beat.situation)}</div>
      <div class="explore-options">
        ${beat.options
					.map((opt, i) => {
						const isChosen = i === beat.chosenOption;
						const cls = isChosen ? "chosen" : "not-chosen";
						return `
            <button class="explore-option option-${opt.type} ${cls}" disabled>
              <span class="explore-option-type">${opt.type}</span>
              <span class="explore-option-text">${escapeHtml(opt.text)}</span>
            </button>
          `;
					})
					.join("")}
      </div>
    </div>
  `;
}

function renderInterlude() {
	state.phase = "interlude";
	const root = document.getElementById("explore-root");

	// Show past beats + interlude
	let html = "";
	for (let i = 0; i <= state.currentBeat; i++) {
		html += renderPastBeat(i);
	}

	html += `
    <div class="explore-interlude">
      <div class="explore-interlude-title">Considering your position...</div>
      <div class="explore-interlude-chain">
        ${state.positionChain
					.map((entry, i) => {
						const isLatest = i === state.positionChain.length - 1;
						return `
            <div class="explore-chain-entry ${isLatest ? "latest" : ""}">
              <span class="explore-chain-badge badge-${entry.choiceType}">${entry.choiceType}</span>
              <span>${escapeHtml(entry.summary)}</span>
            </div>
          `;
					})
					.join("")}
      </div>
      <div class="explore-loading-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;

	root.innerHTML = html;
	renderChainSidebar();
}

function renderChainSidebar() {
	const entries = document.getElementById("chain-entries");
	if (state.positionChain.length === 0) {
		entries.innerHTML =
			'<p style="font-size:13px;color:var(--li-secondary)">Your choices will appear here as you explore.</p>';
		return;
	}

	entries.innerHTML = state.positionChain
		.map(
			(entry, i) => `
    <div class="explore-chain-entry ${i === state.positionChain.length - 1 ? "latest" : ""}">
      <span class="explore-chain-badge badge-${entry.choiceType}">${entry.choiceType}</span>
      <span>${escapeHtml(entry.summary)}</span>
    </div>
  `,
		)
		.join("");
}

function renderSummary() {
	state.phase = "summary";
	const root = document.getElementById("explore-root");

	const chainHtml = state.positionChain
		.map(
			(entry) => `
    <div class="explore-chain-entry">
      <span class="explore-chain-badge badge-${entry.choiceType}">${entry.choiceType}</span>
      <span>${escapeHtml(entry.summary)}</span>
    </div>
  `,
		)
		.join("");

	// Group all options by type across beats
	const optionsByType = { stance: [], concession: [], escalation: [] };
	state.beats.forEach((beat, bi) => {
		beat.options.forEach((opt, oi) => {
			const chosen = beat.chosenOption === oi;
			optionsByType[opt.type]?.push({ ...opt, chosen, beatIndex: bi });
		});
	});

	const optionsHtml = ["stance", "concession", "escalation"]
		.map((type) => {
			const opts = optionsByType[type];
			if (!opts || opts.length === 0) return "";
			return `
      <div style="margin-bottom:12px">
        <span class="explore-chain-badge badge-${type}" style="font-size:11px">${type}</span>
        ${opts
					.map(
						(o) => `
          <div style="font-size:13px;padding:4px 0;${o.chosen ? "font-weight:500" : "color:var(--li-secondary)"}">
            ${o.chosen ? "> " : ""}${escapeHtml(o.text)}
          </div>
        `,
					)
					.join("")}
      </div>
    `;
		})
		.join("");

	const coherenceText =
		state.coherence.tensions === 0
			? "Your positions were internally consistent throughout."
			: state.coherence.tensions === 1
				? "One point of tension was detected in your deliberation."
				: `${state.coherence.tensions} points of tension emerged in your deliberation.`;

	const coherenceBg =
		state.coherence.tensions === 0
			? "background:var(--pnyx-surface);color:var(--pnyx-primary)"
			: "background:var(--pnyx-attention-light);color:var(--pnyx-attention)";

	root.innerHTML = `
    <div class="explore-summary">
      <div class="explore-summary-title">Deliberation Complete</div>
      <div class="explore-summary-claim">"${escapeHtml(state.claim)}"</div>

      <div class="explore-summary-section">
        <h3>Your Position Chain</h3>
        <div style="display:flex;flex-direction:column;gap:6px">${chainHtml}</div>
      </div>

      <div class="explore-summary-section">
        <h3>Coherence</h3>
        <div class="explore-summary-coherence" style="${coherenceBg}">${coherenceText}</div>
      </div>

      <div class="explore-summary-section">
        <h3>All Options Explored</h3>
        ${optionsHtml}
      </div>

      <div class="explore-summary-actions">
        <button class="explore-btn-restart">Start Over</button>
        <button class="explore-btn-back">Back to Feed</button>
      </div>
    </div>
  `;

	root
		.querySelector(".explore-btn-restart")
		?.addEventListener("click", () => restartExploration());
	root
		.querySelector(".explore-btn-back")
		?.addEventListener("click", () => window.close());

	renderChainSidebar();
}

function renderError(message, retryFn) {
	const root = document.getElementById("explore-root");

	// Preserve past beats
	let html = "";
	for (let i = 0; i <= state.currentBeat; i++) {
		if (state.beats[i]) html += renderPastBeat(i);
	}

	html += `
    <div class="explore-error">
      ${escapeHtml(message)}
      ${retryFn ? `<button class="explore-retry-btn">Retry</button>` : ""}
    </div>
  `;
	root.innerHTML = html;

	if (retryFn) {
		root
			.querySelector(".explore-retry-btn")
			?.addEventListener("click", retryFn);
	}
}

// --- Actions ---

async function beginExploration() {
	if (!state.apiConfig.key) {
		renderError("Please configure your API key in Settings.");
		return;
	}

	state.beats = [];
	state.positionChain = [];
	state.coherence = { tensions: 0 };
	state.currentBeat = 0;

	renderInterludeInitial();

	try {
		const beat = await generateBeat(0);
		state.beats[0] = beat;
		renderBeat(0);
	} catch (err) {
		renderError(`Failed to start exploration: ${err.message}`, () =>
			beginExploration(),
		);
	}
}

function renderInterludeInitial() {
	state.phase = "interlude";
	const root = document.getElementById("explore-root");
	root.innerHTML = `
    <div class="explore-interlude">
      <div class="explore-interlude-title">Preparing your exploration...</div>
      <div class="explore-loading-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
}

async function selectOption(beatIndex, optionIndex) {
	const beat = state.beats[beatIndex];
	if (!beat || beat.chosenOption !== undefined) return;

	beat.chosenOption = optionIndex;
	beat.choiceType = beat.options[optionIndex].type;

	// Update position chain
	state.positionChain.push({
		beat: beatIndex + 1,
		choiceType: beat.choiceType,
		summary: beat.options[optionIndex].text,
	});

	// Check if we've completed all beats
	if (beatIndex + 1 >= MAX_BEATS) {
		renderSummary();
		return;
	}

	// Show interlude, then generate next beat
	renderInterlude();

	try {
		const nextBeat = await generateBeat(beatIndex + 1);
		state.beats[beatIndex + 1] = nextBeat;

		// Update coherence after beat 2+
		if (beatIndex + 1 >= 2) {
			updateCoherence(nextBeat);
		}

		renderBeat(beatIndex + 1);
	} catch (err) {
		renderError(`Failed to generate next beat: ${err.message}`, () =>
			retryBeat(beatIndex + 1),
		);
	}
}

async function retryBeat(beatIndex) {
	renderInterlude();
	try {
		const beat = await generateBeat(beatIndex);
		state.beats[beatIndex] = beat;
		renderBeat(beatIndex);
	} catch (err) {
		renderError(`Failed to generate beat: ${err.message}`, () =>
			retryBeat(beatIndex),
		);
	}
}

function restartExploration() {
	state.beats = [];
	state.positionChain = [];
	state.coherence = { tensions: 0 };
	state.currentBeat = 0;
	renderOpening();
}

// --- API ---

async function callApi(messages) {
	const { key, baseUrl, model } = state.apiConfig;

	const response = await fetch(`${baseUrl}/chat/completions`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `Bearer ${key}`,
		},
		body: JSON.stringify({
			model,
			messages,
			response_format: { type: "json_object" },
			temperature: 0.8,
		}),
	});

	if (!response.ok) {
		const errText = await response.text().catch(() => "Unknown error");
		throw new Error(`API error ${response.status}: ${errText}`);
	}

	const data = await response.json();
	const content = data.choices?.[0]?.message?.content;
	if (!content) throw new Error("Empty API response");

	try {
		return JSON.parse(content);
	} catch {
		throw new Error("API returned invalid JSON");
	}
}

function buildSystemPrompt() {
	return `You are the deliberation engine for Pnyx, a platform that makes the structure of public discourse visible. You don't judge claims or push toward consensus. You help people SEE what they're actually arguing about.

Your theoretical foundation (use these as lenses, never as jargon in your output):

HABERMAS - Every claim implicitly raises validity dimensions:
- Wahrheit (truth): What factual claims are being made? What evidence would ground or undermine them?
- Richtigkeit (rightness): What norms or values does this argument assume everyone shares? Are those shared?
- Wahrhaftigkeit (sincerity): Is this position held authentically, or performed for an audience?
Analyze claims through these dimensions to find where the real disagreement lives. Most arguments fail because people argue about truth when they actually disagree about values, or vice versa.

ARENDT - The "space of appearance" requires plurality. Every position contains a "who" - a situated perspective that cannot be reduced to abstract logic. When someone claims "AI will replace jobs," the claim means something different from a displaced factory worker than from a tech executive. Surface whose experience is centered and whose is invisible.

AGONISTIC THEORY (Mouffe) - Disagreement is not a failure of deliberation; it IS deliberation. Don't engineer consensus. Instead, transform antagonism (enemy-logic) into agonism (adversary-logic): the other position deserves to exist and be heard, even if you oppose it. The goal is not agreement but mutual intelligibility.

YOUR TASK: Generate a deliberation beat - a moment of genuine choice that reveals the structure of an argument.

OUTPUT FORMAT - Valid JSON:
{
  "thinking": {
    "validity_dimensions": "Which Habermasian dimension is this claim ACTUALLY about? Truth, rightness, or sincerity? Where is the real disagreement hiding?",
    "hidden_assumptions": "What does this claim take for granted that could be questioned? Whose perspective does it center?",
    "steel_man": "The most intellectually honest version of this claim. Not a strawman dressed up - the version that would make a smart opponent pause.",
    "tensions": "What genuine tensions exist WITHIN this position (not just objections FROM outside)?"
  },
  "resolution": "One sentence narrating the epistemic consequence of the user's last choice. Not 'you chose X' but what that choice REVEALED or COMMITTED them to. Empty string for beat 1.",
  "situation": "2-4 sentences. Set up a genuine deliberative moment by exposing a tension the user hasn't considered yet. Don't restate the claim - push into the territory their previous choice opened up. Write as direct address ('You've committed to... but this means...'). Make the user feel the weight of their position.",
  "options": [
    {
      "text": "1-2 sentences. A genuine move in the deliberation, not a quiz answer.",
      "type": "stance",
      "preview": "Where this leads (under 8 words)"
    },
    {
      "text": "1-2 sentences. Must feel genuinely costly - not 'consider the other side' but 'accept that your position requires giving up...'",
      "type": "concession",
      "preview": "What this costs (under 8 words)"
    },
    {
      "text": "1-2 sentences. Push the claim into territory where its assumptions become visible and potentially unstable.",
      "type": "escalation",
      "preview": "The risk (under 8 words)"
    }
  ]
}

OPTION DESIGN:
- "stance": Deepen the current position by exploring a new validity dimension. If beat 1 was about truth, stance might explore the normative assumptions underneath. This is not "more of the same" - it's "what else does this commit you to?"
- "concession": The agonistic move. Acknowledge that the opposing position has legitimate ground - not as a rhetorical trick but because intellectual honesty demands it. This should feel like it COSTS something. The user should think "this weakens my position but I can't deny it."
- "escalation": Follow the claim's logic further than most people go. What happens if we take this seriously? This often reveals hidden assumptions by pushing past comfortable territory. Should feel bold and slightly dangerous.

QUALITY STANDARDS:
- Never present a false dilemma. All three options should be intellectually defensible.
- The situation text should make the user think "I hadn't considered that." If it doesn't surprise, it's not good enough.
- Concessions that don't cost anything are worthless. "Consider that some people disagree" is not a concession.
- Escalations that are just "the extreme version" are lazy. Good escalations follow the LOGIC to uncomfortable places.
- Resolution text should make the user feel the weight of what they chose. They committed to something - what?
- Write for someone who is smart but not academic. No jargon, but don't condescend.
- Each beat should shift the deliberation to a different validity dimension or perspective than the previous one.`;
}

function buildUserPrompt(beatIndex) {
	const chainContext =
		state.positionChain.length > 0
			? `\n\nPosition chain so far:\n${state.positionChain
					.map(
						(p) =>
							`- Beat ${p.beat}: [${p.choiceType.toUpperCase()}] "${p.summary}"`,
					)
					.join("\n")}`
			: "";

	const coherenceNote =
		beatIndex >= 2 && state.coherence.tensions > 0
			? `\n\nIMPORTANT: The user's position chain contains internal tension. Their choices don't fully cohere. Don't lecture them about this - instead, design the situation to make the tension FELT. Let them discover it through the options rather than being told.`
			: "";

	// Compute resolution guidance from previous choice
	let resolutionGuidance = "";
	if (beatIndex > 0) {
		const prevChoice = state.positionChain[state.positionChain.length - 1];
		const resolutionFrames = {
			stance: `The user chose to deepen their position with: "${prevChoice.summary}". The resolution should narrate what this commitment MEANS - what ground have they now staked out? What have they implicitly accepted by choosing this?`,
			concession: `The user made a concession: "${prevChoice.summary}". The resolution should narrate the epistemic cost - what did they give up? How does their overall position look now that they've acknowledged this? This is agonistic honesty, not defeat.`,
			escalation: `The user escalated: "${prevChoice.summary}". The resolution should narrate the exposure - they pushed further than most would. What assumptions are now visible that were hidden before? What are they now committed to defending?`,
		};
		resolutionGuidance = `\n\n${resolutionFrames[prevChoice.choiceType] || "Write the resolution to narrate the consequence of their choice."}`;
	}

	// Beat-specific framing to ensure progression
	const beatFrames = [
		// Beat 1: Surface the claim's structure
		"This is the OPENING beat. The user is encountering this claim fresh. Your job: reveal that this claim is more complex than it appears. Find the validity dimension where the real disagreement lives. Is this actually about facts (Wahrheit), about values (Richtigkeit), or about who is speaking (Wahrhaftigkeit)?",
		// Beat 2: Shift dimensions based on beat 1's choice
		"This is the MIDDLE beat. The user has committed to a direction. Now SHIFT the deliberation to a different validity dimension than beat 1 explored. If beat 1 was about empirical truth, push into normative territory. If it was about values, push into whose experience is centered. The user should feel the claim opening up, not narrowing down.",
		// Beat 3: Synthesize toward the fundamental tension
		"This is the FINAL beat. The user has a position chain showing their deliberative path. Now surface the FUNDAMENTAL tension that their choices reveal. This beat should feel like arriving at the core disagreement that was there all along. The options should represent genuinely different ways of holding that tension - not resolving it, but choosing how to live with it.",
	];

	return `Claim: "${state.claim}" (risk level: ${state.risk})
Beat: ${beatIndex + 1} of ${MAX_BEATS}${beatIndex === 0 ? " (first beat - leave resolution as empty string)" : ""}

${beatFrames[beatIndex]}${chainContext}${resolutionGuidance}${coherenceNote}`;
}

async function generateBeat(beatIndex) {
	const messages = [
		{ role: "system", content: buildSystemPrompt() },
		{ role: "user", content: buildUserPrompt(beatIndex) },
	];

	const result = await callApi(messages);

	// Validate response shape
	if (
		!result.situation ||
		!Array.isArray(result.options) ||
		result.options.length < 3
	) {
		throw new Error("Invalid beat structure from API");
	}

	for (const opt of result.options) {
		if (
			!opt.text ||
			!opt.type ||
			!["stance", "concession", "escalation"].includes(opt.type)
		) {
			throw new Error("Invalid option structure from API");
		}
	}

	return {
		situation: result.situation,
		resolution: result.resolution || "",
		options: result.options.slice(0, 3).map((opt) => ({
			text: opt.text,
			type: opt.type,
			preview: opt.preview || "",
		})),
		thinking: result.thinking || null,
		chosenOption: undefined,
		choiceType: undefined,
	};
}

function updateCoherence(beat) {
	if (!beat.thinking) return;

	// Scan all thinking fields for tension indicators
	const thinkingText = Object.values(beat.thinking).join(" ").toLowerCase();
	const tensionKeywords = [
		"contradict",
		"tension",
		"inconsisten",
		"conflict",
		"at odds",
		"reversed",
		"incompatible",
		"undermines",
		"cannot simultaneously",
		"incoher",
	];
	const found = tensionKeywords.some((kw) => thinkingText.includes(kw));
	if (found) {
		state.coherence.tensions++;
	}
}

// --- Utilities ---

function escapeHtml(str) {
	const div = document.createElement("div");
	div.textContent = str;
	return div.innerHTML;
}

// --- Init ---
document.addEventListener("DOMContentLoaded", () => {
	loadConfig();

	document.getElementById("config-toggle").addEventListener("click", () => {
		const panel = document.getElementById("config-panel");
		panel.hidden = !panel.hidden;
	});

	document.getElementById("config-save").addEventListener("click", saveConfig);

	initFromParams();
});
