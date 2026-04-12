/**
 * Content script - scans feeds, requests scores from offscreen, injects badges.
 * Patterns adapted from linkedin-detox: MutationObserver + content hashing + WeakSet.
 */

const PLATFORM_SELECTORS = {
	"www.linkedin.com": {
		postContainer: '[data-testid="mainFeed"] [role="listitem"]',
		postText: '[data-testid="expandable-text-box"]',
		injectAfter: null, // resolved dynamically via findSocialBar()
		commentButton: null, // resolved dynamically via findCommentButton()
		commentBox: '[contenteditable="true"]',
	},
	"x.com": {
		postContainer: 'article[data-testid="tweet"]',
		postText: '[data-testid="tweetText"]',
		injectAfter: '[role="group"]',
		commentButton: '[data-testid="reply"]',
		commentBox: '[data-testid="tweetTextarea_0"]',
	},
};

const platform = PLATFORM_SELECTORS[window.location.hostname];
if (!platform) {
	console.log("Pnyx Lens: unsupported platform", window.location.hostname);
}

const scoreCache = new Map(); // hash -> scores (cap at 2000)
const analyzedElements = new WeakSet();
const pendingElements = new WeakSet(); // guard against duplicate processPost during async scoring
const postScores = new WeakMap(); // postEl -> scores
const postTexts = new WeakMap(); // postEl -> text
const pauseTimers = new Map(); // post hash -> intervalId

// Thresholds from popup (defaults match background.js)
let thresholds = { habermas: 60, erscheinung: 50 };
chrome.storage.local.get({ habermas: 60, erscheinung: 50 }, (t) => {
	thresholds = t;
});
chrome.storage.onChanged.addListener((changes) => {
	if (changes.habermas) thresholds.habermas = changes.habermas.newValue;
	if (changes.erscheinung)
		thresholds.erscheinung = changes.erscheinung.newValue;
});

// V2: Layer toggle settings
const layerSettings = { pauseEnabled: true, tagsEnabled: true };
chrome.storage.local.get({ pauseEnabled: true, tagsEnabled: true }, (t) => {
	layerSettings.pauseEnabled = t.pauseEnabled;
	layerSettings.tagsEnabled = t.tagsEnabled;
});
chrome.storage.onChanged.addListener((changes) => {
	if (changes.pauseEnabled)
		layerSettings.pauseEnabled = changes.pauseEnabled.newValue;
	if (changes.tagsEnabled)
		layerSettings.tagsEnabled = changes.tagsEnabled.newValue;
});

// LinkedIn uses obfuscated class names - find elements by text content and structure
function findCommentButton(postEl) {
	return [...postEl.querySelectorAll("button")].find(
		(b) => b.innerText.trim() === "Comment",
	);
}

function findSocialBar(postEl) {
	const commentBtn = findCommentButton(postEl);
	if (!commentBtn) return null;
	// Walk up to find the container with Like/Comment/Repost/Send (4 buttons)
	let p = commentBtn.parentElement;
	while (
		p &&
		p !== postEl &&
		p.querySelectorAll(":scope > button, :scope > div > button").length < 3
	) {
		p = p.parentElement;
	}
	return p !== postEl ? p : null;
}

function hashText(text) {
	let h = 0;
	for (let i = 0; i < text.length; i++) {
		h = ((h << 5) - h + text.charCodeAt(i)) | 0;
	}
	return `${h >>> 0}:${text.length}`;
}

function extractText(postEl) {
	const textEl = postEl.querySelector(platform.postText);
	if (!textEl) return "";
	return textEl.innerText.trim().slice(0, 1500); // cap at 1500 chars
}

function crLabel(p) {
	const hi = thresholds.habermas / 100;
	const lo = 1 - hi;
	if (p > hi)
		return { label: "W: high", cls: "pnyx-danger", full: "Wahrheit: high" };
	if (p >= lo)
		return { label: "W: med", cls: "pnyx-warn", full: "Wahrheit: medium" };
	return { label: "W: low", cls: "pnyx-good", full: "Wahrheit: low" };
}

function aqLabel(p) {
	if (p > 0.65)
		return {
			label: "R: strong",
			cls: "pnyx-good",
			full: "Richtigkeit: strong",
		};
	if (p >= 0.35)
		return { label: "R: mixed", cls: "pnyx-warn", full: "Richtigkeit: mixed" };
	return { label: "R: weak", cls: "pnyx-danger", full: "Richtigkeit: weak" };
}

// ─── V2: Claim extraction (inline, same algo as demo page) ───
function extractClaims(text, claimRiskProb) {
	if (!text || text.length < 20) return [];
	const ABBREV = /(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|vs|etc|i\.e|e\.g)\./gi;
	const normalized = text.replace(ABBREV, (m) => m.replace(".", "\u0000"));
	const sentences = normalized
		.split(/(?<=[.!?])\s+/)
		.map((s) => s.replace(/\u0000/g, ".").trim())
		.filter((s) => s.length > 10);
	if (sentences.length === 0) return [];
	const risk =
		claimRiskProb > 0.65 ? "high" : claimRiskProb >= 0.35 ? "medium" : "low";
	const scored = sentences.map((sentence, index) => {
		const lower = sentence.toLowerCase();
		let score = 0;
		const assertions = [
			"proves",
			"shows",
			"confirms",
			" is ",
			" was ",
			" are ",
			" will ",
			"demonstrates",
		];
		score += assertions.filter((v) => lower.includes(v)).length * 2;
		const hedges = [
			"may ",
			"might",
			"suggest",
			"appears",
			"could ",
			"possibly",
			"perhaps",
			"seems",
		];
		score -= hedges.filter((h) => lower.includes(h)).length * 3;
		const causal = [
			"because",
			"therefore",
			"due to",
			"caused by",
			"as a result",
			"which means",
		];
		score += causal.filter((c) => lower.includes(c)).length * 3;
		const absolutes = [
			"always",
			"never",
			"everyone",
			"nobody",
			"all ",
			"none ",
			"every ",
			"no one",
		];
		score += absolutes.filter((a) => lower.includes(a)).length * 4;
		if (/\d/.test(sentence)) score += 1;
		if (sentence.trim().endsWith("?")) score -= 10;
		return { text: sentence, risk, index, score };
	});
	return scored
		.filter((s) => s.score >= 1)
		.sort((a, b) => b.score - a.score)
		.slice(0, 3)
		.map(({ text, risk, index }) => ({ text, risk, index }));
}

function createBadge(scores, text) {
	const cr = crLabel(scores.habermas.claimRiskProb);
	const aq = aqLabel(scores.habermas.argQualityProb);
	const crPct = Math.round(scores.habermas.claimRiskProb * 100);
	const aqPct = Math.round(scores.habermas.argQualityProb * 100);
	const claims = extractClaims(text || "", scores.habermas.claimRiskProb);
	const claimCount = claims.length;

	const aiThresh = thresholds.erscheinung / 100;
	let aiHtml = "";
	if (scores.erscheinung.aiProb >= aiThresh) {
		const ind = scores.erscheinung.tier === "heuristic" ? " (pattern)" : "";
		aiHtml = `<span class="pnyx-dot">&middot;</span><span class="pnyx-tag pnyx-warn">Wahrhaftigkeit: AI-likely${ind}</span>`;
	} else if (scores.erscheinung.aiProb >= aiThresh * 0.6) {
		aiHtml = `<span class="pnyx-dot">&middot;</span><span class="pnyx-tag pnyx-muted">Wahrhaftigkeit: uncertain</span>`;
	}

	// Build claims list HTML
	let claimsHtml = "";
	if (claims.length > 0) {
		const claimItems = claims
			.map((c) => {
				const riskCls =
					c.risk === "high" ? "danger" : c.risk === "medium" ? "warn" : "good";
				const riskText =
					c.risk === "high"
						? "unsupported"
						: c.risk === "medium"
							? "unverified"
							: "grounded";
				const exploreUrl =
					chrome.runtime.getURL("explore.html") +
					"?claim=" +
					encodeURIComponent(c.text) +
					"&risk=" +
					c.risk;
				return `<li class="pnyx-claim-item">
					<span class="pnyx-claim-text">"${escapeHtml(c.text)}"</span>
					<span class="pnyx-claim-risk ${riskCls}">${riskText}</span>
					<a class="pnyx-explore-link" href="${exploreUrl}" target="_blank">Explore</a>
				</li>`;
			})
			.join("");
		claimsHtml = `<div class="pnyx-claims-expanded" style="display:none">
			<div class="pnyx-claims-label">Claims detected:</div>
			<ul class="pnyx-claims-list">${claimItems}</ul>
		</div>`;
	}

	const wrapper = document.createElement("div");
	wrapper.className = "pnyx-badge-wrapper";
	wrapper.innerHTML = `
    <div class="pnyx-badge-row" style="cursor:pointer" title="Click to expand claims">
      <span class="pnyx-tag ${cr.cls}">${cr.full} ${crPct}%</span>
      <span class="pnyx-dot">&middot;</span>
      <span class="pnyx-tag ${aq.cls}">${aq.full} ${aqPct}%</span>
      ${aiHtml}
      <span class="pnyx-dot">&middot;</span>
      <span class="pnyx-tag pnyx-good">${claimCount} claim${claimCount === 1 ? "" : "s"}</span>
    </div>
    ${claimsHtml}
  `;

	// Toggle claims on badge click
	const badgeRow = wrapper.querySelector(".pnyx-badge-row");
	const claimsPanel = wrapper.querySelector(".pnyx-claims-expanded");
	if (badgeRow && claimsPanel) {
		badgeRow.addEventListener("click", () => {
			const visible = claimsPanel.style.display !== "none";
			claimsPanel.style.display = visible ? "none" : "block";
		});
	}

	return wrapper;
}

async function scoreText(text) {
	return new Promise((resolve, reject) => {
		chrome.runtime.sendMessage(
			{ type: "score", target: "offscreen", text },
			(response) => {
				if (chrome.runtime.lastError) {
					reject(new Error(chrome.runtime.lastError.message));
				} else {
					resolve(response);
				}
			},
		);
	});
}

async function processPost(postEl) {
	if (pendingElements.has(postEl)) return;
	pendingElements.add(postEl);

	const text = extractText(postEl);
	if (!text || text.length < 20) return;

	const hash = hashText(text);

	// Use cached score or fetch new one
	let scores;
	if (scoreCache.has(hash)) {
		scores = scoreCache.get(hash);
	} else {
		// Cap cache
		if (scoreCache.size > 2000) {
			const iter = scoreCache.keys();
			for (let i = 0; i < 500; i++) scoreCache.delete(iter.next().value);
		}
		try {
			scores = await scoreText(text);
			scoreCache.set(hash, scores);
		} catch (e) {
			console.warn("Pnyx: scoring failed", e.message);
			return;
		}
	}

	try {
		const injectTarget = platform.injectAfter
			? postEl.querySelector(platform.injectAfter)
			: findSocialBar(postEl);
		if (!injectTarget) return;
		const badge = createBadge(scores, text);
		injectTarget.insertAdjacentElement("afterend", badge);
		analyzedElements.add(postEl);

		// Store scores and text for V2 Pause Layer
		postScores.set(postEl, scores);
		postTexts.set(postEl, text);

		// Blur AI-likely posts
		const aiThresh = thresholds.erscheinung / 100;
		if (scores.erscheinung.aiProb >= aiThresh) {
			const textEl = postEl.querySelector(platform.postText);
			if (textEl) {
				textEl.classList.add("pnyx-ai-blur");
				textEl.addEventListener(
					"click",
					() => {
						textEl.classList.remove("pnyx-ai-blur");
					},
					{ once: true },
				);
			}
		}

		// V2: Intercept comment/reply button for Pause Layer
		{
			const commentBtn = platform.commentButton
				? postEl.querySelector(platform.commentButton)
				: findCommentButton(postEl);
			if (commentBtn) {
				commentBtn.addEventListener(
					"click",
					(e) => {
						// Skip if pause disabled or just completed
						if (!layerSettings.pauseEnabled) return;
						if (postEl.dataset.pauseCompleted) return;

						const postText = postTexts.get(postEl) || extractText(postEl);
						const postScoresObj = postScores.get(postEl);
						if (!postScoresObj) return;

						const claims = extractClaims(
							postText,
							postScoresObj.habermas.claimRiskProb,
						);
						if (claims.length === 0) return;

						e.preventDefault();
						e.stopPropagation();

						if (postEl.querySelector(".pnyx-pause-overlay")) return;

						showPauseOverlay(postEl, claims);
					},
					true,
				); // capture phase
			}
		}
	} catch (e) {
		console.warn("Pnyx: badge injection failed for post", e.message);
	}
}

// ─── V2: Pause overlay ─────────────────────────────────────

function escapeHtml(str) {
	return str
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}

function showPauseOverlay(postEl, claims) {
	const maxClaims = claims.slice(0, 3);

	const overlay = document.createElement("div");
	overlay.className = "pnyx-pause-overlay";
	overlay.innerHTML = `
    <div class="pnyx-pause-card">
      <div class="pnyx-pause-header">Before you respond</div>
      <div class="pnyx-pause-subtitle">Here's what they said:</div>
      <div class="pnyx-pause-claims">
        ${maxClaims
					.map((c, i) => {
						const riskCls =
							c.risk === "high"
								? "danger"
								: c.risk === "medium"
									? "warn"
									: "good";
						const riskText =
							c.risk === "high"
								? "unsupported"
								: c.risk === "medium"
									? "unverified"
									: "grounded";
						return `<label class="pnyx-pause-claim">
              <input type="checkbox" data-claim-index="${i}">
              <span>"${escapeHtml(c.text)}"</span>
              <span class="pnyx-claim-risk ${riskCls}">${riskText}</span>
            </label>`;
					})
					.join("")}
      </div>
      <div class="pnyx-pause-prompt">Which are you responding to?</div>
      <div class="pnyx-pause-actions">
        <button class="pnyx-pause-continue" disabled>Continue to Reply</button>
        <a class="pnyx-pause-skip" href="#">Skip</a>
        <span class="pnyx-pause-timer">\u23F1 0:00</span>
      </div>
    </div>
  `;

	// Insert after badge
	const badge = postEl.querySelector(".pnyx-badge-row");
	if (badge) {
		badge.after(overlay);
	} else {
		const injectTarget = postEl.querySelector(platform.injectAfter);
		if (injectTarget) injectTarget.after(overlay);
	}

	// Enable/disable Continue
	const continueBtn = overlay.querySelector(".pnyx-pause-continue");
	const claimsContainer = overlay.querySelector(".pnyx-pause-claims");
	claimsContainer.addEventListener("change", () => {
		const checked = claimsContainer.querySelectorAll("input:checked");
		continueBtn.disabled = checked.length === 0;
	});

	// Continue action
	continueBtn.addEventListener("click", () => {
		const checked = claimsContainer.querySelectorAll("input:checked");
		const selectedTexts = Array.from(checked).map((cb) => {
			return maxClaims[parseInt(cb.dataset.claimIndex)].text;
		});
		dismissPauseOverlay(postEl);
		reopenNativeComment(postEl);
		if (layerSettings.tagsEnabled && selectedTexts.length > 0) {
			const tagText =
				selectedTexts[0].length > 50
					? selectedTexts[0].substring(0, 47) + "..."
					: selectedTexts[0];
			pollForReplyTag(postEl, tagText);
		}
	});

	// Skip action
	overlay.querySelector(".pnyx-pause-skip").addEventListener("click", (e) => {
		e.preventDefault();
		dismissPauseOverlay(postEl);
		reopenNativeComment(postEl);
	});

	// Timer (counts up)
	let elapsed = 0;
	const timerEl = overlay.querySelector(".pnyx-pause-timer");
	const hash = hashText(extractText(postEl) || "unknown");
	const interval = setInterval(() => {
		elapsed++;
		const m = Math.floor(elapsed / 60);
		const s = (elapsed % 60).toString().padStart(2, "0");
		timerEl.textContent = `\u23F1 ${m}:${s}`;
		if (elapsed >= 15) {
			dismissPauseOverlay(postEl);
			reopenNativeComment(postEl);
		}
	}, 1000);
	pauseTimers.set(hash, interval);
}

function dismissPauseOverlay(postEl) {
	const overlay = postEl.querySelector(".pnyx-pause-overlay");
	if (overlay) overlay.remove();
	const hash = hashText(extractText(postEl) || "unknown");
	const interval = pauseTimers.get(hash);
	if (interval) {
		clearInterval(interval);
		pauseTimers.delete(hash);
	}
}

function reopenNativeComment(postEl) {
	const commentBtn = platform.commentButton
		? postEl.querySelector(platform.commentButton)
		: findCommentButton(postEl);
	if (commentBtn) {
		postEl.dataset.pauseCompleted = "true";
		commentBtn.click();
		delete postEl.dataset.pauseCompleted;
	}
}

function pollForReplyTag(postEl, claimText, attempts = 0) {
	if (attempts > 10) return; // give up after ~2s
	setTimeout(() => {
		const box =
			postEl.querySelector(platform.commentBox) ||
			postEl.parentElement?.querySelector(platform.commentBox) ||
			document.querySelector(platform.commentBox);
		if (box) {
			tryInjectReplyTag(postEl, claimText);
		} else {
			pollForReplyTag(postEl, claimText, attempts + 1);
		}
	}, 200);
}

function tryInjectReplyTag(postEl, claimText) {
	const commentBox =
		postEl.querySelector(platform.commentBox) ||
		postEl.parentElement?.querySelector(platform.commentBox) ||
		document.querySelector(platform.commentBox); // X.com renders reply in a portal/dialog
	if (!commentBox) return;

	const textInput =
		commentBox.querySelector('[contenteditable="true"]') ||
		commentBox.querySelector("textarea");

	if (textInput) {
		if (textInput.getAttribute("contenteditable") === "true") {
			const tag = document.createElement("span");
			tag.className = "pnyx-reply-tag";
			tag.textContent = `[Responding to: "${claimText}"]`;
			tag.contentEditable = "false";
			textInput.prepend(tag);
			textInput.prepend(document.createTextNode(" "));
		} else {
			textInput.value = `[Responding to: "${claimText}"] ` + textInput.value;
		}
	}

	// Visual tag above comment box
	const tagEl = document.createElement("div");
	tagEl.className = "pnyx-reply-tag";
	tagEl.style.margin = "4px 0";
	tagEl.textContent = `Responding to: "${claimText}"`;
	commentBox.parentElement?.insertBefore(tagEl, commentBox);
}

function scanFeed() {
	if (!platform) return;
	const posts = document.querySelectorAll(platform.postContainer);
	for (const post of posts) {
		if (analyzedElements.has(post) || pendingElements.has(post)) continue;
		if (post.offsetHeight < 10) continue;
		processPost(post);
	}
}

// Observe feed mutations
if (platform) {
	const observer = new MutationObserver(() => scanFeed());
	observer.observe(document.body, { childList: true, subtree: true });
	// Initial scan
	scanFeed();
}
