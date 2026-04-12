/**
 * Pause Layer overlay for V2.
 * Creates a friction moment between seeing a post and responding.
 */

const activeTimers = new Map();

function riskLabel(risk) {
	switch (risk) {
		case "high":
			return { cls: "danger", text: "unsupported" };
		case "medium":
			return { cls: "warn", text: "unverified" };
		case "low":
			return { cls: "good", text: "grounded" };
		default:
			return { cls: "muted", text: "unknown" };
	}
}

export function createPauseOverlay(postId, claims, onContinue, onSkip) {
	const maxClaims = claims.slice(0, 3);

	const overlay = document.createElement("div");
	overlay.className = "pnyx-pause-overlay";
	overlay.dataset.postId = postId;

	if (maxClaims.length === 0) {
		overlay.innerHTML = `
      <div class="pnyx-pause-card">
        <div class="pnyx-pause-header">Before you respond</div>
        <div class="pnyx-pause-subtitle">No specific claims detected in this post.</div>
        <div class="pnyx-pause-actions">
          <a class="pnyx-pause-skip" href="#">Continue to reply</a>
        </div>
      </div>
    `;
		overlay
			.querySelector(".pnyx-pause-skip")
			.addEventListener("click", (e) => {
				e.preventDefault();
				onSkip();
			});
		return overlay;
	}

	overlay.innerHTML = `
    <div class="pnyx-pause-card">
      <div class="pnyx-pause-header">Before you respond</div>
      <div class="pnyx-pause-subtitle">Here's what they said:</div>
      <div class="pnyx-pause-claims">
        ${maxClaims
					.map((c, i) => {
						const rl = riskLabel(c.risk);
						return `<label class="pnyx-pause-claim">
            <input type="checkbox" data-claim-index="${i}">
            <span>"${c.text}"</span>
            <span class="pnyx-claim-risk ${rl.cls}">${rl.text}</span>
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

	const continueBtn = overlay.querySelector(".pnyx-pause-continue");
	const claimsContainer = overlay.querySelector(".pnyx-pause-claims");

	// Enable/disable Continue based on checkbox state
	claimsContainer.addEventListener("change", () => {
		const checked = claimsContainer.querySelectorAll("input:checked");
		continueBtn.disabled = checked.length === 0;
	});

	// Continue: pass selected claim texts back
	continueBtn.addEventListener("click", () => {
		const checked = claimsContainer.querySelectorAll("input:checked");
		const selectedTexts = Array.from(checked).map((cb) => {
			return maxClaims[parseInt(cb.dataset.claimIndex)].text;
		});
		onContinue(selectedTexts);
	});

	// Skip
	overlay.querySelector(".pnyx-pause-skip").addEventListener("click", (e) => {
		e.preventDefault();
		onSkip();
	});

	// Timer (counts UP from 0:00)
	let elapsed = 0;
	const timerEl = overlay.querySelector(".pnyx-pause-timer");
	const interval = setInterval(() => {
		elapsed++;
		const m = Math.floor(elapsed / 60);
		const s = (elapsed % 60).toString().padStart(2, "0");
		timerEl.textContent = `\u23F1 ${m}:${s}`;
		if (elapsed >= 15) {
			onSkip();
		}
	}, 1000);
	activeTimers.set(postId, interval);

	return overlay;
}

export function dismissPause(postId) {
	const interval = activeTimers.get(postId);
	if (interval) {
		clearInterval(interval);
		activeTimers.delete(postId);
	}
	const overlay = document.querySelector(
		`.pnyx-pause-overlay[data-post-id="${postId}"]`,
	);
	if (overlay) overlay.remove();
}
