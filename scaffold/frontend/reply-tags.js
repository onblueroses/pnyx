/**
 * Reply tag rendering for V2 Visibility Layer.
 * Pure HTML string functions - no DOM manipulation.
 */

function escapeHtml(str) {
	return str
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}

export function renderReplyTag(claimText) {
	return `<span class="pnyx-reply-tag">Responded to: "${escapeHtml(claimText)}"</span>`;
}

export function renderSkippedTag() {
	return `<span class="pnyx-reply-tag skipped">Responded without engaging with specific claims</span>`;
}

export function renderReply(
	author,
	avatarColor,
	avatarInitials,
	text,
	tagHtml,
) {
	return `
    <div class="pnyx-reply">
      <div class="pnyx-reply-header">
        <div class="pnyx-reply-avatar" style="background:${escapeHtml(avatarColor)}">${escapeHtml(avatarInitials)}</div>
        <span class="pnyx-reply-author">${escapeHtml(author)}</span>
      </div>
      ${tagHtml}
      <p class="pnyx-reply-text">${escapeHtml(text)}</p>
    </div>
  `;
}

export function renderListeningIndicator(
	addressedCount,
	totalClaims,
	ignoredCount,
) {
	return `<div class="pnyx-listening-indicator">${addressedCount}/${totalClaims} claims addressed &middot; ${ignoredCount} ignored</div>`;
}
