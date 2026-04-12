/**
 * Service worker - relays messages between content scripts and offscreen document.
 */

let offscreenPromise = null;

async function ensureOffscreen() {
	if (offscreenPromise) return offscreenPromise;
	offscreenPromise = (async () => {
		const contexts = await chrome.runtime.getContexts({
			contextTypes: ["OFFSCREEN_DOCUMENT"],
		});
		if (contexts.length > 0) return;
		await chrome.offscreen.createDocument({
			url: "offscreen.html",
			reasons: ["WORKERS"],
			justification:
				"ONNX Runtime WASM inference for Habermas + Erscheinung models",
		});
	})();
	return offscreenPromise;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
	if (msg.target === "offscreen") {
		ensureOffscreen()
			.then(() => {
				chrome.runtime.sendMessage(
					{ ...msg, target: "offscreen-doc" },
					(response) => {
						sendResponse(response);
					},
				);
			})
			.catch((err) => {
				sendResponse({ error: err.message });
			});
		return true; // async response
	}

	if (msg.target === "background") {
		if (msg.type === "get-thresholds") {
			chrome.storage.local.get({ habermas: 60, erscheinung: 50 }, sendResponse);
			return true;
		}
		if (msg.type === "set-thresholds") {
			chrome.storage.local.set(msg.thresholds);
			return false;
		}
	}
});
