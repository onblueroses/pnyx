/**
 * Offscreen document - runs both ONNX models in a separate context.
 * Uses vendored libs (CDN imports blocked by MV3 CSP).
 */

import { loadSlopModel, scoreForSlop } from "./detection/index.js";

let discourseSession = null;
let discourseTokenizer = null;
let useDiscourseModel = false;
let ort = null;
let loadPromise = null;

function loadModels() {
	if (loadPromise) return loadPromise;
	loadPromise = (async () => {
		// Erscheinung first (heuristic tier scores immediately)
		try {
			await loadSlopModel(
				chrome.runtime.getURL("model-slop/model.onnx"),
				chrome.runtime.getURL("model-slop/token_remap.json"),
			);
		} catch (e) {
			console.warn("Erscheinung ML load failed, heuristic only:", e.message);
		}

		// Habermas
		try {
			const configUrl = chrome.runtime.getURL("model/config.json");
			const r = await fetch(configUrl, { method: "HEAD" });
			if (r.ok) {
				ort = await import(chrome.runtime.getURL("vendor/ort.all.min.mjs"));
				const { AutoTokenizer } = await import(
					chrome.runtime.getURL("vendor/transformers.min.js")
				);

				// from_pretrained can't resolve chrome-extension:// URLs.
				// Intercept fetch so tokenizer file requests go to local extension files.
				const modelBase = chrome.runtime.getURL("model/");
				const origFetch = globalThis.fetch;
				globalThis.fetch = (url, opts) => {
					if (typeof url === "string" && url.includes("huggingface.co")) {
						const filename = url.split("/").pop().split("?")[0];
						return origFetch(modelBase + filename, opts);
					}
					return origFetch(url, opts);
				};
				try {
					discourseTokenizer =
						await AutoTokenizer.from_pretrained("pnyx-habermas");
				} finally {
					globalThis.fetch = origFetch;
				}

				ort.env.wasm.numThreads = 1;
				ort.env.wasm.simd = true;
				discourseSession = await ort.InferenceSession.create(
					chrome.runtime.getURL("model/model.onnx"),
				);
				useDiscourseModel = true;
				console.log("Habermas model loaded (on-device ONNX)");
			}
		} catch (e) {
			console.warn("Habermas load failed, placeholder mode:", e.message);
		}
	})();
	return loadPromise;
}

function softmax(logits) {
	const max = Math.max(...logits);
	const exps = Array.from(logits).map((x) => Math.exp(x - max));
	const sum = exps.reduce((a, b) => a + b);
	return exps.map((x) => x / sum);
}

async function scoreDiscourse(text) {
	const start = performance.now();
	if (useDiscourseModel && discourseSession && discourseTokenizer) {
		const { input_ids, attention_mask } = await discourseTokenizer(text, {
			padding: true,
			truncation: true,
			max_length: 256,
			return_tensors: "np",
		});
		const results = await discourseSession.run({
			input_ids: new ort.Tensor("int64", input_ids.data, input_ids.dims),
			attention_mask: new ort.Tensor(
				"int64",
				attention_mask.data,
				attention_mask.dims,
			),
		});
		return {
			claimRiskProb: softmax(results.claim_risk.data)[1],
			argQualityProb: softmax(results.argument_quality.data)[1],
			inferenceMs: Math.round(performance.now() - start),
			mode: "onnx",
		};
	}
	// Placeholder
	const lower = text.toLowerCase();
	const risk = [
		"proves",
		"hiding",
		"everyone knows",
		"wake up",
		"naive",
		"share before",
		"cover up",
		"coordinated",
	];
	const arg = [
		"because",
		"according to",
		"report",
		"evidence",
		"suggests",
		"however",
		"although",
		"research",
	];
	return {
		claimRiskProb: Math.min(
			0.95,
			0.15 + risk.filter((w) => lower.includes(w)).length * 0.22,
		),
		argQualityProb: Math.min(
			0.95,
			0.12 + arg.filter((w) => lower.includes(w)).length * 0.18,
		),
		inferenceMs: Math.round(performance.now() - start),
		mode: "placeholder",
	};
}

async function scoreAI(text) {
	try {
		const r = await scoreForSlop(text);
		return {
			aiProb: r.aiProb,
			verdict: r.verdict,
			tier: r.tier,
			inferenceMs: r.inferenceMs || 0,
		};
	} catch {
		return { aiProb: 0, verdict: "clean", tier: "error", inferenceMs: 0 };
	}
}

// Serialize scoring - catch errors to prevent bricking the queue
let queue = Promise.resolve();

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
	if (msg.target !== "offscreen-doc") return false;

	if (msg.type === "score") {
		queue = queue
			.then(async () => {
				try {
					await loadModels();
					const [d, ai] = await Promise.all([
						scoreDiscourse(msg.text),
						scoreAI(msg.text),
					]);
					sendResponse({
						habermas: {
							claimRiskProb: d.claimRiskProb,
							argQualityProb: d.argQualityProb,
						},
						erscheinung: {
							aiProb: ai.aiProb,
							verdict: ai.verdict,
							tier: ai.tier,
						},
						inferenceMs: {
							habermas: d.inferenceMs,
							erscheinung: ai.inferenceMs,
						},
						mode: d.mode,
					});
				} catch (e) {
					sendResponse({ error: e.message });
				}
			})
			.catch(() => {}); // reset queue on unexpected rejection
		return true; // async
	}

	return false;
});

// Pre-load models
loadModels();
