/**
 * Inference Web Worker - runs Habermas model + Erscheinung heuristics off main thread.
 *
 * Messages IN:
 *   { type: 'init' }                     -> load models
 *   { type: 'score', id, text }          -> score a post
 *
 * Messages OUT:
 *   { type: 'model-status', model, status, detail? }  -> loading progress
 *   { type: 'ready' }                                  -> all models loaded
 *   { type: 'result', id, scores }                     -> scoring result
 *   { type: 'error', id?, message }                    -> error
 */

import { scoreForSlop } from "./detection/index.js";

let discourseSession = null;
let discourseTokenizer = null;
let useDiscourseModel = false;
let ort = null;

function status(model, s, detail) {
	self.postMessage({ type: "model-status", model, status: s, detail });
}

async function loadDiscourseModel() {
	status("habermas", "checking");
	try {
		const r = await fetch("./model/config.json", { method: "HEAD" });
		if (!r.ok) {
			status("habermas", "placeholder");
			return;
		}

		status("habermas", "loading");
		ort = await import(
			"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.all.min.mjs"
		);
		const transformers = await import(
			"https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1"
		);
		// Enable local model loading in worker context (relative path = "local")
		transformers.env.allowLocalModels = true;

		status("habermas", "tokenizer");
		discourseTokenizer =
			await transformers.AutoTokenizer.from_pretrained("./model/");

		status("habermas", "model");
		ort.env.wasm.numThreads = 1;
		ort.env.wasm.simd = true;
		discourseSession = await ort.InferenceSession.create("./model/model.onnx");
		useDiscourseModel = true;
		status("habermas", "ready");
	} catch (e) {
		status("habermas", "placeholder", e.message);
	}
}

async function loadErscheinung() {
	// Erscheinung ML model disabled: v0.5/v0.6 models overfit to RAID dataset
	// and classify all out-of-distribution text as AI. Heuristic tier (70+ patterns)
	// provides reliable detection for obvious AI slop without false positives.
	// New model training planned with social-media-distribution data.
	status("erscheinung", "heuristic");
}

const TEMPERATURE = 1.75; // >1 spreads probabilities, reduces overconfidence

function softmax(logits) {
	const scaled = Array.from(logits).map((x) => x / TEMPERATURE);
	const max = Math.max(...scaled);
	const exps = scaled.map((x) => Math.exp(x - max));
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
	return placeholderScore(text, start);
}

function placeholderScore(text, startTime) {
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
		"silence",
		"disappears",
		"obvious",
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
		"data",
		"inspections",
		"compare",
		"council minutes",
		"may be linked",
		"correlation",
		"causation",
		"audit",
		"review",
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
		inferenceMs: Math.round(performance.now() - startTime),
		mode: "placeholder",
	};
}

async function scoreAI(text) {
	try {
		// Heuristic-only: 70+ pattern-based signals, <1ms
		// ML tier disabled pending new model training
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

// Queue to process one post at a time (avoids OOM from parallel inference)
let queue = Promise.resolve();

self.onmessage = async (e) => {
	const { type, id, text } = e.data;

	if (type === "init") {
		try {
			await Promise.all([loadErscheinung(), loadDiscourseModel()]);
			self.postMessage({ type: "ready" });
		} catch (err) {
			self.postMessage({ type: "error", message: err.message });
		}
		return;
	}

	if (type === "score") {
		queue = queue.then(async () => {
			try {
				const [d, ai] = await Promise.all([
					scoreDiscourse(text),
					scoreAI(text),
				]);
				self.postMessage({ type: "result", id, scores: { ...d, ai } });
			} catch (err) {
				self.postMessage({ type: "error", id, message: err.message });
			}
		});
		return;
	}
};
