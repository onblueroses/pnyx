/**
 * ONNX inference pipeline for AI slop detection.
 * Replicates unslop extension's offscreen.js runInference() for browser context.
 *
 * Model: DeBERTa-v3-small with pruned vocab (v0.7, trained on social media data)
 * Inputs: input_ids [1,128] int64, attention_mask [1,128] int64, features [1,8] float32
 * Output: [1,2] raw logits -> softmax -> index [1] is ai_prob
 */

import {
	AutoTokenizer,
	env as transformersEnv,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.all.min.mjs";

// Enable local model loading (tokenizer files served from model-slop/)
transformersEnv.allowLocalModels = true;

import { extractFeatures, featuresToArray, normalizeText } from "./features.js";
import { scoreText } from "./heuristics.js";
import { patterns } from "./patterns-en.js";

const TOKENIZER_ID = "microsoft/deberta-v3-small";
const MAX_SEQ_LEN = 128;
const SLOP_THRESHOLD = 0.5;
const HEURISTIC_SKIP_THRESHOLD = 4.0; // D10: skip ML if heuristic score >= 4.0

// WASM-only, single thread, SIMD enabled (D14)
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let session = null;
let tokenizer = null;
let tokenRemap = null;
let modelReady = false;

/**
 * Load the slop detection ONNX model and tokenizer.
 * Must be called before classifySlop().
 *
 * @param {string} modelPath - Path to model.onnx file
 * @param {string} remapPath - Path to token_remap.json file
 * @param {object} [preloadedTokenizer] - Pre-loaded tokenizer instance (skips from_pretrained if provided)
 */
export async function loadSlopModel(modelPath, remapPath) {
	if (modelReady) return;

	// Load all three resources before committing to global state.
	// If any step fails, no partial state is left behind.
	const newSession = await ort.InferenceSession.create(modelPath, {
		executionProviders: ["wasm"],
	});

	const newTokenizer = await AutoTokenizer.from_pretrained("./model-slop/");

	// Token remap is required for the pruned-vocab model - not optional
	const remapResponse = await fetch(remapPath);
	if (!remapResponse.ok) {
		throw new Error(`Failed to load token remap: ${remapResponse.status}`);
	}
	const remapObj = await remapResponse.json();
	const newRemap = {};
	for (const [oldId, newId] of Object.entries(remapObj)) {
		newRemap[Number(oldId)] = newId;
	}

	// All loaded successfully - commit to global state
	session = newSession;
	tokenizer = newTokenizer;
	tokenRemap = newRemap;
	modelReady = true;
}

/**
 * Run ML inference on text. Requires loadSlopModel() to have been called.
 *
 * @param {string} text - Raw text to classify
 * @returns {{aiProb: number, verdict: string, inferenceMs: number}}
 */
export async function classifySlop(text) {
	if (!session || !tokenizer) {
		return { aiProb: 0, verdict: "skip", inferenceMs: 0 };
	}

	const start = performance.now();

	// Preprocess: NFKC normalize + strip zero-width chars
	const normalized = normalizeText(text);

	// Tokenize
	const encoded = tokenizer(normalized, {
		padding: true,
		truncation: true,
		max_length: MAX_SEQ_LEN,
		return_tensors: "np",
	});

	let inputIds = encoded.input_ids.data;
	const attentionMask = encoded.attention_mask.data;
	const seqLen = encoded.input_ids.dims[1];

	// Remap token IDs for pruned vocabulary.
	// Use Array.from to avoid BigInt64Array.map() type coercion issues -
	// tokenRemap values are Numbers, BigInt64Array.map requires BigInt returns.
	if (tokenRemap) {
		inputIds = Array.from(inputIds, (id) => tokenRemap[Number(id)] ?? 0);
	}

	// Extract 8 hand-crafted features
	const features = extractFeatures(normalized);
	const featureArray = featuresToArray(features);

	// Create ONNX tensors
	const inputIdsTensor = new ort.Tensor(
		"int64",
		BigInt64Array.from(inputIds, (v) => BigInt(v)),
		[1, seqLen],
	);
	const maskTensor = new ort.Tensor(
		"int64",
		BigInt64Array.from(attentionMask, (v) => BigInt(v)),
		[1, seqLen],
	);
	const featuresTensor = new ort.Tensor(
		"float32",
		Float32Array.from(featureArray),
		[1, 8],
	);

	const output = await session.run({
		input_ids: inputIdsTensor,
		attention_mask: maskTensor,
		features: featuresTensor,
	});

	const logits = output.output.data; // [human_logit, ai_logit]
	const maxLogit = Math.max(logits[0], logits[1]);
	const expHuman = Math.exp(logits[0] - maxLogit);
	const expAi = Math.exp(logits[1] - maxLogit);
	const aiProb = expAi / (expHuman + expAi);
	const inferenceMs = performance.now() - start;

	return {
		aiProb,
		verdict: aiProb >= SLOP_THRESHOLD ? "slop" : "clean",
		inferenceMs,
	};
}

/**
 * Map heuristic score to a probability-like value for badge display.
 * Sigmoid with midpoint at 3.0, slope 0.5. Outputs:
 *   score 4.0 -> ~0.62, score 6.0 -> ~0.82, score 10.0 -> ~0.97
 * Not calibrated against the ML model - for UI display only.
 */
function heuristicToProb(score) {
	return 1 / (1 + Math.exp(-(score - 3) * 0.5));
}

/**
 * Full scoring pipeline: heuristic tier first, then ML if needed (D10).
 * Heuristic score >= 4.0 skips ML inference entirely.
 *
 * @param {string} text - Raw text to score
 * @returns {{aiProb: number, verdict: string, tier: string, heuristicScore: number, signals: Array, inferenceMs: number}}
 */
export async function scoreForSlop(text) {
	// Preprocess once
	const normalized = normalizeText(text);

	// Tier 0: Heuristic scoring (< 1ms)
	const heuristic = scoreText(normalized, patterns);

	// D10: Skip ML if heuristic is confident enough
	if (heuristic.score >= HEURISTIC_SKIP_THRESHOLD) {
		return {
			aiProb: heuristicToProb(heuristic.score),
			verdict: "slop",
			tier: "heuristic",
			heuristicScore: heuristic.score,
			signals: heuristic.signals,
			inferenceMs: 0,
		};
	}

	// Tier 1: ML inference
	if (!modelReady) {
		return {
			aiProb: 0,
			verdict: "unknown",
			tier: "heuristic-only",
			heuristicScore: heuristic.score,
			signals: heuristic.signals,
			inferenceMs: 0,
		};
	}

	const ml = await classifySlop(normalized);

	return {
		aiProb: ml.aiProb,
		verdict: ml.verdict,
		tier: "ml",
		heuristicScore: heuristic.score,
		signals: heuristic.signals,
		inferenceMs: ml.inferenceMs,
	};
}
