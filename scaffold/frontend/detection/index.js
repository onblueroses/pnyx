/**
 * AI slop detection - public API.
 * Dual-tier: heuristic (instant) + ML inference (ONNX, ~500ms).
 */

export { extractFeatures, featuresToArray, normalizeText } from "./features.js";
export { scoreText } from "./heuristics.js";
export { patterns } from "./patterns-en.js";
export { classifySlop, loadSlopModel, scoreForSlop } from "./slop-inference.js";
