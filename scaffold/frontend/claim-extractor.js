/**
 * Lightweight claim extraction from post text.
 * Runs AFTER model inference - uses model scores as context.
 * Sentence-level heuristic analysis, not span extraction.
 */

export function extractClaims(text, claimRiskProb) {
	if (!text || text.length < 20) return [];

	// Sentence splitting with abbreviation handling
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

		// Assertion verbs (+2 each)
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

		// Hedging language (-3 each)
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

		// Causal claims (+3 each)
		const causal = [
			"because",
			"therefore",
			"due to",
			"caused by",
			"as a result",
			"which means",
		];
		score += causal.filter((c) => lower.includes(c)).length * 3;

		// Absolute language (+4 each)
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

		// Numbers (+1)
		if (/\d/.test(sentence)) score += 1;

		// Question (-10)
		if (sentence.trim().endsWith("?")) score -= 10;

		return { text: sentence, risk, index, score };
	});

	return scored
		.filter((s) => s.score >= 1)
		.sort((a, b) => b.score - a.score)
		.slice(0, 3)
		.map(({ text, risk, index }) => ({ text, risk, index }));
}
