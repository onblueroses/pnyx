/**
 * Hand-crafted features for AI text detection.
 * Shared between heuristic scoring (content script) and ML inference pipeline.
 *
 * CRITICAL: These 8 features must match training/features.py exactly.
 * Any drift silently degrades model accuracy. See training/MODEL_CONTRACT.md.
 */

const STOP_WORDS = new Set([
  'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
  'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
  'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
  'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
  'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
  'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
  'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
  'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
  'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
  'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
  'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'has',
  'had', 'did', 'does', 'am',
]);

const CONTRACTION_RE = /\b(don't|won't|can't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|didn't|wouldn't|couldn't|shouldn't|it's|that's|there's|here's|what's|who's|I'm|I've|I'll|I'd|we're|we've|we'll|they're|they've|they'll|you're|you've|you'll|he's|she's|let's)\b/i;

const ZERO_WIDTH_RE = /[\u200B\u200C\u200D\uFEFF]/g;

/**
 * NFKC normalization + zero-width character stripping.
 * Defense against homoglyph and zero-width space attacks.
 */
export function normalizeText(text) {
  return text.normalize('NFKC').replace(ZERO_WIDTH_RE, '');
}

/**
 * Extract 8 hand-crafted features from text.
 * Returns values in a fixed order matching the ML model's expected input.
 *
 * @param {string} text - Already normalized text
 * @returns {{ttr: number, hapaxRate: number, sentenceLengthVariance: number, avgSentenceLength: number, bigramUniqueness: number, stopWordDensity: number, contractionPresence: number, lowercaseRatio: number}}
 */
export function extractFeatures(text) {
  const zeros = {
    ttr: 0, hapaxRate: 0, sentenceLengthVariance: 0, avgSentenceLength: 0,
    bigramUniqueness: 0, stopWordDensity: 0, contractionPresence: 0, lowercaseRatio: 0,
  };

  if (!text || text.length < 50) return zeros;

  const words = text.toLowerCase().match(/\b[a-z]+\b/g);
  if (!words || words.length < 10) return zeros;

  const wordFreq = new Map();
  for (const w of words) {
    wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
  }

  // 1. Type-Token Ratio
  const ttr = wordFreq.size / words.length;

  // 2. Hapax legomenon rate (words appearing exactly once)
  let hapaxCount = 0;
  for (const count of wordFreq.values()) {
    if (count === 1) hapaxCount++;
  }
  const hapaxRate = hapaxCount / wordFreq.size;

  // 3-4. Sentence length statistics
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  let sentenceLengthVariance = 0;
  let avgSentenceLength = 0;

  if (sentences.length >= 2) {
    const lengths = sentences.map(s => s.trim().split(/\s+/).length);
    avgSentenceLength = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    sentenceLengthVariance = lengths.reduce((a, b) => a + (b - avgSentenceLength) ** 2, 0) / lengths.length;
  }

  // 5. Bigram uniqueness ratio
  let bigramUniqueness = 0;
  if (words.length >= 2) {
    const bigrams = new Set();
    const totalBigrams = words.length - 1;
    for (let i = 0; i < totalBigrams; i++) {
      bigrams.add(words[i] + ' ' + words[i + 1]);
    }
    bigramUniqueness = bigrams.size / totalBigrams;
  }

  // 6. Stop word density
  let stopCount = 0;
  for (const w of words) {
    if (STOP_WORDS.has(w)) stopCount++;
  }
  const stopWordDensity = stopCount / words.length;

  // 7. Contraction presence (binary: 0 or 1)
  const contractionPresence = CONTRACTION_RE.test(text) ? 1 : 0;

  // 8. Lowercase letter ratio (over all alpha chars)
  let lowerCount = 0;
  let alphaCount = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if ((c >= 65 && c <= 90)) { alphaCount++; }
    else if ((c >= 97 && c <= 122)) { alphaCount++; lowerCount++; }
  }
  const lowercaseRatio = alphaCount > 0 ? lowerCount / alphaCount : 0;

  return {
    ttr, hapaxRate, sentenceLengthVariance, avgSentenceLength,
    bigramUniqueness, stopWordDensity, contractionPresence, lowercaseRatio,
  };
}

/**
 * Convert features object to array in the model's expected order.
 * Order: [ttr, hapaxRate, sentenceLengthVariance, avgSentenceLength,
 *         bigramUniqueness, stopWordDensity, contractionPresence, lowercaseRatio]
 */
export function featuresToArray(features) {
  return [
    features.ttr,
    features.hapaxRate,
    features.sentenceLengthVariance,
    features.avgSentenceLength,
    features.bigramUniqueness,
    features.stopWordDensity,
    features.contractionPresence,
    features.lowercaseRatio,
  ];
}
