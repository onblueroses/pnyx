/**
 * Tier 0: Heuristic AI slop detection.
 * Weighted signal accumulation with per-tier caps.
 * Runs synchronously in content script, <1ms per element.
 *
 * Structural signals based on:
 * - PNAS 2024 Biber feature analysis (because-clause poverty, participial overuse, passive underuse)
 * - GPTZero burstiness measurement
 * - Ghostbuster TTR / token probability features
 * - PubMed excess vocabulary study
 */

import { extractFeatures } from './features.js';

const TIER_WEIGHTS = {
  high: { weight: 3.0, cap: Infinity },
  medium: { weight: 1.5, cap: 4.5 },
  low: { weight: 0.5, cap: 2.0 },
};

const THRESHOLDS = {
  conservative: 4.0,
  balanced: 2.5,
  aggressive: 1.5,
};

export function scoreText(text, patterns, sensitivity = 'balanced') {
  if (!text || text.length < 50) return { score: 0, signals: [], verdict: 'skip' };

  const signals = [];
  const tierScores = { high: 0, medium: 0, low: 0 };

  // Pattern matching
  for (const pattern of patterns) {
    const matches = countMatches(text, pattern.regex);
    if (matches > 0 && matches >= (pattern.minMatches || 1)) {
      const tier = TIER_WEIGHTS[pattern.tier];
      const contribution = Math.min(
        matches * tier.weight,
        tier.cap - tierScores[pattern.tier]
      );
      if (contribution > 0) {
        tierScores[pattern.tier] += contribution;
        signals.push({ name: pattern.name, tier: pattern.tier, matches, contribution });
      }
    }
  }

  // Structural signals
  const structural = analyzeStructure(text);
  for (const signal of structural) {
    const tier = TIER_WEIGHTS[signal.tier];
    const contribution = Math.min(signal.score, tier.cap - tierScores[signal.tier]);
    if (contribution > 0) {
      tierScores[signal.tier] += contribution;
      signals.push({ name: signal.name, tier: signal.tier, matches: 1, contribution });
    }
  }

  const score = tierScores.high + tierScores.medium + tierScores.low;
  const threshold = THRESHOLDS[sensitivity];

  return {
    score,
    signals,
    verdict: score >= threshold ? 'slop' : score >= threshold * 0.6 ? 'ambiguous' : 'clean',
  };
}

function countMatches(text, regex) {
  const matches = text.match(regex);
  return matches ? matches.length : 0;
}

function analyzeStructure(text) {
  const signals = [];
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);

  if (sentences.length < 3) return signals;

  const features = extractFeatures(text);
  const sd = Math.sqrt(features.sentenceLengthVariance);

  // Contraction absence in long text
  if (text.length > 350 && features.contractionPresence === 0) {
    signals.push({ name: 'no-contractions', tier: 'low', score: 0.5 });
  }

  // Sentence length uniformity (burstiness proxy - GPTZero's #2 signal)
  if (sd < 3 && sentences.length >= 5) {
    signals.push({ name: 'uniform-sentence-length', tier: 'low', score: 0.5 });
  }

  // Type-Token Ratio (Ghostbuster feature)
  const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (words.length >= 100 && features.ttr < 0.35) {
    signals.push({ name: 'low-ttr', tier: 'low', score: 0.5 });
  }

  // Emoji engagement bait
  const emojiCount = (text.match(/[\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}]/gu) || []).length;
  if (emojiCount >= 4) {
    signals.push({ name: 'emoji-bait', tier: 'medium', score: 1.5 });
  } else if (emojiCount >= 2) {
    signals.push({ name: 'emoji-moderate', tier: 'low', score: 0.5 });
  }

  // --- NEW SIGNALS (from humanizer research) ---

  // Rule of three: AI forces ideas into groups of exactly 3
  // Detect comma-separated lists with exactly 3 items
  const tripleListCount = (text.match(/\b\w+,\s+\w+,?\s+and\s+\w+\b/gi) || []).length;
  if (tripleListCount >= 2) {
    signals.push({ name: 'rule-of-three', tier: 'low', score: 0.5 });
  }

  // Because-clause poverty (PNAS 2024: AI uses "because" at 19-20% of human rate)
  // In 200+ word text with no causal connectors, that's a signal
  if (words.length >= 40) {
    const causalCount = (text.match(/\b(because|since|so that|that's why|therefore)\b/gi) || []).length;
    const listingCount = (text.match(/\b(additionally|furthermore|moreover|in addition)\b/gi) || []).length;
    if (causalCount === 0 && listingCount >= 2) {
      signals.push({ name: 'no-causal-connectors', tier: 'low', score: 0.5 });
    }
  }

  // Sentence template repetition: 3+ sentences starting with the same structure
  if (sentences.length >= 5) {
    const openers = sentences.map(s => {
      const trimmed = s.trim();
      const firstTwo = trimmed.split(/\s+/).slice(0, 2).join(' ').toLowerCase();
      return firstTwo;
    });
    const openerCounts = {};
    for (const opener of openers) {
      openerCounts[opener] = (openerCounts[opener] || 0) + 1;
    }
    const maxRepeated = Math.max(...Object.values(openerCounts));
    if (maxRepeated >= 3) {
      signals.push({ name: 'template-repetition', tier: 'medium', score: 1.5 });
    }
  }

  // Inline-header list pattern: "**Bold:** description" repeated
  const boldHeaderCount = (text.match(/\*\*[^*]+\*\*:\s/g) || []).length;
  if (boldHeaderCount >= 3) {
    signals.push({ name: 'inline-header-list', tier: 'medium', score: 1.5 });
  }

  // Excessive hedging: multiple hedge phrases in one text
  const hedgeCount = (text.match(/\b(it could potentially|might possibly|could be argued|it is worth noting|generally speaking|to some extent|from a broader perspective)\b/gi) || []).length;
  if (hedgeCount >= 2) {
    signals.push({ name: 'excessive-hedging', tier: 'low', score: 0.5 });
  }

  // Generic positive conclusion
  const hasGenericEnding = /\b(the future looks bright|exciting times lie ahead|a step in the right direction|continues to (thrive|grow|evolve)|remains to be seen)\b/i.test(text);
  if (hasGenericEnding) {
    signals.push({ name: 'generic-conclusion', tier: 'low', score: 0.5 });
  }

  // Synonym cycling: same entity referred to by 3+ different names in close proximity
  // Simplified: detect "the [noun]" / "the [different noun]" pattern for protagonist/subject
  // This is hard to do well heuristically, so we check for a known pattern:
  // four sentences where the subject changes names each time
  const subjectPatterns = sentences.map(s => {
    const match = s.trim().match(/^(the \w+|this \w+)/i);
    return match ? match[0].toLowerCase() : null;
  }).filter(Boolean);
  if (subjectPatterns.length >= 4) {
    const uniqueSubjects = new Set(subjectPatterns);
    if (uniqueSubjects.size >= 4 && uniqueSubjects.size === subjectPatterns.length) {
      signals.push({ name: 'synonym-cycling', tier: 'low', score: 0.5 });
    }
  }

  // False range: "from X to Y" used for comprehensiveness rather than actual range
  const falseRangeCount = (text.match(/\bfrom [\w\s]+ to [\w\s]+,?\s*(from|and from)\b/gi) || []).length;
  if (falseRangeCount >= 1) {
    signals.push({ name: 'false-range', tier: 'low', score: 0.5 });
  }

  return signals;
}
