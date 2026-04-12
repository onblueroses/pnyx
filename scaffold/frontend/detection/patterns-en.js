/**
 * English AI slop detection patterns.
 * Tiered by confidence: high (3pts), medium (1.5pts capped), low (0.5pts capped).
 *
 * Sources:
 * - Wikipedia:Signs of AI writing (WikiProject AI Cleanup)
 * - PubMed excess vocabulary study (280 words, frequency ratios)
 * - Pangram Labs DAMAGE paper (19 humanizer tools tested)
 * - PNAS 2024 Biber feature analysis
 */

export const patterns = [
  // HIGH CONFIDENCE - strong AI tells (self-references, chatbot artifacts)
  { name: 'ai-self-reference', tier: 'high', regex: /\b(as an AI|as a language model|as an artificial intelligence|I don't have personal|I cannot browse|my training data|my knowledge cutoff)\b/gi },
  { name: 'ai-disclaimer', tier: 'high', regex: /\b(I should note that|it's worth noting that|it's important to note)\b/gi },
  { name: 'chatbot-artifacts', tier: 'high', regex: /\b(I hope this helps|let me know if you'?d like|would you like me to|feel free to ask)\b/gi },
  { name: 'sycophantic', tier: 'high', regex: /\b(great question|that's an excellent point|you'?re absolutely right|absolutely!|certainly!)\b/gi },

  // MEDIUM CONFIDENCE - AI vocabulary (PubMed study: 3x+ expected frequency)
  // "delves" at 25x, "showcasing" at 9.2x, "underscores" at 9.1x
  { name: 'delve', tier: 'medium', regex: /\bdelve[sd]?\b/gi },
  { name: 'tapestry', tier: 'medium', regex: /\btapestry\b/gi },
  { name: 'landscape-abstract', tier: 'medium', regex: /\b(the landscape of|broader landscape|shifting landscape|evolving landscape|technological landscape|digital landscape)\b/gi },
  { name: 'crucial', tier: 'medium', regex: /\bcrucial\b/gi },
  { name: 'pivotal', tier: 'medium', regex: /\bpivotal\b/gi },
  { name: 'multifaceted', tier: 'medium', regex: /\bmultifaceted\b/gi },
  { name: 'nuanced', tier: 'medium', regex: /\bnuanced\b/gi },
  { name: 'paradigm-shift', tier: 'medium', regex: /\bparadigm shift\b/gi },
  { name: 'game-changer', tier: 'medium', regex: /\bgame[- ]changer\b/gi },
  { name: 'realm', tier: 'medium', regex: /\b(in the realm of|within the realm)\b/gi },
  { name: 'foster', tier: 'medium', regex: /\bfoster(s|ed|ing)?\b/gi },
  { name: 'leverage', tier: 'medium', regex: /\bleverage[sd]?\b/gi },
  { name: 'embark', tier: 'medium', regex: /\bembark(s|ed|ing)?\b/gi },
  { name: 'unleash', tier: 'medium', regex: /\bunleash(es|ed|ing)?\b/gi },
  { name: 'harness', tier: 'medium', regex: /\bharness(es|ed|ing)?\b/gi },
  { name: 'elevate', tier: 'medium', regex: /\belevate[sd]?\b/gi },
  { name: 'resonate', tier: 'medium', regex: /\bresonate[sd]?\b/gi },
  { name: 'robust', tier: 'medium', regex: /\brobust\b/gi },
  { name: 'seamless', tier: 'medium', regex: /\bseamless(ly)?\b/gi },
  { name: 'cutting-edge', tier: 'medium', regex: /\bcutting[- ]edge\b/gi },
  { name: 'groundbreaking', tier: 'medium', regex: /\bgroundbreaking\b/gi },
  { name: 'revolutionize', tier: 'medium', regex: /\brevolutionize[sd]?\b/gi },
  { name: 'em-dash-overuse', tier: 'medium', regex: /\u2014/g, minMatches: 3 },

  // MEDIUM - additional high-signal vocabulary (PubMed excess frequency study)
  { name: 'beacon', tier: 'medium', regex: /\bbeacon\b/gi },
  { name: 'testament', tier: 'medium', regex: /\b(testament to|stands as a testament)\b/gi },
  { name: 'underscore', tier: 'medium', regex: /\bunderscor(e[sd]?|ing)\b/gi },
  { name: 'showcase', tier: 'medium', regex: /\bshowcas(e[sd]?|ing)\b/gi },
  { name: 'illuminate', tier: 'medium', regex: /\billuminat(e[sd]?|ing)\b/gi },
  { name: 'facilitate', tier: 'medium', regex: /\bfacilitat(e[sd]?|ing)\b/gi },
  { name: 'bolster', tier: 'medium', regex: /\bbolster(s|ed|ing)?\b/gi },
  { name: 'meticulous', tier: 'medium', regex: /\bmeticulous(ly)?\b/gi },
  { name: 'palpable', tier: 'medium', regex: /\bpalpable\b/gi },
  { name: 'intricate', tier: 'medium', regex: /\bintricat(e|es|ely)\b/gi },
  { name: 'vibrant', tier: 'medium', regex: /\bvibrant\b/gi },
  { name: 'enduring', tier: 'medium', regex: /\benduring\b/gi },
  { name: 'garner', tier: 'medium', regex: /\bgarner(s|ed|ing)?\b/gi },
  { name: 'interplay', tier: 'medium', regex: /\binterplay\b/gi },
  { name: 'noteworthy', tier: 'medium', regex: /\bnoteworthy\b/gi },

  // MEDIUM - structural phrases (Wikipedia AI Cleanup patterns)
  { name: 'copula-avoidance', tier: 'medium', regex: /\b(serves as|stands as|functions as|acts as a|represents a)\b/gi },
  { name: 'negative-parallelism', tier: 'medium', regex: /\b(it'?s not just|not only .{5,40} but also|it'?s not .{3,20},? it'?s|this isn'?t .{3,20},? it'?s)\b/gi },
  { name: 'inflated-significance', tier: 'medium', regex: /\b(marking a pivotal|indelible mark|setting the stage|key turning point|shaping the future)\b/gi },
  { name: 'false-objectivity', tier: 'medium', regex: /\b(that'?s not an opinion|not as a recommendation|not a prediction.{0,5}just|just sharing the facts)\b/gi },

  // MEDIUM - discourse patterns
  { name: 'vague-attribution', tier: 'medium', regex: /\b(experts (say|believe|argue|suggest|note)|industry reports|observers have cited|several sources|studies (show|suggest|indicate))\b/gi },
  { name: 'challenges-and-prospects', tier: 'medium', regex: /\b(despite (these|its|the) challenges|future (outlook|prospects)|challenges and (opportunities|legacy))\b/gi },

  // LOW CONFIDENCE - weak signals, need accumulation
  { name: 'comprehensive', tier: 'low', regex: /\bcomprehensive\b/gi },
  { name: 'streamline', tier: 'low', regex: /\bstreamline[sd]?\b/gi },
  { name: 'synergy', tier: 'low', regex: /\bsynerg(y|ies|istic)\b/gi },
  { name: 'stakeholder', tier: 'low', regex: /\bstakeholder[s]?\b/gi },
  { name: 'ecosystem', tier: 'low', regex: /\becosystem\b/gi },
  { name: 'scalable', tier: 'low', regex: /\bscalable\b/gi },
  { name: 'innovative', tier: 'low', regex: /\binnovative\b/gi },
  { name: 'transformative', tier: 'low', regex: /\btransformative\b/gi },
  { name: 'empower', tier: 'low', regex: /\bempower(s|ed|ing|ment)?\b/gi },
  { name: 'holistic', tier: 'low', regex: /\bholistic(ally)?\b/gi },
  { name: 'actionable', tier: 'low', regex: /\bactionable\b/gi },
  { name: 'proactive', tier: 'low', regex: /\bproactive(ly)?\b/gi },
  { name: 'deep-dive', tier: 'low', regex: /\bdeep[- ]dive\b/gi },
  { name: 'takeaway', tier: 'low', regex: /\bkey takeaway[s]?\b/gi },
  { name: 'double-down', tier: 'low', regex: /\bdouble[- ]down\b/gi },
  { name: 'north-star', tier: 'low', regex: /\bnorth star\b/gi },
  { name: 'move-the-needle', tier: 'low', regex: /\bmove the needle\b/gi },
  { name: 'at-the-end-of-the-day', tier: 'low', regex: /\bat the end of the day\b/gi },
  { name: 'lets-unpack', tier: 'low', regex: /\blet'?s unpack\b/gi },
  { name: 'food-for-thought', tier: 'low', regex: /\bfood for thought\b/gi },

  // LOW - additional vocabulary (still AI-elevated but more common in human text)
  { name: 'additionally', tier: 'low', regex: /\badditionally\b/gi },
  { name: 'furthermore', tier: 'low', regex: /\bfurthermore\b/gi },
  { name: 'moreover', tier: 'low', regex: /\bmoreover\b/gi },
  { name: 'endeavor', tier: 'low', regex: /\bendeavou?r(s|ed|ing)?\b/gi },
  { name: 'navigate', tier: 'low', regex: /\bnavigate[sd]?\b/gi },
  { name: 'unlock', tier: 'low', regex: /\bunlock(s|ed|ing)?\b/gi },
  { name: 'camaraderie', tier: 'low', regex: /\bcamaraderie\b/gi },
  { name: 'reshape', tier: 'low', regex: /\breshap(e[sd]?|ing)\b/gi },
  { name: 'nestled', tier: 'low', regex: /\bnestled\b/gi },
  { name: 'breathtaking', tier: 'low', regex: /\bbreathtaking\b/gi },
  { name: 'renowned', tier: 'low', regex: /\brenowned\b/gi },

  // LOW - filler phrases
  { name: 'in-order-to', tier: 'low', regex: /\bin order to\b/gi },
  { name: 'due-to-the-fact', tier: 'low', regex: /\bdue to the fact that\b/gi },
  { name: 'at-this-point', tier: 'low', regex: /\bat this point in time\b/gi },
  { name: 'it-is-important', tier: 'low', regex: /\bit is important to (note|remember|understand|recognize)\b/gi },
  { name: 'in-todays-world', tier: 'low', regex: /\bin today'?s (world|age|era|society|landscape|climate)\b/gi },
  { name: 'in-conclusion', tier: 'low', regex: /\b(in conclusion|to summarize|in summary|all in all)\b/gi },
];
