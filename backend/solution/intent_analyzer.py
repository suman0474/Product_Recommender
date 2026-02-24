# solution/intent_analyzer.py
# =============================================================================
# CONTEXT-AWARE INTENT CLASSIFIER - Solution Workflow Detection
# =============================================================================
#
# Determines whether user input belongs to the Solution workflow (complex
# multi-instrument system design) or should be redirected elsewhere.
#
# Classification pipeline (in priority order):
#   1. Knowledge-question fast-reject  → immediately returns chat intent
#   2. Semantic similarity (embeddings) against dynamic context descriptions
#   3. Taxonomy RAG signal (optional, non-blocking)
#   4. Keyword matching fallback
#   5. Contextual boost from conversation history
#
# Embedding model: gemini-embedding-001 (same as routing/semantic_classifier.py)
# =============================================================================

import logging
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT RESULT
# =============================================================================

@dataclass
class IntentResult:
    """Result of semantic intent classification."""
    is_solution: bool
    confidence: float  # 0.0 - 1.0
    method: str  # "fast_reject", "semantic", "keyword", "contextual", "hybrid"
    intent_type: str  # "solution", "chat", "unknown", "modification", "comparison"
    domain: str  # Detected domain (e.g., "Oil & Gas")
    industry: str  # Detected industry
    solution_indicators: List[str]  # What triggered solution classification
    extracted_info: Dict[str, Any]  # Additional extracted context


# =============================================================================
# SOLUTION WORKFLOW CONTEXT DESCRIPTIONS
# (Dynamic reference content — used instead of hardcoded embedding phrases)
# =============================================================================

# What the Solution workflow handles (solution-positive signal)
_SOLUTION_CONTEXT = """
The Solution workflow handles complex, multi-instrument engineering challenges
that require holistic system design. It is triggered when a user needs to design
or select a complete instrumentation system: multiple instruments and accessories
working together across a facility, plant, reactor, pipeline, or process unit.

Examples of Solution workflow inputs:
- Designing a crude oil distillation unit with temperature and pressure monitoring
- Complete instrumentation package for a chemical reactor with safety shutdown
- Instruments for a heating circuit: flow meters, pressure transmitters, temperature sensors
- Full measurement system for a water treatment plant
- Safety instrumented system for a hydrogen production facility requiring SIL 2
- Metering skid for custody transfer of petroleum products
- Process control instrumentation for a pharmaceutical batch reactor
- Multi-point monitoring system for a natural gas compressor station
"""

# What the Chat workflow handles (solution-negative signal)
_CHAT_CONTEXT = """
The Chat (EnGenie Chat) workflow handles knowledge questions, definitions,
explanations, standards information, and product information queries.
It does NOT identify or specify instruments — it answers questions.

Examples of Chat workflow inputs:
- What is SIL 2 / SIL rating?
- How does a differential pressure transmitter work?
- Explain HART protocol
- What certifications does this product have?
- What is the difference between ATEX and IECEx?
- Tell me about Foundation Fieldbus
- What temperature sensors does Honeywell make?
- Define accuracy vs precision
- Show me Rosemount 3051 specifications
- What is a thermowell used for?
"""


# =============================================================================
# KNOWLEDGE QUESTION PATTERNS (fast-reject from solution workflow)
# =============================================================================

# Queries starting with these patterns are definitively NOT solution requests.
# They are knowledge/information queries → route to Chat.
KNOWLEDGE_QUESTION_STARTS = [
    "what is", "what are", "what's", "what does", "what do",
    "how does", "how do", "how is", "how are", "how can",
    "explain", "define", "definition of", "meaning of",
    "tell me about", "describe", "what's the difference",
    "why is", "why does", "why do",
    "can you explain", "can you tell me",
    "show me", "list the", "give me information",
]

# High-value topics that are CHAT topics even inside solution-sounding sentences
KNOWLEDGE_TOPIC_PATTERNS = [
    "what is sil", "what is atex", "what is hart", "what is profibus",
    "what is foundation fieldbus", "what is iec", "what is iso",
    "sil rating", "sil level", "sil certification", "sil assessment",
    "atex zone", "ex ia", "ex d", "hazardous area classification",
    "hart protocol", "fieldbus protocol", "modbus protocol",
    "accuracy vs", "accuracy definition",
]


# =============================================================================
# SOLUTION KEYWORDS (for keyword fallback — strong positive signals)
# =============================================================================

SOLUTION_KEYWORDS = [
    # Multi-instrument / system design signals
    "complete system", "design a system", "instrument package",
    "instrumentation package", "complete instrumentation", "full solution",
    "monitoring system", "measurement system", "control system",
    "holistic solution", "entire system", "instrumentation for",
    # Process unit / facility signals
    "distillation unit", "heating circuit", "processing unit",
    "production line", "compressor station", "metering skid",
    "tank gauging", "custody transfer", "batch reactor",
    "boiler system", "water treatment plant", "chiller plant",
    # Industry + multi-component
    "multiple instruments", "multiple sensors", "multiple transmitters",
    "crude oil", "refinery", "chemical plant", "natural gas processing",
    "pharmaceutical facility", "hydrogen production",
    # Design / specify actions implying system scope
    "select and specify", "design and specify",
]

# Weak solution signals (single match = low confidence only)
SOLUTION_KEYWORDS_WEAK = [
    "system", "design", "complete", "package", "profiling", "comprehensive",
    "reactor", "distillation", "pipeline", "process control",
    "safety instrumented", "skid", "plant",
]


# =============================================================================
# DOMAIN KEYWORDS
# =============================================================================

DOMAIN_KEYWORDS = {
    "Oil & Gas": ["crude oil", "refinery", "petroleum", "natural gas", "pipeline", "offshore", "downstream", "upstream"],
    "Chemical": ["chemical", "reactor", "distillation", "separation", "mixing", "catalyst"],
    "Pharmaceutical": ["pharmaceutical", "pharma", "batch", "clean room", "gmp", "fda"],
    "Water/Wastewater": ["water treatment", "wastewater", "desalination", "filtration", "sewage"],
    "Power": ["boiler", "turbine", "power plant", "steam", "generator", "chp"],
    "Food & Beverage": ["food", "beverage", "dairy", "brewing", "pasteurization"],
    "HVAC": ["hvac", "chiller", "cooling", "heating", "ventilation", "air conditioning"],
    "Mining": ["mining", "mineral", "ore", "slurry", "crusher"],
}

# Confidence threshold for solution classification
SOLUTION_CONFIDENCE_THRESHOLD = 0.65


# =============================================================================
# SEMANTIC INTENT CLASSIFIER
# =============================================================================

class SolutionIntentClassifier:
    """
    Context-aware intent classifier for solution workflow detection.

    Uses a priority cascade:
    1. Fast-reject: knowledge question patterns (high confidence, no LLM/embedding needed)
    2. Semantic: embedding similarity against workflow context descriptions
    3. Taxonomy RAG: checks if query asks *about* a known instrument term
    4. Keyword: strong/weak keyword matching
    5. Contextual: conversation history boost
    """

    def __init__(self, cache_embeddings: bool = True):
        self._embeddings_model = None
        self._solution_embeddings = None
        self._non_solution_embeddings = None
        self._cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[str, List[float]] = {}

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------

    def _get_embeddings_model(self):
        """Lazy-load the embeddings model. Uses gemini-embedding-001 (available on v1beta)."""
        if self._embeddings_model is None:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                import os
                self._embeddings_model = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                logger.info("[IntentClassifier] Loaded GoogleGenerativeAIEmbeddings (gemini-embedding-001)")
            except Exception as e:
                self._embeddings_model = None
        return self._embeddings_model

    # -------------------------------------------------------------------------
    # EMBEDDING UTILITIES
    # -------------------------------------------------------------------------

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for a single text, with caching."""
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            from common.infrastructure.caching.embedding_cache import get_embedding_cache
            cached = get_embedding_cache().get(text)
            if cached is not None:
                self._embedding_cache[cache_key] = cached
                return cached
        except ImportError:
            pass

        model = self._get_embeddings_model()
        if model is None:
            return None

        try:
            embedding = model.embed_query(text)
            if self._cache_embeddings:
                self._embedding_cache[cache_key] = embedding
                try:
                    from common.infrastructure.caching.embedding_cache import get_embedding_cache
                    get_embedding_cache().put(text, embedding)
                except ImportError:
                    pass
            return embedding
        except Exception as e:
            return None

    def _compute_reference_embeddings(self) -> bool:
        """Pre-compute reference embeddings from workflow context descriptions (not hardcoded phrases)."""
        if self._solution_embeddings is not None:
            return True

        model = self._get_embeddings_model()
        if model is None:
            return False

        try:
            # Split context blocks into sentence-level chunks for better granularity
            solution_chunks = [
                line.strip() for line in _SOLUTION_CONTEXT.strip().splitlines()
                if line.strip() and not line.strip().startswith("-") and len(line.strip()) > 20
            ]
            # Include the bulleted examples too — strip the leading "- "
            solution_chunks += [
                line.strip().lstrip("- ") for line in _SOLUTION_CONTEXT.strip().splitlines()
                if line.strip().startswith("-") and len(line.strip()) > 20
            ]

            non_solution_chunks = [
                line.strip() for line in _CHAT_CONTEXT.strip().splitlines()
                if line.strip() and not line.strip().startswith("-") and len(line.strip()) > 20
            ]
            non_solution_chunks += [
                line.strip().lstrip("- ") for line in _CHAT_CONTEXT.strip().splitlines()
                if line.strip().startswith("-") and len(line.strip()) > 20
            ]

            self._solution_embeddings = model.embed_documents(solution_chunks)
            self._non_solution_embeddings = model.embed_documents(non_solution_chunks)

            logger.info(
                f"[IntentClassifier] Context embeddings ready: "
                f"{len(self._solution_embeddings)} solution + "
                f"{len(self._non_solution_embeddings)} non-solution chunks"
            )
            return True
        except Exception as e:
            return False

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # -------------------------------------------------------------------------
    # STEP 1: KNOWLEDGE QUESTION FAST-REJECT
    # -------------------------------------------------------------------------

    def _fast_reject_knowledge_question(self, user_input: str) -> Optional[Tuple[bool, float, str]]:
        """
        Check if the query is a knowledge/information question that definitively
        does NOT belong in the solution workflow.

        Returns:
            (is_solution=False, confidence=0.9, method="fast_reject") if matched,
            None if the query could still be a solution request.
        """
        input_lower = user_input.lower().strip()

        # Check high-value knowledge topic patterns first (e.g. "what is SIL2")
        for pattern in KNOWLEDGE_TOPIC_PATTERNS:
            if pattern in input_lower:
                logger.info(
                    f"[IntentClassifier] Fast-reject: knowledge topic '{pattern}' matched"
                )
                return (False, 0.92, "fast_reject")

        # Check question-starter prefixes
        for starter in KNOWLEDGE_QUESTION_STARTS:
            if input_lower.startswith(starter):
                # Exception: "what is ... [system/package/design]" could still be solution-adjacent
                # but only if it contains strong solution keywords after the question word
                strong_solution_check = any(
                    kw in input_lower for kw in [
                        "complete system", "design a", "instrumentation package",
                        "measurement system for", "monitoring system for",
                        "control system for", "instrumentation for",
                    ]
                )
                if not strong_solution_check:
                    logger.info(
                        f"[IntentClassifier] Fast-reject: knowledge question prefix "
                        f"'{starter}' matched (no strong solution indicator)"
                    )
                    return (False, 0.88, "fast_reject")

        return None

    # -------------------------------------------------------------------------
    # STEP 2: SEMANTIC CLASSIFICATION
    # -------------------------------------------------------------------------

    def _semantic_classify(self, user_input: str) -> Optional[Tuple[bool, float]]:
        """
        Classify intent using embedding similarity against workflow context descriptions.

        Returns:
            Tuple of (is_solution, confidence) or None if embeddings unavailable
        """
        if not self._compute_reference_embeddings():
            return None

        user_embedding = self._compute_embedding(user_input)
        if user_embedding is None:
            return None

        solution_similarities = [
            self._cosine_similarity(user_embedding, ref)
            for ref in self._solution_embeddings
        ]
        max_solution_sim = max(solution_similarities) if solution_similarities else 0.0

        non_solution_similarities = [
            self._cosine_similarity(user_embedding, ref)
            for ref in self._non_solution_embeddings
        ]
        max_non_solution_sim = max(non_solution_similarities) if non_solution_similarities else 0.0

        similarity_gap = max_solution_sim - max_non_solution_sim
        confidence = 0.6 * max_solution_sim + 0.4 * max(0, similarity_gap + 0.5)
        confidence = min(1.0, max(0.0, confidence))
        is_solution = confidence >= SOLUTION_CONFIDENCE_THRESHOLD

        logger.info(
            f"[IntentClassifier] Semantic: max_sol={max_solution_sim:.3f}, "
            f"max_nonsol={max_non_solution_sim:.3f}, gap={similarity_gap:.3f}, "
            f"confidence={confidence:.3f}, is_solution={is_solution}"
        )
        return (is_solution, confidence)

    # -------------------------------------------------------------------------
    # STEP 3: TAXONOMY RAG CONTEXT SIGNAL (optional, non-blocking)
    # -------------------------------------------------------------------------

    def _taxonomy_context_signal(self, user_input: str) -> Optional[Tuple[bool, float]]:
        """
        Check if the query is asking *about* a known instrument/accessory term
        (chat intent) vs *requesting* instruments (solution/search intent).

        Returns (is_solution, confidence_boost) or None if unavailable.
        """
        try:
            from taxonomy_rag import get_taxonomy_rag
            rag = get_taxonomy_rag()
            results = rag.retrieve(user_input, top_k=3)

            if not results:
                return None

            input_lower = user_input.lower()
            top_result = results[0]
            term_name = (top_result.get("name") or "").lower()
            score = top_result.get("score", 0.0)

            if score is None or score < 0.6:
                return None

            # If the query starts with a knowledge question prefix AND mentions a
            # known instrument term → definite chat intent
            is_knowledge_query = any(
                input_lower.startswith(s) for s in KNOWLEDGE_QUESTION_STARTS
            )
            if is_knowledge_query and term_name and term_name in input_lower:
                logger.info(
                    f"[IntentClassifier] TaxonomyRAG: knowledge query about known "
                    f"term '{term_name}' (score={score:.2f}) → chat"
                )
                return (False, 0.85)

            # If query mentions instrument term with no question prefix → lean solution/search
            if not is_knowledge_query and score > 0.8:
                return (True, 0.55)

            return None

        except Exception as e:
            logger.debug(f"[IntentClassifier] TaxonomyRAG signal unavailable: {e}")
            return None

    # -------------------------------------------------------------------------
    # STEP 4: KEYWORD CLASSIFICATION (fallback)
    # -------------------------------------------------------------------------

    def _keyword_classify(self, user_input: str) -> Tuple[bool, float]:
        """
        Keyword-based classification using strong and weak keyword sets.

        Strong keywords: multi-word phrases that unambiguously indicate system design.
        Weak keywords: single words that need multiple matches to be confident.
        """
        input_lower = user_input.lower()

        # Strong keywords — single match is sufficient for a solution signal
        strong_matches = [kw for kw in SOLUTION_KEYWORDS if kw in input_lower]
        if strong_matches:
            confidence = min(0.90, 0.65 + len(strong_matches) * 0.07)
            logger.info(
                f"[IntentClassifier] Keyword (strong): {strong_matches} → "
                f"is_solution=True, confidence={confidence:.2f}"
            )
            return (True, confidence)

        # Weak keywords — need 2+ matches
        weak_matches = [kw for kw in SOLUTION_KEYWORDS_WEAK if kw in input_lower]
        match_count = len(weak_matches)

        if match_count >= 3:
            confidence = min(0.85, 0.60 + match_count * 0.05)
            return (True, confidence)
        elif match_count == 2:
            return (True, 0.60)
        elif match_count == 1:
            return (True, 0.35)  # Low confidence — single weak keyword
        else:
            return (False, 0.10)

    # -------------------------------------------------------------------------
    # STEP 5: CONTEXTUAL CLASSIFY
    # -------------------------------------------------------------------------

    def _contextual_classify(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
    ) -> Optional[Tuple[bool, float]]:
        """
        Use conversation history to boost solution confidence.
        Only boosts if prior conversation indicates a system-design context.
        """
        if not conversation_history:
            return None

        recent_messages = conversation_history[-5:]
        solution_context_count = 0

        for msg in recent_messages:
            content = msg.get("content", "").lower()
            if any(kw in content for kw in ["solution", "system", "design", "instrument package", "multiple instruments"]):
                solution_context_count += 1

        if solution_context_count > 0:
            boost = min(0.20, solution_context_count * 0.05)
            return (True, boost)

        return None

    # -------------------------------------------------------------------------
    # DOMAIN EXTRACTION
    # -------------------------------------------------------------------------

    def _extract_domain(self, user_input: str) -> Tuple[str, str]:
        """Extract domain and industry from user input."""
        input_lower = user_input.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in input_lower:
                    return (domain, domain)
        return ("General Industrial", "Industrial")

    # -------------------------------------------------------------------------
    # MAIN CLASSIFY METHOD
    # -------------------------------------------------------------------------

    def classify(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        Classify user input intent using the priority cascade.

        Priority:
        1. Knowledge-question fast-reject (fast, no LLM/embedding)
        2. Semantic similarity against workflow context descriptions
        3. Taxonomy RAG context signal (optional, non-blocking)
        4. Keyword matching (strong → weak)
        5. Contextual boost from conversation history

        Args:
            user_input: The user's input text
            conversation_history: Previous conversation messages

        Returns:
            IntentResult with classification details
        """
        conversation_history = conversation_history or []
        input_lower = user_input.lower()
        domain, industry = self._extract_domain(user_input)

        # ------------------------------------------------------------------
        # STEP 1: Knowledge-question fast-reject
        # ------------------------------------------------------------------
        fast_reject = self._fast_reject_knowledge_question(user_input)
        if fast_reject is not None:
            is_solution, confidence, method = fast_reject
            indicators = []
            result = IntentResult(
                is_solution=False,
                confidence=confidence,
                method=method,
                intent_type="chat",
                domain=domain,
                industry=industry,
                solution_indicators=indicators,
                extracted_info={
                    "domain": domain,
                    "industry": industry,
                    "keyword_matches": 0,
                    "semantic_available": False,
                    "fast_reject": True,
                },
            )
            logger.info(
                f"[IntentClassifier] Result: is_solution=False, "
                f"confidence={confidence:.3f}, method={method}, "
                f"intent_type=chat (fast reject)"
            )
            return result

        # ------------------------------------------------------------------
        # STEP 2: Semantic classification (embeddings against context docs)
        # ------------------------------------------------------------------
        semantic_result = self._semantic_classify(user_input)

        # ------------------------------------------------------------------
        # STEP 3: Taxonomy RAG context signal (non-blocking)
        # ------------------------------------------------------------------
        taxonomy_result = self._taxonomy_context_signal(user_input)

        # ------------------------------------------------------------------
        # STEP 4: Keyword classification (fallback)
        # ------------------------------------------------------------------
        keyword_result = self._keyword_classify(user_input)

        # ------------------------------------------------------------------
        # STEP 5: Contextual boost
        # ------------------------------------------------------------------
        contextual_result = self._contextual_classify(user_input, conversation_history)

        # ------------------------------------------------------------------
        # DECISION: Combine signals
        # ------------------------------------------------------------------
        if semantic_result is not None:
            is_solution, confidence = semantic_result
            method = "semantic"

            # Taxonomy RAG override: if taxonomy strongly says NOT solution, trust it
            if taxonomy_result is not None:
                tax_is_solution, tax_confidence = taxonomy_result
                if not tax_is_solution and tax_confidence > 0.8:
                    is_solution = False
                    confidence = max(confidence, tax_confidence)
                    method = "hybrid_taxonomy"

            # Contextual boost (only boosts, never lowers)
            if contextual_result is not None and is_solution:
                _, context_boost = contextual_result
                confidence = min(1.0, confidence + context_boost)
                method = "hybrid"

            # Cross-validate with strong keywords
            kw_is_solution, kw_confidence = keyword_result
            if is_solution != kw_is_solution and kw_confidence > 0.70:
                confidence = (confidence + kw_confidence) / 2
                is_solution = confidence >= SOLUTION_CONFIDENCE_THRESHOLD
                method = "hybrid"

        else:
            # Embeddings unavailable: use taxonomy signal if available, else keyword
            if taxonomy_result is not None:
                is_solution, confidence = taxonomy_result
                method = "taxonomy_fallback"
            else:
                is_solution, confidence = keyword_result
                method = "keyword"

            if contextual_result is not None:
                _, context_boost = contextual_result
                if is_solution:
                    confidence = min(1.0, confidence + context_boost)
                method = "contextual" if method == "keyword" else method

        # ------------------------------------------------------------------
        # INTENT TYPE RESOLUTION
        # ------------------------------------------------------------------
        indicators = [kw for kw in SOLUTION_KEYWORDS + SOLUTION_KEYWORDS_WEAK if kw in input_lower]

        if is_solution:
            intent_type = "solution"
        elif any(kw in input_lower for kw in KNOWLEDGE_QUESTION_STARTS):
            intent_type = "chat"
        elif any(kw in input_lower for kw in ["modify", "change", "update", "remove", "add"]):
            intent_type = "modification"
        elif any(kw in input_lower for kw in ["compare", "versus", "vs", "difference"]):
            intent_type = "comparison"
        else:
            intent_type = "unknown"

        result = IntentResult(
            is_solution=is_solution,
            confidence=confidence,
            method=method,
            intent_type=intent_type,
            domain=domain,
            industry=industry,
            solution_indicators=indicators[:10],
            extracted_info={
                "domain": domain,
                "industry": industry,
                "keyword_matches": len(indicators),
                "semantic_available": semantic_result is not None,
                "taxonomy_available": taxonomy_result is not None,
                "contextual_boost": contextual_result[1] if contextual_result else 0.0,
            },
        )

        logger.info(
            f"[IntentClassifier] Result: is_solution={result.is_solution}, "
            f"confidence={result.confidence:.3f}, method={result.method}, "
            f"intent_type={result.intent_type}, domain={result.domain}"
        )
        return result
