"""
Intent Classification Routing Agent

Routes user input from the UI textarea to the appropriate agentic workflow:
1. Solution Workflow - Complex engineering challenges requiring multiple instruments
2. Instrument Identifier Workflow - Single product requirements
3. Product Info Workflow - Questions about products, standards, vendors

Also rejects out-of-domain queries (unrelated to industrial automation).

Usage:
    agent = IntentClassificationRoutingAgent()
    result = agent.classify(query="I need a pressure transmitter 0-100 PSI", session_id="abc123")
    # Returns: WorkflowRoutingResult with target workflow and reasoning
"""

import logging
import time
import random  # Added for greeting responses
from typing import Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

# Debug imports
from debug_flags import debug_log, timed_execution, is_debug_enabled

logger = logging.getLogger(__name__)

# Level 4.5: Import WorkflowRegistry for registry-based matching
try:
    from .workflow_registry import get_workflow_registry, WorkflowRegistry
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False
    logger.debug("[Router] WorkflowRegistry not available - using IntentConfig fallback")

# PATH 2: Import metrics module
# Note: Domain validation is now handled by SemanticIntentClassifier's fast-path rejection
try:
    from .classification_metrics import get_classification_metrics
    METRICS_AVAILABLE = True
    logger.info("[INTENT_ROUTER] PATH 2 metrics module available")
except ImportError as e:
    logger.warning(f"[INTENT_ROUTER] PATH 2 metrics not available: {e}")
    METRICS_AVAILABLE = False

# Domain validation flag (removed - now handled by semantic classifier)
DOMAIN_VALIDATOR_AVAILABLE = False


# =============================================================================
# WORKFLOW STATE MEMORY (Session-based memory for workflow locking)
# =============================================================================

class WorkflowStateMemory:
    """
    CACHE for workflow state from frontend SessionManager.
    
    ARCHITECTURE: Frontend SessionManager is the SOURCE OF TRUTH.
    This class only:
    - Caches workflow hints from frontend for the current request
    - Validates hints against known workflows  
    - Clears state ONLY on explicit exit detection
    - Does NOT make independent workflow decisions
    
    The response contains target_workflow which frontend uses to update
    SessionManager. Frontend then passes workflow_hint on next request.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._workflow_states: Dict[str, str] = {}
        self._workflow_timestamps: Dict[str, datetime] = {}
        self._state_lock = Lock()
        logger.info("[WorkflowStateMemory] Initialized - Backend workflow state tracking enabled")
    
    def get_workflow(self, session_id: str) -> Optional[str]:
        """Get current workflow for a session."""
        with self._state_lock:
            workflow = self._workflow_states.get(session_id)
            if workflow:
                logger.debug(f"[WorkflowStateMemory] Session {session_id[:8]}...: current workflow = {workflow}")
            return workflow
    
    def set_workflow(self, session_id: str, workflow: str) -> None:
        """Set workflow for a session."""
        with self._state_lock:
            old_workflow = self._workflow_states.get(session_id)
            self._workflow_states[session_id] = workflow
            self._workflow_timestamps[session_id] = datetime.now()
            logger.info(f"[WorkflowStateMemory] Session {session_id[:8]}...: workflow changed {old_workflow} -> {workflow}")
    
    def clear_workflow(self, session_id: str) -> None:
        """Clear workflow for a session (allows re-classification)."""
        with self._state_lock:
            old_workflow = self._workflow_states.pop(session_id, None)
            self._workflow_timestamps.pop(session_id, None)
            if old_workflow:
                logger.info(f"[WorkflowStateMemory] Session {session_id[:8]}...: workflow cleared (was {old_workflow})")
    
    def is_locked(self, session_id: str) -> bool:
        """Check if session has an active workflow."""
        return self.get_workflow(session_id) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._state_lock:
            return {
                "active_sessions": len(self._workflow_states),
                "workflows": dict(self._workflow_states)
            }

# Global singleton instance
_workflow_memory = WorkflowStateMemory()

def get_workflow_memory() -> WorkflowStateMemory:
    """Get the global workflow state memory instance."""
    return _workflow_memory


# =============================================================================
# EXIT DETECTION
# =============================================================================

# Phrases that indicate user wants to exit current workflow (rule-based fallback)
EXIT_PHRASES = [
    "start over", "new search", "reset", "clear", "begin again", 
    "start new", "exit", "quit", "cancel", "back to start"
]

# Greetings that indicate new conversation (rule-based fallback)
GREETING_PHRASES = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def _should_exit_rule_based(user_input: str) -> bool:
    """Rule-based fallback for exit detection (used when LLM is unavailable)."""
    lower_input = user_input.lower().strip()
    
    # Check for exit phrases
    if any(phrase in lower_input for phrase in EXIT_PHRASES):
        return True
    
    # Check for pure greetings (new conversation)
    if lower_input in GREETING_PHRASES:
        return True
    
    return False

def should_exit_workflow(user_input: str) -> bool:
    """
    Check if user wants to exit current workflow using LLM classification.
    
    Uses the classify_intent_tool with temperature 0.0 for deterministic results.
    Falls back to rule-based detection if LLM fails.
    """
    try:
        # Import here to avoid circular imports
        from common.tools.intent_tools import _classify_llm_fast
        
        result = _classify_llm_fast(user_input)
        
        if result is not None:
            intent = result.get("intent", "")
            # Exit if classified as 'exit' or 'greeting' (new conversation)
            if intent in {"exit", "greeting"}:
                logger.info(f"[ExitDetection] LLM detected exit intent: '{intent}'")
                return True
            return False
        
        # LLM returned 'unknown' - not an exit intent
        return False
        
    except Exception as e:
        logger.warning(f"[ExitDetection] LLM classification failed: {e}, using rule-based fallback")
        return _should_exit_rule_based(user_input)


# =============================================================================
# WORKFLOW TARGETS (Import canonical enum from semantic_classifier)
# =============================================================================

# Import canonical WorkflowType from semantic_classifier (SOURCE OF TRUTH)
try:
    from .semantic_classifier import WorkflowType as _WorkflowType
    # Create backward-compatible alias
    WorkflowTarget = _WorkflowType
    logger.debug("[INTENT_ROUTER] Using canonical WorkflowType from semantic_classifier")
except ImportError:
    # Fallback: Define locally if semantic_classifier unavailable
    class WorkflowTarget(Enum):
        """Available workflow routing targets (fallback definition)."""
        SOLUTION_WORKFLOW = "solution"
        INSTRUMENT_IDENTIFIER = "instrument_identifier"
        ENGENIE_CHAT = "engenie_chat"
        OUT_OF_DOMAIN = "out_of_domain"
        GREETING = "greeting"
        CONVERSATIONAL = "conversational"
    logger.warning("[INTENT_ROUTER] Using local WorkflowTarget (semantic_classifier unavailable)")


# =============================================================================
# WORKFLOW ROUTING RESULT
# =============================================================================

@dataclass
class WorkflowRoutingResult:
    """Result of workflow routing classification."""
    query: str                          # Original query
    target_workflow: WorkflowTarget     # Which workflow to route to
    intent: str                         # Raw intent from classify_intent_tool
    confidence: float                   # Confidence (0.0-1.0)
    reasoning: str                      # Explanation for routing decision
    is_solution: bool                   # Whether this is a solution-type request
    target_rag: Optional[str]           # For ProductInfo: standards_rag, strategy_rag, product_info_rag
    solution_indicators: list           # Indicators that triggered solution detection
    extracted_info: Dict                # Any extracted information
    classification_time_ms: float       # Time taken to classify
    timestamp: str                      # ISO timestamp
    reject_message: Optional[str]       # Message for out-of-domain queries
    direct_response: Optional[str] = None  # Direct response (e.g., for greetings) to skip workflow execution

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "target_workflow": self.target_workflow.value,
            "intent": self.intent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "is_solution": self.is_solution,
            "target_rag": self.target_rag,
            "solution_indicators": self.solution_indicators,
            "extracted_info": self.extracted_info,
            "classification_time_ms": self.classification_time_ms,
            "timestamp": self.timestamp,
            "reject_message": self.reject_message,
            "direct_response": self.direct_response
        }


# =============================================================================
# OUT OF DOMAIN RESPONSE
# =============================================================================

OUT_OF_DOMAIN_MESSAGE = """I'm EnGenie, your industrial automation assistant. I can help with:

✓ Industrial automation products (transmitters, sensors, valves, PLCs, controllers)
✓ Technical specifications and certifications  
✓ Standards and compliance (IEC, ISO, ATEX, SIL, API, ASME)
✓ Vendor information and product comparisons
✓ Process instrumentation and control systems

Your query appears to be outside my expertise area. Please ask about industrial automation equipment, technical standards, or related topics."""


# =============================================================================
# INTENT TO WORKFLOW MAPPING
# =============================================================================
# Unified intent-to-workflow mapping with validation and is_solution override logic.
# This is the SINGLE SOURCE OF TRUTH for intent routing decisions.

class IntentConfig:
    """
    Centralized intent-to-workflow mapping with validation.

    Handles the complex logic of routing based on both intent name and is_solution flag.
    PHASE 1 FIX: Replaces hardcoded INTENT_TO_WORKFLOW_MAP with robust configuration.
    """

    # Primary intent-to-workflow mappings
    # These are the ONLY valid intent values the LLM should return
    # Supports both uppercase (from new prompt) and lowercase (legacy) intent names
    INTENT_MAPPINGS = {
        # ===== NEW 4-INTENT ARCHITECTURE (uppercase from LLM, lowercase normalized) =====
        # Solution Workflow - Complex systems with multiple instruments
        "solution": WorkflowTarget.SOLUTION_WORKFLOW,

        # Instrument Identifier Workflow - Single product with clear specifications
        "search": WorkflowTarget.INSTRUMENT_IDENTIFIER,

        # EnGenie Chat Workflow - Knowledge, education, conversational queries
        "chat": WorkflowTarget.ENGENIE_CHAT,

        # Greeting (NEW) - handled directly
        "greeting": WorkflowTarget.GREETING,
        "greetings": WorkflowTarget.GREETING,

        # Conversational Intents (Direct Response)
        "farewell": WorkflowTarget.CONVERSATIONAL,
        "gratitude": WorkflowTarget.CONVERSATIONAL,
        "help": WorkflowTarget.CONVERSATIONAL,
        "chitchat": WorkflowTarget.CONVERSATIONAL,
        "complaint": WorkflowTarget.CONVERSATIONAL,
        "gibberish": WorkflowTarget.CONVERSATIONAL,

        # Out of Domain - Unrelated to industrial automation
        "invalid_input": WorkflowTarget.OUT_OF_DOMAIN,

        # ===== LEGACY MAPPINGS (for backwards compatibility) =====
        # Solution Workflow variants
        "system": WorkflowTarget.SOLUTION_WORKFLOW,
        "systems": WorkflowTarget.SOLUTION_WORKFLOW,
        "complex_system": WorkflowTarget.SOLUTION_WORKFLOW,
        "design": WorkflowTarget.SOLUTION_WORKFLOW,

        # Instrument Identifier Workflow (legacy names)
        "requirements": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "additional_specs": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "instrument": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "accessories": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "spec_request": WorkflowTarget.INSTRUMENT_IDENTIFIER,

        # EnGenie Chat Workflow (legacy names)
        "question": WorkflowTarget.ENGENIE_CHAT,
        "productinfo": WorkflowTarget.ENGENIE_CHAT,
        "product_info": WorkflowTarget.ENGENIE_CHAT,
        # "greeting": WorkflowTarget.ENGENIE_CHAT, # Replaced by dedicated GREETING target
        "confirm": WorkflowTarget.ENGENIE_CHAT,
        "reject": WorkflowTarget.ENGENIE_CHAT,
        "standards": WorkflowTarget.ENGENIE_CHAT,
        "vendor_strategy": WorkflowTarget.ENGENIE_CHAT,
        "grounded_chat": WorkflowTarget.ENGENIE_CHAT,
        "comparison": WorkflowTarget.ENGENIE_CHAT,
        "productcomparison": WorkflowTarget.ENGENIE_CHAT,

        # Out of Domain (legacy names)
        "chitchat": WorkflowTarget.CONVERSATIONAL, # Remapped from OUT_OF_DOMAIN
        "unrelated": WorkflowTarget.OUT_OF_DOMAIN,
    }

    # Intents that FORCE Solution Workflow regardless of other signals
    # These are "strong indicators" that override even low confidence
    # Includes both new (4-intent) and legacy intent names
    SOLUTION_FORCING_INTENTS = {
        "solution",  # New 4-intent architecture
        "system", "systems", "complex_system", "design"  # Legacy names
    }

    @classmethod
    def get_workflow(cls, intent: str, is_solution: bool = False) -> WorkflowTarget:
        """
        Determine workflow target based on intent and is_solution flag.

        ROUTING LOGIC (in priority order):
        1. If intent is in SOLUTION_FORCING_INTENTS → SOLUTION_WORKFLOW
        2. If is_solution=True → SOLUTION_WORKFLOW (override via flag)
        3. Otherwise → Use intent mapping (default: ENGENIE_CHAT for unknown)

        Args:
            intent: Intent classification string from LLM or rule-based classifier
            is_solution: Boolean flag indicating complex system detection

        Returns:
            WorkflowTarget enum value

        Examples:
            get_workflow("solution", is_solution=False) → SOLUTION_WORKFLOW
            get_workflow("question", is_solution=True) → SOLUTION_WORKFLOW
            get_workflow("unknown_intent", is_solution=False) → ENGENIE_CHAT
        """
        intent_lower = intent.lower().strip() if intent else "unrelated"

        # Priority 1: Intent is a solution forcing intent
        if intent_lower in cls.SOLUTION_FORCING_INTENTS:
            logger.debug(f"[IntentConfig] Solution forcing intent: '{intent_lower}'")
            return WorkflowTarget.SOLUTION_WORKFLOW

        # Priority 2: is_solution flag is set (explicit complex system detection)
        if is_solution:
            logger.debug(f"[IntentConfig] is_solution flag set, routing to SOLUTION")
            return WorkflowTarget.SOLUTION_WORKFLOW

        # Priority 3: Use intent mapping (defaults to ENGENIE_CHAT for unknowns)
        workflow = cls.INTENT_MAPPINGS.get(intent_lower, WorkflowTarget.ENGENIE_CHAT)
        logger.debug(f"[IntentConfig] Intent '{intent_lower}' → {workflow.value}")
        return workflow

    @classmethod
    def is_known_intent(cls, intent: str) -> bool:
        """Check if intent is in the recognized list."""
        if not intent:
            return False
        return intent.lower().strip() in cls.INTENT_MAPPINGS

    @classmethod
    def get_all_valid_intents(cls) -> list:
        """Get list of all recognized intent values for validation and documentation."""
        return list(cls.INTENT_MAPPINGS.keys())

    # Intent-specific confidence thresholds (PATH 2 IMPROVEMENT)
    # Different workflows have different complexity and cost, so require different confidence levels
    CONFIDENCE_THRESHOLDS = {
        WorkflowTarget.ENGENIE_CHAT: 0.60,              # Low risk, fast, can handle ambiguity
        WorkflowTarget.INSTRUMENT_IDENTIFIER: 0.75,     # Medium risk, expensive (30-60s workflow)
        WorkflowTarget.SOLUTION_WORKFLOW: 0.80,         # High risk, very expensive (20-40s workflow)
        WorkflowTarget.OUT_OF_DOMAIN: 0.90,             # Critical, user rejection, must be certain
        WorkflowTarget.GREETING: 0.90,                  # High confidence required for exact matches
        WorkflowTarget.CONVERSATIONAL: 0.90             # High confidence for conversational patterns
    }

    @classmethod
    def should_accept_classification(
        cls,
        workflow: WorkflowTarget,
        confidence: float
    ) -> bool:
        """
        Check if confidence meets threshold for target workflow.

        Args:
            workflow: Target workflow
            confidence: Classification confidence (0.0-1.0)

        Returns:
            True if confidence is acceptable, False if too low

        Examples:
            >>> should_accept_classification(WorkflowTarget.ENGENIE_CHAT, 0.65)
            True  # CHAT accepts lower confidence (0.60)

            >>> should_accept_classification(WorkflowTarget.SOLUTION_WORKFLOW, 0.75)
            False  # SOLUTION requires higher confidence (0.80)
        """
        threshold = cls.CONFIDENCE_THRESHOLDS.get(workflow, 0.70)
        accepted = confidence >= threshold

        if not accepted:
            logger.debug(
                f"[IntentConfig] Confidence {confidence:.2f} below threshold "
                f"{threshold:.2f} for {workflow.value}"
            )

        return accepted

    @classmethod
    def needs_disambiguation(
        cls,
        workflow: WorkflowTarget,
        confidence: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if classification needs user disambiguation.

        Returns:
            (needs_disambiguation, question_text)

        Examples:
            >>> needs_disambiguation(WorkflowTarget.ENGENIE_CHAT, 0.55)
            (True, "Do you want to learn about this topic, or find products?")

            >>> needs_disambiguation(WorkflowTarget.ENGENIE_CHAT, 0.85)
            (False, None)
        """
        threshold = cls.CONFIDENCE_THRESHOLDS.get(workflow, 0.70)

        if confidence < threshold:
            # Generate disambiguation question based on workflow
            questions = {
                WorkflowTarget.ENGENIE_CHAT:
                    "Do you want to learn about this topic, or find specific products?",
                WorkflowTarget.INSTRUMENT_IDENTIFIER:
                    "Are you looking for a specific product to purchase?",
                WorkflowTarget.SOLUTION_WORKFLOW:
                    "Are you designing a complete system with multiple instruments?",
                WorkflowTarget.OUT_OF_DOMAIN:
                    "Your query might be outside my scope. Could you rephrase it in terms of instrumentation?"
            }

            question = questions.get(workflow, "I'm not sure how to help. Could you clarify your request?")
            logger.info(f"[IntentConfig] Disambiguation needed for {workflow.value} (conf={confidence:.2f})")
            return True, question

        return False, None

    @classmethod
    def get_confidence_threshold(cls, workflow: WorkflowTarget) -> float:
        """Get confidence threshold for a specific workflow."""
        return cls.CONFIDENCE_THRESHOLDS.get(workflow, 0.70)

    @classmethod
    def print_config(cls):
        """Print configuration for debugging (logging)."""
        logger.info("[IntentConfig] ===== INTENT CONFIGURATION =====")
        logger.info(f"[IntentConfig] Total intents: {len(cls.INTENT_MAPPINGS)}")
        logger.info(f"[IntentConfig] Solution forcing intents: {cls.SOLUTION_FORCING_INTENTS}")
        logger.info("[IntentConfig] Confidence thresholds:")
        for workflow, threshold in cls.CONFIDENCE_THRESHOLDS.items():
            logger.info(f"[IntentConfig]   {workflow.value}: {threshold:.2f}")
        logger.info("[IntentConfig] Intent mappings by workflow:")
        for workflow in WorkflowTarget:
            intents = [i for i, w in cls.INTENT_MAPPINGS.items() if w == workflow]
            logger.info(f"[IntentConfig]   {workflow.value}: {intents}")


# Legacy alias for backward compatibility (can be removed in next version)
INTENT_TO_WORKFLOW_MAP = IntentConfig.INTENT_MAPPINGS


# =============================================================================
# INTENT CLASSIFICATION ROUTING AGENT
# =============================================================================

class IntentClassificationRoutingAgent:
    """
    Agent that classifies user queries and routes to appropriate workflows.

    Uses classify_intent_tool as the core classifier and maps intents to
    workflow targets.
    
    WORKFLOW STATE LOCKING:
    - Tracks which workflow each session is in via WorkflowStateMemory
    - Once a session enters a workflow, subsequent queries stay in that workflow
    - User can exit workflow by saying "start over", "reset", etc.

    Workflow Routing:
    - solution → Solution Workflow (complex systems)
    - requirements, additional_specs → Instrument Identifier Workflow
    - question, productInfo, greeting, confirm, reject → EnGenie Chat Workflow
    - chitchat, unrelated → OUT_OF_DOMAIN (reject)
    """

    def __init__(self, name: str = "WorkflowRouter", use_registry: bool = True):
        """Initialize the agent.
        
        Args:
            name: Agent name for logging
            use_registry: If True, use WorkflowRegistry for matching (Level 4.5).
                         Falls back to IntentConfig if registry unavailable.
        """
        self.name = name
        self.classification_count = 0
        self.last_classification_time_ms = 0.0
        self._memory = get_workflow_memory()  # Use singleton workflow memory
        self._use_registry = use_registry and _REGISTRY_AVAILABLE
        
        if self._use_registry:
            self._registry = get_workflow_registry()
            logger.info(f"[{self.name}] Initialized with WorkflowRegistry (Level 4.5)")
        else:
            self._registry = None
            logger.info(f"[{self.name}] Initialized with IntentConfig fallback")

    def _is_query_relevant_to_workflow(self, query: str, workflow: str) -> bool:
        """
        Check if query is relevant to the currently locked workflow.
        
        This prevents knowledge/question queries from being incorrectly routed
        to SOLUTION workflow when session is locked.
        
        Args:
            query: User query string
            workflow: Currently locked workflow name
            
        Returns:
            True if query is relevant to workflow, False if it should break the lock
        """
        query_lower = query.lower().strip()
        
        # Knowledge/question patterns that are NOT relevant to solution workflow
        if workflow == "solution":
            # Question starters - these are knowledge queries, not solution continuation
            knowledge_patterns = [
                "what is", "what are", "what's the", "what does",
                "how does", "how do", "how is", "how are",
                "explain", "tell me about", "describe",
                "why is", "why does", "why do",
                "can you explain", "can you tell",
                "input current", "output range", "accuracy",
                "specification", "specs of", "datasheet"
            ]
            
            # Check for knowledge patterns
            if any(query_lower.startswith(p) or f" {p}" in query_lower for p in knowledge_patterns):
                logger.info(
                    f"[{self.name}] Query not relevant to solution workflow "
                    f"(knowledge pattern detected): '{query[:50]}...'"
                )
                return False
            
            # Solution-relevant patterns (should stay locked)
            solution_relevant = [
                "add", "include", "remove", "select", "confirm",
                "i've selected", "use this", "continue", "proceed",
                "next", "done", "finished", "submit"
            ]
            if any(p in query_lower for p in solution_relevant):
                return True
        
        # Default: consider relevant to respect workflow lock
        return True

    def _has_strong_solution_indicators(self, query: str) -> bool:
        """
        Check if query has strong indicators of a solution/system request.
        
        These indicators should preempt fast pattern matching for standards/keywords.
        Examples: "select and specify", "complete system", "design a system"
        """
        query_lower = query.lower().strip()
        
        strong_indicators = [
            "select and specify",
            "complete system",
            "design a system",
            "design a solution",
            "pump system",
            "measurement system",
            "control system",
            "instrumentation system",
            "complete instrumentation",
            "holistic solution",
            "full solution",
            "entire system"
        ]
        
        return any(indicator in query_lower for indicator in strong_indicators)

    def _check_conversational_patterns(self, query: str) -> Optional[Dict]:
        """
        Check for fast conversational patterns to bypass LLM classification.
        
        Handles:
        - Greetings (hi, hello)
        - Farewells (bye, exit)
        - Gratitude (thanks)
        - Help requests (help, stuck)
        
        Returns:
            Dict with 'intent' and 'direct_response' if matched, else None
        """
        query_lower = query.lower().strip()
        
        # 1. Greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if query_lower in greetings:
            return {
                "intent": "greeting",
                "direct_response": random.choice([
                    "Hello! I'm EnGenie. How can I help you with industrial automation today?",
                    "Hi there! Ready to assist with your instrumentation needs.",
                    "Greetings! What product or specification are you looking for?"
                ])
            }
            
        # 2. Farewells / Exit
        farewells = ["bye", "goodbye", "see you", "exit", "quit", "end"]
        if query_lower in farewells:
            return {
                "intent": "farewell",
                "direct_response": "Goodbye! Let me know if you need anything else for your projects."
            }
            
        # 3. Gratitude
        gratitude = ["thanks", "thank you", "thx", "appreciate it"]
        if any(g in query_lower for g in gratitude):
            return {
                "intent": "gratitude",
                "direct_response": "You're welcome! Happy to help."
            }
            
        # 4. Help
        if query_lower in ["help", "support", "assist"]:
            return {
                "intent": "help",
                "direct_response": "I can help you find products, compare specifications, or design complete instrumentation systems. Try asking for a pressure transmitter or a flow meter."
            }
            
        return None

    @debug_log("INTENT", log_args=False)
    @timed_execution("INTENT", threshold_ms=2000)
    def classify(
        self,
        query: str,
        session_id: str = "default",
        context: Optional[Dict] = None
    ) -> WorkflowRoutingResult:
        """
        Classify a query and determine which workflow to route to.

        SMART LOCKING:
        - Helper logic checks if the user is stuck in a workflow.
        - Strong intents (greeting, new solution) BREAK the lock.
        - Contextual intents (questions, confirmations) RESPECT the lock.

        Args:
            query: User query string from UI textarea
            session_id: Session ID for workflow state tracking
            context: Optional context (current workflow step, conversation history)

        Returns:
            WorkflowRoutingResult with target workflow and details
        """
        start_time = datetime.now()
        
        logger.info(f"[{self.name}] Classifying: '{query[:80]}...' (session: {session_id[:8]}...)")

        # =====================================================================
        # STEP 1: CHECK FOR EXPRESS EXIT & CONVERSATIONAL PATTERNS
        # =====================================================================
        if should_exit_workflow(query):
            self._memory.clear_workflow(session_id)
            logger.info(f"[{self.name}] Exit detected - clearing workflow state")
            # Proceed to classify as a fresh query
        
        # FAST CONVERSATIONAL CHECK (Optimization)
        # Check if query is a greeting, farewell, etc. to avoid expensive LLM calls
        conversational_result = self._check_conversational_patterns(query)
        if conversational_result:
            end_time = datetime.now()
            classification_time_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"[{self.name}] Fast conversational pattern detected: {conversational_result['intent']}")
            
            return WorkflowRoutingResult(
                query=query,
                target_workflow=WorkflowTarget.GREETING if conversational_result['intent'] == 'greeting' else WorkflowTarget.CONVERSATIONAL,
                intent=conversational_result['intent'],
                confidence=1.0,
                reasoning=f"Fast rule-based {conversational_result['intent']} detection",
                is_solution=False,
                target_rag=None,
                solution_indicators=[],
                extracted_info={"fast_pattern": True},
                classification_time_ms=classification_time_ms,
                timestamp=datetime.now().isoformat(),
                reject_message=None,
                direct_response=conversational_result["direct_response"]
            )

        # =====================================================================
        # STEP 1.5: DOMAIN VALIDATION (Removed - now handled by SemanticIntentClassifier)
        # SemanticIntentClassifier now has fast-path rejection using STRONG_REJECT_KEYWORDS
        # and OUT_OF_DOMAIN detection via embedding similarity.
        # =====================================================================



        # =====================================================================
        # STEP 2: HANDLE WORKFLOW HINT FROM FRONTEND
        # =====================================================================
        context = context or {}
        workflow_hint = context.get('workflow_hint')
        valid_workflows = {'engenie_chat', 'solution', 'instrument_identifier'}
        
        if workflow_hint and workflow_hint not in valid_workflows:
            logger.warning(f"[{self.name}] Invalid workflow hint ignored: {workflow_hint}")
            workflow_hint = None
        
        # =====================================================================
        # STEP 3: CHECK WORKFLOW LOCK (with session type differentiation)
        # =====================================================================
        current_workflow = self._memory.get_workflow(session_id)
        
        # If no backend state but valid hint from frontend, restore it
        if not current_workflow and workflow_hint and not should_exit_workflow(query):
            logger.info(f"[{self.name}] Restoring workflow from frontend hint: {workflow_hint}")
            current_workflow = workflow_hint
            self._memory.set_workflow(session_id, current_workflow)
        
        if current_workflow and not should_exit_workflow(query):
            # FIX: Differentiate between main interface sessions and dedicated workflow sessions
            # Main interface sessions (e.g., "main_Daman_DEFAULT") should allow free workflow switching
            # Dedicated workflow sessions (e.g., "engenie_chat_1234567890") should stay locked
            
            is_main_interface = session_id.startswith("main_")
            
            if is_main_interface:
                # Check if query is an in-workflow confirmation/rejection (yes/no)
                # These short responses are context-dependent and must stay in the current workflow
                _in_workflow_confirms = {
                    'yes', 'y', 'yep', 'yeah', 'sure', 'ok', 'okay',
                    'proceed', 'continue', 'go ahead', 'confirm', 'approved'
                }
                _in_workflow_rejects = {
                    'no', 'n', 'nope', 'reject', 'decline',
                    'never mind', 'nevermind', 'forget it'
                }
                query_lower = query.lower().strip()

                if query_lower in _in_workflow_confirms or query_lower in _in_workflow_rejects:
                    # Short-circuit: confirmation/rejection stays in current workflow
                    workflow_map = {
                        "engenie_chat": WorkflowTarget.ENGENIE_CHAT,
                        "instrument_identifier": WorkflowTarget.INSTRUMENT_IDENTIFIER,
                        "solution": WorkflowTarget.SOLUTION_WORKFLOW
                    }
                    target_workflow = workflow_map.get(current_workflow, WorkflowTarget.INSTRUMENT_IDENTIFIER)
                    intent = "confirm" if query_lower in _in_workflow_confirms else "reject"

                    logger.info(
                        f"[{self.name}] In-workflow {intent} detected: '{query}' → staying in "
                        f"{current_workflow} workflow (session: {session_id[:30]}...)"
                    )

                    end_time = datetime.now()
                    classification_time_ms = (end_time - start_time).total_seconds() * 1000

                    return WorkflowRoutingResult(
                        query=query,
                        target_workflow=target_workflow,
                        intent=intent,
                        confidence=1.0,
                        reasoning=f"In-workflow {intent}: User response '{query}' stays in {current_workflow} workflow",
                        is_solution=(current_workflow == "solution"),
                        target_rag=None,
                        solution_indicators=[],
                        extracted_info={
                            "in_workflow_response": True,
                            "current_workflow": current_workflow
                        },
                        classification_time_ms=classification_time_ms,
                        timestamp=datetime.now().isoformat(),
                        reject_message=None
                    )

                # Main interface - allow re-classification for workflow switching
                logger.info(
                    f"[{self.name}] Main interface session detected (session: {session_id[:30]}...). "
                    f"Previous workflow: '{current_workflow}'. Will re-classify to allow workflow switching."
                )
                # Clear the workflow lock to allow re-classification
                self._memory.clear_workflow(session_id)
                # Continue to classification steps below
            else:
                # Dedicated workflow session - respect the workflow lock
                logger.info(
                    f"[{self.name}] WORKFLOW LOCKED: Dedicated session in '{current_workflow}' "
                    f"(session: {session_id[:30]}...)"
                )
                
                # Map workflow to target
                workflow_map = {
                    "engenie_chat": WorkflowTarget.ENGENIE_CHAT,
                    "instrument_identifier": WorkflowTarget.INSTRUMENT_IDENTIFIER,
                    "solution": WorkflowTarget.SOLUTION_WORKFLOW
                }
                
                target_workflow = workflow_map.get(current_workflow, WorkflowTarget.ENGENIE_CHAT)
                
                # Calculate time
                end_time = datetime.now()
                classification_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return WorkflowRoutingResult(
                    query=query,
                    target_workflow=target_workflow,
                    intent="workflow_locked",
                    confidence=1.0,
                    reasoning=f"Dedicated session locked in {current_workflow} workflow",
                    is_solution=(current_workflow == "solution"),
                    target_rag=None,
                    solution_indicators=[],
                    extracted_info={
                        "workflow_locked": True, 
                        "current_workflow": current_workflow,
                        "session_type": "dedicated"
                    },
                    classification_time_ms=classification_time_ms,
                    timestamp=datetime.now().isoformat(),
                    reject_message=None
                )

        # =====================================================================
        # STEP 3.5: SEMANTIC CLASSIFICATION (PRIMARY - Run First for Accuracy)
        # =====================================================================
        # IMPROVED: Run semantic classification BEFORE fast pattern matching
        # Rationale: Semantic (300ms) is more accurate than pattern (0ms) for
        # detecting out-of-domain queries and avoiding false positives from
        # industrial keywords in wrong context
        
        semantic_result = None
        try:
            from common.agentic.agents.routing.semantic_classifier import (
                get_semantic_classifier, 
                WorkflowType,
                ClassificationResult
            )
            
            semantic_classifier = get_semantic_classifier()
            semantic_result = semantic_classifier.classify(query)
            
            logger.info(
                f"[{self.name}] SEMANTIC: {semantic_result.workflow.value} "
                f"(conf={semantic_result.confidence:.3f}, match='{semantic_result.matched_signature[:50]}...')"
            )
            
            # PRIORITY 1: OUT_OF_DOMAIN detection with HIGH confidence
            # Reject immediately if strongly classified as out-of-domain
            if (semantic_result.workflow == WorkflowType.OUT_OF_DOMAIN and 
                semantic_result.confidence >= 0.70):
                logger.warning(
                    f"[{self.name}] OUT_OF_DOMAIN detected by semantic classifier "
                    f"(conf={semantic_result.confidence:.3f})"
                )
                
                # Use rejection message defined at module level
                
                end_time = datetime.now()
                classification_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return WorkflowRoutingResult(
                    query=query,
                    target_workflow=WorkflowTarget.OUT_OF_DOMAIN,
                    intent="out_of_domain",
                    confidence=semantic_result.confidence,
                    reasoning=f"Semantic OUT_OF_DOMAIN: {semantic_result.reasoning}",
                    is_solution=False,
                    target_rag=None,
                    solution_indicators=[],
                    extracted_info={
                        "semantic_ood_detection": True,
                        "matched_signature": semantic_result.matched_signature,
                        "all_scores": semantic_result.all_scores
                    },
                    classification_time_ms=classification_time_ms,
                    timestamp=datetime.now().isoformat(),
                    reject_message=OUT_OF_DOMAIN_MESSAGE
                )
            
            # PRIORITY 2: HIGH confidence valid workflow classification
            # Use semantic result directly if confidence >= 0.80 for valid workflows
            if semantic_result.confidence >= 0.80 and semantic_result.workflow != WorkflowType.OUT_OF_DOMAIN:
                # Map semantic workflow to target
                semantic_workflow_map = {
                    WorkflowType.ENGENIE_CHAT: WorkflowTarget.ENGENIE_CHAT,
                    WorkflowType.SOLUTION_WORKFLOW: WorkflowTarget.SOLUTION_WORKFLOW,
                    WorkflowType.INSTRUMENT_IDENTIFIER: WorkflowTarget.INSTRUMENT_IDENTIFIER,
                    WorkflowType.OUT_OF_DOMAIN: WorkflowTarget.OUT_OF_DOMAIN  # Added OUT_OF_DOMAIN
                }
                target_workflow = semantic_workflow_map.get(
                    semantic_result.workflow, 
                    WorkflowTarget.ENGENIE_CHAT
                )
                is_solution = (semantic_result.workflow == WorkflowType.SOLUTION_WORKFLOW)
                
                # Set target RAG for EnGenie Chat
                target_rag = None
                if target_workflow == WorkflowTarget.ENGENIE_CHAT:
                    target_rag = "product_info_rag"
                
                logger.info(
                    f"[{self.name}] HIGH CONFIDENCE SEMANTIC: Returning {target_workflow.value} immediately"
                )
                
                end_time = datetime.now()
                classification_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return WorkflowRoutingResult(
                    query=query,
                    target_workflow=target_workflow,
                    intent="semantic_match",
                    confidence=semantic_result.confidence,
                    reasoning=f"High confidence semantic: {semantic_result.reasoning}",
                    is_solution=is_solution,
                    target_rag=target_rag,
                    solution_indicators=["semantic_match"] if is_solution else [],
                    extracted_info={
                        "semantic_classification": True,
                        "matched_signature": semantic_result.matched_signature,
                        "all_scores": semantic_result.all_scores
                    },
                    classification_time_ms=classification_time_ms,
                    timestamp=datetime.now().isoformat(),
                    reject_message=None
                )
            
            # PRIORITY 3: MEDIUM confidence (0.65-0.80) - try fast pattern for refinement
            elif 0.65 <= semantic_result.confidence < 0.80:
                logger.info(
                    f"[{self.name}] MEDIUM semantic confidence ({semantic_result.confidence:.3f}), "
                    f"checking fast patterns for refinement..."
                )
                # Continue to fast pattern check below
            
            # PRIORITY 4: LOW confidence (<0.65) - definitely try fast pattern
            else:
                logger.info(
                    f"[{self.name}] LOW semantic confidence ({semantic_result.confidence:.3f}), "
                    f"falling back to fast patterns..."
                )
                # Continue to fast pattern check below
                
        except ImportError as e:
            logger.warning(f"[{self.name}] Semantic classifier not available: {e}")
        except Exception as e:
            logger.warning(f"[{self.name}] Semantic classification failed: {e}, using fallback")

        # =====================================================================
        # STEP 4: FAST PATTERN CLASSIFICATION (Refinement/Fallback)
        # =====================================================================
        # IMPROVED: Only use fast patterns if semantic was ambiguous or failed
        # This prevents false positives from keyword matching
        
        # Skip fast pattern if we have strong solution indicators (same as before)
        if self._has_strong_solution_indicators(query):
            logger.info(f"[{self.name}] Strong solution indicators detected - Skipping Fast Pattern Check")
        # ALSO skip if semantic already gave us a confident OUT_OF_DOMAIN classification
        elif semantic_result and semantic_result.workflow == WorkflowType.OUT_OF_DOMAIN and semantic_result.confidence >= 0.65:
            logger.info(f"[{self.name}] Semantic OOD detected, skipping fast pattern")
        else:
            try:
                from chat.engenie_chat_intent_agent import classify_by_patterns, DataSource
                
                # Fast pattern check
                pattern_result = classify_by_patterns(query.lower().strip())
                
                if pattern_result:
                    source, conf, reason = pattern_result
                    # Only use if high confidence
                    if conf >= 0.80:
                        # If semantic also gave a result, compare confidences
                        if semantic_result and semantic_result.confidence >= 0.70:
                            logger.info(
                                f"[{self.name}] Both semantic ({semantic_result.confidence:.3f}) and "
                                f"pattern ({conf:.2f}) confident - using semantic (more accurate)"
                            )
                            # Semantic wins ties - skip pattern match
                        else:
                            # Pattern has higher confidence or semantic was low
                            logger.info(
                                f"[{self.name}] FAST PATTERN: {source.value} (conf={conf:.2f})"
                            )
                            
                            target_rag = None
                            intent_name = "question"
                            
                            # Map Source -> Target RAG & Intent
                            if source == DataSource.STANDARDS_RAG:
                                target_rag = "standards_rag" 
                                intent_name = "standards"
                            elif source == DataSource.STRATEGY_RAG:
                                target_rag = "strategy_rag"
                                intent_name = "vendor_strategy"
                            elif source == DataSource.INDEX_RAG:
                                target_rag = "product_info_rag"
                                intent_name = "product_info"
                            elif source == DataSource.DEEP_AGENT:
                                target_rag = "product_info_rag"
                                intent_name = "deep_agent"
                                
                            end_time = datetime.now()
                            classification_time_ms = (end_time - start_time).total_seconds() * 1000
                            
                            return WorkflowRoutingResult(
                                query=query,
                                target_workflow=WorkflowTarget.ENGENIE_CHAT,
                                intent=intent_name,
                                confidence=conf,
                                reasoning=f"Fast Pattern Match: {reason}",
                                is_solution=False,
                                target_rag=target_rag,
                                solution_indicators=[],
                                extracted_info={
                                    "fast_pattern_match": True,
                                    "source": source.value, 
                                    "reason": reason
                                },
                                classification_time_ms=classification_time_ms,
                                timestamp=datetime.now().isoformat(),
                                reject_message=None
                            )
                        
            except ImportError:
                logger.debug(f"[{self.name}] Fast pattern classifier not available")
            except Exception as e:
                logger.warning(f"[{self.name}] Fast pattern check failed: {e}")
        
        # =====================================================================
        # STEP 4.5: USE MEDIUM-CONFIDENCE SEMANTIC IF PATTERNS FAILED
        # =====================================================================
        # If we have a medium-confidence semantic result (0.65-0.80) but patterns didn't match,
        # use the semantic result
        if semantic_result and 0.65 <= semantic_result.confidence < 0.80:
            logger.info(
                f"[{self.name}] Using medium-confidence semantic result "
                f"(patterns didn't provide better match)"
            )
            
            # Handle OUT_OF_DOMAIN (use module-level constant)
            if semantic_result.workflow == WorkflowType.OUT_OF_DOMAIN:
                end_time = datetime.now()
                classification_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return WorkflowRoutingResult(
                    query=query,
                    target_workflow=WorkflowTarget.OUT_OF_DOMAIN,
                    intent="out_of_domain",
                    confidence=semantic_result.confidence,
                    reasoning=f"Semantic OUT_OF_DOMAIN: {semantic_result.reasoning}",
                    is_solution=False,
                    target_rag=None,
                    solution_indicators=[],
                    extracted_info={
                        "semantic_ood_detection": True,
                        "matched_signature": semantic_result.matched_signature,
                        "all_scores": semantic_result.all_scores
                    },
                    classification_time_ms=classification_time_ms,
                    timestamp=datetime.now().isoformat(),
                    reject_message=OUT_OF_DOMAIN_MESSAGE
                )
            
            # Handle valid workflows
            # FIX: Added OUT_OF_DOMAIN to prevent fallback to ENGENIE_CHAT for OOD queries
            semantic_workflow_map = {
                WorkflowType.ENGENIE_CHAT: WorkflowTarget.ENGENIE_CHAT,
                WorkflowType.SOLUTION_WORKFLOW: WorkflowTarget.SOLUTION_WORKFLOW,
                WorkflowType.INSTRUMENT_IDENTIFIER: WorkflowTarget.INSTRUMENT_IDENTIFIER,
                WorkflowType.OUT_OF_DOMAIN: WorkflowTarget.OUT_OF_DOMAIN
            }
            target_workflow = semantic_workflow_map.get(
                semantic_result.workflow, WorkflowTarget.ENGENIE_CHAT
            )
            is_solution = (semantic_result.workflow == WorkflowType.SOLUTION_WORKFLOW)
            target_rag = "product_info_rag" if target_workflow == WorkflowTarget.ENGENIE_CHAT else None
            
            end_time = datetime.now()
            classification_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return WorkflowRoutingResult(
                query=query,
                target_workflow=target_workflow,
                intent="semantic_match",
                confidence=semantic_result.confidence,
                reasoning=f"Medium confidence semantic: {semantic_result.reasoning}",
                is_solution=is_solution,
                target_rag=target_rag,
                solution_indicators=["semantic_match"] if is_solution else [],
                extracted_info={
                    "semantic_classification": True,
                    "matched_signature": semantic_result.matched_signature,
                    "all_scores": semantic_result.all_scores
                },
                classification_time_ms=classification_time_ms,
                timestamp=datetime.now().isoformat(),
                reject_message=None
            )

        # =====================================================================
        # STEP 4: USE LOW-CONFIDENCE SEMANTIC IF AVAILABLE (Performance Optimization)
        # =====================================================================
        # OPTIMIZATION: If we have any semantic result (even <0.65), use it instead of
        # expensive LLM fallback. Semantic classifier is more accurate for workflow routing.
        if semantic_result and semantic_result.confidence >= 0.50:
            logger.info(
                f"[{self.name}] Using low-confidence semantic result instead of LLM fallback "
                f"(conf={semantic_result.confidence:.3f})"
            )

            # Map workflow
            semantic_workflow_map = {
                WorkflowType.ENGENIE_CHAT: WorkflowTarget.ENGENIE_CHAT,
                WorkflowType.SOLUTION_WORKFLOW: WorkflowTarget.SOLUTION_WORKFLOW,
                WorkflowType.INSTRUMENT_IDENTIFIER: WorkflowTarget.INSTRUMENT_IDENTIFIER,
                WorkflowType.OUT_OF_DOMAIN: WorkflowTarget.OUT_OF_DOMAIN
            }
            target_workflow = semantic_workflow_map.get(
                semantic_result.workflow, WorkflowTarget.ENGENIE_CHAT
            )

            # Handle OUT_OF_DOMAIN
            reject_message = None
            if target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
                reject_message = OUT_OF_DOMAIN_MESSAGE

            is_solution = (semantic_result.workflow == WorkflowType.SOLUTION_WORKFLOW)
            target_rag = "product_info_rag" if target_workflow == WorkflowTarget.ENGENIE_CHAT else None

            end_time = datetime.now()
            classification_time_ms = (end_time - start_time).total_seconds() * 1000

            return WorkflowRoutingResult(
                query=query,
                target_workflow=target_workflow,
                intent="semantic_match_low_conf",
                confidence=semantic_result.confidence,
                reasoning=f"Low confidence semantic (skipped LLM): {semantic_result.reasoning}",
                is_solution=is_solution,
                target_rag=target_rag,
                solution_indicators=["semantic_match"] if is_solution else [],
                extracted_info={
                    "semantic_classification": True,
                    "skipped_llm_fallback": True,
                    "matched_signature": semantic_result.matched_signature,
                    "all_scores": semantic_result.all_scores
                },
                classification_time_ms=classification_time_ms,
                timestamp=datetime.now().isoformat(),
                reject_message=reject_message
            )

        # =====================================================================
        # STEP 5: RULE-BASED CLASSIFICATION (FALLBACK - Only if no semantic result)
        # =====================================================================

        # Import classify_intent_tool here to avoid circular imports
        try:
            from tools.intent_tools import classify_intent_tool
        except ImportError:
            logger.error("Could not import classify_intent_tool")
            return self._create_error_result(query, start_time, "Import error")

        # Get context values
        current_step = context.get("current_step") if context else None
        context_str = context.get("context") if context else None

        # Call the core classifier
        try:
            intent_result = classify_intent_tool.invoke({
                "user_input": query,
                "current_step": current_step,
                "context": context_str
            })
        except Exception as e:
            logger.error(f"[{self.name}] classify_intent_tool failed: {e}")
            return self._create_error_result(query, start_time, str(e))

        # Extract intent details
        intent = intent_result.get("intent", "unrelated")
        confidence = intent_result.get("confidence", 0.5)
        is_solution = intent_result.get("is_solution", False)
        solution_indicators = intent_result.get("solution_indicators", [])
        extracted_info = intent_result.get("extracted_info", {})

        # =================================================================
        # PHASE 3 FIX: Semantic validation before solution routing
        # =================================================================
        # If is_solution is True, verify with semantic classifier that this
        # is actually a design/build request, not a knowledge query
        
        # FIX: Don't run semantic override if we have strong solution indicators
        has_strong_indicators = self._has_strong_solution_indicators(query)
        
        if is_solution and not has_strong_indicators:
            try:
                from chat.engenie_chat_intent_agent import classify_query, DataSource
                semantic_source, semantic_conf, semantic_reason = classify_query(query, use_semantic_llm=False)
                
                rag_sources = {DataSource.INDEX_RAG, DataSource.STANDARDS_RAG, DataSource.STRATEGY_RAG, DataSource.DEEP_AGENT}
                
                if semantic_source in rag_sources and semantic_conf >= 0.7:
                    # Override: This is a RAG query, not a solution request
                    logger.info(
                        f"[{self.name}] PHASE 3 OVERRIDE: is_solution=True overridden to False. "
                        f"Semantic analysis: {semantic_source.value} (conf={semantic_conf:.2f})"
                    )
                    is_solution = False
                    solution_indicators = []
                    extracted_info["semantic_override"] = {
                        "original_is_solution": True,
                        "overridden_to": False,
                        "semantic_source": semantic_source.value,
                        "semantic_confidence": semantic_conf,
                        "reason": semantic_reason
                    }
                    # Also update intent to match semantic classification
                    if semantic_source == DataSource.STANDARDS_RAG:
                        intent = "standards"
                    elif semantic_source == DataSource.STRATEGY_RAG:
                        intent = "vendor_strategy"
                    else:
                        intent = "question"
                else:
                    logger.debug(f"[{self.name}] Phase 3 validation passed (Source: {semantic_source.value})")
                        
            except ImportError:
                logger.debug(f"[{self.name}] Semantic classifier not available for validation")
            except Exception as e:
                logger.warning(f"[{self.name}] Semantic validation failed: {e}")
        elif is_solution and has_strong_indicators:
             logger.info(f"[{self.name}] Phase 3 validation skipped (Strong solution indicators present)")

        # =================================================================
        # LEVEL 4.5: Registry-based workflow matching with fallback
        # =================================================================
        registry_match = None
        
        if self._use_registry and self._registry:
            try:
                # Use registry for intent matching
                registry_match = self._registry.match_intent(
                    intent=intent,
                    is_solution=is_solution,
                    confidence=confidence
                )
                
                if registry_match.workflow:
                    # Map registry workflow name to WorkflowTarget enum
                    workflow_name_to_target = {
                        "solution": WorkflowTarget.SOLUTION_WORKFLOW,
                        "instrument_identifier": WorkflowTarget.INSTRUMENT_IDENTIFIER,
                        "engenie_chat": WorkflowTarget.ENGENIE_CHAT
                    }
                    target_workflow = workflow_name_to_target.get(
                        registry_match.workflow.name, 
                        WorkflowTarget.ENGENIE_CHAT
                    )
                    logger.info(
                        f"[{self.name}] Registry match: {registry_match.workflow.name} "
                        f"(confidence={registry_match.confidence:.2f})"
                    )
                    
                    # Check if disambiguation is needed
                    if registry_match.needs_disambiguation:
                        logger.info(
                            f"[{self.name}] Low confidence match - disambiguation suggested: "
                            f"{registry_match.disambiguation_question}"
                        )
                        extracted_info["needs_disambiguation"] = True
                        extracted_info["disambiguation_question"] = registry_match.disambiguation_question
                else:
                    # No registry match - fall back to IntentConfig
                    logger.debug(f"[{self.name}] No registry match, using IntentConfig fallback")
                    target_workflow = IntentConfig.get_workflow(intent, is_solution)
                    
            except Exception as e:
                logger.warning(f"[{self.name}] Registry matching failed: {e}, using IntentConfig")
                target_workflow = IntentConfig.get_workflow(intent, is_solution)
        else:
            # FALLBACK: Use IntentConfig (Level 3 behavior)
            if not IntentConfig.is_known_intent(intent):
                logger.warning(
                    f"[{self.name}] Unknown intent '{intent}' from classifier. "
                    f"Valid intents: {IntentConfig.get_all_valid_intents()[:5]}... "
                    f"Mapping to 'unrelated'"
                )
                intent = "unrelated"
            target_workflow = IntentConfig.get_workflow(intent, is_solution)

        # Log the routing decision
        if is_solution or intent in IntentConfig.SOLUTION_FORCING_INTENTS:
            logger.info(
                f"[{self.name}] Solution detected: intent='{intent}', is_solution={is_solution}, "
                f"indicators={solution_indicators}"
            )

        # Determine Target RAG for Product Info
        target_rag = None
        if target_workflow == WorkflowTarget.ENGENIE_CHAT:
            if intent == "standards":
                target_rag = "standards_rag"
            elif intent == "vendor_strategy":
                target_rag = "strategy_rag"
            else:
                target_rag = "product_info_rag"

        # Override: Out of Domain Logic
        reject_message = None
        if target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
            reject_message = "Invalid question. Please ask a domain-related question about industrial instrumentation, standards, or product selection."
            reasoning = f"Query classified as '{intent}' which is out of domain."
        else:
            reasoning = self._build_reasoning(intent, target_workflow, is_solution, solution_indicators)

        # =====================================================================
        # STEP 4: SET WORKFLOW STATE FOR SESSION
        # =====================================================================
        workflow_name = {
            WorkflowTarget.ENGENIE_CHAT: "engenie_chat",
            WorkflowTarget.INSTRUMENT_IDENTIFIER: "instrument_identifier",
            WorkflowTarget.SOLUTION_WORKFLOW: "solution",
            WorkflowTarget.GREETING: "greeting"
        }.get(target_workflow)
        
        # CONSOLIDATION: Don't set backend state here
        # Frontend receives target_workflow in response and updates SessionManager
        # Frontend then passes workflow_hint on next request
        if workflow_name:
            logger.debug(f"[{self.name}] Target workflow: {workflow_name} (frontend will store)")

        # Prepare reject message for out-of-domain
        # reject_message logic handled above
        
        # Determine Direct Response for GREETING (if not already set by fast path)
        direct_response = extracted_info.get("direct_response")
        if target_workflow == WorkflowTarget.GREETING and not direct_response:
            # Generate random greeting response if LLM classified as greeting but skipped fast path
            # (or if query didn't match fast patterns but semantics matched greeting)
            responses = [
                "Hello! How can I help you with your industrial automation needs today?",
                "Hi there! I'm here to help with process instrumentation and control systems.",
                "Greetings! What can I assist you with?",
                "Hello! Whether it's product selection or system design, I'm ready to help.",
                "Hi! I'm EnGenie. How can I support your project today?"
            ]
            direct_response = random.choice(responses)

        # Calculate classification time
        end_time = datetime.now()
        classification_time_ms = (end_time - start_time).total_seconds() * 1000
        self.last_classification_time_ms = classification_time_ms
        self.classification_count += 1

        result = WorkflowRoutingResult(
            query=query,
            target_workflow=target_workflow,
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            is_solution=is_solution,
            target_rag=target_rag,
            solution_indicators=solution_indicators,
            extracted_info=extracted_info,
            classification_time_ms=classification_time_ms,
            timestamp=datetime.now().isoformat(),
            reject_message=reject_message,
            direct_response=direct_response
        )

        logger.info(f"[{self.name}] Result: {target_workflow.value} (intent={intent}, conf={confidence:.2f}) in {classification_time_ms:.1f}ms")

        # =====================================================================
        # METRICS RECORDING (PATH 2)
        # =====================================================================
        if METRICS_AVAILABLE:
            try:
                metrics = get_classification_metrics()

                # Determine which layer was used
                layer_used = "llm"  # Default
                if extracted_info.get("fast_pattern_match"):
                    layer_used = "pattern"
                elif extracted_info.get("semantic_classification"):
                    layer_used = "semantic"

                metrics.record_classification(
                    query=query,
                    intent=intent,
                    confidence=confidence,
                    target_workflow=target_workflow.value,
                    classification_time_ms=classification_time_ms,
                    layer_used=layer_used,
                    is_solution=is_solution,
                    extracted_info=extracted_info,
                    session_id=session_id
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Metrics recording failed: {e}")

        return result

    def _build_reasoning(
        self,
        intent: str,
        target_workflow: WorkflowTarget,
        is_solution: bool,
        solution_indicators: list
    ) -> str:
        """Build human-readable reasoning for the routing decision."""

        if target_workflow == WorkflowTarget.SOLUTION_WORKFLOW:
            if solution_indicators:
                return f"Solution detected: {', '.join(solution_indicators[:3])}"
            return "Complex system requiring multiple instruments detected"

        elif target_workflow == WorkflowTarget.INSTRUMENT_IDENTIFIER:
            return "Single product requirements detected"

        elif target_workflow == WorkflowTarget.ENGENIE_CHAT:
            if intent == "greeting":
                return "Greeting detected"
            elif intent == "confirm":
                return "User confirmation detected"
            elif intent == "reject":
                return "User rejection/cancellation detected"
            elif intent == "standards":
                return "Standards/compliance question detected"
            elif intent == "vendor_strategy":
                return "Vendor strategy question detected"
            return "Product knowledge question detected"

        elif target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
            return f"Out of domain: '{intent}' is not related to industrial automation"
            
        elif target_workflow == WorkflowTarget.GREETING:
            return "Greeting detected"

        return f"Classified as '{intent}'"

    def _create_error_result(self, query: str, start_time: datetime, error: str) -> WorkflowRoutingResult:
        """Create an error result."""
        end_time = datetime.now()
        classification_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return WorkflowRoutingResult(
            query=query,
            target_workflow=WorkflowTarget.OUT_OF_DOMAIN,
            intent="error",
            confidence=0.0,
            reasoning=f"Classification error: {error}",
            is_solution=False,
            target_rag=None,
            solution_indicators=[],
            extracted_info={},
            classification_time_ms=classification_time_ms,
            timestamp=datetime.now().isoformat(),
            reject_message=OUT_OF_DOMAIN_MESSAGE
        )

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        stats = {
            "name": self.name,
            "classification_count": self.classification_count,
            "last_classification_time_ms": self.last_classification_time_ms,
            "using_registry": self._use_registry,
            "memory_stats": self._memory.get_stats()
        }
        
        # Add registry stats if available
        if self._use_registry and self._registry:
            stats["registry_stats"] = self._registry.get_stats()
        
        return stats

    def _check_fast_greeting(self, query: str) -> Optional[Dict]:
        """Check if query is a simple greeting and return a random response."""
        lower_query = query.lower().strip().rstrip("!.,")
        
        # Extended greeting phrases
        greetings = set(GREETING_PHRASES + [
            "hello there", "hi there", "greetings", "good day"
        ])
        
        if lower_query in greetings:
            responses = [
                "Hello! How can I help you with your industrial automation needs today?",
                "Hi there! I'm here to help with process instrumentation and control systems.",
                "Greetings! What can I assist you with?",
                "Hello! Whether it's product selection or system design, I'm ready to help.",
                "Hi! I'm EnGenie. How can I support your project today?"
            ]
            return {"direct_response": random.choice(responses)}
            
        return None

    def invoke_target_workflow(self, query: str, routing_result: WorkflowRoutingResult, session_id: str) -> Dict[str, Any]:
        """
        Dynamically invoke the workflow targeted by the routing result.
        
        Args:
            query: The original user question/input
            routing_result: The classification result
            session_id: The active session ID
            
        Returns:
            The raw response dictionary from the underlying workflow API.
        """
        logger.info(f"[{self.name}] Natively invoking target workflow: {routing_result.target_workflow.value}")
        
        from common.infrastructure.api.internal import api_client
        target = routing_result.target_workflow
        
        try:
            if target == WorkflowTarget.ENGENIE_CHAT:
                return api_client.call_engenie_chat(query=query, session_id=session_id)
            elif target == WorkflowTarget.INSTRUMENT_IDENTIFIER:
                return api_client.call_instrument_identifier(message=query, session_id=session_id)
            elif target == WorkflowTarget.SOLUTION_WORKFLOW:
                return api_client.call_solution_workflow(message=query, session_id=session_id)
            elif target in (WorkflowTarget.OUT_OF_DOMAIN, WorkflowTarget.CONVERSATIONAL, WorkflowTarget.GREETING):
                msg = routing_result.direct_response or routing_result.reject_message or OUT_OF_DOMAIN_MESSAGE
                return {
                    "success": True,
                    "response": msg,
                    "response_text": msg,
                    "response_data": {"intent": routing_result.intent, "routed_to": target.value}
                }
            else:
                logger.warning(f"[{self.name}] Unhandled workflow target: {target}")
                return {"success": False, "error": f"Unhandled target: {target}"}
        except Exception as e:
            logger.error(f"[{self.name}] Error invoking target workflow {target.value}: {e}")
            return {"success": False, "error": str(e)}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def route_to_workflow(query: str, context: Optional[Dict] = None) -> WorkflowRoutingResult:
    """
    Convenience function for quick workflow routing.

    Args:
        query: User query string
        context: Optional context dict

    Returns:
        WorkflowRoutingResult
    """
    agent = IntentClassificationRoutingAgent()
    session_id = context.get('session_id', 'default') if context else 'default'
    return agent.classify(query, session_id=session_id, context=context)


def get_workflow_target(query: str) -> str:
    """
    Get just the workflow target name.

    Args:
        query: User query string

    Returns:
        Workflow target value (e.g., "solution", "instrument_identifier")
    """
    result = route_to_workflow(query)
    return result.target_workflow.value


def is_valid_domain_query(query: str) -> bool:
    """
    Check if a query is within the valid domain.

    Args:
        query: User query string

    Returns:
        True if valid, False if out-of-domain
    """
    result = route_to_workflow(query)
    return result.target_workflow != WorkflowTarget.OUT_OF_DOMAIN


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WorkflowTarget',
    'WorkflowRoutingResult',
    'IntentClassificationRoutingAgent',
    'WorkflowStateMemory',
    'get_workflow_memory',
    'should_exit_workflow',
    'route_to_workflow',
    'get_workflow_target',
    'is_valid_domain_query',
    'OUT_OF_DOMAIN_MESSAGE',
    'EXIT_PHRASES',
    'GREETING_PHRASES'
]
