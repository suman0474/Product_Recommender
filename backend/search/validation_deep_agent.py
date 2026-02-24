"""
Validation Deep Agent - Pure LangGraph Agent Architecture
=========================================================

This is now the PRIMARY validation module. Contains:
- ValidationDeepAgent: Pure LangGraph agent implementation
- ValidationTool: Backward-compatible wrapper class
- Session caching functions (enrichment & context)
- Request context functions (thread-local isolation)

Converts the monolithic ValidationTool.validate() method (1,130 lines) into a
pure LangGraph deep agent with discrete nodes, state management, and proper
observability.

Architecture:
- State: TypedDict with full workflow data
- Nodes: 8 pure functions for each major operation
- Workflow: LangGraph state machine with conditional routing
- Entry Point: ValidationDeepAgent class (backward compatible)

Node Flow:
1. detect_ui_decision → Early filter for UI patterns
2. detect_hitl_response → Detect YES/NO before validation
3. trigger_advanced_specs → Run AdvancedSpecificationAgent
4. extract_product_type → Extract & normalize product type
5. load_schema → Load or generate schema
6. enrich_schema → Multi-step standards enrichment
7. validate_requirements → Validate against schema
8. build_result → Assemble final result
"""

import logging
import time
import contextvars
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START

# Import BoundedCache for session caching
from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

# Import ExecutionContext for proper context integration
from common.infrastructure.context import (
    ExecutionContext,
    get_context as get_execution_context,
    get_session_id as get_ctx_session_id,
    get_cache_key as get_ctx_cache_key
)

logger = logging.getLogger(__name__)

# Debug flags for validation tool debugging
try:
    from debug_flags import debug_log, timed_execution, is_debug_enabled
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    def debug_log(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def timed_execution(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def is_debug_enabled(module):
        return False


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  THREAD-LOCAL CONTEXT FOR REQUEST ISOLATION                               ║
# ║  [FIX Feb 2026] Ensures each concurrent request has isolated state        ║
# ║  [UPGRADE Feb 2026] Now integrates with ExecutionContext system           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Thread-local context for current request - prevents cross-request contamination
# NOTE: These are maintained for backward compatibility. New code should use ExecutionContext.
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar('current_session_id', default='')
_current_workflow_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar('current_workflow_thread_id', default='')


def set_request_context(session_id: str, workflow_thread_id: str = ''):
    """
    Set thread-local context for current request.
    Call at the start of request processing.

    Args:
        session_id: Session identifier for this request
        workflow_thread_id: Optional workflow thread ID

    Note:
        This maintains backward compatibility. The ExecutionContext is set
        separately by the API layer using execution_context() context manager.
    """
    _current_session_id.set(session_id)
    _current_workflow_thread_id.set(workflow_thread_id)
    logger.debug(f"[RequestContext] Set context: session={session_id}, thread={workflow_thread_id}")


def get_request_session_id() -> str:
    """
    Get session_id for current request from thread-local context.

    Checks in order:
    1. ExecutionContext (preferred, set by API layer)
    2. Legacy _current_session_id contextvar

    Returns:
        Session ID or empty string if not set
    """
    # First try ExecutionContext (preferred)
    ctx = get_execution_context()
    if ctx and ctx.session_id:
        return ctx.session_id

    # Fallback to legacy contextvar
    return _current_session_id.get()


def get_request_workflow_thread_id() -> str:
    """
    Get workflow_thread_id for current request from thread-local context.

    Checks in order:
    1. ExecutionContext (preferred, set by API layer)
    2. Legacy _current_workflow_thread_id contextvar

    Returns:
        Workflow thread ID or empty string if not set
    """
    # First try ExecutionContext (preferred)
    ctx = get_execution_context()
    if ctx and ctx.workflow_id:
        return ctx.workflow_id

    # Fallback to legacy contextvar
    return _current_workflow_thread_id.get()


def get_isolated_cache_key(suffix: str) -> str:
    """
    Get a cache key scoped to the current execution context.

    Uses ExecutionContext for proper session/workflow isolation.

    Args:
        suffix: Cache key suffix (e.g., "enrichment:pressure_transmitter")

    Returns:
        Isolated cache key in format: ctx:{session_id}:{workflow_id}:{suffix}
    """
    ctx = get_execution_context()
    if ctx:
        return ctx.to_cache_key(suffix)

    # Fallback to session_id-based key
    session_id = get_request_session_id()
    if session_id:
        return f"session:{session_id}:{suffix}"

    logger.warning(f"[CacheKey] No context for cache key '{suffix}' - using global scope")
    return f"global:{suffix}"


def clear_request_context():
    """Clear thread-local context after request completes."""
    _current_session_id.set('')
    _current_workflow_thread_id.set('')


# ═══════════════════════════════════════════════════════════════════════════
# SESSION CACHING - Self-contained (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  [FIX #A1] SESSION-LEVEL ENRICHMENT DEDUPLICATION                       ║
# ║  Prevents redundant Standards RAG calls for same product in one session ║
# ║  Cache: product_type -> enrichment_result (thread-safe)                 ║
# ║  [PHASE 1] Using BoundedCache with TTL/LRU to prevent memory leaks      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
_session_enrichment_cache: BoundedCache = get_or_create_cache(
    name="session_enrichment",
    max_size=200,           # Max 200 concurrent sessions
    ttl_seconds=1800        # 30 minute session TTL
)


def _get_session_enrichment(product_type: str, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached enrichment result for this session.

    Args:
        product_type: Product type to look up
        session_id: Session identifier (REQUIRED for isolation between concurrent requests)

    Returns:
        Cached enrichment result or None if not found/session_id missing

    Note:
        [FIX Feb 2026] Made session_id required to prevent cross-session contamination.
    """
    if not session_id:
        logger.warning("[FIX #A1] _get_session_enrichment called without session_id - skipping cache lookup")
        return None

    normalized_type = product_type.lower().strip()
    key = f"enrichment:{session_id}:{normalized_type}"
    return _session_enrichment_cache.get(key)


def _cache_session_enrichment(product_type: str, enrichment_result: Dict[str, Any], session_id: str):
    """
    Cache enrichment result for this session.

    Args:
        product_type: Product type to cache
        enrichment_result: Data to cache
        session_id: Session identifier (REQUIRED for isolation between concurrent requests)

    Note:
        [FIX Feb 2026] Made session_id required to prevent cross-session contamination.
    """
    if not session_id:
        logger.warning("[FIX #A1] _cache_session_enrichment called without session_id - skipping cache write")
        return

    normalized_type = product_type.lower().strip()
    key = f"enrichment:{session_id}:{normalized_type}"
    _session_enrichment_cache.set(key, enrichment_result)
    logger.info(f"[FIX #A1] Cached enrichment for {key} (size: {len(_session_enrichment_cache)})")


def clear_session_enrichment_cache():
    """Clear session enrichment cache (call at start of new session)."""
    count = _session_enrichment_cache.clear()
    logger.info(f"[FIX #A1] Session enrichment cache cleared ({count} entries)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION CONTEXT CACHE - Stores product_type for HITL YES/NO responses    ║
# ║  When user says "YES" without product_type, retrieve from this cache      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
_session_context_cache: BoundedCache = get_or_create_cache(
    name="session_context",
    max_size=500,           # Max 500 concurrent sessions
    ttl_seconds=3600        # 1 hour session context TTL
)


def _get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached session context (product_type, schema, etc.) for HITL responses.

    Args:
        session_id: Session identifier (REQUIRED for isolation)

    Returns:
        Cached context dict or None if not found

    Note:
        [FIX Feb 2026] Added 'context:' prefix to key for namespace isolation
    """
    if not session_id:
        logger.warning("[SessionContext] _get_session_context called without session_id")
        return None
    key = f"context:{session_id}"
    return _session_context_cache.get(key)


def _cache_session_context(session_id: str, context: Dict[str, Any]):
    """
    Cache session context after validation for HITL response handling.

    Args:
        session_id: Session identifier (REQUIRED for isolation)
        context: Dict containing product_type, schema, provided_requirements, etc.

    Note:
        [FIX Feb 2026] Added 'context:' prefix to key for namespace isolation
    """
    if not session_id:
        logger.warning("[SessionContext] _cache_session_context called without session_id - skipping")
        return
    key = f"context:{session_id}"
    _session_context_cache.set(key, context)
    logger.info(f"[SessionContext] Cached context for {key}: product_type={context.get('product_type')}")


def clear_session_cache(session_id: str):
    """
    Clear all cached data for a specific session.
    Call this when a session ends or user starts a new search.

    Args:
        session_id: Session identifier to clear

    Note:
        [FIX Feb 2026] Added to support proper session cleanup.
    """
    if not session_id:
        logger.warning("[SessionCleanup] clear_session_cache called without session_id")
        return

    context_key = f"context:{session_id}"
    _session_context_cache.delete(context_key)
    logger.info(f"[SessionCleanup] Cleared session context cache for: {session_id}")
    logger.info(f"[SessionCleanup] Note: Enrichment cache entries will expire via TTL (30 min)")


# ═══════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

class ValidationDeepAgentState(TypedDict, total=False):
    """LangGraph state for Validation Deep Agent."""

    # ═══════════════════════════════════════════════════════════════
    # INPUTS (set once at graph entry)
    # ═══════════════════════════════════════════════════════════════
    session_id: str
    user_input: str
    expected_product_type: Optional[str]
    enable_standards_enrichment: bool
    source_workflow: Optional[str]
    is_standards_enriched: bool  # Skip standards enrichment if input is pre-enriched
    is_taxonomy_normalized: bool  # NEW: Skip taxonomy normalization if pre-normalized

    # ═══════════════════════════════════════════════════════════════
    # HITL DETECTION
    # ═══════════════════════════════════════════════════════════════
    is_ui_decision: bool
    is_hitl_yes: bool
    is_hitl_no: bool
    hitl_response: Optional[str]
    validation_bypassed: bool

    # ═══════════════════════════════════════════════════════════════
    # PRODUCT TYPE EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    product_type: str
    original_product_type: str
    product_type_refined: bool
    normalized_category: Optional[str]
    canonical_product_type: Optional[str]  # NEW: Taxonomy-normalized product type
    taxonomy_matched: bool  # NEW: Whether product type matched taxonomy
    taxonomy_normalization_skipped: bool  # NEW: Track if taxonomy normalization was bypassed

    # ═══════════════════════════════════════════════════════════════
    # SCHEMA LOADING
    # ═══════════════════════════════════════════════════════════════
    schema: Dict[str, Any]
    schema_source: str
    ppi_workflow_used: bool
    from_database: bool

    # ═══════════════════════════════════════════════════════════════
    # STANDARDS ENRICHMENT
    # ═══════════════════════════════════════════════════════════════
    session_cache_hit: bool
    standards_info: Optional[Dict[str, Any]]
    enrichment_result: Optional[Dict[str, Any]]
    standards_rag_invoked: bool
    standards_rag_invocation_time: Optional[str]
    schema_population_info: Dict[str, Any]
    standards_enrichment_skipped: bool  # NEW: Track if enrichment was bypassed

    # ═══════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════
    provided_requirements: Dict[str, Any]
    missing_fields: List[str]
    optional_fields: List[str]
    is_valid: bool
    hitl_message: Optional[str]

    # ═══════════════════════════════════════════════════════════════
    # ADVANCED SPECIFICATIONS
    # ═══════════════════════════════════════════════════════════════
    advanced_specs_info: Optional[Dict[str, Any]]

    # ═══════════════════════════════════════════════════════════════
    # RESULT ASSEMBLY
    # ═══════════════════════════════════════════════════════════════
    result: Dict[str, Any]

    # ═══════════════════════════════════════════════════════════════
    # OBSERVABILITY
    # ═══════════════════════════════════════════════════════════════
    tools_called: List[str]
    current_node: str
    phases_completed: List[str]
    error: Optional[str]
    start_time: float
    processing_time_ms: int


# ═══════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _node_detect_ui_decision(state: ValidationDeepAgentState) -> dict:
    """Node 1 — Detect UI decision patterns BEFORE calling LLM."""
    user_input = state.get("user_input", "")

    try:
        from debug_flags import is_ui_decision_input, get_ui_decision_error_message

        if is_ui_decision_input(user_input):
            logger.warning(f"[ValidationDeepAgent] UI decision pattern detected: '{user_input}'")
            return {
                "is_ui_decision": True,
                "error": get_ui_decision_error_message(user_input),
                "current_node": "detect_ui_decision"
            }
    except ImportError:
        # Fallback detection logic
        ui_patterns = ["user selected:", "user clicked:", "decision:", "continue", "proceed"]
        if any(p in user_input.lower() for p in ui_patterns):
            return {
                "is_ui_decision": True,
                "error": f"Input '{user_input}' is a UI action, not a product requirement.",
                "current_node": "detect_ui_decision"
            }

    return {
        "is_ui_decision": False,
        "current_node": "detect_ui_decision"
    }


def _node_detect_hitl_response(state: ValidationDeepAgentState) -> dict:
    """Node 2 — Detect YES/NO HITL responses before validation."""
    # _get_session_context is now defined locally in this file

    user_input = state.get("user_input", "").lower().strip()
    expected_product_type = state.get("expected_product_type")
    session_id = state.get("session_id")

    is_yes = user_input in ["yes", "y"]
    is_no = user_input in ["no", "n"] or user_input.startswith("no ")

    if is_yes:
        product_type = expected_product_type

        if not product_type:
            cached_context = _get_session_context(session_id)
            if cached_context and cached_context.get("product_type"):
                product_type = cached_context["product_type"]

        if product_type:
            return {
                "is_hitl_yes": True,
                "hitl_response": "yes",
                "product_type": product_type,
                "validation_bypassed": True,
                "current_node": "detect_hitl_response"
            }
        else:
            return {
                "is_hitl_yes": True,
                "error": "Missing product context for YES response",
                "current_node": "detect_hitl_response"
            }

    elif is_no:
        product_type = expected_product_type
        if not product_type:
            cached_context = _get_session_context(session_id)
            if cached_context:
                product_type = cached_context.get("product_type")

        if product_type:
            return {
                "is_hitl_no": True,
                "hitl_response": "no",
                "product_type": product_type,
                "current_node": "detect_hitl_response"
            }
        else:
            return {
                "is_hitl_no": True,
                "error": "Missing product context for NO response",
                "current_node": "detect_hitl_response"
            }

    return {
        "is_hitl_yes": False,
        "is_hitl_no": False,
        "current_node": "detect_hitl_response"
    }


def _node_trigger_advanced_specs(state: ValidationDeepAgentState) -> dict:
    """Node 3 — Trigger AdvancedSpecificationAgent when user says YES."""
    # _get_session_context is now defined locally in this file

    product_type = state.get("product_type")
    session_id = state.get("session_id")

    logger.info("=" * 70)
    logger.info("[ValidationDeepAgent] → Triggering AdvancedSpecificationAgent")
    logger.info(f"   Product Type: {product_type}")
    logger.info("=" * 70)

    try:
        from search.advanced_specification_agent import AdvancedSpecificationAgent

        adv_agent = AdvancedSpecificationAgent()
        advanced_specs_result = adv_agent.discover(
            product_type=product_type,
            session_id=session_id
        )

        if advanced_specs_result.get("success"):
            num_specs = advanced_specs_result.get("total_unique_specifications", 0)
            logger.info(f"[ValidationDeepAgent] ✓ Advanced specs discovered: {num_specs} parameters")

            cached_context = _get_session_context(session_id) or {}

            return {
                "advanced_specs_info": advanced_specs_result,
                "schema": cached_context.get("schema", {}),
                "provided_requirements": cached_context.get("provided_requirements", {}),
                "is_valid": True,
                "current_node": "trigger_advanced_specs"
            }
        else:
            logger.warning(f"[ValidationDeepAgent] ⚠ Advanced specs failed: {advanced_specs_result.get('error')}")
            return {
                "error": advanced_specs_result.get("error", "Advanced specification discovery failed"),
                "current_node": "trigger_advanced_specs"
            }

    except Exception as e:
        logger.error(f"[ValidationDeepAgent] Advanced specs error: {e}")
        return {
            "error": str(e),
            "current_node": "trigger_advanced_specs"
        }


def _node_extract_product_type(state: ValidationDeepAgentState) -> dict:
    """
    Node 4 — Use provided product type (already extracted at orchestration level).

    ARCHITECTURE CHANGE (Feb 2026):
    - Intent extraction now happens at orchestration level (run_product_search_workflow)
    - This node ALWAYS receives expected_product_type
    - Performs only normalization and validation
    """
    user_input = state.get("user_input", "")
    expected_product_type = state.get("expected_product_type")

    logger.info("[ValidationDeepAgent] Step 1.1: Using provided product type")

    try:
        # Product type should ALWAYS be provided by orchestration level
        if not expected_product_type:
            logger.error("[ValidationDeepAgent] ✗ No product type provided - orchestration level should extract it")
            return {
                "error": "Product type not provided by orchestration level",
                "current_node": "extract_product_type"
            }

        product_type = expected_product_type
        logger.info(
            f"[ValidationDeepAgent] ✓ Using product type from orchestration: {product_type}"
        )

        # Normalize bare measurement words (e.g., "pressure" → "Pressure Transmitter")
        BARE_MEASUREMENT_FIX = {
            "level": "Level Transmitter",
            "pressure": "Pressure Transmitter",
            "flow": "Flow Meter",
            "temperature": "Temperature Transmitter",
            "differential pressure": "Differential Pressure Transmitter",
            "dp": "Differential Pressure Transmitter",
        }

        if product_type.lower().strip() in BARE_MEASUREMENT_FIX:
            original_pt = product_type
            product_type = BARE_MEASUREMENT_FIX[product_type.lower().strip()]

            # Context-aware normalization for flow and level
            if original_pt.lower() == "flow":
                if "transmitter" in user_input.lower() and "meter" not in user_input.lower():
                    product_type = "Flow Transmitter"
                elif "switch" in user_input.lower():
                    product_type = "Flow Switch"

            elif original_pt.lower() == "level":
                if "switch" in user_input.lower():
                    product_type = "Level Switch"
                elif "gauge" in user_input.lower():
                    product_type = "Level Gauge"

            logger.warning(f"[ValidationDeepAgent] ⚠ Normalized '{original_pt}' → '{product_type}'")

        return {
            "product_type": product_type,
            "original_product_type": expected_product_type,
            "current_node": "extract_product_type",
            "tools_called": state.get("tools_called", []) + ["orchestration_level_intent_extraction"]
        }

    except Exception as e:
        logger.error(f"[ValidationDeepAgent] Product type processing error: {e}")
        return {
            "error": str(e),
            "current_node": "extract_product_type"
        }


def _node_normalize_product_type(state: ValidationDeepAgentState) -> dict:
    """Node 4b — Normalize product type using Taxonomy RAG."""
    product_type = state.get("product_type", "")
    user_input = state.get("user_input", "")
    source_workflow = state.get("source_workflow")
    is_taxonomy_normalized = state.get("is_taxonomy_normalized", False)

    logger.info("[ValidationDeepAgent] Step 1.1b: Normalizing product type with Taxonomy RAG")

    try:
        # NEW: Skip if already normalized by Solution Deep Agent
        if is_taxonomy_normalized:
            logger.info(
                f"[ValidationDeepAgent] ⚡ Skipping taxonomy normalization - "
                f"already normalized by {source_workflow}"
            )
            return {
                "canonical_product_type": product_type,
                "taxonomy_matched": False,
                "taxonomy_normalization_skipped": True,
                "current_node": "normalize_product_type"
            }

        # NEW: Call Taxonomy RAG for normalization
        from taxonomy_rag.normalization_agent import TaxonomyNormalizationAgent

        logger.info(f"[ValidationDeepAgent] Calling Taxonomy RAG for: {product_type}")
        norm_agent = TaxonomyNormalizationAgent()
        normalized_items = norm_agent.normalize_with_context(
            items=[{"name": product_type, "type": "instrument"}],
            user_input=user_input,
            history=[],
            item_type="instrument"
        )

        if normalized_items and len(normalized_items) > 0:
            canonical = normalized_items[0].get("canonical_name", product_type)
            matched = normalized_items[0].get("taxonomy_matched", False)

            logger.info(
                f"[ValidationDeepAgent] ✓ Taxonomy normalized: "
                f"{product_type} → {canonical} (matched={matched})"
            )

            return {
                "canonical_product_type": canonical,
                "product_type": canonical,  # Update product_type with canonical name
                "taxonomy_matched": matched,
                "taxonomy_normalization_skipped": False,
                "current_node": "normalize_product_type"
            }
        else:
            logger.warning("[ValidationDeepAgent] Taxonomy RAG returned no results")
            return {
                "canonical_product_type": product_type,
                "taxonomy_matched": False,
                "taxonomy_normalization_skipped": False,
                "current_node": "normalize_product_type"
            }

    except Exception as e:
        logger.warning(f"[ValidationDeepAgent] Taxonomy normalization failed: {e}")
        # Fallback: use original product_type
        return {
            "canonical_product_type": product_type,
            "taxonomy_matched": False,
            "taxonomy_normalization_skipped": False,
            "current_node": "normalize_product_type"
        }


def _node_load_schema(state: ValidationDeepAgentState) -> dict:
    """Node 5 — Load or generate schema."""
    product_type = state.get("product_type")

    logger.info("[ValidationDeepAgent] Step 1.2: Loading/generating schema")

    try:
        from common.tools.schema_tools import load_schema_tool

        schema_result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": True
        })

        schema = schema_result.get("schema", {})
        schema_source = schema_result.get("source", "unknown")
        ppi_used = schema_result.get("ppi_used", False)
        from_database = schema_result.get("from_database", False)

        if from_database:
            logger.info("[ValidationDeepAgent] ✓ Schema loaded from Azure Blob Storage")
        elif ppi_used:
            logger.info("[ValidationDeepAgent] ✓ Schema generated via PPI workflow")
        else:
            logger.warning("[ValidationDeepAgent] ⚠ Using default schema (fallback)")

        return {
            "schema": schema,
            "schema_source": schema_source,
            "ppi_workflow_used": ppi_used,
            "from_database": from_database,
            "current_node": "load_schema",
            "tools_called": state.get("tools_called", []) + ["load_schema_tool"]
        }

    except Exception as e:
        logger.error(f"[ValidationDeepAgent] Schema loading error: {e}")
        return {
            "error": str(e),
            "current_node": "load_schema"
        }


def _fallback_simple_enrichment(product_type: str, schema: dict) -> dict:
    """Fallback to simple enrichment when Standards Deep Agent is not available."""
    try:
        from common.rag.standards import enrich_identified_items_with_standards
        product_item = [{
            "name": product_type,
            "category": product_type,
            "specifications": schema.get("mandatory", {})
        }]
        enriched_items = enrich_identified_items_with_standards(
            items=product_item,
            product_type=product_type,
            top_k=3
        )
        if enriched_items and len(enriched_items) > 0:
            enrichment_result = enriched_items[0].get("standards_info", {})
            enrichment_result["source"] = "simple_enrichment_fallback"
            if enriched_items[0].get("normalized_category"):
                schema["normalized_category"] = enriched_items[0]["normalized_category"]
            return enrichment_result
    except Exception as e:
        logger.debug(f"[ValidationDeepAgent] Fallback enrichment also failed: {e}")
    return {"source": "fallback_failed", "success": False}


def _node_enrich_schema(state: ValidationDeepAgentState) -> dict:
    """Node 6 — Enrich schema with Standards RAG."""
    # _get_session_enrichment and _cache_session_enrichment are now defined locally
    import datetime

    product_type = state.get("product_type")
    schema = state.get("schema", {})
    session_id = state.get("session_id")
    enable_standards = state.get("enable_standards_enrichment", True)
    is_standards_enriched = state.get("is_standards_enriched", False)

    # NEW: Early exit if input is already enriched
    if is_standards_enriched:
        logger.info(
            f"[ValidationDeepAgent] ⚡ Skipping Standards Deep Agent - "
            f"input already enriched by {state.get('source_workflow')}"
        )
        return {
            "session_cache_hit": False,
            "schema": schema,  # Use as-is
            "standards_info": {},
            "enrichment_result": {
                "success": True,
                "source": "pre_enriched",
                "specifications_count": 0
            },
            "standards_rag_invoked": False,
            "standards_enrichment_skipped": True,
            "standards_rag_invocation_time": None,
            "current_node": "enrich_schema"
        }

    if not enable_standards:
        logger.info("[ValidationDeepAgent] Standards enrichment disabled")
        return {"current_node": "enrich_schema"}

    logger.info("[ValidationDeepAgent] Step 1.2.1: Enriching schema with Standards RAG")

    standards_rag_invocation_time = datetime.datetime.now().isoformat()

    try:
        # Check session cache first (FIX #A1)
        cached_enrichment = _get_session_enrichment(product_type, session_id)

        if cached_enrichment:
            logger.info(f"[ValidationDeepAgent] 🎯 SESSION CACHE HIT for {product_type}")
            return {
                "session_cache_hit": True,
                "standards_info": cached_enrichment.get('standards_info'),
                "enrichment_result": cached_enrichment.get('enrichment_result'),
                "schema": cached_enrichment.get('schema', schema),
                "standards_rag_invoked": True,
                "standards_rag_invocation_time": standards_rag_invocation_time,
                "current_node": "enrich_schema"
            }

        logger.info(f"[ValidationDeepAgent] 🔴 SESSION CACHE MISS for {product_type}")

        # Sub-step 1: Populate standards fields
        try:
            from common.tools.standards_enrichment_tool import populate_schema_fields_from_standards
            schema = populate_schema_fields_from_standards(product_type, schema)
        except Exception as e:
            logger.warning(f"[ValidationDeepAgent] Standards field population failed: {e}")

        # Sub-step 2: Get applicable standards
        try:
            from common.tools.standards_enrichment_tool import get_applicable_standards
            standards_info = get_applicable_standards(product_type, top_k=5)

            if standards_info.get('success'):
                if 'standards' not in schema:
                    schema['standards'] = {
                        'applicable_standards': standards_info.get('applicable_standards', []),
                        'certifications': standards_info.get('certifications', []),
                        'sources': standards_info.get('sources', []),
                    }
        except Exception as e:
            logger.warning(f"[ValidationDeepAgent] Standards retrieval failed: {e}")
            standards_info = None

        # Sub-step 3: Run Standards Deep Agent with ITERATIVE LOOP for minimum 60 specs
        enrichment_result = None
        deep_agent_result = None
        try:
            from common.standards.generation.deep_agent import run_standards_deep_agent, MIN_STANDARDS_SPECS_COUNT

            # Use the iterative deep agent that ensures minimum specs
            logger.info(f"[ValidationDeepAgent] 🚀 Running Standards Deep Agent (min: {MIN_STANDARDS_SPECS_COUNT} specs)")

            deep_agent_result = run_standards_deep_agent(
                user_requirement=f"Product type: {product_type}. Specifications needed for: {product_type}",
                session_id=session_id,
                inferred_specs=schema.get("mandatory", {}),
                min_specs=MIN_STANDARDS_SPECS_COUNT  # Ensure minimum 60 specifications
            )

            if deep_agent_result and deep_agent_result.get("success"):
                final_specs = deep_agent_result.get("final_specifications", {})
                specs_dict = final_specs.get("specifications", {}) if isinstance(final_specs, dict) else {}
                specs_count = len(specs_dict)

                logger.info(f"[ValidationDeepAgent] ✓ Standards Deep Agent generated {specs_count} specifications")

                # Store raw deep agent specs
                if "deep_agent_specs" not in schema:
                    schema["deep_agent_specs"] = {}
                schema["deep_agent_specs"].update(specs_dict)

                # ═══════════════════════════════════════════════════════════════════
                # FIX: Merge enriched values INTO existing field definitions
                # The frontend expects field values inside each field definition object
                # ═══════════════════════════════════════════════════════════════════
                enriched_field_count = 0
                for category in ["mandatory", "optional"]:
                    if category in schema and isinstance(schema[category], dict):
                        for field_name, field_def in schema[category].items():
                            # Normalize field name for matching
                            normalized_key = field_name.lower().replace(" ", "_").replace("-", "_")

                            # Check multiple key formats for the enriched value
                            enriched_value = None
                            for check_key in [normalized_key, field_name, field_name.lower()]:
                                if check_key in specs_dict:
                                    enriched_value = specs_dict[check_key]
                                    break

                            if enriched_value and str(enriched_value).lower() not in ["null", "none", ""]:
                                if isinstance(field_def, dict):
                                    # Update existing field definition with enriched value
                                    field_def["value"] = enriched_value
                                    field_def["enrichment_source"] = "standards_deep_agent"
                                    enriched_field_count += 1
                                else:
                                    # Replace primitive with dict containing value
                                    schema[category][field_name] = {
                                        "value": enriched_value,
                                        "enrichment_source": "standards_deep_agent"
                                    }
                                    enriched_field_count += 1

                logger.info(f"[ValidationDeepAgent] ✓ Enriched {enriched_field_count} field definitions with standards values")

                enrichment_result = {
                    "success": True,
                    "specifications_count": specs_count,
                    "target_reached": deep_agent_result.get("final_specifications", {}).get("target_reached", False),
                    "iterations": deep_agent_result.get("iterations_performed", 1),
                    "domains_analyzed": deep_agent_result.get("domains_analyzed", []),
                    "source": "standards_deep_agent"
                }

                if deep_agent_result.get("final_specifications", {}).get("normalized_category"):
                    schema["normalized_category"] = deep_agent_result["final_specifications"]["normalized_category"]
            else:
                logger.warning(f"[ValidationDeepAgent] ⚠ Standards Deep Agent returned no results, falling back to simple enrichment")
                # Fallback to simple enrichment if deep agent fails
                enrichment_result = _fallback_simple_enrichment(product_type, schema)

        except ImportError as ie:
            logger.warning(f"[ValidationDeepAgent] Standards Deep Agent not available: {ie}, using fallback")
            enrichment_result = _fallback_simple_enrichment(product_type, schema)
        except Exception as e:
            logger.error(f"[ValidationDeepAgent] Standards Deep Agent error: {e}, using fallback")
            enrichment_result = _fallback_simple_enrichment(product_type, schema)

        # Cache enrichment result for session
        enrichment_cache_data = {
            'standards_info': standards_info or {},
            'enrichment_result': enrichment_result,
            'schema': schema,
            'standards_section': schema.get('standards')
        }
        _cache_session_enrichment(product_type, enrichment_cache_data, session_id)

        return {
            "session_cache_hit": False,
            "schema": schema,
            "standards_info": standards_info or {},
            "enrichment_result": enrichment_result,
            "standards_rag_invoked": True,
            "standards_rag_invocation_time": standards_rag_invocation_time,
            "current_node": "enrich_schema"
        }

    except Exception as e:
        logger.error(f"[ValidationDeepAgent] Standards enrichment error: {e}")
        return {
            "error": str(e),
            "current_node": "enrich_schema"
        }


def _node_validate_requirements(state: ValidationDeepAgentState) -> dict:
    """Node 7 — Validate requirements against schema."""
    user_input = state.get("user_input", "")
    product_type = state.get("product_type")
    schema = state.get("schema", {})

    logger.info("[ValidationDeepAgent] Step 1.3: Validating requirements")

    try:
        from common.tools.schema_tools import validate_requirements_tool

        validation_result = validate_requirements_tool.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "product_schema": schema
        })

        provided_requirements = validation_result.get("provided_requirements", {})

        # Flatten nested field values for UI
        def flatten_field_value(field_data):
            if isinstance(field_data, dict):
                if "value" in field_data:
                    val = field_data.get("value", "")
                    unit = field_data.get("unit", "")
                    return f"{val} {unit}".strip()
                elif "min" in field_data and "max" in field_data:
                    unit = field_data.get("unit", "")
                    return f"{field_data['min']} - {field_data['max']} {unit}".strip()
            return field_data

        for k, v in provided_requirements.items():
            provided_requirements[k] = flatten_field_value(v)

        missing_fields = validation_result.get("missing_fields", [])
        is_valid = validation_result.get("is_valid", False)

        return {
            "provided_requirements": provided_requirements,
            "missing_fields": missing_fields,
            "optional_fields": validation_result.get("optional_fields", []),
            "is_valid": is_valid,
            "schema": schema,
            "current_node": "validate_requirements",
            "tools_called": state.get("tools_called", []) + ["validate_requirements_tool"]
        }

    except Exception as e:
        logger.error(f"[ValidationDeepAgent] Validation error: {e}")
        return {
            "error": str(e),
            "current_node": "validate_requirements"
        }


def _node_build_result(state: ValidationDeepAgentState) -> dict:
    """Node 8 — Assemble comprehensive validation result."""
    # _cache_session_context is now defined locally in this file

    product_type = state.get("product_type", "product")
    missing_fields = state.get("missing_fields", [])
    provided_requirements = state.get("provided_requirements", {})
    is_valid = state.get("is_valid", False)

    # ═══════════════════════════════════════════════════════════════════════════
    # HITL MESSAGE GENERATION
    # Generate human-readable message asking user for YES/NO confirmation
    # ═══════════════════════════════════════════════════════════════════════════
    hitl_message = state.get("hitl_message")  # Check if already set (e.g., from advanced specs)

    if not hitl_message:
        num_provided = len(provided_requirements)
        num_missing = len(missing_fields)

        if num_missing > 0:
            # Format missing fields for display
            missing_display = ", ".join(missing_fields[:5])
            if len(missing_fields) > 5:
                missing_display += f", and {len(missing_fields) - 5} more"

            hitl_message = (
                f"I've extracted {num_provided} specification(s) for **{product_type}**.\n\n"
                f"**Missing specifications:** {missing_display}\n\n"
                f"Would you like to provide these missing details for better search results, "
                f"or shall I continue with the current specifications?\n\n"
                f"Reply **'YES'** to add advanced specifications, or **'NO'** to continue with the search."
            )
        else:
            hitl_message = (
                f"I've extracted {num_provided} specification(s) for **{product_type}**.\n\n"
                f"All required specifications have been provided.\n\n"
                f"Would you like to add more advanced specifications for better matching, "
                f"or shall I proceed with the search?\n\n"
                f"Reply **'YES'** to add advanced specifications, or **'NO'** to continue with the search."
            )

        logger.info(f"[ValidationDeepAgent] Generated HITL message: {len(hitl_message)} chars")

    result = {
        "success": True,
        "session_id": state.get("session_id"),
        "product_type": product_type,
        "product_type_refined": state.get("product_type_refined", False),
        "original_product_type": state.get("original_product_type"),
        "normalized_category": state.get("normalized_category"),
        "schema": state.get("schema", {}),
        "provided_requirements": provided_requirements,
        "missing_fields": missing_fields,
        "optional_fields": state.get("optional_fields", []),
        "is_valid": is_valid,
        "hitl_message": hitl_message,
        "ppi_workflow_used": state.get("ppi_workflow_used", False),
        "schema_source": state.get("schema_source", "unknown"),
        "from_database": state.get("from_database", False),
        "hitl_response": state.get("hitl_response"),
        "validation_bypassed": state.get("validation_bypassed", False),
        "rag_invocations": {
            "standards_rag": {
                "invoked": state.get("standards_rag_invoked", False),
                "invocation_time": state.get("standards_rag_invocation_time"),
                "success": state.get("standards_info", {}).get('success', False),
                "product_type": state.get("product_type"),
                "results_count": len(state.get("standards_info", {}).get('applicable_standards', []))
            }
        },
        "standards_info": state.get("standards_info", {}),
        "standards_enrichment_skipped": state.get("standards_enrichment_skipped", False),
        "canonical_product_type": state.get("canonical_product_type"),
        "taxonomy_matched": state.get("taxonomy_matched", False),
        "taxonomy_normalization_skipped": state.get("taxonomy_normalization_skipped", False),
        "advanced_specs_info": state.get("advanced_specs_info", {}),
        "tools_called": state.get("tools_called", []),
    }

    # Cache session context for future HITL responses
    if state.get("session_id") and state.get("product_type"):
        _cache_session_context(state["session_id"], {
            "product_type": state["product_type"],
            "schema": state.get("schema", {}),
            "provided_requirements": state.get("provided_requirements", {}),
            "missing_fields": state.get("missing_fields", []),
            "is_valid": state.get("is_valid", False)
        })

    return {
        "result": result,
        "current_node": "build_result"
    }


# ═══════════════════════════════════════════════════════════════════════════
# DECISION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _decide_after_ui_detection(state: ValidationDeepAgentState) -> str:
    """Decide next node after UI decision detection."""
    if state.get("is_ui_decision"):
        return "error"
    return "continue"


def _decide_after_hitl_detection(state: ValidationDeepAgentState) -> str:
    """Decide next node after HITL detection."""
    if state.get("is_hitl_yes"):
        return "yes"
    elif state.get("is_hitl_no"):
        return "no"
    return "continue"


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def create_validation_workflow() -> StateGraph:
    """Create the Validation Deep Agent LangGraph workflow."""

    workflow = StateGraph(ValidationDeepAgentState)

    # Add all nodes
    workflow.add_node("detect_ui_decision", _node_detect_ui_decision)
    workflow.add_node("detect_hitl_response", _node_detect_hitl_response)
    workflow.add_node("trigger_advanced_specs", _node_trigger_advanced_specs)
    workflow.add_node("extract_product_type", _node_extract_product_type)
    workflow.add_node("normalize_product_type", _node_normalize_product_type)  # NEW
    workflow.add_node("load_schema", _node_load_schema)
    workflow.add_node("enrich_schema", _node_enrich_schema)
    workflow.add_node("validate_requirements", _node_validate_requirements)
    workflow.add_node("build_result", _node_build_result)

    # Entry point
    workflow.set_entry_point("detect_ui_decision")

    # Conditional routing from detect_ui_decision
    workflow.add_conditional_edges(
        "detect_ui_decision",
        _decide_after_ui_detection,
        {
            "error": END,
            "continue": "detect_hitl_response"
        }
    )

    # Conditional routing from detect_hitl_response
    workflow.add_conditional_edges(
        "detect_hitl_response",
        _decide_after_hitl_detection,
        {
            "yes": "trigger_advanced_specs",
            "no": "extract_product_type",
            "continue": "extract_product_type"
        }
    )

    # Advanced specs → build result
    workflow.add_edge("trigger_advanced_specs", "build_result")

    # Normal validation flow (NEW: added taxonomy normalization)
    workflow.add_edge("extract_product_type", "normalize_product_type")  # NEW
    workflow.add_edge("normalize_product_type", "load_schema")  # NEW
    workflow.add_edge("load_schema", "enrich_schema")
    workflow.add_edge("enrich_schema", "validate_requirements")
    workflow.add_edge("validate_requirements", "build_result")
    workflow.add_edge("build_result", END)

    return workflow


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ValidationDeepAgent:
    """Pure LangGraph Deep Agent for validation workflow.

    This agent converts the monolithic ValidationTool.validate() method
    (1,130 lines) into a pure LangGraph agent with discrete nodes and
    proper observability.

    Backward compatible with existing ValidationTool API.
    """

    def __init__(
        self,
        enable_ppi: bool = True,
        enable_standards_enrichment: bool = True
    ):
        """Initialize the Validation Deep Agent.

        Args:
            enable_ppi: Enable PPI workflow for schema generation
            enable_standards_enrichment: Enable Standards RAG enrichment
        """
        self.enable_ppi = enable_ppi
        self.enable_standards_enrichment = enable_standards_enrichment
        self.workflow = create_validation_workflow().compile()

        logger.info("[ValidationDeepAgent] Initialized with PPI workflow: %s",
                   "enabled" if enable_ppi else "disabled")
        logger.info("[ValidationDeepAgent] Standards enrichment: %s",
                   "enabled" if enable_standards_enrichment else "disabled")

    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_standards_enrichment: Optional[bool] = None,
        source_workflow: Optional[str] = None,
        is_standards_enriched: Optional[bool] = None,
        is_taxonomy_normalized: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Validate user input and requirements.

        BACKWARD COMPATIBLE with existing ValidationTool.validate() API.

        Args:
            user_input: User's requirement description
            expected_product_type: Expected product type (optional)
            session_id: Session identifier (for logging/tracking)
            enable_standards_enrichment: Override standards enrichment setting
            source_workflow: Source workflow identifier
            is_standards_enriched: Skip enrichment if input is pre-enriched (auto-derived from source_workflow)
            is_taxonomy_normalized: Skip taxonomy normalization if pre-normalized (auto-derived from source_workflow)

        Returns:
            Validation result with comprehensive metadata
        """
        start_time = time.time()

        if not session_id:
            session_id = f"session_{int(start_time * 1000)}"

        logger.info("[ValidationDeepAgent] Starting validation")
        logger.info("[ValidationDeepAgent] Session: %s", session_id)
        logger.info("[ValidationDeepAgent] Input: %s", user_input[:100] + "..." if len(user_input) > 100 else user_input)

        # Initialize state
        initial_state: ValidationDeepAgentState = {
            "user_input": user_input,
            "expected_product_type": expected_product_type,
            "session_id": session_id,
            "enable_standards_enrichment": (
                enable_standards_enrichment if enable_standards_enrichment is not None
                else self.enable_standards_enrichment
            ),
            "source_workflow": source_workflow,
            "is_standards_enriched": (
                is_standards_enriched if is_standards_enriched is not None
                else (source_workflow == "solution_deep_agent")
            ),
            "is_taxonomy_normalized": (
                is_taxonomy_normalized if is_taxonomy_normalized is not None
                else (source_workflow == "solution_deep_agent")
            ),
            "tools_called": [],
            "phases_completed": [],
            "start_time": start_time,
            "is_ui_decision": False,
            "is_hitl_yes": False,
            "is_hitl_no": False,
            "validation_bypassed": False,
            "session_cache_hit": False,
            "standards_rag_invoked": False
        }

        try:
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Return result (backward compatible)
            if final_state.get("error"):
                logger.error("[ValidationDeepAgent] Validation failed: %s", final_state["error"])
                return {
                    "success": False,
                    "error": final_state["error"],
                    "session_id": session_id,
                    "error_type": "ValidationError",
                    "processing_time_ms": processing_time_ms
                }

            result = final_state.get("result", {})
            result["processing_time_ms"] = processing_time_ms

            logger.info("[ValidationDeepAgent] ✓ Validation completed successfully")
            logger.info("[ValidationDeepAgent] Product Type: %s", result.get("product_type"))
            logger.info("[ValidationDeepAgent] Valid: %s", result.get("is_valid"))
            logger.info("[ValidationDeepAgent] Processing Time: %dms", processing_time_ms)

            return result

        except Exception as e:
            logger.error("[ValidationDeepAgent] ✗ Validation failed: %s", e, exc_info=True)
            processing_time_ms = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": session_id,
                "processing_time_ms": processing_time_ms
            }


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION TOOL WRAPPER CLASS (Backward Compatibility)
# ═══════════════════════════════════════════════════════════════════════════

class ValidationTool:
    """
    Validation Tool - Step 1 of Product Search Workflow (LangGraph Agent Wrapper)

    This is now a pure LangGraph Deep Agent wrapper.
    All logic has been moved to ValidationDeepAgent.

    Responsibilities:
    1. Extract product type from user input
    2. Load or generate product schema (PPI workflow if needed)
    3. Validate user requirements against schema
    4. Return structured validation result
    """

    def __init__(
        self,
        enable_ppi: bool = True,
        enable_phase2: bool = True,
        enable_phase3: bool = True,
        use_async_workflow: bool = True,
        enable_standards_enrichment: bool = True
    ):
        """
        Initialize the validation tool (backward compatibility wrapper).

        This is now a thin wrapper around ValidationDeepAgent.

        Args:
            enable_ppi: Enable PPI workflow for schema generation
            enable_phase2: Enable Phase 2 parallel optimization
            enable_phase3: Enable Phase 3 async optimization (highest priority)
            use_async_workflow: Use new async SchemaWorkflow (recommended)
            enable_standards_enrichment: Enable Standards RAG enrichment (Step 1.2.1)
        """
        # Store configuration for backward compatibility (used by other methods)
        self.enable_ppi = enable_ppi
        self.enable_phase2 = enable_phase2
        self.enable_phase3 = enable_phase3
        self.use_async_workflow = use_async_workflow
        self.enable_standards_enrichment = enable_standards_enrichment

        # Create the pure LangGraph deep agent
        self._agent = ValidationDeepAgent(
            enable_ppi=enable_ppi,
            enable_standards_enrichment=enable_standards_enrichment
        )
        logger.info("[ValidationTool] ✓ Initialized with ValidationDeepAgent (pure LangGraph)")

        logger.info("[ValidationTool] Configuration: PPI=%s, Phase2=%s, Phase3=%s, Standards=%s",
                   "enabled" if enable_ppi else "disabled",
                   "enabled" if enable_phase2 else "disabled",
                   "enabled" if enable_phase3 else "disabled",
                   "enabled" if enable_standards_enrichment else "disabled")

    def update_specifications(self, current_specs: Dict[str, Any], new_input: str) -> Dict[str, Any]:
        """Update existing specifications with newly provided values from the user."""
        from common.tools.intent_tools import extract_requirements_tool
        logger.info(f"[ValidationTool] Updating specifications with new input: '{new_input}'")
        try:
            extract_result = extract_requirements_tool.invoke({
                "user_input": new_input
            })
            new_specs = extract_result.get("specifications", {})
            updated_specs = dict(current_specs)
            updated_specs.update(new_specs)
            return updated_specs
        except Exception as e:
            logger.error(f"[ValidationTool] Failed to update specifications: {e}")
            return current_specs

    @timed_execution("VALIDATION_TOOL", threshold_ms=20000)
    @debug_log("VALIDATION_TOOL", log_args=True, log_result=False)
    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_standards_enrichment: Optional[bool] = None,
        source_workflow: Optional[str] = None,
        is_standards_enriched: Optional[bool] = None,
        is_taxonomy_normalized: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Validate user input and requirements.

        BACKWARD COMPATIBLE with original ValidationTool.validate() API.
        Implementation now delegated to ValidationDeepAgent (pure LangGraph).

        Args:
            user_input: User's requirement description
            expected_product_type: Expected product type (optional, for validation)
            session_id: Session identifier (for logging/tracking)
            enable_standards_enrichment: Override standards enrichment setting
            source_workflow: Source workflow identifier
            is_standards_enriched: Skip standards enrichment if pre-enriched
            is_taxonomy_normalized: Skip taxonomy normalization if pre-normalized

        Returns:
            Validation result with schema, requirements, validity, etc.
        """
        # Delegate to the LangGraph deep agent
        result = self._agent.validate(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id,
            enable_standards_enrichment=enable_standards_enrichment if enable_standards_enrichment is not None else self.enable_standards_enrichment,
            source_workflow=source_workflow,
            is_standards_enriched=is_standards_enriched,
            is_taxonomy_normalized=is_taxonomy_normalized
        )

        return result

    def get_schema_only(
        self,
        product_type: str,
        enable_ppi: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Load or generate schema without validation.

        Args:
            product_type: Product type to get schema for
            enable_ppi: Override PPI setting (uses instance setting if None)

        Returns:
            Schema result with source information
        """
        logger.info("[ValidationTool] Loading schema for: %s", product_type)

        try:
            from common.tools.schema_tools import load_schema_tool

            schema_result = load_schema_tool.invoke({
                "product_type": product_type,
                "enable_ppi": enable_ppi if enable_ppi is not None else self.enable_ppi
            })

            logger.info("[ValidationTool] ✓ Schema loaded from: %s",
                       schema_result.get("source", "unknown"))

            return schema_result

        except Exception as e:
            logger.error("[ValidationTool] ✗ Schema loading failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "schema": {}
            }

    def validate_with_schema(
        self,
        user_input: str,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate user input against a provided schema.

        Args:
            user_input: User's requirement description
            product_type: Product type
            schema: Pre-loaded schema

        Returns:
            Validation result
        """
        logger.info("[ValidationTool] Validating with provided schema")

        try:
            from common.tools.schema_tools import validate_requirements_tool

            validation_result = validate_requirements_tool.invoke({
                "user_input": user_input,
                "product_type": product_type,
                "schema": schema
            })

            return {
                "success": True,
                "product_type": product_type,
                "provided_requirements": validation_result.get("provided_requirements", {}),
                "missing_fields": validation_result.get("missing_fields", []),
                "is_valid": validation_result.get("is_valid", False)
            }

        except Exception as e:
            logger.error("[ValidationTool] ✗ Validation failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 2: PARALLEL OPTIMIZATION METHODS
    # ════════════════════════════════════════════════════════════════════════════

    def validate_multiple_products_parallel(
        self,
        product_types: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate schemas for multiple products in parallel (Phase 2 optimization).

        Use this when user needs schemas for multiple products at once.
        Example: "I need schemas for Temperature Transmitter, Pressure Gauge, Level Switch"

        Args:
            product_types: List of product types
            session_id: Session identifier

        Returns:
            Dictionary mapping product_type -> validation result
        """
        if not self.enable_phase2:
            logger.warning("[Phase 2] Parallel optimization disabled, falling back to sequential")
            return self._validate_sequentially(product_types, session_id)

        try:
            from agentic.deep_agent.schema.generation.parallel_generator import ParallelSchemaGenerator

            logger.info(f"[Phase 2] Starting parallel schema generation for {len(product_types)} products")

            generator = ParallelSchemaGenerator(max_workers=min(3, len(product_types)))
            schemas = generator.generate_schemas_in_parallel(product_types, force_regenerate=False)

            # Validate each schema
            results = {}
            for product_type, schema_result in schemas.items():
                if schema_result.get('success'):
                    results[product_type] = {
                        "success": True,
                        "product_type": product_type,
                        "schema": schema_result.get('schema'),
                        "schema_source": schema_result.get('source'),
                        "optimization": "phase2_parallel"
                    }
                else:
                    results[product_type] = {
                        "success": False,
                        "product_type": product_type,
                        "error": schema_result.get('error')
                    }

            logger.info(f"[Phase 2] Parallel generation completed for {len(results)} products")
            return results

        except ImportError:
            logger.warning("[Phase 2] Parallel Schema Generator not available, using sequential")
            return self._validate_sequentially(product_types, session_id)
        except Exception as e:
            logger.error(f"[Phase 2] Error in parallel validation: {e}")
            return self._validate_sequentially(product_types, session_id)

    def _validate_sequentially(
        self,
        product_types: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback to sequential validation when parallel not available."""
        results = {}
        for product_type in product_types:
            result = self.validate(product_type, session_id=session_id)
            results[product_type] = result
        return results

    def enrich_schema_parallel(
        self,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich schema using parallel field group queries (Phase 2 optimization).

        Instead of querying Standards RAG sequentially for each field group,
        query all field groups in parallel (5x faster!).

        Args:
            product_type: Product type
            schema: Schema to enrich

        Returns:
            Enriched schema with populated fields
        """
        if not self.enable_phase2:
            logger.warning("[Phase 2] Parallel enrichment disabled")
            return schema

        try:
            from agentic.workflows.standards_rag.parallel_standards_enrichment import ParallelStandardsEnrichment

            logger.info(f"[Phase 2] Starting parallel enrichment for {product_type}")

            enricher = ParallelStandardsEnrichment(max_workers=5)
            enriched = enricher.enrich_schema_in_parallel(product_type, schema)

            logger.info(f"[Phase 2] Parallel enrichment completed for {product_type}")
            return enriched

        except ImportError:
            logger.warning("[Phase 2] Parallel Standards Enrichment not available")
            return schema
        except Exception as e:
            logger.error(f"[Phase 2] Error in parallel enrichment: {e}")
            return schema

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 3: ASYNC WORKFLOW (Complete Schema Lifecycle)
    # ════════════════════════════════════════════════════════════════════════════

    async def get_or_generate_schema_async(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Get or generate schema using complete async workflow (Phase 3).

        Complete lifecycle:
        1. Check session cache (FIX #A1)
        2. Check database
        3. Generate via PPI (with Phase 1+2+3 optimizations)
        4. Enrich with standards (async parallel)
        5. Store to database
        6. Return to user

        This is the RECOMMENDED method for single or multiple products.

        Args:
            product_type: Product type to get schema for
            session_id: Session identifier
            force_regenerate: Force regeneration (skip caches)

        Returns:
            Dictionary with schema and metadata
        """

        if not self.enable_phase3:
            logger.warning("[Phase 3] Async workflow disabled")
            # Fall back to sync validation
            return self.validate(product_type, session_id=session_id)

        try:
            from agentic.workflows.schema.schema_workflow import SchemaWorkflow

            logger.info(f"[Phase 3] Starting async workflow for: {product_type}")

            workflow = SchemaWorkflow(use_phase3_async=True)
            result = await workflow.get_or_generate_schema(
                product_type,
                session_id=session_id,
                force_regenerate=force_regenerate
            )

            return result

        except ImportError:
            logger.warning("[Phase 3] SchemaWorkflow not available, falling back to Phase 2")
            return self.validate_multiple_products_parallel([product_type], session_id)
        except Exception as e:
            logger.error(f"[Phase 3] Error in async workflow: {e}")
            return self.validate(product_type, session_id=session_id)

    async def get_or_generate_schemas_batch_async(
        self,
        product_types: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get or generate schemas for multiple products concurrently (Phase 3).

        Uses async concurrent execution for multiple products with benefits:
        - All products generated concurrently
        - Shared database lookups
        - Combined enrichment queries
        - Non-blocking I/O throughout

        Expected Performance (3 products):
        - Before: (437 + 210) × 3 = 1941 seconds
        - After Phase 3: 100-120 seconds (19x faster!)

        Args:
            product_types: List of product types
            session_id: Session identifier

        Returns:
            Dictionary mapping product_type -> schema result
        """

        if not self.enable_phase3:
            logger.warning("[Phase 3] Async batch disabled, using Phase 2")
            return self.validate_multiple_products_parallel(product_types, session_id)

        try:
            from agentic.workflows.schema.schema_workflow import SchemaWorkflow

            logger.info(f"[Phase 3] Starting async batch workflow for {len(product_types)} products")

            workflow = SchemaWorkflow(use_phase3_async=True)
            results = await workflow.get_or_generate_schemas_batch(
                product_types,
                session_id=session_id
            )

            logger.info(f"[Phase 3] Batch workflow completed")

            return results

        except ImportError:
            logger.warning("[Phase 3] SchemaWorkflow not available, falling back to Phase 2")

            # Fallback: Use Phase 2 in parallel
            results = {}
            for product_type in product_types:
                result = self.validate_multiple_products_parallel(
                    [product_type],
                    session_id
                )
                results.update(result)
            return results

        except Exception as e:
            logger.error(f"[Phase 3] Error in async batch workflow: {e}")

            # Fallback: Sequential validation
            results = {}
            for product_type in product_types:
                result = self.validate(product_type, session_id=session_id)
                results[product_type] = result
            return results


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example usage of ValidationTool"""
    print("\n" + "="*70)
    print("VALIDATION TOOL - STANDALONE EXAMPLE")
    print("="*70)

    # Initialize tool
    tool = ValidationTool(enable_ppi=True)

    # Example 1: Validate user input
    print("\n[Example 1] Validate user input:")
    result = tool.validate(
        user_input="I need a pressure transmitter with 4-20mA output, 0-100 PSI range",
        session_id="test_session_001"
    )

    print(f"✓ Success: {result['success']}")
    print(f"✓ Product Type: {result.get('product_type')}")
    print(f"✓ Valid: {result.get('is_valid')}")
    print(f"✓ Schema Source: {result.get('schema_source')}")
    print(f"✓ PPI Used: {result.get('ppi_workflow_used')}")
    print(f"✓ Missing Fields: {result.get('missing_fields', [])}")

    # Example 2: Get schema only
    print("\n[Example 2] Get schema only:")
    schema_result = tool.get_schema_only("flow meter")
    print(f"✓ Schema Source: {schema_result.get('source')}")
    print(f"✓ Has Mandatory Fields: {bool(schema_result.get('schema', {}).get('mandatory'))}")


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    example_usage()
