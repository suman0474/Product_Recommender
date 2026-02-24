"""
Search Orchestration Workflow - Unified LangGraph Orchestrator
==============================================================

This module provides a unified LangGraph orchestrator that coordinates three
existing modules as discrete nodes:

Node 1: ValidationTool (validation_functions.py) - Product type detection, schema validation, HITL detection
Node 2: AdvancedSpecificationAgent (advanced_specs_functions.py) - Advanced parameter discovery
Node 3: VendorAnalysisDeepAgent (9 internal nodes) - Vendor matching with Strategy RAG
Node 4: RankingTool - Final product ranking

Architecture:
- Each module/agent remains internally unchanged
- Parent orchestrator calls ValidationTool.validate(), discover_advanced_specs(), agent.analyze() as black-box functions
- ValidationTool (Node 1) handles both validation AND HITL detection
- HITL routing: YES -> advanced_specs -> vendor_analysis; NO -> skip to vendor_analysis
- Session isolation via set_request_context() and BoundedCache

Graph Flow:
    validation (Node 1) -> [hitl_decision | advanced_specs | vendor_analysis | missing_fields_response]
                                  |              |                  |
                           [end | proceed]       |                  |
                                  |              |                  |
                           advanced_specs <------+                  |
                                  |                                 |
                                  v                                 |
                           vendor_analysis <------------------------+
                                  |
                       [ranking | no_matches_response]
                                  |
                                  v
                           compose_response
                                  |
                                 END
"""

import logging
import time
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

# Import session context utilities
from .validation_functions import (
    set_request_context,
    clear_request_context,
    get_session_context,
)

# Import BoundedCache for session-isolated caching
from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

# Import ExecutionContext
from common.infrastructure.context import (
    ExecutionContext,
    set_context,
    clear_context,
)

# Import workflow locking
try:
    from common.infrastructure.locking import with_workflow_lock
except ImportError:
    # Fallback: no-op decorator if locking not available
    def with_workflow_lock(**kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SearchOrchestrationState(TypedDict, total=False):
    """
    Unified LangGraph state for Search Orchestration Workflow.

    Carries data between the orchestrator nodes that wrap each agent.
    Non-serializable objects (agent instances) are stored in module-level
    registry keyed by session_id.
    """

    # =========================================================================
    # INPUTS (set once at graph entry)
    # =========================================================================
    session_id: str
    user_input: str
    expected_product_type: Optional[str]
    user_provided_fields: Optional[Dict[str, Any]]
    enable_ppi: bool
    auto_mode: bool
    source_workflow: Optional[str]
    user_decision: Optional[str]  # "yes", "no", or None
    current_phase: Optional[str]  # Frontend-provided phase hint
    standards_enriched: bool  # [FIX Feb 2026] Skip standards enrichment if pre-enriched by solution workflow

    # =========================================================================
    # HITL STATE
    # =========================================================================
    is_hitl_yes: bool
    is_hitl_no: bool
    hitl_prompt_shown: bool
    advanced_specs_presented: bool
    awaiting_user_input: bool

    # =========================================================================
    # VALIDATION NODE OUTPUT
    # =========================================================================
    validation_result: Dict[str, Any]
    product_type: str
    schema: Dict[str, Any]
    provided_requirements: Dict[str, Any]
    missing_fields: List[str]
    optional_fields: List[str]
    is_valid: bool
    validation_bypassed: bool
    hitl_message: Optional[str]

    # =========================================================================
    # ADVANCED SPECS NODE OUTPUT
    # =========================================================================
    advanced_specs_result: Dict[str, Any]
    advanced_parameters: List[Dict[str, Any]]

    # =========================================================================
    # VENDOR ANALYSIS NODE OUTPUT
    # =========================================================================
    vendor_analysis_result: Dict[str, Any]
    vendor_matches: List[Dict[str, Any]]

    # =========================================================================
    # RANKING NODE OUTPUT
    # =========================================================================
    ranking_result: Dict[str, Any]
    ranked_products: List[Dict[str, Any]]
    top_product: Optional[Dict[str, Any]]

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    response: str
    response_data: Dict[str, Any]

    # =========================================================================
    # OBSERVABILITY
    # =========================================================================
    tools_called: List[str]
    tool_results_summary: Dict[str, Any]
    current_node: str
    phases_completed: List[str]
    error: Optional[str]

    # =========================================================================
    # TIMING
    # =========================================================================
    start_time: float
    processing_time_ms: int


# =============================================================================
# SESSION MEMORY REGISTRY (non-serializable objects stored outside state)
# =============================================================================

_SESSION_ORCHESTRATION_CACHE: BoundedCache = get_or_create_cache(
    name="search_orchestration",
    max_size=500,
    ttl_seconds=0  # [FIX Feb 2026] No TTL — persist for session lifetime
)


def _get_orchestration_context(session_id: str) -> Optional[Dict[str, Any]]:
    """Get cached orchestration context for HITL resume."""
    if not session_id:
        return None
    key = f"orch:{session_id}"
    return _SESSION_ORCHESTRATION_CACHE.get(key)


def _cache_orchestration_context(session_id: str, context: Dict[str, Any]):
    """Cache orchestration context for HITL resume."""
    if not session_id:
        return
    key = f"orch:{session_id}"
    _SESSION_ORCHESTRATION_CACHE.set(key, context)
    logger.debug(f"[SearchOrchestrator] Cached context for {session_id}")


# =============================================================================
# STATE FACTORY
# =============================================================================

def create_search_orchestration_state(
    user_input: str,
    session_id: str,
    expected_product_type: Optional[str] = None,
    user_provided_fields: Optional[Dict[str, Any]] = None,
    enable_ppi: bool = True,
    auto_mode: bool = True,
    source_workflow: Optional[str] = None,
    user_decision: Optional[str] = None,
    current_phase: Optional[str] = None,
    standards_enriched: bool = False,
) -> SearchOrchestrationState:
    """Create initial state for search orchestration workflow."""
    return SearchOrchestrationState(
        # Inputs
        session_id=session_id,
        user_input=user_input,
        expected_product_type=expected_product_type,
        user_provided_fields=user_provided_fields or {},
        enable_ppi=enable_ppi,
        auto_mode=auto_mode,
        source_workflow=source_workflow or "direct",
        user_decision=user_decision,
        current_phase=current_phase,
        standards_enriched=standards_enriched,

        # HITL State
        is_hitl_yes=False,
        is_hitl_no=False,
        hitl_prompt_shown=False,
        advanced_specs_presented=False,
        awaiting_user_input=False,

        # Validation Output
        validation_result={},
        product_type="",
        schema={},
        provided_requirements={},
        missing_fields=[],
        optional_fields=[],
        is_valid=False,
        validation_bypassed=False,
        hitl_message=None,

        # Advanced Specs Output
        advanced_specs_result={},
        advanced_parameters=[],

        # Vendor Analysis Output
        vendor_analysis_result={},
        vendor_matches=[],

        # Ranking Output
        ranking_result={},
        ranked_products=[],
        top_product=None,

        # Final Output
        response="",
        response_data={},

        # Observability
        tools_called=[],
        tool_results_summary={},
        current_node="init",
        phases_completed=[],
        error=None,

        # Timing
        start_time=time.time(),
        processing_time_ms=0,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _is_hitl_yes_response(user_input: str) -> bool:
    """Check if user input is a YES response to HITL prompt."""
    normalized = user_input.lower().strip()
    return normalized in ["yes", "y", "yeah", "yep", "sure", "ok", "okay"]


def _is_hitl_no_response(user_input: str) -> bool:
    """Check if user input is a NO response to HITL prompt."""
    normalized = user_input.lower().strip()
    return normalized in ["no", "n", "nope", "skip", "continue", "proceed"] or normalized.startswith("no ")


def _mark_phase_complete(state: SearchOrchestrationState, phase: str) -> List[str]:
    """Mark a phase as complete and return updated list."""
    phases = list(state.get("phases_completed", []))
    if phase not in phases:
        phases.append(phase)
    return phases


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def validation_node(state: SearchOrchestrationState) -> dict:
    """
    Node 1: Entry point - ValidationDeepAgent wrapper with HITL detection.

    This node serves as the entry point for the orchestration workflow.
    It first checks for HITL responses (YES/NO from previous conversation),
    then calls ValidationDeepAgent which has its own 8-node internal workflow:
    - detect_ui_decision
    - detect_hitl_response
    - trigger_advanced_specs
    - extract_product_type
    - load_schema
    - enrich_schema
    - validate_requirements
    - build_result

    Routes to:
    - advanced_specs: User said YES to HITL prompt
    - vendor_analysis: User said NO to HITL prompt
    - hitl_decision: New search, validation complete, check if HITL needed
    - missing_fields_response: Critical fields missing
    """
    session_id = state.get("session_id", "")
    user_input = state.get("user_input", "")
    user_decision = state.get("user_decision")
    current_phase = state.get("current_phase")
    expected_product_type = state.get("expected_product_type")
    enable_ppi = state.get("enable_ppi", True)
    source_workflow = state.get("source_workflow")

    logger.info("=" * 70)
    logger.info("[SearchOrchestrator] Node 1: validation (entry point)")
    logger.info(f"   Session: {session_id}")
    logger.info(f"   Input: {user_input[:100]}..." if len(user_input) > 100 else f"   Input: {user_input}")
    logger.info(f"   User Decision: {user_decision}")
    logger.info(f"   Current Phase: {current_phase}")
    logger.info("=" * 70)

    # Set request context for session isolation
    set_request_context(session_id)

    # =========================================================================
    # PHASE 1: Check for HITL responses (YES/NO from previous conversation)
    # =========================================================================
    cached_ctx = _get_orchestration_context(session_id)
    validation_ctx = get_session_context(session_id)

    # Load cached state
    hitl_prompt_shown = False
    advanced_specs_presented = False
    product_type = expected_product_type or ""
    schema = {}
    provided_requirements = state.get("user_provided_fields", {}) or {}

    if cached_ctx:
        hitl_prompt_shown = cached_ctx.get("hitl_prompt_shown", False)
        advanced_specs_presented = cached_ctx.get("advanced_specs_presented", False)
        product_type = cached_ctx.get("product_type", product_type)
        schema = cached_ctx.get("schema", {})
        provided_requirements = cached_ctx.get("provided_requirements", provided_requirements)
        logger.info(f"[SearchOrchestrator] Cached context: hitl_shown={hitl_prompt_shown}, adv_specs={advanced_specs_presented}")

    if validation_ctx:
        product_type = validation_ctx.get("product_type", product_type)
        schema = validation_ctx.get("schema", schema)
        provided_requirements = validation_ctx.get("provided_requirements", provided_requirements)

    # Check for YES/NO responses
    is_yes = _is_hitl_yes_response(user_input) or user_decision == "yes"
    is_no = _is_hitl_no_response(user_input) or user_decision == "no"

    # Check phase hints from frontend
    if current_phase in ["collect_advanced_specs", "awaiting_advanced_input"]:
        advanced_specs_presented = True
    if current_phase in ["hitl_prompt", "awaiting_confirmation"]:
        hitl_prompt_shown = True

    logger.info(f"[SearchOrchestrator] HITL detection: is_yes={is_yes}, is_no={is_no}, hitl_shown={hitl_prompt_shown}")

    # Handle HITL YES response - skip validation, go to advanced specs
    if (hitl_prompt_shown or advanced_specs_presented) and is_yes:
        logger.info("[SearchOrchestrator] HITL YES - bypassing validation, routing to advanced_specs")
        return {
            "is_hitl_yes": True,
            "is_hitl_no": False,
            "hitl_prompt_shown": True,
            "advanced_specs_presented": True,
            "product_type": product_type,
            "schema": schema,
            "provided_requirements": provided_requirements,
            "validation_bypassed": True,
            "is_valid": True,
            "current_node": "validation",
            "phases_completed": _mark_phase_complete(state, "validation"),
            "tools_called": state.get("tools_called", []) + ["validation_hitl_check"]
        }

    # Handle HITL NO response - skip validation, go to vendor analysis
    if (hitl_prompt_shown or advanced_specs_presented) and is_no:
        logger.info("[SearchOrchestrator] HITL NO - bypassing validation, routing to advanced_specs (mandatory)")
        return {
            "is_hitl_yes": False,
            "is_hitl_no": True,
            "hitl_prompt_shown": True,
            "advanced_specs_presented": False,
            "product_type": product_type,
            "schema": schema,
            "provided_requirements": provided_requirements,
            "validation_bypassed": True,
            "is_valid": True,
            "current_node": "validation",
            "phases_completed": _mark_phase_complete(state, "validation"),
            "tools_called": state.get("tools_called", []) + ["validation_hitl_check"]
        }

    # =========================================================================
    # PHASE 2: Run ValidationTool (function-based validation)
    # =========================================================================
    logger.info("[SearchOrchestrator] Running ValidationTool...")

    try:
        from .validation_functions import ValidationTool

        validation_agent = ValidationTool(
            enable_ppi=enable_ppi
        )

        # disable standards enrichment in ValidationTool to avoid redundancy.
        standards_enriched = state.get("standards_enriched", False)
        enable_standards = enable_ppi and (not standards_enriched)
        if standards_enriched:
            logger.info(
                "[SearchOrchestrator] Standards enrichment SKIPPED "
                "(pre-enriched by solution workflow)"
            )

        validation_result = validation_agent.validate(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id,
            enable_standards_enrichment=enable_standards,
            source_workflow=source_workflow
        )

        logger.info(f"[SearchOrchestrator] ValidationTool result: success={validation_result.get('success')}, "
                    f"product_type={validation_result.get('product_type')}, "
                    f"is_valid={validation_result.get('is_valid')}")

        # Extract fields from validation result
        product_type = validation_result.get("product_type", "")
        schema = validation_result.get("schema", {})
        provided_requirements = validation_result.get("provided_requirements", {})
        missing_fields = validation_result.get("missing_fields", [])
        optional_fields = validation_result.get("optional_fields", [])
        is_valid = validation_result.get("is_valid", False)
        hitl_message = validation_result.get("hitl_message")
        validation_bypassed = validation_result.get("validation_bypassed", False)
        advanced_specs_info = validation_result.get("advanced_specs_info")

        # Cache context for HITL resume
        _cache_orchestration_context(session_id, {
            "product_type": product_type,
            "schema": schema,
            "provided_requirements": provided_requirements,
            "hitl_prompt_shown": False,
            "advanced_specs_presented": bool(advanced_specs_info)
        })

        return {
            "validation_result": validation_result,
            "product_type": product_type,
            "schema": schema,
            "provided_requirements": provided_requirements,
            "missing_fields": missing_fields,
            "optional_fields": optional_fields,
            "is_valid": is_valid,
            "is_hitl_yes": False,
            "is_hitl_no": False,
            "hitl_prompt_shown": False,
            "validation_bypassed": validation_bypassed,
            "hitl_message": hitl_message,
            "current_node": "validation",
            "phases_completed": _mark_phase_complete(state, "validation"),
            "tools_called": state.get("tools_called", []) + ["ValidationDeepAgent"],
            "tool_results_summary": {
                **state.get("tool_results_summary", {}),
                "validation": {
                    "success": validation_result.get("success"),
                    "product_type": product_type,
                    "is_valid": is_valid,
                    "missing_fields_count": len(missing_fields)
                }
            }
        }

    except Exception as e:
        logger.error(f"[SearchOrchestrator] ValidationDeepAgent failed: {e}", exc_info=True)
        return {
            "error": f"Validation failed: {str(e)}",
            "is_hitl_yes": False,
            "is_hitl_no": False,
            "current_node": "validation",
            "phases_completed": _mark_phase_complete(state, "validation")
        }


def hitl_decision_node(state: SearchOrchestrationState) -> dict:
    """
    Node 3: HITL decision point - pause and return for user input if needed.

    Conditions for HITL pause:
    - auto_mode=False AND hitl_prompt_shown=False
    - Validation passed but user hasn't confirmed

    Caches session context for HITL resume.
    """
    logger.info("=" * 70)
    logger.info("[SearchOrchestrator] Node 3: hitl_decision")
    logger.info("=" * 70)

    session_id = state.get("session_id", "")
    auto_mode = state.get("auto_mode", True)
    hitl_prompt_shown = state.get("hitl_prompt_shown", False)
    hitl_message = state.get("hitl_message")
    product_type = state.get("product_type", "")
    schema = state.get("schema", {})
    provided_requirements = state.get("provided_requirements", {})
    missing_fields = state.get("missing_fields", [])

    # If auto_mode, skip HITL
    if auto_mode:
        logger.info("[SearchOrchestrator] Auto mode enabled - skipping HITL prompt")
        return {
            "awaiting_user_input": False,
            "current_node": "hitl_decision",
            "phases_completed": _mark_phase_complete(state, "hitl_decision")
        }

    # If HITL prompt already shown, don't show again
    if hitl_prompt_shown:
        logger.info("[SearchOrchestrator] HITL prompt already shown - proceeding")
        return {
            "awaiting_user_input": False,
            "current_node": "hitl_decision",
            "phases_completed": _mark_phase_complete(state, "hitl_decision")
        }

    # Generate HITL prompt
    if not hitl_message:
        num_provided = len(provided_requirements)
        num_missing = len(missing_fields)

        if num_missing > 0:
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

    # Cache context for HITL resume
    _cache_orchestration_context(session_id, {
        "product_type": product_type,
        "schema": schema,
        "provided_requirements": provided_requirements,
        "hitl_prompt_shown": True,
        "advanced_specs_presented": False
    })

    logger.info("[SearchOrchestrator] Pausing for HITL input")

    return {
        "awaiting_user_input": True,
        "hitl_prompt_shown": True,
        "hitl_message": hitl_message,
        "response": hitl_message,
        "response_data": {
            "product_type": product_type,
            "schema": schema,
            "provided_requirements": provided_requirements,
            "missing_fields": missing_fields,
            "awaiting_user_input": True,
            "current_phase": "hitl_prompt",
            "hitl_prompt_shown": True
        },
        "current_node": "hitl_decision",
        "phases_completed": _mark_phase_complete(state, "hitl_decision")
    }


def advanced_specs_node(state: SearchOrchestrationState) -> dict:
    """
    Node 4: Wrapper for AdvancedSpecificationAgent.discover().

    Calls the existing AdvancedSpecificationAgent which has its own 6-node
    internal LangGraph workflow.
    """
    logger.info("=" * 70)
    logger.info("[SearchOrchestrator] Node 4: advanced_specs")
    logger.info("=" * 70)

    session_id = state.get("session_id", "")
    product_type = state.get("product_type", "")
    schema = state.get("schema", {})

    if not product_type:
        logger.warning("[SearchOrchestrator] No product_type - skipping advanced specs")
        return {
            "advanced_specs_result": {"success": False, "error": "No product type"},
            "advanced_parameters": [],
            "current_node": "advanced_specs",
            "phases_completed": _mark_phase_complete(state, "advanced_specs")
        }

    try:
        from .advanced_specs_functions import AdvancedSpecificationAgent

        adv_agent = AdvancedSpecificationAgent()
        adv_result = adv_agent.discover(
            product_type=product_type,
            session_id=session_id,
            existing_schema=schema
        )

        advanced_parameters = adv_result.get("unique_specifications", [])
        total_specs = adv_result.get("total_unique_specifications", 0)

        logger.info(f"[SearchOrchestrator] Advanced specs result: success={adv_result.get('success')}, "
                    f"total_specs={total_specs}")

        # Cache context with advanced_specs_presented=True
        _cache_orchestration_context(session_id, {
            "product_type": product_type,
            "schema": schema,
            "provided_requirements": state.get("provided_requirements", {}),
            "hitl_prompt_shown": True,
            "advanced_specs_presented": True
        })

        return {
            "advanced_specs_result": adv_result,
            "advanced_parameters": advanced_parameters,
            "current_node": "advanced_specs",
            "phases_completed": _mark_phase_complete(state, "advanced_specs"),
            "tools_called": state.get("tools_called", []) + ["AdvancedSpecificationAgent"],
            "tool_results_summary": {
                **state.get("tool_results_summary", {}),
                "advanced_specs": {
                    "success": adv_result.get("success"),
                    "total_specs": total_specs
                }
            }
        }

    except Exception as e:
        logger.error(f"[SearchOrchestrator] Advanced specs failed: {e}", exc_info=True)
        return {
            "advanced_specs_result": {"success": False, "error": str(e)},
            "advanced_parameters": [],
            "current_node": "advanced_specs",
            "phases_completed": _mark_phase_complete(state, "advanced_specs")
        }


def vendor_analysis_node(state: SearchOrchestrationState) -> dict:
    """
    Node 5: Wrapper for VendorAnalysisDeepAgent.analyze().

    Calls the existing VendorAnalysisDeepAgent which has its own 9-node
    internal LangGraph workflow including Strategy RAG.
    """
    logger.info("=" * 70)
    logger.info("[SearchOrchestrator] Node 5: vendor_analysis")
    logger.info("=" * 70)

    session_id = state.get("session_id", "")
    product_type = state.get("product_type", "")
    schema = state.get("schema", {})
    provided_requirements = state.get("provided_requirements", {})
    advanced_parameters = state.get("advanced_parameters", [])

    if not product_type:
        logger.warning("[SearchOrchestrator] No product_type - cannot analyze vendors")
        return {
            "vendor_analysis_result": {"success": False, "error": "No product type"},
            "vendor_matches": [],
            "current_node": "vendor_analysis",
            "phases_completed": _mark_phase_complete(state, "vendor_analysis")
        }

    # Merge advanced parameters into provided requirements
    structured_requirements = dict(provided_requirements)
    if advanced_parameters:
        adv_specs = {}
        for param in advanced_parameters:
            key = param.get("key", "")
            name = param.get("name", key)
            if key:
                adv_specs[key] = name
        if adv_specs:
            structured_requirements["advancedSpecs"] = adv_specs

    try:
        from .vendor_analysis_deep_agent import VendorAnalysisDeepAgent

        vendor_agent = VendorAnalysisDeepAgent()
        vendor_result = vendor_agent.analyze(
            structured_requirements=structured_requirements,
            product_type=product_type,
            session_id=session_id,
            schema=schema
        )

        vendor_matches = vendor_result.get("vendor_matches", [])
        total_matches = vendor_result.get("total_matches", 0)

        logger.info(f"[SearchOrchestrator] Vendor analysis result: success={vendor_result.get('success')}, "
                    f"total_matches={total_matches}")

        return {
            "vendor_analysis_result": vendor_result,
            "vendor_matches": vendor_matches,
            "current_node": "vendor_analysis",
            "phases_completed": _mark_phase_complete(state, "vendor_analysis"),
            "tools_called": state.get("tools_called", []) + ["VendorAnalysisDeepAgent"],
            "tool_results_summary": {
                **state.get("tool_results_summary", {}),
                "vendor_analysis": {
                    "success": vendor_result.get("success"),
                    "total_matches": total_matches,
                    "vendors_analyzed": vendor_result.get("vendors_analyzed", 0)
                }
            }
        }

    except Exception as e:
        logger.error(f"[SearchOrchestrator] Vendor analysis failed: {e}", exc_info=True)
        return {
            "vendor_analysis_result": {"success": False, "error": str(e)},
            "vendor_matches": [],
            "current_node": "vendor_analysis",
            "phases_completed": _mark_phase_complete(state, "vendor_analysis")
        }


def ranking_node(state: SearchOrchestrationState) -> dict:
    """
    Node 6: Wrapper for RankingTool.rank().

    Normalizes vendor_matches to list format before calling.
    """
    logger.info("=" * 70)
    logger.info("[SearchOrchestrator] Node 6: ranking")
    logger.info("=" * 70)

    session_id = state.get("session_id", "")
    vendor_matches = state.get("vendor_matches", [])
    provided_requirements = state.get("provided_requirements", {})
    vendor_analysis_result = state.get("vendor_analysis_result", {})

    if not vendor_matches:
        logger.warning("[SearchOrchestrator] No vendor matches - skipping ranking")
        return {
            "ranking_result": {"success": False, "error": "No vendor matches"},
            "ranked_products": [],
            "top_product": None,
            "current_node": "ranking",
            "phases_completed": _mark_phase_complete(state, "ranking")
        }

    try:
        from .ranking_tool import RankingTool

        ranking_tool = RankingTool(use_llm_ranking=True)
        ranking_result = ranking_tool.rank(
            vendor_analysis={
                "vendor_matches": vendor_matches,
                "strategy_context": vendor_analysis_result.get("strategy_context", {}),
                "rag_invocations": vendor_analysis_result.get("rag_invocations", {})
            },
            session_id=session_id,
            structured_requirements=provided_requirements
        )

        ranked_products = ranking_result.get("overall_ranking", [])
        top_product = ranking_result.get("top_product")

        logger.info(f"[SearchOrchestrator] Ranking result: success={ranking_result.get('success')}, "
                    f"ranked_count={len(ranked_products)}")

        return {
            "ranking_result": ranking_result,
            "ranked_products": ranked_products,
            "top_product": top_product,
            "current_node": "ranking",
            "phases_completed": _mark_phase_complete(state, "ranking"),
            "tools_called": state.get("tools_called", []) + ["RankingTool"],
            "tool_results_summary": {
                **state.get("tool_results_summary", {}),
                "ranking": {
                    "success": ranking_result.get("success"),
                    "ranked_count": len(ranked_products)
                }
            }
        }

    except Exception as e:
        logger.error(f"[SearchOrchestrator] Ranking failed: {e}", exc_info=True)
        # Fall back to using vendor_matches as ranked_products
        return {
            "ranking_result": {"success": False, "error": str(e)},
            "ranked_products": vendor_matches,
            "top_product": vendor_matches[0] if vendor_matches else None,
            "current_node": "ranking",
            "phases_completed": _mark_phase_complete(state, "ranking")
        }


def compose_response_node(state: SearchOrchestrationState) -> dict:
    """
    Node 7: Final node - compose response using SalesAgentTool.
    """
    logger.info("=" * 70)
    logger.info("[SearchOrchestrator] Node 7: compose_response")
    logger.info("=" * 70)

    session_id = state.get("session_id", "")
    user_input = state.get("user_input", "")
    product_type = state.get("product_type", "")
    ranked_products = state.get("ranked_products", [])
    vendor_matches = state.get("vendor_matches", [])
    schema = state.get("schema", {})
    provided_requirements = state.get("provided_requirements", {})
    missing_fields = state.get("missing_fields", [])
    vendor_analysis_result = state.get("vendor_analysis_result", {})

    processing_time_ms = int((time.time() - state.get("start_time", time.time())) * 1000)

    try:
        from .sales_agent_tool import SalesAgentTool

        sales_agent = SalesAgentTool()
        response_result = sales_agent.process_step(
            step="finalAnalysis",
            user_message=user_input,
            data_context={
                "productType": product_type,
                "rankedProducts": ranked_products
            },
            session_id=session_id
        )

        response_message = response_result.get("content", "Search completed successfully.")

    except Exception as e:
        logger.warning(f"[SearchOrchestrator] SalesAgent failed: {e}")
        if ranked_products:
            top = ranked_products[0]
            response_message = (
                f"Based on your requirements for **{product_type}**, "
                f"I recommend the **{top.get('productName', 'product')}** "
                f"from **{top.get('vendor', 'the vendor')}** "
                f"with a match score of {top.get('matchScore', 0)}%."
            )
        else:
            response_message = f"Search completed for {product_type}. No matching products found."

    # Build final response_data
    response_data = {
        "success": True,
        "product_type": product_type,
        "schema": schema,
        "provided_requirements": provided_requirements,
        "missing_fields": missing_fields,
        "ranked_products": ranked_products,
        "vendor_matches": vendor_analysis_result.get("vendor_matches", vendor_matches),
        "vendor_analysis": vendor_analysis_result,
        "top_recommendation": ranked_products[0] if ranked_products else None,
        "total_matches": len(ranked_products),
        "processing_time_ms": processing_time_ms,
        "awaiting_user_input": False,
        "current_phase": "complete"
    }

    logger.info(f"[SearchOrchestrator] Response composed: {len(ranked_products)} products, {processing_time_ms}ms")

    return {
        "response": response_message,
        "response_data": response_data,
        "processing_time_ms": processing_time_ms,
        "current_node": "compose_response",
        "phases_completed": _mark_phase_complete(state, "compose_response")
    }


def no_matches_response_node(state: SearchOrchestrationState) -> dict:
    """Handle case when no vendor matches found."""
    logger.info("[SearchOrchestrator] No matches response node")

    product_type = state.get("product_type", "product")
    processing_time_ms = int((time.time() - state.get("start_time", time.time())) * 1000)

    response_message = (
        f"I searched for **{product_type}** matching your requirements, "
        f"but no matching products were found in our catalog.\n\n"
        f"Suggestions:\n"
        f"- Try broadening your requirements\n"
        f"- Check if the product type is correct\n"
        f"- Contact us for custom sourcing"
    )

    return {
        "response": response_message,
        "response_data": {
            "success": True,
            "product_type": product_type,
            "ranked_products": [],
            "vendor_matches": [],
            "total_matches": 0,
            "processing_time_ms": processing_time_ms,
            "awaiting_user_input": False,
            "current_phase": "complete",
            "no_matches": True
        },
        "processing_time_ms": processing_time_ms,
        "current_node": "no_matches_response",
        "phases_completed": _mark_phase_complete(state, "no_matches_response")
    }


def missing_fields_response_node(state: SearchOrchestrationState) -> dict:
    """Handle case when validation finds missing required fields."""
    logger.info("[SearchOrchestrator] Missing fields response node")

    product_type = state.get("product_type", "product")
    missing_fields = state.get("missing_fields", [])
    provided_requirements = state.get("provided_requirements", {})
    processing_time_ms = int((time.time() - state.get("start_time", time.time())) * 1000)

    missing_display = ", ".join(missing_fields[:5])
    if len(missing_fields) > 5:
        missing_display += f", and {len(missing_fields) - 5} more"

    response_message = (
        f"To search for **{product_type}**, I need some additional information.\n\n"
        f"**Please provide:** {missing_display}\n\n"
        f"You can specify these details in your next message."
    )

    return {
        "response": response_message,
        "response_data": {
            "success": True,
            "product_type": product_type,
            "provided_requirements": provided_requirements,
            "missing_fields": missing_fields,
            "processing_time_ms": processing_time_ms,
            "awaiting_user_input": True,
            "current_phase": "collect_requirements"
        },
        "processing_time_ms": processing_time_ms,
        "current_node": "missing_fields_response",
        "phases_completed": _mark_phase_complete(state, "missing_fields_response")
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_validation(state: SearchOrchestrationState) -> str:
    """
    Route based on validation result (Node 1 routing).

    [FIX Feb 2026] advanced_specs is MANDATORY after validation.
    HITL prompt is shown AFTER advanced_specs, not before.

    Returns:
    - "advanced_specs": Always proceed to advanced specs (mandatory)
    - "missing_fields_response": Missing required fields, return for user input
    - "compose_response": Error case
    """
    # Check for HITL YES - route to advanced_specs (specs may not have run yet)
    if state.get("is_hitl_yes"):
        logger.info("[SearchOrchestrator] Routing: validation -> advanced_specs (HITL YES)")
        return "advanced_specs"

    # Check for HITL NO - route to advanced_specs (mandatory, not skippable)
    if state.get("is_hitl_no"):
        logger.info("[SearchOrchestrator] Routing: validation -> advanced_specs (HITL NO, mandatory)")
        return "advanced_specs"

    # Check for errors
    if state.get("error"):
        logger.info("[SearchOrchestrator] Routing: validation -> compose_response (error)")
        return "compose_response"

    # Check if validation was bypassed and advanced specs already triggered
    validation_result = state.get("validation_result", {})
    if validation_result.get("validation_bypassed") and validation_result.get("advanced_specs_info"):
        logger.info("[SearchOrchestrator] Routing: validation -> compose_response (bypassed with adv specs)")
        return "compose_response"

    # Check for missing fields (only if many are missing and is_valid=False)
    missing_fields = state.get("missing_fields", [])
    is_valid = state.get("is_valid", False)

    # Only block if there are critical missing fields and validation explicitly failed
    if not is_valid and len(missing_fields) > 3:
        logger.info(f"[SearchOrchestrator] Routing: validation -> missing_fields_response ({len(missing_fields)} missing)")
        return "missing_fields_response"

    # [FIX Feb 2026] Always go to advanced_specs after validation (mandatory)
    logger.info("[SearchOrchestrator] Routing: validation -> advanced_specs (mandatory)")
    return "advanced_specs"


def route_after_hitl_decision(state: SearchOrchestrationState) -> str:
    """
    Route based on HITL decision.

    [FIX Feb 2026] HITL decision now comes AFTER advanced_specs.
    So we route to vendor_analysis (not advanced_specs).

    Returns:
    - "end": User needs to respond (HITL pause)
    - "vendor_analysis": Auto mode or HITL complete, proceed
    """
    if state.get("awaiting_user_input"):
        logger.info("[SearchOrchestrator] Routing: hitl_decision -> END (awaiting input)")
        return "end"

    logger.info("[SearchOrchestrator] Routing: hitl_decision -> vendor_analysis")
    return "vendor_analysis"


def route_after_vendor_analysis(state: SearchOrchestrationState) -> str:
    """
    Route based on vendor analysis result.

    Returns:
    - "ranking": Found matches, proceed to ranking
    - "no_matches_response": No matches found, compose empty response
    """
    vendor_matches = state.get("vendor_matches", [])

    if not vendor_matches:
        logger.info("[SearchOrchestrator] Routing: vendor_analysis -> no_matches_response")
        return "no_matches_response"

    logger.info(f"[SearchOrchestrator] Routing: vendor_analysis -> ranking ({len(vendor_matches)} matches)")
    return "ranking"


# =============================================================================
# WORKFLOW CONSTRUCTION
# =============================================================================

def create_search_orchestration_workflow() -> StateGraph:
    """
    Create the Search Orchestration Workflow.

    Architecture (FIX Feb 2026 — advanced_specs is mandatory BEFORE HITL):
                        +-------------------+
                        |    validation     |  <- Node 1 (Entry Point)
                        |  (8 internal      |     ValidationDeepAgent
                        |   LangGraph nodes)|
                        +---------+---------+
               +------------------+------------------+
               |(HITL YES/NO)     |(normal)          |(error)
               v                  v                  v
        +---------------+   +---------------+   +----------------+
        | advanced_specs |   | advanced_specs|   | compose_resp   |
        | (Node 2)       |   | (Node 2)      |   +----------------+
        +-------+-------+   +-------+-------+
                |                    |
                v                    v
        +---------------+   +---------------+
        |vendor_analysis|   | hitl_decision |
        | (Node 3)      |   +-------+-------+
        +-------+-------+          |
                |           (wait) | (proceed)
                |                  v
                |           +----------------+
                +---------->| vendor_analysis|
                            +--------+-------+
                                     |
                            +--------+--------+
                            |(matches)        |(no matches)
                            v                 v
                        +--------+    +------------------+
                        | ranking|    | no_matches_resp  |
                        +----+---+    +------------------+
                             |
                             v
                        +----------------+
                        | compose_resp   |
                        +----------------+
    """
    workflow = StateGraph(SearchOrchestrationState)

    # Add nodes
    # Node 1: ValidationDeepAgent (entry point)
    workflow.add_node("validation", validation_node)
    # Node 2: AdvancedSpecificationAgent
    workflow.add_node("advanced_specs", advanced_specs_node)
    # Node 3: VendorAnalysisDeepAgent
    workflow.add_node("vendor_analysis", vendor_analysis_node)
    # Supporting nodes
    workflow.add_node("hitl_decision", hitl_decision_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("compose_response", compose_response_node)
    workflow.add_node("no_matches_response", no_matches_response_node)
    workflow.add_node("missing_fields_response", missing_fields_response_node)

    # Entry point: validation (Node 1)
    workflow.set_entry_point("validation")

    # Conditional routing from validation (Node 1)
    # [FIX Feb 2026] All paths go to advanced_specs (mandatory)
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "advanced_specs": "advanced_specs",
            "missing_fields_response": "missing_fields_response",
            "compose_response": "compose_response"
        }
    )

    # [FIX Feb 2026] advanced_specs -> hitl_decision (HITL comes AFTER specs)
    workflow.add_edge("advanced_specs", "hitl_decision")

    # Conditional routing from HITL decision
    # [FIX Feb 2026] Routes to vendor_analysis (not advanced_specs)
    workflow.add_conditional_edges(
        "hitl_decision",
        route_after_hitl_decision,
        {
            "end": END,
            "vendor_analysis": "vendor_analysis"
        }
    )

    # Conditional routing from vendor analysis
    workflow.add_conditional_edges(
        "vendor_analysis",
        route_after_vendor_analysis,
        {
            "ranking": "ranking",
            "no_matches_response": "no_matches_response"
        }
    )

    # Linear edges to END
    workflow.add_edge("ranking", "compose_response")
    workflow.add_edge("compose_response", END)
    workflow.add_edge("no_matches_response", END)
    workflow.add_edge("missing_fields_response", END)

    return workflow


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class SearchOrchestrator:
    """
    Search Orchestrator - Unified LangGraph workflow for product search.

    Provides backward-compatible API while using the new orchestrated workflow.

    Usage:
        orchestrator = SearchOrchestrator()
        result = orchestrator.run(
            user_input="I need a pressure transmitter with 4-20mA output",
            session_id="session_123"
        )
    """

    def __init__(
        self,
        enable_ppi: bool = True,
        auto_mode: bool = True
    ):
        """
        Initialize the search orchestrator.

        Args:
            enable_ppi: Enable PPI workflow for schema generation
            auto_mode: Skip HITL prompts and proceed automatically
        """
        self.enable_ppi = enable_ppi
        self.auto_mode = auto_mode
        self._workflow = None

        logger.info(f"[SearchOrchestrator] Initialized: enable_ppi={enable_ppi}, auto_mode={auto_mode}")

    @property
    def workflow(self):
        """Lazy initialization of compiled workflow."""
        if self._workflow is None:
            raw_workflow = create_search_orchestration_workflow()
            self._workflow = raw_workflow.compile()
            logger.info("[SearchOrchestrator] Workflow compiled")
        return self._workflow

    @with_workflow_lock(session_id_param="session_id", timeout=120.0)
    def run(
        self,
        user_input: str,
        session_id: str = "",
        expected_product_type: Optional[str] = None,
        user_provided_fields: Optional[Dict[str, Any]] = None,
        auto_mode: Optional[bool] = None,
        user_decision: Optional[str] = None,
        current_phase: Optional[str] = None,
        source_workflow: Optional[str] = None,
        standards_enriched: bool = False,
        ctx: Optional[ExecutionContext] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the search orchestration workflow.

        Backward compatible with run_product_search_workflow() signature.

        Args:
            user_input: User's search query or requirements
            session_id: Session identifier for isolation
            expected_product_type: Hint for expected product type
            user_provided_fields: Pre-extracted user requirements
            auto_mode: Override auto_mode setting (None uses instance setting)
            user_decision: User's YES/NO response to HITL prompt
            current_phase: Frontend-provided phase hint
            source_workflow: Source workflow identifier
            ctx: Execution context for thread isolation

        Returns:
            {
                "success": bool,
                "response": str,
                "response_data": {
                    "product_type": str,
                    "schema": dict,
                    "ranked_products": list,
                    "vendor_matches": dict,
                    "missing_fields": list,
                    "awaiting_user_input": bool
                },
                "error": str (if failed)
            }
        """
        start_time = time.time()

        # Generate session_id if not provided
        if not session_id:
            session_id = f"search_{int(start_time * 1000)}"

        # Determine auto_mode
        effective_auto_mode = auto_mode if auto_mode is not None else self.auto_mode

        logger.info("=" * 70)
        logger.info("[SearchOrchestrator] Starting search workflow")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   Auto Mode: {effective_auto_mode}")
        logger.info(f"   Input: {user_input[:100]}..." if len(user_input) > 100 else f"   Input: {user_input}")
        logger.info("=" * 70)

        # Set execution context if provided
        if ctx:
            set_context(ctx)
        else:
            effective_ctx = ExecutionContext(
                session_id=session_id,
                workflow_type="product_search",
                workflow_id=f"search_{session_id}"
            )
            set_context(effective_ctx)

        try:
            # Create initial state
            initial_state = create_search_orchestration_state(
                user_input=user_input,
                session_id=session_id,
                expected_product_type=expected_product_type,
                user_provided_fields=user_provided_fields,
                enable_ppi=self.enable_ppi,
                auto_mode=effective_auto_mode,
                source_workflow=source_workflow,
                user_decision=user_decision,
                current_phase=current_phase,
                standards_enriched=standards_enriched,
            )

            # Run workflow
            final_state = self.workflow.invoke(initial_state)

            # Extract results
            response = final_state.get("response", "")
            response_data = final_state.get("response_data", {})
            error = final_state.get("error")
            processing_time_ms = final_state.get("processing_time_ms", 0)

            if error:
                logger.error(f"[SearchOrchestrator] Workflow error: {error}")
                return {
                    "success": False,
                    "response": f"Search failed: {error}",
                    "response_data": response_data,
                    "error": error,
                    "processing_time_ms": processing_time_ms
                }

            logger.info(f"[SearchOrchestrator] Workflow completed: {processing_time_ms}ms")

            return {
                "success": True,
                "response": response,
                "response_data": response_data,
                "processing_time_ms": processing_time_ms
            }

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[SearchOrchestrator] Workflow failed: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Search workflow error: {str(e)}",
                "response_data": {},
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time_ms
            }

        finally:
            # Clear context
            clear_request_context()
            clear_context()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_search_workflow(
    user_input: str,
    session_id: str = "",
    expected_product_type: Optional[str] = None,
    user_provided_fields: Optional[Dict[str, Any]] = None,
    enable_ppi: bool = True,
    auto_mode: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run search workflow.

    Creates a SearchOrchestrator instance and runs the workflow.

    Args:
        user_input: User's search query
        session_id: Session identifier
        expected_product_type: Expected product type hint
        user_provided_fields: Pre-extracted requirements
        enable_ppi: Enable PPI workflow
        auto_mode: Skip HITL prompts

    Returns:
        Workflow result dict
    """
    orchestrator = SearchOrchestrator(
        enable_ppi=enable_ppi,
        auto_mode=auto_mode
    )
    return orchestrator.run(
        user_input=user_input,
        session_id=session_id,
        expected_product_type=expected_product_type,
        user_provided_fields=user_provided_fields,
        **kwargs
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SearchOrchestrationState",
    "SearchOrchestrator",
    "create_search_orchestration_workflow",
    "create_search_orchestration_state",
    "run_search_workflow",
]
