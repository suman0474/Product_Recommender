"""
Search Workflow - Simplified Single Orchestrator
=================================================

Single LangGraph StateGraph with 6 main nodes using simplified functions.
Only VendorAnalysisDeepAgent remains as a deep agent (justified complexity).

Architecture:
    Node 1: hitl_detection_node      - Simple function call
    Node 2: validation_node          - 4 function calls (extract -> load -> enrich -> validate)
    Node 3: advanced_specs_node      - 1 function call (cache -> LLM -> persist)
    Node 4: vendor_analysis_node     - VendorAnalysisDeepAgent.analyze() [DEEP AGENT]
    Node 5: ranking_node             - RankingTool.rank()
    Node 6: compose_response_node    - SalesAgentTool.process_step()
"""

import logging
import time
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache
from common.infrastructure.state.context.lock_monitor import with_workflow_lock

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION (Simplified - 25 fields instead of 40+65)
# =============================================================================

class SearchWorkflowState(TypedDict, total=False):
    """Simplified state for search workflow."""

    # =========================================================================
    # INPUTS (9)
    # =========================================================================
    session_id: str
    user_input: str
    expected_product_type: Optional[str]
    user_provided_fields: Optional[Dict[str, Any]]
    enable_ppi: bool
    auto_mode: bool
    source_workflow: Optional[str]
    user_decision: Optional[str]
    current_phase: Optional[str]
    standards_enriched: bool  # [FIX Feb 2026] Skip standards enrichment if pre-enriched by solution workflow

    # =========================================================================
    # HITL STATE (9)
    # =========================================================================
    is_hitl_yes: bool
    is_hitl_no: bool
    is_skip: bool  # [FIX Feb 2026] True when user responds with SKIP (treated as NO)
    hitl_prompt_shown: bool
    advanced_specs_hitl_shown: bool  # [FIX Feb 2026] True after advanced specs HITL prompt shown
    awaiting_user_input: bool
    awaiting_missing_fields_response: bool  # [FIX Feb 2026] True when waiting for user to provide missing fields
    awaiting_additional_specs_response: bool  # [FIX Feb 2026] True when asking if user wants to add more specs
    all_fields_provided: bool  # [FIX Feb 2026] True when all required fields are provided

    # =========================================================================
    # VALIDATION OUTPUT (5)
    # =========================================================================
    product_type: str
    schema: Dict[str, Any]
    provided_requirements: Dict[str, Any]
    missing_fields: List[str]
    is_valid: bool

    # =========================================================================
    # ADVANCED SPECS OUTPUT (1)
    # =========================================================================
    advanced_parameters: List[Dict[str, Any]]

    # =========================================================================
    # VENDOR ANALYSIS OUTPUT (2) - from VendorAnalysisDeepAgent
    # =========================================================================
    vendor_analysis_result: Dict[str, Any]
    vendor_matches: List[Dict[str, Any]]

    # =========================================================================
    # RANKING OUTPUT (2)
    # =========================================================================
    ranked_products: List[Dict[str, Any]]
    top_product: Optional[Dict[str, Any]]

    # =========================================================================
    # OUTPUT (5)
    # =========================================================================
    response: str
    response_data: Dict[str, Any]
    hitl_message: Optional[str]
    summary: Optional[str]  # [FIX Feb 2026] Summary of specifications before vendor analysis
    error: Optional[str]

    # =========================================================================
    # TIMING (2)
    # =========================================================================
    start_time: float
    processing_time_ms: int


# =============================================================================
# SESSION CACHE
# =============================================================================

_SESSION_WORKFLOW_CACHE: BoundedCache = get_or_create_cache(
    name="search_workflow_sessions",
    max_size=500,
    ttl_seconds=0  # [FIX Feb 2026] No TTL — persist for session lifetime
)


def _cache_workflow_state(session_id: str, state: Dict[str, Any]) -> None:
    """Cache workflow state for HITL resume."""
    if session_id:
        key = f"workflow:{session_id}"
        _SESSION_WORKFLOW_CACHE.set(key, state)


def _get_workflow_state(session_id: str) -> Optional[Dict[str, Any]]:
    """Get cached workflow state."""
    if not session_id:
        return None
    key = f"workflow:{session_id}"
    return _SESSION_WORKFLOW_CACHE.get(key)


# =============================================================================
# NODE 1: HITL DETECTION (Simple function call)
# =============================================================================

def hitl_detection_node(state: SearchWorkflowState) -> dict:
    """
    Node 1: Detect YES/NO HITL responses BEFORE validation.
    Uses detect_hitl_response() function.

    Two-phase HITL:
    - Phase 1 (validation HITL): User says YES/NO after seeing extracted specs
    - Phase 2 (advanced_specs HITL): User says YES/NO after seeing discovered advanced specs

    We distinguish phases using the `advanced_specs_hitl_shown` flag in cached state.
    """
    from .validation_functions import detect_hitl_response

    user_input = state.get("user_input", "")
    session_id = state.get("session_id", "")
    expected_product_type = state.get("expected_product_type")

    logger.info("[SearchWorkflow] Node 1: hitl_detection")

    result = detect_hitl_response(
        user_input=user_input,
        session_id=session_id,
        expected_product_type=expected_product_type
    )

    # Also check user_decision parameter (from frontend)
    user_decision = (state.get("user_decision") or "").lower().strip()
    if user_decision == "yes":
        result["is_hitl_yes"] = True
        result["validation_bypassed"] = True
    elif user_decision == "no":
        result["is_hitl_no"] = True

    # [FIX Feb 2026] Check cached workflow state to determine HITL phase
    cached_wf_state = _get_workflow_state(session_id) or {}
    advanced_specs_hitl_shown = cached_wf_state.get("advanced_specs_hitl_shown", False)

    return {
        "is_hitl_yes": result.get("is_hitl_yes", False),
        "is_hitl_no": result.get("is_hitl_no", False),
        "product_type": result.get("product_type", state.get("product_type", "")),
        "schema": result.get("cached_context", {}).get("schema", state.get("schema", {})),
        "provided_requirements": result.get("cached_context", {}).get("provided_requirements", {}),
        "advanced_parameters": cached_wf_state.get("advanced_parameters", []),
        "advanced_specs_hitl_shown": advanced_specs_hitl_shown,
        "hitl_prompt_shown": cached_wf_state.get("hitl_prompt_shown", False)
    }


# =============================================================================
# NODE 2: VALIDATION (4 sequential function calls)
# =============================================================================

def validation_node(state: SearchWorkflowState) -> dict:
    """
    Node 2: Extract product type, load schema, enrich, validate.
    Calls 4 simple functions in sequence.
    """
    from .validation_functions import (
        extract_product_type,
        load_schema,
        enrich_schema_with_standards,
        validate_requirements,
        generate_hitl_message,
        cache_session_context
    )

    session_id = state.get("session_id", "")
    user_input = state.get("user_input", "")
    enable_ppi = state.get("enable_ppi", True)
    expected_product_type = state.get("expected_product_type")
    source_workflow = state.get("source_workflow")

    logger.info("[SearchWorkflow] Node 2: validation")

    # Step 1: Extract product type
    extract_result = extract_product_type(
        user_input=user_input,
        expected_product_type=expected_product_type,
        source_workflow=source_workflow
    )

    if not extract_result.get("success"):
        return {
            "error": extract_result.get("error", "Product type extraction failed"),
            "product_type": ""
        }

    product_type = extract_result["product_type"]
    logger.info(f"[SearchWorkflow] Extracted product type: {product_type}")

    # Step 2: Load schema
    schema_result = load_schema(product_type, enable_ppi)
    schema = schema_result.get("schema", {})

    # Step 3: Enrich schema with standards
    # [FIX Feb 2026] Skip standards enrichment if already done by Solution workflow
    standards_enriched = state.get("standards_enriched", False)
    if standards_enriched:
        logger.info(
            "[SearchWorkflow] Standards enrichment SKIPPED "
            "(pre-enriched by solution workflow)"
        )
    elif enable_ppi:
        enrich_result = enrich_schema_with_standards(
            product_type=product_type,
            schema=schema,
            session_id=session_id
        )
        schema = enrich_result.get("schema", schema)

    # Step 4: Validate requirements
    validation_result = validate_requirements(user_input, product_type, schema)
    provided_requirements = validation_result.get("provided_requirements", {})
    missing_fields = validation_result.get("missing_fields", [])
    is_valid = validation_result.get("is_valid", False)

    # Generate HITL message
    hitl_message = None
    if not state.get("auto_mode"):
        hitl_message = generate_hitl_message(
            product_type=product_type,
            provided_requirements=provided_requirements,
            missing_fields=missing_fields
        )

    # Cache session context for HITL resume
    cache_session_context(session_id, {
        "product_type": product_type,
        "schema": schema,
        "provided_requirements": provided_requirements,
        "missing_fields": missing_fields,
        "is_valid": is_valid
    })

    return {
        "product_type": product_type,
        "schema": schema,
        "provided_requirements": provided_requirements,
        "missing_fields": missing_fields,
        "is_valid": is_valid,
        "hitl_message": hitl_message
    }


# =============================================================================
# NODE 3: ADVANCED SPECS (1 function call)
# =============================================================================

def advanced_specs_node(state: SearchWorkflowState) -> dict:
    """
    Node 3: Discover advanced specifications.
    Calls discover_advanced_specs() function.
    """
    from .advanced_specs_functions import discover_advanced_specs

    product_type = state.get("product_type", "")
    session_id = state.get("session_id", "")
    schema = state.get("schema", {})

    logger.info(f"[SearchWorkflow] Node 3: advanced_specs for {product_type}")

    result = discover_advanced_specs(
        product_type=product_type,
        session_id=session_id,
        existing_schema=schema
    )

    advanced_params = result.get("unique_specifications", [])
    logger.info(f"[SearchWorkflow] Discovered {len(advanced_params)} advanced specs")

    return {
        "advanced_parameters": advanced_params
    }


# =============================================================================
# NODE 4: VENDOR ANALYSIS (DEEP AGENT - black box call)
# =============================================================================

def vendor_analysis_node(state: SearchWorkflowState) -> dict:
    """
    Node 4: Run VendorAnalysisDeepAgent.analyze() as black box.
    This is the ONLY deep agent call in the workflow.
    """
    from .vendor_analysis_deep_agent import VendorAnalysisDeepAgent

    product_type = state.get("product_type", "")
    session_id = state.get("session_id", "")
    schema = state.get("schema", {})
    provided_requirements = state.get("provided_requirements", {})
    advanced_parameters = state.get("advanced_parameters", [])

    logger.info(f"[SearchWorkflow] Node 4: vendor_analysis for {product_type}")

    # Merge advanced params into requirements
    requirements = dict(provided_requirements)
    if advanced_parameters:
        requirements["advancedSpecs"] = {
            p["key"]: p["name"] for p in advanced_parameters if p.get("key")
        }

    try:
        agent = VendorAnalysisDeepAgent()
        result = agent.analyze(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
            schema=schema
        )

        vendor_matches = result.get("vendor_matches", [])
        logger.info(f"[SearchWorkflow] Found {len(vendor_matches)} vendor matches")

        return {
            "vendor_analysis_result": result,
            "vendor_matches": vendor_matches
        }

    except Exception as e:
        logger.error(f"[SearchWorkflow] Vendor analysis failed: {e}")
        return {
            "vendor_analysis_result": {"error": str(e)},
            "vendor_matches": [],
            "error": str(e)
        }


# =============================================================================
# NODE 5: RANKING (tool call)
# =============================================================================

def ranking_node(state: SearchWorkflowState) -> dict:
    """
    Node 5: Rank products using RankingTool.rank().
    """
    from .ranking_tool import RankingTool

    vendor_matches = state.get("vendor_matches", [])
    session_id = state.get("session_id", "")
    provided_requirements = state.get("provided_requirements", {})

    logger.info(f"[SearchWorkflow] Node 5: ranking ({len(vendor_matches)} products)")

    if not vendor_matches:
        return {
            "ranked_products": [],
            "top_product": None
        }

    try:
        tool = RankingTool(use_llm_ranking=True)
        result = tool.rank(
            vendor_analysis={"vendor_matches": vendor_matches},
            session_id=session_id,
            structured_requirements=provided_requirements
        )

        ranked_products = result.get("overall_ranking", [])
        top_product = result.get("top_product")

        logger.info(f"[SearchWorkflow] Ranked {len(ranked_products)} products")

        return {
            "ranked_products": ranked_products,
            "top_product": top_product
        }

    except Exception as e:
        logger.error(f"[SearchWorkflow] Ranking failed: {e}")
        return {
            "ranked_products": vendor_matches,  # Fallback to unranked
            "top_product": vendor_matches[0] if vendor_matches else None
        }


# =============================================================================
# NODE 6: COMPOSE RESPONSE
# =============================================================================

def compose_response_node(state: SearchWorkflowState) -> dict:
    """
    Node 6: Build final response using SalesAgentTool.
    """
    from .sales_agent_tool import SalesAgentTool

    product_type = state.get("product_type", "")
    user_input = state.get("user_input", "")
    session_id = state.get("session_id", "")
    ranked_products = state.get("ranked_products", [])
    top_product = state.get("top_product")
    start_time = state.get("start_time", time.time())

    logger.info("[SearchWorkflow] Node 6: compose_response")

    processing_time_ms = int((time.time() - start_time) * 1000)

    try:
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
        response = response_result.get("content", "Search completed.")

    except Exception as e:
        logger.warning(f"[SearchWorkflow] SalesAgentTool failed: {e}")
        # Fallback response
        if top_product:
            response = f"Top recommendation: {top_product.get('productName', 'N/A')} by {top_product.get('vendor', 'N/A')}"
        elif ranked_products:
            response = f"Found {len(ranked_products)} matching products for {product_type}."
        else:
            response = f"No matching products found for {product_type}."

    return {
        "response": response,
        "response_data": {
            "success": True,
            "product_type": product_type,
            "ranked_products": ranked_products,
            "top_product": top_product,
            "vendor_matches": state.get("vendor_matches", []),
            "processing_time_ms": processing_time_ms
        },
        "processing_time_ms": processing_time_ms
    }


# =============================================================================
# RESPONSE NODES
# =============================================================================

def hitl_response_node(state: SearchWorkflowState) -> dict:
    """Return HITL prompt to user and pause — AFTER advanced_specs has run."""
    session_id = state.get("session_id", "")
    product_type = state.get("product_type", "")
    advanced_parameters = state.get("advanced_parameters", [])

    logger.info(f"[SearchWorkflow] HITL pause - showing {len(advanced_parameters)} advanced specs to user")

    # Build HITL message that includes the discovered advanced specs
    num_provided = len(state.get("provided_requirements", {}))
    if advanced_parameters:
        bullets = []
        for p in advanced_parameters[:15]:
            name = str(p.get("name") or p.get("key", "")).replace("_", " ").title()
            bullets.append(f"- {name}")
        spec_list = "\n".join(bullets)
        hitl_message = (
            f"I've extracted {num_provided} specification(s) for **{product_type}** "
            f"and discovered {len(advanced_parameters)} advanced specifications:\n\n"
            f"{spec_list}\n\n"
            f"Would you like to add any of these advanced specifications to your requirements?\n\n"
            f"Reply **'YES'** to add advanced specifications, or **'NO'** to continue with the search."
        )
    else:
        hitl_message = state.get("hitl_message", "")
        if not hitl_message:
            hitl_message = (
                f"I've extracted {num_provided} specification(s) for **{product_type}**.\n\n"
                f"Would you like to add more advanced specifications for better matching, "
                f"or shall I proceed with the search?\n\n"
                f"Reply **'YES'** to add advanced specifications, or **'NO'** to continue with the search."
            )

    # Cache state for resume — set advanced_specs_hitl_shown flag
    _cache_workflow_state(session_id, {
        "product_type": product_type,
        "schema": state.get("schema", {}),
        "provided_requirements": state.get("provided_requirements", {}),
        "advanced_parameters": advanced_parameters,
        "hitl_prompt_shown": True,
        "advanced_specs_hitl_shown": True  # [FIX Feb 2026] Mark that advanced specs HITL has been shown
    })

    return {
        "response": hitl_message,
        "response_data": {
            "awaiting_user_input": True,
            "current_phase": "advanced_specs_discovered",
            "product_type": product_type,
            "schema": state.get("schema", {}),
            "provided_requirements": state.get("provided_requirements", {}),
            "advanced_parameters": advanced_parameters,
            "missing_fields": state.get("missing_fields", [])
        },
        "awaiting_user_input": True,
        "hitl_prompt_shown": True,
        "advanced_specs_hitl_shown": True
    }


def no_matches_node(state: SearchWorkflowState) -> dict:
    """Handle no vendor matches case."""
    product_type = state.get("product_type", "")
    start_time = state.get("start_time", time.time())
    processing_time_ms = int((time.time() - start_time) * 1000)

    logger.info("[SearchWorkflow] No matches found")

    return {
        "response": f"No matching products found for {product_type}. Please try refining your requirements.",
        "response_data": {
            "success": True,
            "product_type": product_type,
            "ranked_products": [],
            "no_matches": True,
            "processing_time_ms": processing_time_ms
        },
        "processing_time_ms": processing_time_ms
    }


def error_response_node(state: SearchWorkflowState) -> dict:
    """Handle error case."""
    error = state.get("error", "Unknown error")
    start_time = state.get("start_time", time.time())
    processing_time_ms = int((time.time() - start_time) * 1000)

    logger.error(f"[SearchWorkflow] Error: {error}")

    return {
        "response": f"Search failed: {error}",
        "response_data": {
            "success": False,
            "error": error,
            "processing_time_ms": processing_time_ms
        },
        "processing_time_ms": processing_time_ms
    }


def missing_fields_response_node(state: SearchWorkflowState) -> dict:
    """
    Show missing required fields and end workflow.
    User must provide missing fields before continuing.
    """
    session_id = state.get("session_id", "")
    product_type = state.get("product_type", "product")
    missing_fields = state.get("missing_fields", [])

    logger.info(f"[SearchWorkflow] Missing fields response - {len(missing_fields)} missing fields")

    if missing_fields:
        field_list = "\n".join([f"- {f}" for f in missing_fields])
        response = (
            f"I've validated your request for **{product_type}**, "
            f"but the following required specifications are missing:\n\n"
            f"{field_list}\n\n"
            f"Please provide the missing specifications to proceed with the search."
        )
    else:
        # Should not reach here, but handle gracefully
        response = (
            f"Validation complete for **{product_type}**. "
            f"Please provide any additional specifications if needed."
        )

    # Cache state and end workflow
    _cache_workflow_state(session_id, {
        "product_type": product_type,
        "schema": state.get("schema", {}),
        "provided_requirements": state.get("provided_requirements", {}),
        "missing_fields": missing_fields,
        "awaiting_missing_fields_response": True,
        "hitl_prompt_shown": True
    })

    return {
        "response": response,
        "response_data": {
            "awaiting_user_input": True,
            "current_phase": "missing_fields",
            "product_type": product_type,
            "schema": state.get("schema", {}),
            "provided_requirements": state.get("provided_requirements", {}),
            "missing_fields": missing_fields
        },
        "awaiting_user_input": True,
        "awaiting_missing_fields_response": True,
        "hitl_prompt_shown": True
    }


def ask_additional_specs_node(state: SearchWorkflowState) -> dict:
    """
    Ask user if they want to add additional specifications (BEFORE discovery).
    This is shown when all required fields are provided.
    """
    session_id = state.get("session_id", "")
    product_type = state.get("product_type", "product")
    provided_requirements = state.get("provided_requirements", {})
    provided_count = len(provided_requirements.get("specifications", []))

    logger.info(f"[SearchWorkflow] Asking if user wants to add additional specs - {provided_count} specs provided")

    response = (
        f"I've extracted {provided_count} specification(s) for **{product_type}**.\n\n"
        f"Would you like to add additional specifications for better product matching?\n\n"
        f"Reply **'YES'** to discover and add more specifications, or **'SKIP'** to continue with the current specifications."
    )

    # Cache state for resume
    _cache_workflow_state(session_id, {
        "product_type": product_type,
        "schema": state.get("schema", {}),
        "provided_requirements": provided_requirements,
        "awaiting_additional_specs_response": True,
        "all_fields_provided": True,
        "hitl_prompt_shown": True
    })

    return {
        "response": response,
        "response_data": {
            "awaiting_user_input": True,
            "current_phase": "ask_additional_specs",
            "product_type": product_type,
            "schema": state.get("schema", {}),
            "provided_requirements": provided_requirements,
            "provided_count": provided_count
        },
        "awaiting_user_input": True,
        "awaiting_additional_specs_response": True,
        "all_fields_provided": True,
        "hitl_prompt_shown": True
    }


def summary_node(state: SearchWorkflowState) -> dict:
    """
    Show complete specification summary before proceeding to vendor analysis.
    This gives the user a final view of what specs will be used for matching.
    """
    product_type = state.get("product_type", "product")
    provided_requirements = state.get("provided_requirements", {})
    advanced_parameters = state.get("advanced_parameters", [])

    logger.info(f"[SearchWorkflow] Generating summary - {len(advanced_parameters)} advanced specs")

    # Build specification lists
    provided_specs = provided_requirements.get("specifications", [])
    provided_list = "\n".join([
        f"  - {s.get('name', s.get('key', 'N/A'))}: {s.get('value', 'N/A')}"
        for s in provided_specs
    ]) if provided_specs else "  (None)"

    advanced_list = ""
    if advanced_parameters:
        advanced_list = "\n\n**Advanced Specifications Added:**\n" + "\n".join([
            f"  - {p.get('name', p.get('key', 'N/A'))}"
            for p in advanced_parameters[:15]  # Show first 15
        ])
        if len(advanced_parameters) > 15:
            advanced_list += f"\n  ... and {len(advanced_parameters) - 15} more"

    summary = (
        f"## Summary of Your Requirements\n\n"
        f"**Product Type:** {product_type}\n\n"
        f"**Provided Specifications:**\n{provided_list}"
        f"{advanced_list}\n\n"
        f"Proceeding with vendor analysis to find matching products..."
    )

    return {
        "summary": summary,
        "response": summary  # Also set as response for display
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_hitl_detection(state: SearchWorkflowState) -> str:
    """
    Route after HITL detection — multi-phase HITL support.

    New flow (Feb 2026):
    1. If awaiting_missing_fields_response:
       - User provided missing fields → validation (re-validate)

    2. If awaiting_additional_specs_response:
       - YES → advanced_specs (discover and show specs)
       - NO/SKIP → summary (skip discovery, show current specs)

    3. Legacy advanced_specs_hitl_shown (for backward compatibility):
       - YES → vendor_analysis
       - NO → vendor_analysis

    4. First call (no HITL flags):
       - Go to validation
    """
    is_yes = state.get("is_hitl_yes", False)
    is_no = state.get("is_hitl_no", False)
    is_skip = state.get("is_skip", False)

    # Check which phase we're in based on cached context
    cached_wf_state = _get_workflow_state(state.get("session_id", "")) or {}
    awaiting_missing_fields = cached_wf_state.get("awaiting_missing_fields_response", False)
    awaiting_additional_specs = cached_wf_state.get("awaiting_additional_specs_response", False)
    advanced_specs_hitl_shown = cached_wf_state.get("advanced_specs_hitl_shown", False)

    # Phase 1: Missing fields response
    if awaiting_missing_fields:
        # User provided missing fields, re-run validation
        logger.info("[SearchWorkflow] Route: HITL (missing fields) -> validation (re-validate)")
        return "validation"

    # Phase 2: Additional specs question (asked BEFORE discovery)
    if awaiting_additional_specs:
        if is_yes:
            # User wants to add specs → discover advanced specs
            logger.info("[SearchWorkflow] Route: HITL YES (additional specs) -> advanced_specs (discover)")
            return "advanced_specs"
        elif is_no or is_skip:
            # User skips additional specs → show summary with current specs
            logger.info("[SearchWorkflow] Route: HITL NO/SKIP (additional specs) -> summary (no discovery)")
            return "summary"

    # Phase 3: Legacy advanced specs HITL (for backward compatibility)
    if advanced_specs_hitl_shown:
        if is_yes or is_no:
            # Old flow: go to vendor_analysis
            logger.info("[SearchWorkflow] Route: HITL (legacy advanced specs) -> vendor_analysis")
            return "vendor_analysis"

    # Not a HITL response — first call, go to validation
    logger.info("[SearchWorkflow] Route: HITL (first call) -> validation")
    return "validation"


def route_after_validation(state: SearchWorkflowState) -> str:
    """
    Route after validation based on missing fields and auto_mode.

    New flow (Feb 2026):
    - If error → error_response
    - If auto_mode → advanced_specs (skip HITL)
    - If missing fields AND not auto_mode → missing_fields_response (pause)
    - Otherwise (all fields provided) → ask_additional_specs (pause)
    """
    if state.get("error"):
        return "error_response"

    auto_mode = state.get("auto_mode", False)
    missing_fields = state.get("missing_fields", [])

    # If auto_mode, skip HITL and go directly to advanced specs
    if auto_mode:
        logger.info("[SearchWorkflow] Route: validation -> advanced_specs (auto_mode)")
        return "advanced_specs"

    # If missing fields, show missing fields response and pause
    if missing_fields:
        logger.info(f"[SearchWorkflow] Route: validation -> missing_fields_response ({len(missing_fields)} missing)")
        return "missing_fields_response"

    # All fields provided, ask if user wants to add additional specs
    logger.info("[SearchWorkflow] Route: validation -> ask_additional_specs (all fields provided)")
    return "ask_additional_specs"


def route_after_ask_additional_specs(state: SearchWorkflowState) -> str:
    """
    Route after asking user if they want additional specs.

    This node only pauses the workflow. The actual routing happens when
    user responds (in hitl_detection_node).

    This routing function should never be called since ask_additional_specs_node
    returns to END. But we include it for completeness.
    """
    # This is a pause node, so it always goes to END
    logger.info("[SearchWorkflow] Route: ask_additional_specs -> END (pause)")
    return "END"


def route_after_advanced_specs(state: SearchWorkflowState) -> str:
    """
    Route after advanced_specs — always go to summary to show complete specs.

    New flow (Feb 2026):
    - Advanced specs discovery has completed
    - Always show summary of all specs (provided + advanced) before vendor analysis
    """
    logger.info("[SearchWorkflow] Route: advanced_specs -> summary (show complete spec list)")
    return "summary"


def route_after_vendor_analysis(state: SearchWorkflowState) -> str:
    """Route after vendor analysis."""
    if state.get("error"):
        return "error_response"
    if not state.get("vendor_matches"):
        return "no_matches"
    return "ranking"


# =============================================================================
# WORKFLOW CONSTRUCTION
# =============================================================================

def create_search_workflow() -> StateGraph:
    """
    Create simplified search workflow with 6 main nodes + 3 new HITL nodes.

    New flow (Feb 2026):
        hitl_detection -> [validation | advanced_specs | summary | vendor_analysis]
        validation -> [missing_fields_response | ask_additional_specs | advanced_specs | error_response]
        missing_fields_response -> END (pause, await missing fields)
        ask_additional_specs -> END (pause, await YES/SKIP)
        advanced_specs -> summary
        summary -> vendor_analysis
        vendor_analysis -> [ranking | no_matches | error_response]
        ranking -> compose_response
        compose_response -> END
        no_matches -> END
        error_response -> END

    HITL Resume Paths:
        - Missing fields provided → hitl_detection → validation (re-validate)
        - User says YES to additional specs → hitl_detection → advanced_specs
        - User says SKIP to additional specs → hitl_detection → summary
    """
    workflow = StateGraph(SearchWorkflowState)

    # Add main nodes
    workflow.add_node("hitl_detection", hitl_detection_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("advanced_specs", advanced_specs_node)
    workflow.add_node("vendor_analysis", vendor_analysis_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("compose_response", compose_response_node)

    # Add new HITL/response nodes
    workflow.add_node("missing_fields_response", missing_fields_response_node)
    workflow.add_node("ask_additional_specs", ask_additional_specs_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("hitl_response", hitl_response_node)  # Legacy, for backward compat
    workflow.add_node("no_matches", no_matches_node)
    workflow.add_node("error_response", error_response_node)

    # Entry point
    workflow.set_entry_point("hitl_detection")

    # Routing from hitl_detection (handles HITL resume + first call)
    workflow.add_conditional_edges(
        "hitl_detection",
        route_after_hitl_detection,
        {
            "validation": "validation",
            "advanced_specs": "advanced_specs",
            "summary": "summary",
            "vendor_analysis": "vendor_analysis"  # Legacy path
        }
    )

    # Routing from validation
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "missing_fields_response": "missing_fields_response",
            "ask_additional_specs": "ask_additional_specs",
            "advanced_specs": "advanced_specs",  # auto_mode path
            "error_response": "error_response"
        }
    )

    # advanced_specs -> summary (always show summary)
    workflow.add_conditional_edges(
        "advanced_specs",
        route_after_advanced_specs,
        {
            "summary": "summary"
        }
    )

    # summary -> vendor_analysis (always proceed)
    workflow.add_edge("summary", "vendor_analysis")

    # Routing from vendor_analysis
    workflow.add_conditional_edges(
        "vendor_analysis",
        route_after_vendor_analysis,
        {
            "ranking": "ranking",
            "no_matches": "no_matches",
            "error_response": "error_response"
        }
    )

    # ranking -> compose_response
    workflow.add_edge("ranking", "compose_response")

    # Terminal edges (END workflow)
    workflow.add_edge("compose_response", END)
    workflow.add_edge("missing_fields_response", END)  # Pause
    workflow.add_edge("ask_additional_specs", END)  # Pause
    workflow.add_edge("hitl_response", END)  # Legacy pause
    workflow.add_edge("no_matches", END)
    workflow.add_edge("error_response", END)

    return workflow


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

_compiled_workflow = None


def _get_compiled_workflow():
    """Get or create compiled workflow (singleton)."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = create_search_workflow().compile()
        logger.info("[SearchWorkflow] Workflow compiled successfully")
    return _compiled_workflow


class SearchWorkflowOrchestrator:
    """Simplified search orchestrator using the new workflow."""

    def __init__(
        self,
        enable_ppi: bool = True,
        auto_mode: bool = True
    ):
        self.enable_ppi = enable_ppi
        self.auto_mode = auto_mode

    @property
    def workflow(self):
        """Get compiled workflow (lazy singleton)."""
        return _get_compiled_workflow()

    @with_workflow_lock(session_id_param="session_id", timeout=120.0)
    def run(
        self,
        user_input: str,
        session_id: str = "",
        expected_product_type: Optional[str] = None,
        user_provided_fields: Optional[Dict[str, Any]] = None,
        auto_mode: Optional[bool] = None,
        user_decision: Optional[str] = None,
        source_workflow: Optional[str] = None,
        current_phase: Optional[str] = None,
        standards_enriched: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run search workflow.

        Args:
            user_input: User's search query
            session_id: Session identifier
            expected_product_type: Optional product type hint
            user_provided_fields: Pre-extracted user requirements
            auto_mode: Skip HITL pause if True
            user_decision: "yes" or "no" for HITL response
            source_workflow: Source workflow identifier
            current_phase: Frontend phase hint

        Returns:
            {
                "success": bool,
                "response": str,
                "response_data": dict,
                "error": str (if failed)
            }
        """
        if not session_id:
            session_id = f"search_{int(time.time() * 1000)}"

        logger.info("=" * 70)
        logger.info("[SearchWorkflow] Starting workflow")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   Input: {user_input[:100]}..." if len(user_input) > 100 else f"   Input: {user_input}")
        logger.info(f"   Auto Mode: {auto_mode if auto_mode is not None else self.auto_mode}")
        logger.info("=" * 70)

        # Create initial state
        initial_state: SearchWorkflowState = {
            "session_id": session_id,
            "user_input": user_input,
            "expected_product_type": expected_product_type,
            "user_provided_fields": user_provided_fields or {},
            "enable_ppi": self.enable_ppi,
            "auto_mode": auto_mode if auto_mode is not None else self.auto_mode,
            "source_workflow": source_workflow,
            "user_decision": user_decision,
            "current_phase": current_phase,
            "start_time": time.time(),
            "is_hitl_yes": False,
            "is_hitl_no": False,
            "is_skip": False,
            "hitl_prompt_shown": False,
            "awaiting_user_input": False,
            "awaiting_missing_fields_response": False,
            "awaiting_additional_specs_response": False,
            "all_fields_provided": False,
            "advanced_parameters": [],
            "vendor_matches": [],
            "ranked_products": [],
            "standards_enriched": standards_enriched,
        }

        try:
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)

            processing_time_ms = final_state.get("processing_time_ms", 0)

            # Return result
            return {
                "success": True,
                "response": final_state.get("response", ""),
                "response_data": final_state.get("response_data", {}),
                "processing_time_ms": processing_time_ms
            }

        except Exception as e:
            logger.error(f"[SearchWorkflow] Workflow failed: {e}", exc_info=True)
            processing_time_ms = int((time.time() - initial_state["start_time"]) * 1000)
            return {
                "success": False,
                "error": str(e),
                "response": f"Search failed: {e}",
                "response_data": {"error": str(e)},
                "processing_time_ms": processing_time_ms
            }


# =============================================================================
# CONVENIENCE FUNCTION (backward compatible)
# =============================================================================

def run_search_workflow(
    user_input: str,
    session_id: str = "",
    enable_ppi: bool = True,
    auto_mode: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for running search workflow.
    Backward compatible entry point.
    """
    orchestrator = SearchWorkflowOrchestrator(
        enable_ppi=enable_ppi,
        auto_mode=auto_mode
    )
    return orchestrator.run(
        user_input=user_input,
        session_id=session_id,
        **kwargs
    )
