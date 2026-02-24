# search/__init__.py
"""
Product Search Workflow - LangGraph Orchestrated Architecture
=============================================================

This module provides a LangGraph-orchestrated product search workflow with the following components:

Orchestrator
------------
SearchOrchestrator      -- Unified LangGraph workflow coordinating all agents as nodes

Function Modules (Feb 2026 refactor — replaces legacy deep agent files)
------------------------------------------------------------------------
validation_functions.py     -- Node 1: Product type detection, schema generation, validation
advanced_specs_functions.py -- Node 2: Discover advanced parameters from vendors
VendorAnalysisDeepAgent     -- Node 3: Analyze vendors and match products (9 internal nodes)
RankingTool                 -- Node 4: Rank matched products with detailed analysis

(ValidationDeepAgent and AdvancedSpecificationAgent are kept as backward-compat aliases)

Workflow Functions
------------------
run_product_search_workflow()     -- Full end-to-end product search (via orchestrator)
run_search_workflow()             -- Convenience function for orchestrator
run_validation_only()             -- Run only validation step
run_advanced_params_only()        -- Run only parameter discovery
run_analysis_only()               -- Run vendor analysis + ranking
process_from_solution_workflow()  -- Batch processing for solution workflow

Utility Functions
-----------------
get_schema_only()                 -- Get schema without validation
validate_with_schema()            -- Validate input against schema
"""

# =============================================================================
# TOOL CLASSES
# =============================================================================

# NEW: Simplified function modules (Feb 2026 refactor) - PRIMARY
from .validation_functions import (
    # Core validation functions
    detect_hitl_response,
    extract_product_type,
    load_schema,
    enrich_schema_with_standards,
    validate_requirements,
    generate_hitl_message,
    run_validation,
    BARE_MEASUREMENT_FIX,
    # Session caching
    cache_session_context,
    get_session_context,
    get_session_enrichment,
    cache_session_enrichment,
    clear_session_cache,
    clear_session_enrichment_cache,
    # Request context (thread-local)
    set_request_context,
    get_request_session_id,
    get_request_workflow_thread_id,
    clear_request_context,
    # Backward-compat classes
    ValidationTool,
    ValidationDeepAgent,
)

from .advanced_specs_functions import (
    # Core functions
    discover_advanced_specs,
    get_existing_schema_params,
    check_advanced_specs_cache,
    discover_advanced_specs_llm,
    parse_advanced_specs_response,
    persist_advanced_specs,
    format_specs_for_display,
    get_spec_keys,
    # Backward-compat
    discover_advanced_parameters,
    AdvancedSpecificationAgent,
)

# NEW: Simplified single orchestrator (Feb 2026 refactor)
from .search_workflow import (
    SearchWorkflowOrchestrator,
    SearchWorkflowState,
    create_search_workflow,
    run_search_workflow,
)
from .vendor_analysis_deep_agent import VendorAnalysisDeepAgent, VendorAnalysisTool
from .ranking_tool import RankingTool
from .sales_agent_tool import SalesAgentTool

# Legacy orchestrator (kept for fallback, to be removed in future)
from .search_orchestration_workflow import (
    SearchOrchestrator,
    SearchOrchestrationState,
    create_search_orchestration_workflow,
)

# =============================================================================
# WORKFLOW ORCHESTRATION FUNCTIONS
# =============================================================================

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

# ExecutionContext for proper session/workflow isolation
from common.infrastructure.context import (
    ExecutionContext,
    execution_context,
    get_context,
    set_context,      # [FIX Feb 2026 #4] Added for proper context registration
    clear_context,    # [FIX Feb 2026 #4] Added for proper context cleanup
    get_session_id as ctx_get_session_id
)

if TYPE_CHECKING:
    from common.infrastructure.context import ExecutionContext

logger = logging.getLogger(__name__)


def run_product_search_workflow(
    user_input: str,
    session_id: str = "",
    expected_product_type: Optional[str] = None,
    user_provided_fields: Optional[Dict[str, Any]] = None,
    enable_ppi: bool = True,
    auto_mode: bool = True,
    ctx: Optional["ExecutionContext"] = None,
    user_decision: Optional[str] = None,
    current_phase: Optional[str] = None,
    source_workflow: Optional[str] = "direct",
    use_orchestrator: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run complete product search workflow via LangGraph orchestrator.

    This function delegates to the SearchOrchestrator which coordinates
    ValidationTool, discover_advanced_specs, and VendorAnalysisDeepAgent
    as discrete nodes in a unified LangGraph workflow.

    Args:
        user_input: User's search query
        session_id: Session identifier (DEPRECATED - use ctx instead)
        expected_product_type: Optional product type hint
        user_provided_fields: User-provided specification fields
        enable_ppi: Enable Potential Product Index workflow if no schema exists
        auto_mode: Run automatically without HITL pauses
        ctx: ExecutionContext for proper session/workflow isolation (preferred)
        user_decision: User's YES/NO response to HITL prompt
        current_phase: Frontend-provided phase hint
        source_workflow: Source workflow identifier
        use_orchestrator: Use new LangGraph orchestrator (True) or procedural fallback (False)
        **kwargs: Additional parameters (main_thread_id, parent_workflow_id, etc.)

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

    Note:
        [Feb 2026] Now uses LangGraph orchestrator by default.
        Falls back to procedural implementation on error.
    """
    # [FIX Feb 2026] Extract standards_enriched flag from kwargs
    # When True, search workflow skips redundant standards deep agent enrichment
    standards_enriched = kwargs.pop("standards_enriched", False)

    if use_orchestrator:
        try:
            # NEW: Use simplified SearchWorkflowOrchestrator (Feb 2026 refactor)
            orchestrator = SearchWorkflowOrchestrator(
                enable_ppi=enable_ppi,
                auto_mode=auto_mode
            )
            result = orchestrator.run(
                user_input=user_input,
                session_id=session_id,
                expected_product_type=expected_product_type,
                user_provided_fields=user_provided_fields,
                auto_mode=auto_mode,
                user_decision=user_decision,
                current_phase=current_phase,
                source_workflow=source_workflow,
                standards_enriched=standards_enriched,
                **kwargs
            )
            return result
        except Exception as e:
            logger.warning(f"[ProductSearch] Orchestrator failed, using procedural fallback: {e}")
            # Fall through to procedural implementation

    # Procedural implementation (fallback)
    return _run_product_search_workflow_procedural(
        user_input=user_input,
        session_id=session_id,
        expected_product_type=expected_product_type,
        user_provided_fields=user_provided_fields,
        enable_ppi=enable_ppi,
        auto_mode=auto_mode,
        ctx=ctx,
        user_decision=user_decision,
        current_phase=current_phase,
        source_workflow=source_workflow,
        **kwargs
    )


def _run_product_search_workflow_procedural(
    user_input: str,
    session_id: str = "",
    expected_product_type: Optional[str] = None,
    user_provided_fields: Optional[Dict[str, Any]] = None,
    enable_ppi: bool = True,
    auto_mode: bool = True,
    ctx: Optional["ExecutionContext"] = None,
    user_decision: Optional[str] = None,
    current_phase: Optional[str] = None,
    source_workflow: Optional[str] = "direct",
    **kwargs
) -> Dict[str, Any]:
    """
    Run complete product search workflow (procedural implementation).

    This is the original procedural implementation kept as fallback.
    Prefer using run_product_search_workflow() which uses the LangGraph orchestrator.

    Args:
        user_input: User's search query
        session_id: Session identifier (DEPRECATED - use ctx instead)
        expected_product_type: Optional product type hint
        user_provided_fields: User-provided specification fields
        enable_ppi: Enable Potential Product Index workflow if no schema exists
        auto_mode: Run automatically without HITL pauses
        ctx: ExecutionContext for proper session/workflow isolation (preferred)
        **kwargs: Additional parameters (main_thread_id, parent_workflow_id, etc.)

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

    Note:
        [FIX Feb 2026] Added ExecutionContext for proper session isolation.
        This prevents concurrent requests from interfering with each other's state.
    """
    import uuid

    # Fix frontend parameter mismatch: if the frontend sends a user_decision but no
    # user_input, assign the user_decision to user_input so downstream nodes work.
    if user_decision and not user_input:
        user_input = user_decision

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT RESOLUTION: Prefer ExecutionContext, fallback to session_id
    # ═══════════════════════════════════════════════════════════════════════════

    # Try to get context from: 1) parameter, 2) thread-local storage, 3) create from session_id
    effective_ctx = ctx or get_context()
    context_was_created = False  # [FIX Feb 2026 #4] Track if we created the context

    if effective_ctx:
        # Use ExecutionContext for isolation
        session_id = effective_ctx.session_id
        workflow_thread_id = effective_ctx.workflow_id
        logger.info(f"[ProductSearch] Using ExecutionContext: {effective_ctx.to_log_context()}")
    else:
        # Legacy path: create context from session_id
        if not session_id:
            session_id = f"search_{uuid.uuid4().hex[:16]}"
            logger.warning(f"[ProductSearch] No context or session_id provided, generated: {session_id}")

        workflow_thread_id = kwargs.get('workflow_thread_id', '')

        # Create an ExecutionContext for this request
        effective_ctx = ExecutionContext(
            session_id=session_id,
            workflow_type="product_search",
            workflow_id=workflow_thread_id
        )

        # [FIX Feb 2026 #4] Set context to thread-local storage so nested calls can access it
        set_context(effective_ctx)
        context_was_created = True
        logger.info(f"[ProductSearch] Created and registered ExecutionContext: {session_id}")

    logger.info(f"[ProductSearch] Starting workflow for session: {session_id}")

    # Set thread-local context for this request (enables isolation in nested calls)
    set_request_context(session_id, workflow_thread_id)

    try:
        # =====================================================================
        # FIX: Check if we are waiting for response to advanced specs prompt
        # We check both cached state AND explicit frontend parameters
        # =====================================================================
        # Using get_session_context and cache_session_context from validation_functions (already imported)
        cached_context = get_session_context(session_id) or {}
        advanced_specs_presented = cached_context.get("advanced_specs_presented", False)
        hitl_prompt_shown = cached_context.get("hitl_prompt_shown", False)

        normalized_input = user_input.lower().strip()
        is_hitl_no_response = normalized_input in ["no", "n", "none"] or normalized_input.startswith("no ")
        is_hitl_yes_response = normalized_input in ["yes", "y", "sure", "yeah", "yep"]

        validation_result = None

        if advanced_specs_presented or hitl_prompt_shown or current_phase in ["advanced_specs_discovered", "await_advanced_selection"]:
            if is_hitl_no_response or user_decision == "no":
                logger.info("[ProductSearch] User declined advanced specs, proceeding directly to analysis")
                cached_context["advanced_specs_presented"] = False
                cached_context["hitl_prompt_shown"] = True  # Mark HITL as complete (NOT False!) to skip re-prompting
                cache_session_context(session_id, cached_context)
                
                current_schema = cached_context.get("schema")
                if not current_schema and expected_product_type:
                    from common.tools.schema_tools import load_schema_tool
                    schema_res = load_schema_tool(expected_product_type)
                    current_schema = schema_res.get("schema", {}) if isinstance(schema_res, dict) else {}

                # Setup validation_result as if validation succeeded to continue to next phase
                validation_result = {
                    "is_valid": True,
                    "product_type": cached_context.get("product_type", expected_product_type),
                    "schema": current_schema or {},
                    "provided_requirements": user_provided_fields or cached_context.get("provided_requirements", {})
                }
            elif is_hitl_yes_response or user_decision == "yes":
                if advanced_specs_presented or current_phase in ["advanced_specs_discovered"]:
                    logger.info("[ProductSearch] User accepted advanced specs, prompting for input")

                    # Reset HITL flags
                    cached_context["hitl_prompt_shown"] = False
                    cached_context["advanced_specs_presented"] = True  # Mark that we're in advanced specs collection
                    cache_session_context(session_id, cached_context)

                    current_schema = cached_context.get("schema")
                    if not current_schema and expected_product_type:
                        from common.tools.schema_tools import load_schema_tool
                        schema_res = load_schema_tool(expected_product_type)
                        current_schema = schema_res.get("schema", {}) if isinstance(schema_res, dict) else {}

                    return {
                        "success": True,
                        "response": "Please type the names or details of the advanced specifications you'd like to add.",
                        "response_data": {
                            "product_type": cached_context.get("product_type", expected_product_type),
                            "schema": current_schema or {},
                            "provided_requirements": user_provided_fields or cached_context.get("provided_requirements", {}),
                            "advanced_parameters": cached_context.get("advanced_parameters", []),
                            "missing_fields": [],
                            "awaiting_user_input": True,
                            "current_phase": "collect_advanced_specs",
                            "validation_bypassed": True
                        }
                    }
                else:
                    logger.info("[ProductSearch] User wants advanced specs discovery. Falling through to ValidationTool.")
                    pass
            else:
                # User provided specifications directly instead of yes/no
                logger.info("[ProductSearch] User provided advanced specs directly, routing to validation")
                cached_context["advanced_specs_presented"] = False
                cache_session_context(session_id, cached_context)
                # Let it fall through to ValidationTool

        # =====================================================================
        # FIX: Short-circuit conversational inputs ("ok", etc.)
        # =====================================================================
        conversational_phrases = {'ok', 'okay', 'proceed', 'continue', 'thanks', 'thank you', 'stop', 'cancel'}
        
        words = normalized_input.split()
        # Only bypass if it's very short and contains conversational words, OR it's an exact match
        if not validation_result and ((len(words) < 4 and any(word in conversational_phrases for word in words)) or normalized_input in conversational_phrases or normalized_input == 'show me'):
            logger.info(f"[ProductSearch] Detected conversational input '{user_input}', routing to SalesAgentTool")
            sales_agent = SalesAgentTool()
            response_message = sales_agent.process_step(
                step="conversational", 
                user_message=user_input, 
                data_context={}, 
                session_id=session_id
            ).get("content", "I understand. Please let me know if you need anything else.")
            
            return {
                "success": True,
                "response": response_message,
                "response_data": {
                    "product_type": expected_product_type or "unknown",
                    "schema": {},
                    "ranked_products": [],
                    "vendor_matches": {},
                    "missing_fields": [],
                    "awaiting_user_input": True,
                    "current_phase": "conversational"
                }
            }

        # Step 1: Validation
        # NEW: Use simplified run_validation() function (Feb 2026 refactor)
        if not validation_result:
            validation_result = run_validation(
                user_input=user_input,
                session_id=session_id,
                expected_product_type=expected_product_type,
                enable_ppi=enable_ppi,
                enable_standards_enrichment=enable_ppi,
                source_workflow=source_workflow
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # HANDLE VALIDATION BYPASSED (User said "YES" to HITL prompt)
        # When validation is bypassed, advanced_specs may have already ran inside
        # validation_tool. Return the advanced specs and proceed to vendor analysis.
        # ═══════════════════════════════════════════════════════════════════════════
        if validation_result.get("validation_bypassed"):
            logger.info("[ProductSearch] Validation was bypassed (user said YES)")
            product_type = validation_result.get("product_type", expected_product_type)
            advanced_specs_info = validation_result.get("advanced_specs_info", {})
            advanced_params = advanced_specs_info.get("unique_specifications", [])

            # If advanced specs discovery failed, return that error
            if not validation_result.get("success"):
                return {
                    "success": False,
                    "response": validation_result.get("error", "Advanced specification discovery failed"),
                    "response_data": {
                        "product_type": product_type,
                        "validation_bypassed": True,
                        "error": validation_result.get("error")
                    },
                    "error": validation_result.get("error")
                }

            # Return advanced specs to user - they can now provide additional specs
            # or proceed to vendor analysis
            num_specs = len(advanced_params)
            logger.info(f"[ProductSearch] ✓ Advanced specs already discovered: {num_specs} parameters")

            # Formulate the response with bullet points and a Yes/No prompt
            base_msg = validation_result.get("message", f"Discovered {num_specs} advanced specifications for {product_type}")
            
            # Format up to 10 parameters as a bulleted list
            formatted_params = ""
            if advanced_params:
                bullets = []
                for p in advanced_params[:10]:
                    name = str(p.get("name") or p.get("key", "")).replace("_", " ").title()
                    bullets.append(f"- {name}")
                formatted_params = "\n\n" + "\n".join(bullets)
                
            prompt_msg = "\n\nWould you like to add any of these advanced specifications to your requirements? (Yes / No)"
            full_response = f"{base_msg}{formatted_params}{prompt_msg}"

            # Set flag in session context that advanced specs were presented
            # Using get_session_context and cache_session_context from validation_functions (already imported)
            cached_context = get_session_context(session_id) or {}
            cached_context["advanced_specs_presented"] = True
            cached_context["advanced_parameters"] = advanced_params
            cache_session_context(session_id, cached_context)

            return {
                "success": True,
                "response": full_response,
                "response_data": {
                    "product_type": product_type,
                    "schema": validation_result.get("schema", {}),  # BUG FIX: preserve schema from Turn 1
                    "provided_requirements": validation_result.get("provided_requirements", {}),  # BUG FIX: preserve data
                    "advanced_parameters": advanced_params,
                    "advanced_specs_info": advanced_specs_info,
                    "ranked_products": [],
                    "vendor_matches": {},
                    "missing_fields": [],
                    "awaiting_user_input": True,  # User should provide specs now
                    "current_phase": "advanced_specs_discovered",
                    "validation_bypassed": True
                }
            }

        if not validation_result.get("is_valid"):
            # Missing fields - return for HITL
            return {
                "success": True,
                "response": validation_result.get("hitl_message", "Please provide missing specifications"),
                "response_data": {
                    "product_type": validation_result.get("product_type"),
                    "schema": validation_result.get("schema"),
                    "missing_fields": validation_result.get("missing_fields", []),
                    "provided_requirements": validation_result.get("provided_requirements", {}),
                    "awaiting_user_input": True,
                    "current_phase": "validation",
                    "ranked_products": [],
                    "vendor_matches": {}
                }
            }

        # ═══════════════════════════════════════════════════════════════════════════
        # [FIX Feb 2026] Run advanced specs discovery BEFORE showing HITL prompt.
        # Previously the HITL prompt was shown first, making it meaningless because
        # the user couldn't see what specs were available.
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Re-fetch cached_context because ValidationTool may have overwritten it
        # for a new product, clearing the stale hitl_prompt_shown state.
        # Using get_session_context from validation_functions (already imported)
        cached_context = get_session_context(session_id) or {}
        
        if not auto_mode and not cached_context.get("hitl_prompt_shown"):
            product_type = validation_result.get("product_type", expected_product_type)

            # [FIX Feb 2026] Run advanced_specs discovery FIRST
            advanced_params = []
            try:
                params_result = discover_advanced_specs(
                    product_type=product_type,
                    session_id=session_id
                )
                advanced_params = params_result.get("unique_specifications", [])
                logger.info(f"[ProductSearch] Pre-HITL: Discovered {len(advanced_params)} advanced parameters")
            except Exception as e:
                logger.warning(f"[ProductSearch] Pre-HITL: Advanced params discovery failed: {e}")

            # Build HITL message that includes discovered specs
            num_provided = len(validation_result.get("provided_requirements", {}))
            if advanced_params:
                bullets = []
                for p in advanced_params[:15]:
                    name = str(p.get("name") or p.get("key", "")).replace("_", " ").title()
                    bullets.append(f"- {name}")
                spec_list = "\n".join(bullets)
                hitl_message = (
                    f"I've extracted {num_provided} specification(s) for **{product_type}** "
                    f"and discovered {len(advanced_params)} advanced specifications:\n\n"
                    f"{spec_list}\n\n"
                    f"Would you like to add any of these advanced specifications to your requirements?\n\n"
                    f"Reply **'YES'** to add advanced specifications, or **'NO'** to continue with the search."
                )
            else:
                hitl_message = validation_result.get("hitl_message")
                if not hitl_message:
                    hitl_message = (
                        f"I've extracted {num_provided} specification(s) for **{product_type}**.\n\n"
                        f"All required specifications have been provided.\n\n"
                        f"Would you like to add more advanced specifications for better matching, "
                        f"or shall I proceed with the search?\n\n"
                        f"Reply **'YES'** to add advanced specifications, or **'NO'** to continue with the search."
                    )

            # Mark HITL prompt as shown in cache (include advanced_parameters)
            cached_context["hitl_prompt_shown"] = True
            cached_context["product_type"] = product_type
            cached_context["schema"] = validation_result.get("schema", {})
            cached_context["provided_requirements"] = validation_result.get("provided_requirements", {})
            cached_context["advanced_parameters"] = advanced_params
            cache_session_context(session_id, cached_context)

            logger.info(f"[ProductSearch] Showing HITL prompt (auto_mode=False): {len(hitl_message)} chars")

            return {
                "success": True,
                "response": hitl_message,
                "response_data": {
                    "product_type": product_type,
                    "schema": validation_result.get("schema", {}),
                    "missing_fields": validation_result.get("missing_fields", []),
                    "provided_requirements": validation_result.get("provided_requirements", {}),
                    "advanced_parameters": advanced_params,
                    "awaiting_user_input": True,
                    "current_phase": "advanced_specs_discovered",
                    "ranked_products": [],
                    "vendor_matches": {}
                }
            }

        product_type = validation_result["product_type"]
        schema = validation_result["schema"]
        provided_requirements = validation_result.get("provided_requirements", {})

        # Merge user-provided fields if any
        if user_provided_fields:
            provided_requirements.update(user_provided_fields)

        # Step 2: Advanced Parameters Discovery (optional)
        # NEW: Use simplified discover_advanced_specs() function (Feb 2026 refactor)
        advanced_params = []
        try:
            params_result = discover_advanced_specs(
                product_type=product_type,
                session_id=session_id
            )
            advanced_params = params_result.get("unique_specifications", [])
            logger.info(f"[ProductSearch] Discovered {len(advanced_params)} advanced parameters")
        except Exception as e:
            logger.warning(f"[ProductSearch] Advanced params discovery failed: {e}")

        # Step 3: Vendor Analysis
        vendor_tool = VendorAnalysisTool()
        vendor_result = vendor_tool.analyze(
            structured_requirements=provided_requirements,
            product_type=product_type,
            session_id=session_id,
            schema=schema
        )

        vendor_matches = vendor_result.get("vendor_matches", {})
        if not vendor_matches:
            return {
                "success": True,
                "response": "No matching products found for your requirements",
                "response_data": {
                    "product_type": product_type,
                    "schema": schema,
                    "ranked_products": [],
                    "vendor_matches": {},
                    "missing_fields": [],
                    "awaiting_user_input": False
                }
            }

        # Step 4: Ranking
        ranking_tool = RankingTool(use_llm_ranking=True)
        ranking_result = ranking_tool.rank(
            vendor_analysis={"vendor_matches": vendor_matches if isinstance(vendor_matches, list) else list(vendor_matches.values()) if isinstance(vendor_matches, dict) else []},
            session_id=session_id,
            structured_requirements=provided_requirements
        )

        ranked_products = ranking_result.get("overall_ranking", ranking_result.get("ranked_products", []))

        # Step 5: Format response
        sales_agent = SalesAgentTool()
        response_message = sales_agent.process_step(
            step="finalAnalysis",
            user_message=user_input,
            data_context={
                "productType": product_type,
                "rankedProducts": ranked_products
            },
            session_id=session_id
        ).get("content", "Search completed successfully")

        return {
            "success": True,
            "response": response_message,
            "response_data": {
                "product_type": product_type,
                "schema": schema,
                "ranked_products": ranked_products,
                "vendor_matches": vendor_matches,
                "advanced_parameters": advanced_params,
                "provided_requirements": provided_requirements,
                "missing_fields": [],
                "awaiting_user_input": False,
                "completed": True,
                "current_phase": "completed"
            }
        }

    except Exception as e:
        logger.exception(f"[ProductSearch] Workflow failed: {e}")
        return {
            "success": False,
            "response": f"Product search failed: {str(e)}",
            "response_data": {},
            "error": str(e)
        }

    finally:
        # ═══════════════════════════════════════════════════════════════════════════
        # FIX: Always clear request context to prevent leakage to other requests
        # ═══════════════════════════════════════════════════════════════════════════
        # [FIX Feb 2026 #4] Clear ExecutionContext if we created it
        if context_was_created:
            clear_context()
        clear_request_context()


def run_validation_only(
    user_input: str,
    expected_product_type: Optional[str] = None,
    session_id: str = "default",
    enable_ppi: bool = True
) -> Dict[str, Any]:
    """
    Run only validation step (product type detection + schema generation).

    NOW USES: run_validation() from validation_functions.py (Feb 2026 refactor)

    Args:
        user_input: User's search query
        expected_product_type: Optional product type hint
        session_id: Session identifier
        enable_ppi: Enable PPI workflow if no schema exists

    Returns:
        {
            "product_type": str,
            "schema": dict,
            "provided_requirements": dict,
            "missing_fields": list,
            "is_valid": bool,
            "ppi_workflow_used": bool
        }
    """
    logger.info(f"[ValidationOnly] Session: {session_id}")

    try:
        # NEW: Use simplified run_validation() function
        result = run_validation(
            user_input=user_input,
            session_id=session_id,
            expected_product_type=expected_product_type,
            enable_ppi=enable_ppi,
            enable_standards_enrichment=enable_ppi
        )
        return result

    except Exception as e:
        logger.exception(f"[ValidationOnly] Failed: {e}")
        return {
            "product_type": "",
            "schema": {},
            "provided_requirements": {},
            "missing_fields": [],
            "is_valid": False,
            "error": str(e)
        }


def run_advanced_params_only(
    product_type: str,
    user_input: str = "",
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Run only advanced parameters discovery.

    NOW USES: discover_advanced_specs() from advanced_specs_functions.py (Feb 2026 refactor)

    Args:
        product_type: Product type
        user_input: User's search query (optional, unused in new implementation)
        session_id: Session identifier

    Returns:
        {
            "unique_specifications": list,
            "total_unique_specifications": int,
            "success": bool
        }
    """
    logger.info(f"[AdvancedParamsOnly] Product: {product_type}, Session: {session_id}")

    try:
        # NEW: Use simplified discover_advanced_specs() function
        result = discover_advanced_specs(
            product_type=product_type,
            session_id=session_id
        )
        return result

    except Exception as e:
        logger.exception(f"[AdvancedParamsOnly] Failed: {e}")
        return {
            "unique_specifications": [],
            "total_unique_specifications": 0,
            "success": False,
            "error": str(e)
        }


def run_analysis_only(
    product_type: str,
    structured_requirements: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    session_id: str = "default",
    user_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run vendor analysis + ranking (skip validation).

    Args:
        product_type: Product type
        structured_requirements: User-provided requirements (structured format)
        schema: Product schema
        session_id: Session identifier
        user_input: User's search query (optional)

    Returns:
        {
            "success": bool,
            "ranked_products": list,
            "overall_ranking": list,
            "vendor_matches": dict,
            "response": str
        }
    """
    logger.info(f"[AnalysisOnly] Product: {product_type}, Session: {session_id}")

    try:
        requirements = structured_requirements or {}

        # Vendor Analysis
        vendor_tool = VendorAnalysisTool()
        vendor_result = vendor_tool.analyze(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
            schema=schema
        )

        vendor_matches = vendor_result.get("vendor_matches", {})

        # Ranking
        ranking_tool = RankingTool(use_llm_ranking=True)
        # Normalise vendor_matches into a flat list for RankingTool
        if isinstance(vendor_matches, dict):
            vendor_matches_list = []
            for match_list in vendor_matches.values():
                if isinstance(match_list, list):
                    vendor_matches_list.extend(match_list)
                elif isinstance(match_list, dict):
                    vendor_matches_list.append(match_list)
        elif isinstance(vendor_matches, list):
            vendor_matches_list = vendor_matches
        else:
            vendor_matches_list = []

        ranking_result = ranking_tool.rank(
            vendor_analysis={"vendor_matches": vendor_matches_list},
            session_id=session_id,
            structured_requirements=requirements
        )

        ranked_products = ranking_result.get("overall_ranking", ranking_result.get("ranked_products", []))
        top_product = ranking_result.get("top_product", ranked_products[0] if ranked_products else None)

        return {
            "success": True,
            "ranked_products": ranked_products,
            "overall_ranking": ranked_products,  # Alias for compatibility
            "top_product": top_product,
            "vendor_matches": vendor_matches,
            "totalRanked": len(ranked_products),
            "exactMatchCount": sum(1 for p in ranked_products if p.get("requirementsMatch") or p.get("overallScore", 0) >= 80),
            "approximateMatchCount": sum(1 for p in ranked_products if not (p.get("requirementsMatch") or p.get("overallScore", 0) >= 80)),
            "response": "Analysis completed successfully"
        }

    except Exception as e:
        logger.exception(f"[AnalysisOnly] Failed: {e}")
        return {
            "success": False,
            "ranked_products": [],
            "overall_ranking": [],
            "vendor_matches": {},
            "response": f"Analysis failed: {str(e)}",
            "error": str(e)
        }


def process_from_solution_workflow(
    items: List[Dict[str, Any]],
    session_id: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Batch processing for solution workflow.
    Process multiple items in parallel.

    Args:
        items: List of items to process (each with sample_input, category, etc.)
        session_id: Session identifier
        **kwargs: Additional parameters

    Returns:
        List of results for each item
    """
    logger.info(f"[BatchProcess] Processing {len(items)} items for session: {session_id}")

    # [FIX Feb 2026] Detect if Solution workflow already enriched items with standards.
    # If ANY item has standards_applied=True, set the flag to skip redundant
    # standards deep agent enrichment in the Search validation workflow.
    standards_enriched = any(
        item.get("standards_applied", False) for item in items
    )
    if standards_enriched:
        logger.info(
            "[BatchProcess] Standards enrichment already applied by Solution workflow — "
            "search validation will SKIP redundant standards deep agent enrichment"
        )

    results = []
    for idx, item in enumerate(items):
        try:
            sample_input = item.get("sample_input", "")
            product_type = item.get("category", "")
            item_session = f"{session_id}_item_{idx}"

            result = run_product_search_workflow(
                user_input=sample_input,
                session_id=item_session,
                expected_product_type=product_type,
                auto_mode=True,
                source_workflow="solution",
                standards_enriched=standards_enriched,
            )

            results.append({
                "item_number": idx + 1,
                "category": product_type,
                "search_result": result
            })

        except Exception as e:
            logger.exception(f"[BatchProcess] Item {idx} failed: {e}")
            results.append({
                "item_number": idx + 1,
                "error": str(e)
            })

    return results


def get_schema_only(
    product_type: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Get product schema without validation.

    NOW USES: load_schema() from validation_functions.py (Feb 2026 refactor)

    Args:
        product_type: Product type
        session_id: Session identifier (unused in new implementation)

    Returns:
        Schema dictionary
    """
    try:
        # NEW: Use simplified load_schema() function
        result = load_schema(product_type=product_type, enable_ppi=True)
        return result

    except Exception as e:
        logger.exception(f"[GetSchemaOnly] Failed: {e}")
        return {"success": False, "schema": {}, "error": str(e)}


def validate_with_schema(
    user_input: str,
    schema: Dict[str, Any],
    product_type: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Validate user input against provided schema.

    NOW USES: validate_requirements() from validation_functions.py (Feb 2026 refactor)

    Args:
        user_input: User's input
        schema: Product schema
        product_type: Product type
        session_id: Session identifier (unused in new implementation)

    Returns:
        Validation result
    """
    try:
        # NEW: Use simplified validate_requirements() function
        result = validate_requirements(
            user_input=user_input,
            product_type=product_type,
            schema=schema
        )
        return result

    except Exception as e:
        logger.exception(f"[ValidateWithSchema] Failed: {e}")
        return {
            "success": False,
            "is_valid": False,
            "error": str(e)
        }


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for old function names
product_search_workflow = run_product_search_workflow
run_single_product_workflow = run_product_search_workflow


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # NEW: Simplified orchestrator (Feb 2026 refactor)
    "SearchWorkflowOrchestrator",
    "SearchWorkflowState",
    "create_search_workflow",
    "run_search_workflow",

    # NEW: Simplified validation functions (Feb 2026 refactor)
    "detect_hitl_response",
    "extract_product_type",
    "load_schema",
    "enrich_schema_with_standards",
    "validate_requirements",
    "generate_hitl_message",
    "run_validation",
    "cache_session_context",
    "get_session_context",
    "BARE_MEASUREMENT_FIX",

    # NEW: Simplified advanced specs functions (Feb 2026 refactor)
    "discover_advanced_specs",
    "get_existing_schema_params",
    "check_advanced_specs_cache",
    "format_specs_for_display",

    # Legacy orchestrator (kept for backward compatibility)
    "SearchOrchestrator",
    "SearchOrchestrationState",
    "create_search_orchestration_workflow",

    # Backward-compat aliases (ValidationDeepAgent → ValidationTool, AdvancedSpecificationAgent → discover_advanced_specs wrapper)
    "ValidationTool",
    "ValidationDeepAgent",
    "AdvancedSpecificationAgent",
    "VendorAnalysisDeepAgent",
    "VendorAnalysisTool",
    "RankingTool",
    "SalesAgentTool",

    # Workflow functions
    "run_product_search_workflow",
    "run_validation_only",
    "run_advanced_params_only",
    "run_analysis_only",
    "process_from_solution_workflow",

    # Utility functions
    "get_schema_only",
    "validate_with_schema",
    "clear_session_enrichment_cache",
    "clear_session_cache",

    # Backward compatibility aliases
    "product_search_workflow",
    "run_single_product_workflow",
]
