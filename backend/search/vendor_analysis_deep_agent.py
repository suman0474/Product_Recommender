"""
Vendor Analysis Tool for Product Search Workflow
=================================================

Step 4 of Product Search Workflow:
- Loads vendors matching product type
- **APPLIES STRATEGY RAG** to filter/prioritize vendors before analysis
- Retrieves PDF datasheets and JSON product catalogs for approved vendors
- Runs parallel vendor analysis using get_vendor_prompt
- Returns matched products with detailed analysis + strategy context

STRATEGY RAG INTEGRATION (Step 2.5):
- Retrieves strategic context from Pinecone vector store (TRUE RAG)
- Falls back to LLM inference if vector store is empty
- Filters out FORBIDDEN vendors defined in strategy documents
- Prioritizes PREFERRED vendors for analysis
- Enriches final matches with strategy_priority scores

Strategic context includes:
- Cost optimization priorities
- Sustainability requirements
- Compliance alignment
- Supplier reliability metrics
- Long-term partnership alignment

This tool integrates:
- Vendor loading from MongoDB/Azure
- Strategy RAG for vendor filtering (NEW)
- PDF content extraction
- JSON product catalog loading
- LLM-powered vendor analysis

Deep Agent Architecture (LangGraph):
- VendorAnalysisTool.analyze() is now a thin shell over a LangGraph graph
- Each pipeline phase runs as a discrete graph node (_node_* functions)
- Graph with conditional edges handles empty-vendor / no-data early exits
- All helper methods (_analyze_vendor, _format_requirements, etc.) unchanged
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# [FIX Feb 2026] Import BoundedCache for session-isolated caching with TTL/LRU
from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

# [FIX Feb 2026] Import ExecutionContext for thread context propagation
from common.infrastructure.context import get_context, execution_context

# Configure logging
logger = logging.getLogger(__name__)

# Debug flags for vendor analysis debugging
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

# Issue-specific debug logging
try:
    from debug_flags import issue_debug
except ImportError:
    issue_debug = None


# =============================================================================
# DEEP AGENT STATE
# =============================================================================

class VendorAnalysisDeepAgentState(TypedDict, total=False):
    """
    LangGraph state for the Vendor Analysis Deep Agent.

    Carries all intermediate data between nodes so each node is a pure,
    testable function that reads from and writes to this shared state.
    """
    # --- Inputs (set once at graph entry) ---
    session_id: Optional[str]
    product_type: str
    structured_requirements: Dict[str, Any]
    schema: Optional[Dict[str, Any]]
    tool: Any                          # VendorAnalysisTool instance (for helper methods)

    # --- Node 1: setup_components ---
    components: Optional[Dict[str, Any]]

    # --- Node 2: load_vendors ---
    vendors: List[str]

    # --- Node 3: apply_strategy_rag ---
    strategy_context: Optional[Dict[str, Any]]
    filtered_vendors: List[str]
    vendor_priorities: Dict[str, int]
    excluded_vendors: List[Dict[str, Any]]
    strategy_rag_invoked: bool
    strategy_rag_invocation_time: str

    # --- Node 4: load_product_data ---
    products_data: Dict[str, Any]

    # --- Node 5: prepare_payloads ---
    vendor_payloads: Dict[str, Dict[str, Any]]

    # --- Node 6: load_standards ---
    applicable_standards: List[str]
    standards_specs: str

    # --- Node 7: run_parallel_analysis ---
    vendor_matches: List[Dict[str, Any]]
    run_details: List[Dict[str, Any]]

    # --- Node 8: enrich_matches ---
    enriched_matches: List[Dict[str, Any]]

    # --- Node 9: assemble_result ---
    result: Dict[str, Any]

    # --- Observability ---
    current_node: str
    error: Optional[str]
    start_time: float


# =============================================================================
# GRAPH NODE FUNCTIONS
# =============================================================================

def _node_setup_components(state: VendorAnalysisDeepAgentState) -> dict:
    """Node 1 — Set up LangChain LLM components."""
    logger.info("[VendorAnalysisDeepAgent] Node 1: setup_components")
    try:
        from core.chaining import setup_langchain_components
        components = setup_langchain_components()
        return {"components": components, "current_node": "setup_components"}
    except Exception as e:
        logger.error("[VendorAnalysisDeepAgent] setup_components failed: %s", e)
        return {"components": None, "error": str(e), "current_node": "setup_components"}


def _node_load_vendors(state: VendorAnalysisDeepAgentState) -> dict:
    """Node 2 — Load vendors from Azure Blob Storage for the given product type."""
    product_type = state.get("product_type", "")
    logger.info("[VendorAnalysisDeepAgent] Node 2: load_vendors — product_type='%s'", product_type)
    try:
        from services.azure.blob_utils import get_vendors_for_product_type
        vendors = get_vendors_for_product_type(product_type) if product_type else []

        logger.info("[VendorAnalysisDeepAgent] ===== DIAGNOSTIC: VENDOR LOADING =====")
        logger.info("[VendorAnalysisDeepAgent] Product type: '%s'", product_type)
        logger.info("[VendorAnalysisDeepAgent] Found vendors: %d - %s", len(vendors), vendors[:5] if vendors else [])

        if not vendors:
            logger.warning("[VendorAnalysisDeepAgent] NO VENDORS FOUND")

        return {"vendors": vendors, "current_node": "load_vendors"}
    except Exception as e:
        logger.error("[VendorAnalysisDeepAgent] load_vendors failed: %s", e)
        return {"vendors": [], "error": str(e), "current_node": "load_vendors"}


def _node_apply_strategy_rag(state: VendorAnalysisDeepAgentState) -> dict:
    """
    Node 3 — Apply Strategy RAG to filter and prioritize vendors.

    - TRUE RAG from Pinecone (or LLM inference fallback)
    - Removes FORBIDDEN vendors
    - Orders PREFERRED vendors first
    """
    vendors = state.get("vendors", [])
    product_type = state.get("product_type", "")
    structured_requirements = state.get("structured_requirements", {})
    session_id = state.get("session_id")

    import datetime
    invocation_time = datetime.datetime.now().isoformat()
    rag_invoked = False

    logger.info("=" * 70)
    logger.info("STRATEGY RAG INVOKED")
    logger.info("   Timestamp: %s", invocation_time)
    logger.info("   Product Type: %s", product_type)
    logger.info("   Vendors to Filter: %d", len(vendors))
    logger.info("   Session: %s", session_id)
    logger.info("=" * 70)

    strategy_context = None
    filtered_vendors = vendors.copy()
    excluded_vendors: List[Dict[str, Any]] = []
    vendor_priorities: Dict[str, int] = {}

    try:
        rag_invoked = True
        from common.rag.strategy.enrichment import get_strategy_with_auto_fallback
        from common.rag.strategy.mongodb_loader import filter_vendors_by_strategy

        strategy_context = get_strategy_with_auto_fallback(
            product_type=product_type,
            requirements=structured_requirements,
            top_k=7
        )

        if strategy_context.get("success"):
            rag_type = strategy_context.get("rag_type", "unknown")
            preferred = strategy_context.get("preferred_vendors", [])
            forbidden = strategy_context.get("forbidden_vendors", [])
            confidence = strategy_context.get("confidence", 0.0)
            priorities = strategy_context.get("procurement_priorities", {})
            strategy_notes = strategy_context.get("strategy_notes", "")

            logger.info("=" * 70)
            logger.info("STRATEGY RAG APPLIED SUCCESSFULLY (%s)", rag_type)
            logger.info("   Preferred vendors: %s", preferred)
            logger.info("   Forbidden vendors: %s", forbidden)
            logger.info("   Confidence: %.2f", confidence)
            logger.info("=" * 70)
            logger.info("[VendorAnalysisDeepAgent] Priorities: %s", priorities)
            logger.info("[VendorAnalysisDeepAgent] Strategy notes: %s...", (strategy_notes[:200] if strategy_notes else "None"))

            filter_result = filter_vendors_by_strategy(vendors, strategy_context)
            accepted = filter_result.get("accepted_vendors", [])
            excluded_vendors = filter_result.get("excluded_vendors", [])

            if accepted:
                filtered_vendors = [v["vendor"] for v in accepted]
                vendor_priorities = {v["vendor"]: v.get("priority_score", 0) for v in accepted}
                logger.info("[VendorAnalysisDeepAgent] Strategy filtering: %d accepted, %d excluded",
                            len(filtered_vendors), len(excluded_vendors))
                for ex in excluded_vendors:
                    logger.info("[VendorAnalysisDeepAgent] Excluded: %s - %s", ex["vendor"], ex["reason"])
            else:
                logger.warning("[VendorAnalysisDeepAgent] No vendors passed strategy filter — using original list")
                filtered_vendors = vendors
        else:
            logger.warning("[VendorAnalysisDeepAgent] Strategy RAG returned no results: %s",
                           strategy_context.get("error", "Unknown"))

    except Exception as strategy_error:
        logger.warning("[VendorAnalysisDeepAgent] ⚠ Strategy RAG failed (proceeding without): %s", strategy_error)
        logger.error("=" * 70)
        logger.error("[STRATEGY RAG] ERROR: %s", strategy_error)
        logger.error("=" * 70)
        filtered_vendors = vendors

    return {
        "strategy_context": strategy_context,
        "filtered_vendors": filtered_vendors,
        "vendor_priorities": vendor_priorities,
        "excluded_vendors": excluded_vendors,
        "strategy_rag_invoked": rag_invoked,
        "strategy_rag_invocation_time": invocation_time,
        "current_node": "apply_strategy_rag",
    }


def _node_load_product_data(state: VendorAnalysisDeepAgentState) -> dict:
    """Node 4 — Load product data from Azure Blob for strategy-filtered vendors."""
    filtered_vendors = state.get("filtered_vendors", [])
    product_type = state.get("product_type", "")
    vendor_priorities = state.get("vendor_priorities", {})

    logger.info("[VendorAnalysisDeepAgent] Node 4: load_product_data — %d vendors", len(filtered_vendors))
    try:
        from services.azure.blob_utils import get_products_for_vendors
        products_data = get_products_for_vendors(filtered_vendors, product_type)

        logger.info("[VendorAnalysisDeepAgent] ===== DIAGNOSTIC: PRODUCT DATA (STRATEGY-FILTERED) =====")
        logger.info("[VendorAnalysisDeepAgent] Products loaded for %d vendors", len(products_data))
        for v, products in list(products_data.items())[:3]:
            product_count = len(products) if products else 0
            priority = vendor_priorities.get(v, 0)
            logger.info("[VendorAnalysisDeepAgent]   - %s: %d product entries (priority: %d)", v, product_count, priority)

        return {"products_data": products_data, "current_node": "load_product_data"}
    except Exception as e:
        logger.error("[VendorAnalysisDeepAgent] load_product_data failed: %s", e)
        return {"products_data": {}, "error": str(e), "current_node": "load_product_data"}


def _node_prepare_payloads(state: VendorAnalysisDeepAgentState) -> dict:
    """Node 5 — Build per-vendor JSON payloads for LLM analysis."""
    products_data = state.get("products_data", {})

    logger.info("[VendorAnalysisDeepAgent] Node 5: prepare_payloads")
    vendor_payloads: Dict[str, Dict[str, Any]] = {}

    for vendor_name, products in products_data.items():
        if products:
            products_json = json.dumps(products, indent=2, ensure_ascii=False)
            vendor_payloads[vendor_name] = {
                "products": products,
                "pdf_text": products_json,  # product JSON serves as analysis content
            }

    logger.info("[VendorAnalysisDeepAgent] ===== DIAGNOSTIC: VENDOR PAYLOADS =====")
    logger.info("[VendorAnalysisDeepAgent] Total payloads prepared: %d", len(vendor_payloads))
    for v, data in list(vendor_payloads.items())[:5]:
        content_len = len(data.get("pdf_text", "")) if data.get("pdf_text") else 0
        products_list = data.get("products", [])
        products_count = len(products_list)

        models_count = submodels_count = specs_count = 0
        if products_list and isinstance(products_list[0], dict):
            first_product = products_list[0]
            if "models" in first_product:
                models = first_product["models"]
                models_count = len(models) if models else 0
                for model in (models or []):
                    if "sub_models" in model:
                        submodels_count += len(model["sub_models"])
                        for sub in model["sub_models"]:
                            if "specifications" in sub:
                                specs_count += len(sub.get("specifications", {}))

        logger.info("[VendorAnalysisDeepAgent]   - %s: Content=%d chars, Products=%d, Models=%d, Submodels=%d, Specs=%d",
                    v, content_len, products_count, models_count, submodels_count, specs_count)

    return {"vendor_payloads": vendor_payloads, "current_node": "prepare_payloads"}


def _node_load_standards(state: VendorAnalysisDeepAgentState) -> dict:
    """Node 6 — Extract applicable engineering standards from requirements."""
    structured_requirements = state.get("structured_requirements", {})

    logger.info("[VendorAnalysisDeepAgent] Node 6: load_standards")
    applicable_standards: List[str] = []
    standards_specs = "No specific standards requirements provided."

    try:
        if structured_requirements and isinstance(structured_requirements, dict):
            if "applicable_standards" in structured_requirements:
                applicable_standards = structured_requirements.get("applicable_standards", [])
            if "standards_specifications" in structured_requirements:
                standards_specs = structured_requirements.get("standards_specifications", standards_specs)
        logger.info("[VendorAnalysisDeepAgent] Loaded %d applicable standards", len(applicable_standards))
    except Exception as standards_error:
        logger.warning("[VendorAnalysisDeepAgent] Failed to load standards: %s (continuing)", standards_error)

    return {
        "applicable_standards": applicable_standards,
        "standards_specs": standards_specs,
        "current_node": "load_standards",
    }


def _node_run_parallel_analysis(state: VendorAnalysisDeepAgentState) -> dict:
    """Node 7 — Run parallel LLM vendor analysis using ThreadPoolExecutor."""
    tool: "VendorAnalysisDeepAgent" = state["tool"]
    components = state.get("components", {})
    vendor_payloads = state.get("vendor_payloads", {})
    structured_requirements = state.get("structured_requirements", {})
    applicable_standards = state.get("applicable_standards", [])
    standards_specs = state.get("standards_specs", "No specific standards requirements provided.")

    logger.info("[VendorAnalysisDeepAgent] Node 7: run_parallel_analysis — %d vendors", len(vendor_payloads))

    requirements_str = tool._format_requirements(structured_requirements)
    logger.info("[VendorAnalysisDeepAgent] Requirements: %s...", requirements_str[:200] if requirements_str else "EMPTY")

    vendor_matches: List[Dict[str, Any]] = []
    run_details: List[Dict[str, Any]] = []

    from core.chaining import to_dict_if_pydantic

    # [FIX Feb 2026 #2] Capture parent context for propagation to worker threads
    parent_ctx = get_context()
    session_id = state.get("session_id", "unknown")

    def run_vendor_analysis_with_context(vendor: str, data: Dict[str, Any]):
        """
        Wrapper that propagates ExecutionContext to worker thread.
        This ensures logging, caching, and tracing have correct session/workflow IDs.
        """
        if parent_ctx:
            with execution_context(parent_ctx):
                return tool._analyze_vendor(
                    components, requirements_str, vendor, data,
                    applicable_standards, standards_specs
                )
        else:
            # Fallback: run without context (legacy behavior)
            return tool._analyze_vendor(
                components, requirements_str, vendor, data,
                applicable_standards, standards_specs
            )

    actual_workers = min(len(vendor_payloads), tool.max_workers)
    logger.info("[VendorAnalysisDeepAgent] Using %d workers for %d vendors (session: %s)",
                actual_workers, len(vendor_payloads), session_id[:16] if session_id else "N/A")

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {}
        for vendor, data in vendor_payloads.items():
            # [FIX Feb 2026 #2] Use wrapper function for context propagation
            future = executor.submit(run_vendor_analysis_with_context, vendor, data)
            futures[future] = vendor

        for future in as_completed(futures):
            vendor = futures[future]
            try:
                vendor_result, error = future.result()

                if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                    for match in vendor_result["vendor_matches"]:
                        match_dict = to_dict_if_pydantic(match)
                        normalized_match = {
                            "productName": match_dict.get("product_name", match_dict.get("productName", "")),
                            "vendor": vendor,
                            "modelFamily": match_dict.get("model_family", match_dict.get("modelFamily", "")),
                            "productType": match_dict.get("product_type", match_dict.get("productType", "")),
                            "matchScore": match_dict.get("match_score", match_dict.get("matchScore", 0)),
                            "requirementsMatch": match_dict.get("requirements_match", match_dict.get("requirementsMatch", False)),
                            "reasoning": match_dict.get("reasoning", ""),
                            "limitations": match_dict.get("limitations", ""),
                            "productDescription": match_dict.get("product_description", match_dict.get("productDescription", "")),
                            "standardsCompliance": match_dict.get("standards_compliance", match_dict.get("standardsCompliance", {})),
                            "matchedRequirements": match_dict.get("matched_requirements", match_dict.get("matchedRequirements", {})),
                            "unmatchedRequirements": match_dict.get("unmatched_requirements", match_dict.get("unmatchedRequirements", [])),
                            "keyStrengths": match_dict.get("key_strengths", match_dict.get("keyStrengths", [])),
                            "recommendation": match_dict.get("recommendation", ""),
                        }
                        vendor_matches.append(normalized_match)
                        logger.info("[VendorAnalysisDeepAgent] DEBUG: Normalized match for %s: %s", vendor, normalized_match)

                    run_details.append({"vendor": vendor, "status": "success"})
                    logger.info("[VendorAnalysisDeepAgent] Vendor '%s' returned %d matches",
                                vendor, len(vendor_result["vendor_matches"]))
                else:
                    run_details.append({
                        "vendor": vendor,
                        "status": "failed" if error else "empty",
                        "error": error,
                    })
                    logger.warning("[VendorAnalysisDeepAgent] Vendor '%s' failed: %s", vendor, error)

            except Exception as e:
                logger.error("[VendorAnalysisDeepAgent] Vendor '%s' exception: %s", vendor, str(e))
                run_details.append({"vendor": vendor, "status": "error", "error": str(e)})

    return {
        "vendor_matches": vendor_matches,
        "run_details": run_details,
        "current_node": "run_parallel_analysis",
    }


def _node_enrich_matches(state: VendorAnalysisDeepAgentState) -> dict:
    """
    Node 8 — Enrich vendor matches with images, logos, and pricing links.

    Each enrichment step is wrapped in try/except so partial failures
    never abort the pipeline.
    """
    vendor_matches = list(state.get("vendor_matches", []))

    logger.info("[VendorAnalysisDeepAgent] Node 8: enrich_matches — %d matches", len(vendor_matches))

    # Step 8a: Product images
    logger.info("[VendorAnalysisDeepAgent] Step 8a: Enriching with vendor product images")
    try:
        from agentic.utils.vendor_images import fetch_images_for_vendor_matches

        vendor_groups: Dict[str, list] = {}
        for match in vendor_matches:
            v = match.get("vendor", "")
            vendor_groups.setdefault(v, []).append(match)

        for vendor_name, vendor_match_list in vendor_groups.items():
            try:
                logger.info("[VendorAnalysisDeepAgent] Fetching images for vendor: %s", vendor_name)
                enriched = fetch_images_for_vendor_matches(
                    vendor_name=vendor_name,
                    matches=vendor_match_list,
                    max_workers=2,
                )
                for enriched_match in enriched:
                    for i, match in enumerate(vendor_matches):
                        if (match.get("vendor") == enriched_match.get("vendor") and
                                match.get("productName") == enriched_match.get("productName")):
                            vendor_matches[i] = enriched_match
            except Exception as vendor_image_error:
                logger.warning("[VendorAnalysisDeepAgent] Failed to fetch images for %s: %s",
                               vendor_name, vendor_image_error)

        logger.info("[VendorAnalysisDeepAgent] Image enrichment complete")
    except ImportError:
        logger.debug("[VendorAnalysisDeepAgent] Image utilities not available — skipping")
    except Exception as image_error:
        logger.warning("[VendorAnalysisDeepAgent] Image enrichment failed: %s", image_error)

    # Step 8b: Vendor logos
    logger.info("[VendorAnalysisDeepAgent] Step 8b: Enriching with vendor logos")
    try:
        from agentic.utils.vendor_images import enrich_matches_with_logos
        vendor_matches = enrich_matches_with_logos(matches=vendor_matches, max_workers=2)
        logger.info("[VendorAnalysisDeepAgent] Logo enrichment complete")
    except ImportError:
        logger.debug("[VendorAnalysisDeepAgent] Logo utilities not available — skipping")
    except Exception as logo_error:
        logger.warning("[VendorAnalysisDeepAgent] Logo enrichment failed: %s", logo_error)

    # Step 8c: Pricing links
    logger.info("[VendorAnalysisDeepAgent] Step 8c: Enriching with pricing links")
    try:
        from agentic.utils.pricing_search import enrich_matches_with_pricing
        vendor_matches = enrich_matches_with_pricing(matches=vendor_matches, max_workers=5)
        logger.info("[VendorAnalysisDeepAgent] Pricing enrichment complete")
    except ImportError:
        logger.debug("[VendorAnalysisDeepAgent] Pricing utilities not available — skipping")
    except Exception as pricing_error:
        logger.warning("[VendorAnalysisDeepAgent] Pricing enrichment failed: %s", pricing_error)

    return {"enriched_matches": vendor_matches, "current_node": "enrich_matches"}


def _node_assemble_result(state: VendorAnalysisDeepAgentState) -> dict:
    """
    Node 9 — Assemble the final result dict, enrich matches with strategy
    priority scores, and cache the result on the tool instance.
    """
    tool: "VendorAnalysisTool" = state["tool"]

    product_type = state.get("product_type", "")
    session_id = state.get("session_id")
    vendors = state.get("vendors", [])
    filtered_vendors = state.get("filtered_vendors", [])
    vendor_payloads = state.get("vendor_payloads", {})
    excluded_vendors = state.get("excluded_vendors", [])
    vendor_priorities = state.get("vendor_priorities", {})
    strategy_context = state.get("strategy_context")
    run_details = state.get("run_details", [])
    strategy_rag_invoked = state.get("strategy_rag_invoked", False)
    strategy_rag_invocation_time = state.get("strategy_rag_invocation_time", "")

    # Use enriched_matches if available, else fall back to vendor_matches
    vendor_matches = list(state.get("enriched_matches") or state.get("vendor_matches") or [])

    logger.info("[VendorAnalysisDeepAgent] Node 9: assemble_result — %d matches", len(vendor_matches))

    # Enrich matches with strategy priority scores
    for match in vendor_matches:
        vendor_name = match.get("vendor", "")
        match["strategy_priority"] = vendor_priorities.get(vendor_name, 0)
        match["is_preferred_vendor"] = vendor_name.lower() in [
            p.lower() for p in (strategy_context.get("preferred_vendors", []) if strategy_context else [])
        ]
        if "requirementsMatch" not in match:
            score = match.get("matchScore", 0)
            match["requirementsMatch"] = score >= 80

    # Build summary
    successful = sum(1 for d in run_details if d.get("status") == "success")
    strategy_source = (strategy_context or {}).get("rag_type", "none")
    analysis_summary = (
        f"Strategy RAG ({strategy_source}): Filtered {len(vendors)} → {len(filtered_vendors)} vendors | "
        f"Analyzed {len(vendor_payloads)} vendors, {successful} successful | "
        f"Found {len(vendor_matches)} matching products"
    )

    result: Dict[str, Any] = {
        "success": True,
        "product_type": product_type,
        "session_id": session_id,
        "vendor_matches": vendor_matches,
        "vendor_run_details": run_details,
        "total_matches": len(vendor_matches),
        "vendors_analyzed": len(vendor_payloads),
        "original_vendor_count": len(vendors),
        "filtered_vendor_count": len(filtered_vendors),
        "excluded_by_strategy": len(excluded_vendors),
        "analysis_summary": analysis_summary,
        "strategy_context": {
            "applied": strategy_context is not None and strategy_context.get("success", False),
            "rag_type": strategy_context.get("rag_type") if strategy_context else None,
            "preferred_vendors": strategy_context.get("preferred_vendors", []) if strategy_context else [],
            "forbidden_vendors": strategy_context.get("forbidden_vendors", []) if strategy_context else [],
            "excluded_vendors": excluded_vendors,
            "vendor_priorities": vendor_priorities,
            "confidence": strategy_context.get("confidence", 0.0) if strategy_context else 0.0,
            "strategy_notes": strategy_context.get("strategy_notes", "") if strategy_context else "",
            "sources_used": strategy_context.get("sources_used", []) if strategy_context else [],
        },
        "rag_invocations": {
            "strategy_rag": {
                "invoked": strategy_rag_invoked,
                "invocation_time": strategy_rag_invocation_time,
                "success": strategy_context is not None and strategy_context.get("success", False),
                "rag_type": strategy_context.get("rag_type") if strategy_context else None,
                "product_type": product_type,
                "vendors_before_filter": len(vendors),
                "vendors_after_filter": len(filtered_vendors),
                "excluded_count": len(excluded_vendors),
            },
            "standards_rag": {
                "invoked": False,
                "note": "Standards RAG is applied in validation_tool.py, not during vendor analysis",
            },
        },
    }

    logger.info("[VendorAnalysisDeepAgent] ===== ANALYSIS COMPLETE =====")
    logger.info("[VendorAnalysisDeepAgent] %s", analysis_summary)
    if excluded_vendors:
        logger.info("[VendorAnalysisDeepAgent] Excluded by strategy: %s", [e["vendor"] for e in excluded_vendors])

    # [FIX Feb 2026 #1] Cache with session_id included for isolation
    try:
        import hashlib
        session_id = state.get("session_id", "global")
        structured_requirements = state.get("structured_requirements", {})
        # Include session_id in cache key to prevent cross-session contamination
        cache_key = hashlib.md5(
            f"{session_id}:{product_type}:{json.dumps(structured_requirements, sort_keys=True, default=str)}".encode()
        ).hexdigest()
        # [FIX Feb 2026 #6] Use BoundedCache .set() method
        tool._response_cache.set(cache_key, result)
        logger.info("[VendorAnalysisDeepAgent] ✓ Cached result (key: %s..., session: %s)",
                    cache_key[:8], session_id[:16] if session_id else "global")
    except Exception as cache_write_error:
        logger.warning("[VendorAnalysisDeepAgent] Failed to cache result: %s", cache_write_error)

    return {"result": result, "current_node": "assemble_result"}


# --- Conditional routing helpers ---

def _route_after_load_vendors(state: VendorAnalysisDeepAgentState) -> str:
    """Skip to assemble_result when no vendors found."""
    vendors = state.get("vendors", [])
    if not vendors:
        logger.info("[VendorAnalysisDeepAgent] No vendors — routing to assemble_result (empty)")
        return "assemble_result"
    return "apply_strategy_rag"


def _route_after_load_product_data(state: VendorAnalysisDeepAgentState) -> str:
    """Skip to assemble_result when no product data available."""
    products_data = state.get("products_data", {})
    if not products_data:
        logger.info("[VendorAnalysisDeepAgent] No product data — routing to assemble_result (empty)")
        return "assemble_result"
    return "prepare_payloads"


def _route_after_prepare_payloads(state: VendorAnalysisDeepAgentState) -> str:
    """Skip to assemble_result when no valid vendor payloads."""
    vendor_payloads = state.get("vendor_payloads", {})
    if not vendor_payloads:
        logger.info("[VendorAnalysisDeepAgent] No vendor payloads — routing to assemble_result (empty)")
        return "assemble_result"
    return "load_standards"


# =============================================================================
# GRAPH BUILDER (lazy singleton)
# =============================================================================

_vendor_analysis_graph = None  # module-level singleton


def _build_vendor_analysis_graph():
    """Build (or return cached) the compiled LangGraph for vendor analysis."""
    global _vendor_analysis_graph
    if _vendor_analysis_graph is not None:
        return _vendor_analysis_graph

    graph = StateGraph(VendorAnalysisDeepAgentState)

    # Register nodes
    graph.add_node("setup_components", _node_setup_components)
    graph.add_node("load_vendors", _node_load_vendors)
    graph.add_node("apply_strategy_rag", _node_apply_strategy_rag)
    graph.add_node("load_product_data", _node_load_product_data)
    graph.add_node("prepare_payloads", _node_prepare_payloads)
    graph.add_node("load_standards", _node_load_standards)
    graph.add_node("run_parallel_analysis", _node_run_parallel_analysis)
    graph.add_node("enrich_matches", _node_enrich_matches)
    graph.add_node("assemble_result", _node_assemble_result)

    # Edges
    graph.add_edge(START, "setup_components")
    graph.add_edge("setup_components", "load_vendors")

    graph.add_conditional_edges(
        "load_vendors",
        _route_after_load_vendors,
        {"apply_strategy_rag": "apply_strategy_rag", "assemble_result": "assemble_result"},
    )

    graph.add_edge("apply_strategy_rag", "load_product_data")

    graph.add_conditional_edges(
        "load_product_data",
        _route_after_load_product_data,
        {"prepare_payloads": "prepare_payloads", "assemble_result": "assemble_result"},
    )

    graph.add_conditional_edges(
        "prepare_payloads",
        _route_after_prepare_payloads,
        {"load_standards": "load_standards", "assemble_result": "assemble_result"},
    )

    graph.add_edge("load_standards", "run_parallel_analysis")
    graph.add_edge("run_parallel_analysis", "enrich_matches")
    graph.add_edge("enrich_matches", "assemble_result")
    graph.add_edge("assemble_result", END)

    _vendor_analysis_graph = graph.compile()
    logger.info("[VendorAnalysisDeepAgent] LangGraph compiled successfully")
    return _vendor_analysis_graph


# =============================================================================
# VENDOR ANALYSIS TOOL CLASS
# =============================================================================

class VendorAnalysisDeepAgent:
    """
    Vendor Analysis Deep Agent - Step 4 of Product Search Workflow

    Responsibilities:
    1. Load vendors matching the detected product type
    2. Retrieve vendor documentation (PDFs and JSON)
    3. Run parallel vendor analysis
    4. Return matched products with detailed analysis

    This agent is powered by a LangGraph workflow:
    each pipeline phase runs as a discrete, observable graph node.
    All helper methods (_analyze_vendor, _format_requirements, etc.) are encapsulated here.
    """

    # [FIX Feb 2026 #6] Class-level bounded cache shared across instances
    # Prevents unbounded memory growth with TTL-based eviction
    _response_cache: BoundedCache = None

    def __init__(self, max_workers: int = 10, max_retries: int = 3):
        """
        Initialize the vendor analysis deep agent.

        Args:
            max_workers: Maximum parallel workers for vendor analysis
            max_retries: Maximum retries for rate-limited requests
        """
        self.max_workers = max_workers
        self.max_retries = max_retries

        # [FIX Feb 2026 #6] Use bounded cache with TTL/LRU instead of unbounded dict
        if VendorAnalysisDeepAgent._response_cache is None:
            VendorAnalysisDeepAgent._response_cache = get_or_create_cache(
                name="vendor_analysis_response",
                max_size=200,           # Max 200 cached results
                ttl_seconds=1800        # 30 minute TTL
            )

        logger.info("[VendorAnalysisDeepAgent] Initialized with max_workers=%d", max_workers)

    @timed_execution("VENDOR_ANALYSIS", threshold_ms=45000)
    @debug_log("VENDOR_ANALYSIS", log_args=True, log_result=False)
    def analyze(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        session_id: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze vendors for matching products via the LangGraph Deep Agent.

        The pipeline runs as 9 discrete graph nodes:
          1. setup_components     — LangChain LLM setup
          2. load_vendors         — Azure Blob vendor loading
          3. apply_strategy_rag   — Strategy RAG filter/prioritize
          4. load_product_data    — Product data per vendor
          5. prepare_payloads     — Build LLM analysis payloads
          6. load_standards       — Extract applicable standards
          7. run_parallel_analysis — ThreadPoolExecutor LLM calls
          8. enrich_matches        — Images / logos / pricing links
          9. assemble_result       — Build + cache final result

        Args:
            structured_requirements: Collected user requirements
            product_type: Detected product type
            session_id: Session tracking ID
            schema: Optional product schema

        Returns:
            {
                'success': bool,
                'product_type': str,
                'vendor_matches': list,
                'vendor_run_details': list,
                'total_matches': int,
                'vendors_analyzed': int,
                'original_vendor_count': int,
                'filtered_vendor_count': int,
                'excluded_by_strategy': int,
                'strategy_context': dict,
                'rag_invocations': dict,
                'analysis_summary': str
            }
        """
        logger.info("[VendorAnalysisTool] Starting vendor analysis (Deep Agent)")
        logger.info("[VendorAnalysisTool] Product type: %s", product_type)
        logger.info("[VendorAnalysisTool] Session: %s", session_id or "N/A")

        # [FIX Feb 2026 #1] Include session_id in cache key for isolation
        import hashlib
        effective_session = session_id or "global"
        try:
            cache_key = hashlib.md5(
                f"{effective_session}:{product_type}:{json.dumps(structured_requirements, sort_keys=True, default=str)}".encode()
            ).hexdigest()

            # [FIX Feb 2026 #6] Use BoundedCache .get() method
            cached_result = self._response_cache.get(cache_key)
            if cached_result is not None:
                logger.info("[VendorAnalysisTool] ✓ Cache hit (session: %s) — returning cached result",
                            effective_session[:16])
                if issue_debug:
                    issue_debug.cache_hit("vendor_analysis", cache_key[:16])
                return cached_result
        except Exception as cache_check_error:
            logger.warning("[VendorAnalysisTool] Cache check failed: %s (continuing)", cache_check_error)

        # --- Build initial state ---
        initial_state: VendorAnalysisDeepAgentState = {
            "session_id": session_id,
            "product_type": product_type,
            "structured_requirements": structured_requirements,
            "schema": schema,
            "tool": self,               # nodes use self._analyze_vendor() etc.
            "components": None,
            "vendors": [],
            "strategy_context": None,
            "filtered_vendors": [],
            "vendor_priorities": {},
            "excluded_vendors": [],
            "strategy_rag_invoked": False,
            "strategy_rag_invocation_time": "",
            "products_data": {},
            "vendor_payloads": {},
            "applicable_standards": [],
            "standards_specs": "No specific standards requirements provided.",
            "vendor_matches": [],
            "run_details": [],
            "enriched_matches": [],
            "result": {},
            "current_node": "init",
            "error": None,
            "start_time": time.time(),
        }

        try:
            graph = _build_vendor_analysis_graph()
            final_state = graph.invoke(initial_state)
            result = final_state.get("result")

            if not result:
                return {
                    "success": False,
                    "product_type": product_type,
                    "session_id": session_id,
                    "error": "Deep Agent graph produced no result",
                }

            return result

        except Exception as e:
            logger.error("[VendorAnalysisTool] Deep Agent invocation failed: %s", str(e), exc_info=True)
            return {
                "success": False,
                "product_type": product_type,
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    # -------------------------------------------------------------------------
    # HELPER METHODS  (all unchanged — nodes delegate to these)
    # -------------------------------------------------------------------------

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements dictionary into structured string."""
        lines = []

        if "mandatoryRequirements" in requirements or "mandatory" in requirements:
            mandatory = requirements.get("mandatoryRequirements") or requirements.get("mandatory", {})
            if mandatory:
                lines.append("## Mandatory Requirements")
                for key, value in mandatory.items():
                    if value:
                        lines.append(f"- {self._format_field_name(key)}: {value}")

        if "optionalRequirements" in requirements or "optional" in requirements:
            optional = requirements.get("optionalRequirements") or requirements.get("optional", {})
            if optional:
                lines.append("\n## Optional Requirements")
                for key, value in optional.items():
                    if value:
                        lines.append(f"- {self._format_field_name(key)}: {value}")

        if "selectedAdvancedParams" in requirements or "advancedSpecs" in requirements:
            advanced = requirements.get("selectedAdvancedParams") or requirements.get("advancedSpecs", {})
            if advanced:
                lines.append("\n## Advanced Specifications")
                for key, value in advanced.items():
                    if value:
                        lines.append(f"- {key}: {value}")

        if not lines:
            return """## Requirements Summary
No specific mandatory or optional requirements have been provided for this product search.

## Analysis Instruction
Analyze available products and return JSON with general recommendations based on:
- Standard industrial specifications and certifications
- Product feature completeness and quality
- Typical use case suitability for this product type
- Provide match_score based on product quality (use 85-95 range for well-documented, certified products)"""

        return "\n".join(lines)

    def _format_field_name(self, field: str) -> str:
        """Convert camelCase or snake_case to Title Case."""
        import re
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", field)
        words = words.replace("_", " ")
        return words.title()

    def _prepare_payloads(
        self,
        vendors: List[str],
        pdf_content: Dict[str, str],
        products_json: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare vendor payloads for analysis."""
        payloads = {}

        for vendor in vendors:
            pdf_text = pdf_content.get(vendor, "")
            products = products_json.get(vendor, [])

            if pdf_text or products:
                payloads[vendor] = {
                    "pdf_text": pdf_text,
                    "products": products,
                }
                logger.debug("[VendorAnalysisTool] Prepared payload for '%s': PDF=%s, Products=%d",
                             vendor, bool(pdf_text), len(products) if products else 0)

        return payloads

    def _analyze_vendor(
        self,
        components: Dict[str, Any],
        requirements_str: str,
        vendor: str,
        vendor_data: Dict[str, Any],
        applicable_standards: Optional[List[str]] = None,
        standards_specs: Optional[str] = None
    ) -> tuple:
        """
        Analyze a single vendor.

        Args:
            components: LLM and other component references
            requirements_str: Formatted requirements string
            vendor: Vendor name being analyzed
            vendor_data: Vendor PDF and product data
            applicable_standards: List of applicable engineering standards
            standards_specs: Standards specifications from documents

        Returns:
            Tuple of (result dict, error string or None)
        """
        error = None
        result = None
        base_retry_delay = 15

        logger.info("[VendorAnalysisTool] START analysis for vendor: %s", vendor)

        for attempt in range(self.max_retries):
            try:
                from core.chaining import invoke_vendor_chain

                pdf_text = vendor_data.get("pdf_text", "")
                products = vendor_data.get("products", [])

                pdf_payload = json.dumps({vendor: pdf_text}, ensure_ascii=False) if pdf_text else "{}"
                products_payload = json.dumps(products, ensure_ascii=False)

                result = invoke_vendor_chain(
                    components,
                    vendor,
                    requirements_str,
                    products_payload,
                    pdf_payload,
                    components["vendor_format_instructions"],
                    applicable_standards=applicable_standards or [],
                    standards_specs=standards_specs or "No specific standards requirements provided.",
                )

                from core.chaining import to_dict_if_pydantic, parse_vendor_analysis_response
                result = to_dict_if_pydantic(result)
                result = parse_vendor_analysis_response(result, vendor)

                logger.info("[VendorAnalysisTool] END analysis for vendor: %s (success)", vendor)
                return result, None

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = any(x in error_msg.lower() for x in
                                    ["429", "resource has been exhausted", "quota", "503", "overloaded"])

                if is_rate_limit and attempt < self.max_retries - 1:
                    wait_time = base_retry_delay * (2 ** attempt)
                    logger.warning("[VendorAnalysisTool] Rate limit for %s, retry %d/%d after %ds",
                                   vendor, attempt + 1, self.max_retries, wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("[VendorAnalysisTool] Analysis failed for %s: %s", vendor, error_msg)
                    error = error_msg
                    break

        logger.info("[VendorAnalysisTool] END analysis for vendor: %s (error: %s)", vendor, error)
        return None, error


# =============================================================================
# CONVENIENCE FUNCTION (unchanged)
# =============================================================================

def analyze_vendors(
    structured_requirements: Dict[str, Any],
    product_type: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run vendor analysis.

    Args:
        structured_requirements: Collected user requirements
        product_type: Detected product type
        session_id: Session tracking ID

    Returns:
        Vendor analysis result
    """
    agent = VendorAnalysisDeepAgent()
    return agent.analyze(
        structured_requirements=structured_requirements,
        product_type=product_type,
        session_id=session_id
    )


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Alias for old class name
VendorAnalysisTool = VendorAnalysisDeepAgent
# Alias for the core analysis function (if requested elsewhere)
final_vendor_analysis = VendorAnalysisDeepAgent().analyze
