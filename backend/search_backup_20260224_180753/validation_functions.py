"""
Validation Functions Module
===========================

Simple, pure functions for validation workflow - replacing ValidationDeepAgent.

Functions:
- detect_hitl_response() - Detect YES/NO HITL responses
- extract_product_type() - Extract and normalize product type
- load_schema() - Load or generate schema
- enrich_schema_with_standards() - Multi-step standards enrichment
- validate_requirements() - Validate requirements against schema
- generate_hitl_message() - Build HITL prompt text
- Session caching helpers for HITL resume
"""

import logging
import contextvars
from typing import Dict, Any, Optional, List

from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache
from common.infrastructure.context import (
    ExecutionContext,
    get_context as get_execution_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# THREAD-LOCAL REQUEST CONTEXT
# =============================================================================

# Thread-local context for current request - prevents cross-request contamination
# NOTE: These are maintained for backward compatibility. New code should use ExecutionContext.
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    'current_session_id', default=''
)
_current_workflow_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    'current_workflow_thread_id', default=''
)


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


def clear_request_context():
    """Clear thread-local context after request completes."""
    _current_session_id.set('')
    _current_workflow_thread_id.set('')


def clear_session_enrichment_cache():
    """Clear session enrichment cache (call at start of new session)."""
    count = _session_enrichment_cache.clear()
    logger.info(f"[ValidationFunctions] Session enrichment cache cleared ({count} entries)")


# =============================================================================
# SESSION CACHES
# =============================================================================

_session_enrichment_cache: BoundedCache = get_or_create_cache(
    name="validation_enrichment",
    max_size=200,
    ttl_seconds=0  # [FIX Feb 2026] No TTL — persist for session lifetime (LRU eviction still caps memory)
)

_session_context_cache: BoundedCache = get_or_create_cache(
    name="validation_context",
    max_size=500,
    ttl_seconds=0  # [FIX Feb 2026] No TTL — persist for session lifetime
)


# =============================================================================
# CONSTANTS
# =============================================================================

BARE_MEASUREMENT_FIX = {
    "level": "Level Transmitter",
    "pressure": "Pressure Transmitter",
    "flow": "Flow Meter",
    "temperature": "Temperature Transmitter",
    "differential pressure": "Differential Pressure Transmitter",
    "dp": "Differential Pressure Transmitter",
}

MIN_STANDARDS_SPECS_COUNT = 60


# =============================================================================
# SESSION CACHING HELPERS
# =============================================================================

def cache_session_context(session_id: str, context: Dict[str, Any]) -> None:
    """
    Cache session context for HITL resume.

    Args:
        session_id: Session identifier
        context: Dict containing product_type, schema, provided_requirements, etc.
    """
    if not session_id:
        logger.warning("[ValidationFunctions] cache_session_context called without session_id")
        return
    key = f"context:{session_id}"
    _session_context_cache.set(key, context)
    logger.debug(f"[ValidationFunctions] Cached context for {key}")


def get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached session context for HITL responses.

    Args:
        session_id: Session identifier

    Returns:
        Cached context dict or None if not found
    """
    if not session_id:
        return None
    key = f"context:{session_id}"
    return _session_context_cache.get(key)


def get_session_enrichment(product_type: str, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached enrichment result for this session.

    Args:
        product_type: Product type to look up
        session_id: Session identifier

    Returns:
        Cached enrichment result or None
    """
    if not session_id or not product_type:
        return None
    normalized_type = product_type.lower().strip()
    key = f"enrichment:{session_id}:{normalized_type}"
    return _session_enrichment_cache.get(key)


def cache_session_enrichment(product_type: str, result: Dict[str, Any], session_id: str) -> None:
    """
    Cache enrichment result for this session.

    Args:
        product_type: Product type to cache
        result: Enrichment data to cache
        session_id: Session identifier
    """
    if not session_id or not product_type:
        return
    normalized_type = product_type.lower().strip()
    key = f"enrichment:{session_id}:{normalized_type}"
    _session_enrichment_cache.set(key, result)
    logger.debug(f"[ValidationFunctions] Cached enrichment for {key}")


def clear_session_cache(session_id: str) -> None:
    """Clear all cached data for a specific session."""
    if not session_id:
        return
    context_key = f"context:{session_id}"
    _session_context_cache.delete(context_key)
    logger.info(f"[ValidationFunctions] Cleared session cache for: {session_id}")


# =============================================================================
# FUNCTION 1: HITL DETECTION
# =============================================================================

def detect_hitl_response(
    user_input: str,
    session_id: str,
    expected_product_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect YES/NO HITL responses and retrieve cached context.

    Args:
        user_input: User's input text
        session_id: Session identifier
        expected_product_type: Optional expected product type

    Returns:
        {
            "is_hitl_yes": bool,
            "is_hitl_no": bool,
            "product_type": str,
            "cached_context": dict,
            "validation_bypassed": bool
        }
    """
    input_lower = user_input.lower().strip()

    is_yes = input_lower in ["yes", "y"]
    is_no = input_lower in ["no", "n"] or input_lower.startswith("no ")
    is_skip = input_lower in ["skip", "s"] or input_lower.startswith("skip ")

    result = {
        "is_hitl_yes": False,
        "is_hitl_no": False,
        "is_skip": False,
        "product_type": "",
        "cached_context": {},
        "validation_bypassed": False
    }

    if is_yes:
        product_type = expected_product_type
        cached_context = get_session_context(session_id) or {}

        if not product_type and cached_context.get("product_type"):
            product_type = cached_context["product_type"]

        if product_type:
            result["is_hitl_yes"] = True
            result["product_type"] = product_type
            result["cached_context"] = cached_context
            result["validation_bypassed"] = True
            logger.info(f"[ValidationFunctions] HITL YES detected for {product_type}")
        else:
            result["is_hitl_yes"] = True
            result["error"] = "Missing product context for YES response"

    elif is_no:
        product_type = expected_product_type
        cached_context = get_session_context(session_id) or {}

        if not product_type and cached_context.get("product_type"):
            product_type = cached_context["product_type"]

        if product_type:
            result["is_hitl_no"] = True
            result["product_type"] = product_type
            result["cached_context"] = cached_context
            logger.info(f"[ValidationFunctions] HITL NO detected for {product_type}")
        else:
            result["is_hitl_no"] = True
            result["error"] = "Missing product context for NO response"

    elif is_skip:
        # SKIP is treated same as NO (skip adding additional specs)
        product_type = expected_product_type
        cached_context = get_session_context(session_id) or {}

        if not product_type and cached_context.get("product_type"):
            product_type = cached_context["product_type"]

        if product_type:
            result["is_skip"] = True
            result["is_hitl_no"] = True  # Treat SKIP as NO for routing
            result["product_type"] = product_type
            result["cached_context"] = cached_context
            logger.info(f"[ValidationFunctions] HITL SKIP detected for {product_type} (treated as NO)")
        else:
            result["is_skip"] = True
            result["is_hitl_no"] = True
            result["error"] = "Missing product context for SKIP response"

    return result


# =============================================================================
# FUNCTION 2: PRODUCT TYPE EXTRACTION
# =============================================================================

def extract_product_type(
    user_input: str,
    expected_product_type: Optional[str] = None,
    source_workflow: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract and normalize product type using extract_requirements_tool.

    Applies BARE_MEASUREMENT_FIX dict for bare measurement words.

    Args:
        user_input: User's requirement text
        expected_product_type: Optional expected product type
        source_workflow: Source workflow identifier

    Returns:
        {
            "success": bool,
            "product_type": str,
            "original_product_type": str,
            "error": str (if failed)
        }
    """
    logger.info("[ValidationFunctions] Extracting product type")

    try:
        from common.tools.intent_tools import extract_requirements_tool

        extract_result = extract_requirements_tool.invoke({
            "user_input": user_input
        })

        # Use expected type for instrument_identifier workflow
        if source_workflow == "instrument_identifier" and expected_product_type:
            product_type = expected_product_type
        else:
            product_type = extract_result.get("product_type") or expected_product_type or ""

        original_product_type = product_type

        # Normalize bare measurement words
        if product_type.lower().strip() in BARE_MEASUREMENT_FIX:
            original_pt = product_type
            product_type = BARE_MEASUREMENT_FIX[product_type.lower().strip()]

            # Special case handling for flow
            if original_pt.lower() == "flow":
                if "transmitter" in user_input.lower() and "meter" not in user_input.lower():
                    product_type = "Flow Transmitter"
                elif "switch" in user_input.lower():
                    product_type = "Flow Switch"

            # Special case handling for level
            elif original_pt.lower() == "level":
                if "switch" in user_input.lower():
                    product_type = "Level Switch"
                elif "gauge" in user_input.lower():
                    product_type = "Level Gauge"

            logger.info(f"[ValidationFunctions] Normalized '{original_pt}' -> '{product_type}'")

        if not product_type:
            return {
                "success": False,
                "product_type": "",
                "original_product_type": "",
                "error": "Could not determine product type from input"
            }

        logger.info(f"[ValidationFunctions] Extracted product type: {product_type}")

        return {
            "success": True,
            "product_type": product_type,
            "original_product_type": original_product_type
        }

    except Exception as e:
        logger.error(f"[ValidationFunctions] Product type extraction error: {e}")
        return {
            "success": False,
            "product_type": "",
            "original_product_type": "",
            "error": str(e)
        }


# =============================================================================
# FUNCTION 3: SCHEMA LOADING
# =============================================================================

def load_schema(
    product_type: str,
    enable_ppi: bool = True
) -> Dict[str, Any]:
    """
    Load or generate schema using load_schema_tool.

    Args:
        product_type: Product type to load schema for
        enable_ppi: Enable PPI workflow for schema generation

    Returns:
        {
            "success": bool,
            "schema": dict,
            "schema_source": str,
            "ppi_workflow_used": bool,
            "from_database": bool
        }
    """
    logger.info(f"[ValidationFunctions] Loading schema for: {product_type}")

    try:
        from common.tools.schema_tools import load_schema_tool

        schema_result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": enable_ppi
        })

        schema = schema_result.get("schema", {})
        schema_source = schema_result.get("source", "unknown")
        ppi_used = schema_result.get("ppi_used", False)
        from_database = schema_result.get("from_database", False)

        if from_database:
            logger.info("[ValidationFunctions] Schema loaded from database")
        elif ppi_used:
            logger.info("[ValidationFunctions] Schema generated via PPI workflow")
        else:
            logger.warning("[ValidationFunctions] Using fallback schema")

        return {
            "success": True,
            "schema": schema,
            "schema_source": schema_source,
            "ppi_workflow_used": ppi_used,
            "from_database": from_database
        }

    except Exception as e:
        logger.error(f"[ValidationFunctions] Schema loading error: {e}")
        return {
            "success": False,
            "schema": {},
            "schema_source": "error",
            "ppi_workflow_used": False,
            "from_database": False,
            "error": str(e)
        }


# =============================================================================
# FUNCTION 4: SCHEMA ENRICHMENT
# =============================================================================

def _fallback_simple_enrichment(product_type: str, schema: dict) -> Dict[str, Any]:
    """Fallback to simple enrichment when Standards Deep Agent fails."""
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
            return {"success": True, "result": enrichment_result}

    except Exception as e:
        logger.debug(f"[ValidationFunctions] Fallback enrichment failed: {e}")

    return {"success": False, "result": {"source": "fallback_failed"}}


def enrich_schema_with_standards(
    product_type: str,
    schema: Dict[str, Any],
    session_id: str
) -> Dict[str, Any]:
    """
    Multi-step enrichment using Standards Deep Agent.

    Steps:
    1. Check session cache
    2. populate_schema_fields_from_standards()
    3. get_applicable_standards()
    4. run_standards_deep_agent() with min_specs=60
    5. Fallback to simple_enrichment if deep agent fails
    6. Cache result

    Args:
        product_type: Product type to enrich
        schema: Schema to enrich
        session_id: Session identifier

    Returns:
        {
            "success": bool,
            "schema": dict (enriched),
            "standards_info": dict,
            "enrichment_result": dict,
            "cache_hit": bool
        }
    """
    import datetime

    logger.info(f"[ValidationFunctions] Enriching schema for: {product_type}")

    try:
        # Check session cache first
        cached_enrichment = get_session_enrichment(product_type, session_id)

        if cached_enrichment:
            logger.info(f"[ValidationFunctions] Cache HIT for {product_type}")
            return {
                "success": True,
                "schema": cached_enrichment.get("schema", schema),
                "standards_info": cached_enrichment.get("standards_info", {}),
                "enrichment_result": cached_enrichment.get("enrichment_result", {}),
                "cache_hit": True
            }

        logger.info(f"[ValidationFunctions] Cache MISS for {product_type}")

        # Make a copy to avoid mutating original
        schema = dict(schema)
        standards_info = {}
        enrichment_result = {}

        # Sub-step 1: Populate standards fields
        try:
            from common.tools.standards_enrichment_tool import populate_schema_fields_from_standards
            schema = populate_schema_fields_from_standards(product_type, schema)
        except Exception as e:
            logger.warning(f"[ValidationFunctions] Standards field population failed: {e}")

        # Sub-step 2: Get applicable standards
        try:
            from common.tools.standards_enrichment_tool import get_applicable_standards
            standards_info = get_applicable_standards(product_type, top_k=5)

            if standards_info.get("success"):
                if "standards" not in schema:
                    schema["standards"] = {
                        "applicable_standards": standards_info.get("applicable_standards", []),
                        "certifications": standards_info.get("certifications", []),
                        "sources": standards_info.get("sources", []),
                    }
        except Exception as e:
            logger.warning(f"[ValidationFunctions] Standards retrieval failed: {e}")
            standards_info = {}

        # Sub-step 3: Run Standards Deep Agent with iterative loop for min 60 specs
        try:
            from common.standards.generation.deep_agent import run_standards_deep_agent

            logger.info(f"[ValidationFunctions] Running Standards Deep Agent (min: {MIN_STANDARDS_SPECS_COUNT} specs)")

            deep_agent_result = run_standards_deep_agent(
                user_requirement=f"Product type: {product_type}. Specifications needed for: {product_type}",
                session_id=session_id,
                inferred_specs=schema.get("mandatory", {}),
                min_specs=MIN_STANDARDS_SPECS_COUNT
            )

            if deep_agent_result and deep_agent_result.get("success"):
                final_specs = deep_agent_result.get("final_specifications", {})
                specs_dict = final_specs.get("specifications", {}) if isinstance(final_specs, dict) else {}
                specs_count = len(specs_dict)

                logger.info(f"[ValidationFunctions] Standards Deep Agent generated {specs_count} specs")

                # Merge deep agent specs into schema
                if "deep_agent_specs" not in schema:
                    schema["deep_agent_specs"] = {}
                schema["deep_agent_specs"].update(specs_dict)

                # Update mandatory fields
                if "mandatory" in schema and isinstance(schema["mandatory"], dict):
                    for key, value in specs_dict.items():
                        if key not in schema["mandatory"] and value and str(value).lower() not in ["null", "none"]:
                            schema["mandatory"][key] = {"value": value, "source": "standards_deep_agent"}

                enrichment_result = {
                    "success": True,
                    "specifications_count": specs_count,
                    "target_reached": final_specs.get("target_reached", False),
                    "iterations": deep_agent_result.get("iterations_performed", 1),
                    "source": "standards_deep_agent"
                }

                if final_specs.get("normalized_category"):
                    schema["normalized_category"] = final_specs["normalized_category"]
            else:
                logger.warning("[ValidationFunctions] Standards Deep Agent returned no results, using fallback")
                fallback = _fallback_simple_enrichment(product_type, schema)
                enrichment_result = fallback.get("result", {})

        except ImportError as ie:
            logger.warning(f"[ValidationFunctions] Standards Deep Agent not available: {ie}")
            fallback = _fallback_simple_enrichment(product_type, schema)
            enrichment_result = fallback.get("result", {})
        except Exception as e:
            logger.error(f"[ValidationFunctions] Standards Deep Agent error: {e}")
            fallback = _fallback_simple_enrichment(product_type, schema)
            enrichment_result = fallback.get("result", {})

        # Cache enrichment result
        cache_data = {
            "standards_info": standards_info,
            "enrichment_result": enrichment_result,
            "schema": schema
        }
        cache_session_enrichment(product_type, cache_data, session_id)

        return {
            "success": True,
            "schema": schema,
            "standards_info": standards_info,
            "enrichment_result": enrichment_result,
            "cache_hit": False
        }

    except Exception as e:
        logger.error(f"[ValidationFunctions] Standards enrichment error: {e}")
        return {
            "success": False,
            "schema": schema,
            "standards_info": {},
            "enrichment_result": {},
            "cache_hit": False,
            "error": str(e)
        }


# =============================================================================
# FUNCTION 5: REQUIREMENTS VALIDATION
# =============================================================================

def validate_requirements(
    user_input: str,
    product_type: str,
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate user requirements against schema.

    Args:
        user_input: User's requirement text
        product_type: Product type
        schema: Schema to validate against

    Returns:
        {
            "success": bool,
            "provided_requirements": dict,
            "missing_fields": list,
            "optional_fields": list,
            "is_valid": bool
        }
    """
    logger.info("[ValidationFunctions] Validating requirements")

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

        return {
            "success": True,
            "provided_requirements": provided_requirements,
            "missing_fields": validation_result.get("missing_fields", []),
            "optional_fields": validation_result.get("optional_fields", []),
            "is_valid": validation_result.get("is_valid", False)
        }

    except Exception as e:
        logger.error(f"[ValidationFunctions] Validation error: {e}")
        return {
            "success": False,
            "provided_requirements": {},
            "missing_fields": [],
            "optional_fields": [],
            "is_valid": False,
            "error": str(e)
        }


# =============================================================================
# FUNCTION 6: HITL MESSAGE GENERATION
# =============================================================================

def generate_hitl_message(
    product_type: str,
    provided_requirements: Dict[str, Any],
    missing_fields: List[str]
) -> str:
    """
    Generate HITL prompt text for YES/NO confirmation.

    Args:
        product_type: Product type
        provided_requirements: Dict of provided requirements
        missing_fields: List of missing field names

    Returns:
        Human-readable HITL message
    """
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

    return hitl_message


# =============================================================================
# COMBINED VALIDATION FUNCTION
# =============================================================================

def run_validation(
    user_input: str,
    session_id: str,
    expected_product_type: Optional[str] = None,
    enable_ppi: bool = True,
    enable_standards_enrichment: bool = True,
    source_workflow: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete validation flow - combines all functions.

    This is the main entry point that replaces ValidationDeepAgent.validate().

    Args:
        user_input: User's requirement text
        session_id: Session identifier
        expected_product_type: Optional expected product type
        enable_ppi: Enable PPI workflow
        enable_standards_enrichment: Enable standards enrichment
        source_workflow: Source workflow identifier

    Returns:
        Complete validation result dict
    """
    import time
    start_time = time.time()

    # Step 1: Extract product type
    extract_result = extract_product_type(
        user_input=user_input,
        expected_product_type=expected_product_type,
        source_workflow=source_workflow
    )

    if not extract_result["success"]:
        return {
            "success": False,
            "error": extract_result.get("error", "Product type extraction failed"),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

    product_type = extract_result["product_type"]

    # Step 2: Load schema
    schema_result = load_schema(product_type, enable_ppi)
    schema = schema_result.get("schema", {})

    # Step 3: Enrich schema with standards
    if enable_standards_enrichment:
        enrich_result = enrich_schema_with_standards(product_type, schema, session_id)
        schema = enrich_result.get("schema", schema)
        standards_info = enrich_result.get("standards_info", {})
    else:
        standards_info = {}

    # Step 4: Validate requirements
    validation_result = validate_requirements(user_input, product_type, schema)
    provided_requirements = validation_result.get("provided_requirements", {})
    missing_fields = validation_result.get("missing_fields", [])
    is_valid = validation_result.get("is_valid", False)

    # Step 5: Generate HITL message
    hitl_message = generate_hitl_message(product_type, provided_requirements, missing_fields)

    # Step 6: Cache session context
    cache_session_context(session_id, {
        "product_type": product_type,
        "schema": schema,
        "provided_requirements": provided_requirements,
        "missing_fields": missing_fields,
        "is_valid": is_valid
    })

    processing_time_ms = int((time.time() - start_time) * 1000)

    return {
        "success": True,
        "session_id": session_id,
        "product_type": product_type,
        "original_product_type": extract_result.get("original_product_type"),
        "schema": schema,
        "provided_requirements": provided_requirements,
        "missing_fields": missing_fields,
        "optional_fields": validation_result.get("optional_fields", []),
        "is_valid": is_valid,
        "hitl_message": hitl_message,
        "ppi_workflow_used": schema_result.get("ppi_workflow_used", False),
        "schema_source": schema_result.get("schema_source", "unknown"),
        "from_database": schema_result.get("from_database", False),
        "standards_info": standards_info,
        "processing_time_ms": processing_time_ms
    }


# =============================================================================
# BACKWARD COMPATIBILITY CLASSES
# =============================================================================

class ValidationTool:
    """
    Backward-compatible wrapper class.

    Provides the .validate() method API that old code expects:
        tool = ValidationTool(enable_ppi=True)
        result = tool.validate(user_input="...", session_id="...")
    """

    def __init__(self, enable_ppi: bool = True):
        """Initialize with PPI flag."""
        self._enable_ppi = enable_ppi

    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_standards_enrichment: bool = True,
        source_workflow: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Delegate to the simplified run_validation function."""
        return run_validation(
            user_input=user_input,
            session_id=session_id or "unknown",
            expected_product_type=expected_product_type,
            enable_ppi=self._enable_ppi,
            enable_standards_enrichment=enable_standards_enrichment,
            source_workflow=source_workflow,
            **kwargs
        )

    def get_schema_only(
        self,
        product_type: str,
        enable_ppi: bool = True
    ) -> Dict[str, Any]:
        """Get schema without validation."""
        return load_schema(product_type, enable_ppi=enable_ppi)

    def validate_with_schema(
        self,
        user_input: str,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate input against provided schema."""
        return validate_requirements(user_input, product_type, schema)


# Alias for code that imports ValidationDeepAgent
ValidationDeepAgent = ValidationTool
