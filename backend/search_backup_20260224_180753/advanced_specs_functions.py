"""
Advanced Specs Functions Module
===============================

Simple, pure functions for advanced specification discovery - replacing AdvancedSpecificationAgent.

Functions:
- get_existing_schema_params() - Load existing params to avoid duplicates
- check_advanced_specs_cache() - Two-tier cache check (in-memory + Azure Blob)
- discover_advanced_specs_llm() - LLM call for parameter discovery
- parse_advanced_specs_response() - Parse + deduplicate LLM response
- persist_advanced_specs() - Save to cache + Azure Blob
- discover_advanced_specs() - Main entry point (replaces AdvancedSpecificationAgent.discover)
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.config.azure_blob_config import get_azure_blob_connection
from common.config import AgenticConfig
from common.services.llm.fallback import create_llm_with_fallback
from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

logger = logging.getLogger(__name__)


# =============================================================================
# CACHES
# =============================================================================

_SPEC_CACHE: BoundedCache = get_or_create_cache(
    name="advanced_spec_cache",
    max_size=500,
    ttl_seconds=600  # 10 min
)

_SCHEMA_PARAM_CACHE: BoundedCache = get_or_create_cache(
    name="schema_param_cache",
    max_size=500,
    ttl_seconds=0  # [FIX Feb 2026] No TTL — schema params persist for session lifetime
)

_COLLECTION_CACHE = None  # Module-level Azure Blob collection singleton


# =============================================================================
# PROMPTS
# =============================================================================

_ADV_PARAMS_PROMPTS = {
    "GENERIC_SPECIFICATIONS": """You are an industrial instrumentation expert identifying advanced parameters and features for specific product types.

PRODUCT_TYPE: {product_type}
CATEGORY: {category}

TASK: Generate 20-30 specifications covering:
- Performance: accuracy, repeatability, range, linearity
- Physical: dimensions, weight, materials, mounting
- Electrical: output, voltage, power, protocols
- Environmental: IP rating, temperature, humidity, vibration
- Compliance: SIL, ATEX, certifications
- Installation: connection, calibration, response time

RULES:
- Clean technical values only (no descriptions)
- Confidence scores 0.0-1.0
- Relevant to product type

OUTPUT (JSON):
{{
  "specifications": {{
    "accuracy": {{"value": "±0.1%", "confidence": 0.9}},
    "pressure_range": {{"value": "0-100 bar", "confidence": 0.9}},
    "output_signal": {{"value": "4-20mA with HART", "confidence": 0.95}}
  }},
  "total_specs": <count>
}}""",

    "PARAMETER_DISCOVERY": """You are an industrial instrumentation expert identifying advanced parameters and features for specific product types.

PRODUCT_TYPE: {product_type}
EXISTING_PARAMETERS: {existing_parameters}

TASK: Generate 10-15 advanced, product-specific parameters NOT in existing parameters.
Focus on cutting-edge features and innovations highly relevant to {product_type}.

ADVANCED FEATURE CATEGORIES (prioritize product-specific ones):
- AI/ML: Predictive diagnostics, anomaly detection, self-learning algorithms, AI-driven drift compensation, pattern recognition
- Wireless/IoT: Bluetooth Low Energy, LoRaWAN, 5G connectivity, mesh networking, WirelessHART, remote configuration
- Cloud Integration: Edge computing, remote monitoring dashboards, cloud analytics, OPC UA, predictive maintenance platforms
- Advanced Diagnostics: Self-calibration, health monitoring, predictive maintenance, in-situ verification, auto-diagnostics
- Energy Efficiency: Low-power modes, energy harvesting, solar-powered operation, ultra-low power consumption
- Cybersecurity: Encrypted protocols, secure boot, certificate-based authentication, firewall integration, intrusion detection
- Digital Twin: Digital twin integration, virtual commissioning, simulation capabilities, real-time mirroring
- Augmented Reality: AR-based maintenance, QR code diagnostics, mobile app integration, remote expert assistance
- Process-Specific: Product-dependent advanced features tailored to measurement principles and applications

PRODUCT-SPECIFIC EXAMPLES TO GUIDE YOU:
- Pressure Transmitter: "AI-Powered Sensor Drift Compensation", "Wireless Mesh Networking", "Integrated Process Seal Leak Detection", "Real-Time Impulse Line Blockage Detection", "Predictive Diaphragm Failure Analysis"
- Flow Meter: "Multi-Phase Flow Profiling", "Predictive Fouling Detection", "Edge-Based Flow Pattern Analysis", "AI-Driven Density Compensation", "Low-Flow Sensitivity Enhancement"
- Level Transmitter: "Adaptive Dielectric Compensation (AI-driven)", "Remote Tank Strapping Integration", "Multi-Echo Processing", "Foam & Turbulence Suppression", "Guided Wave Radar Multi-Path Rejection"
- Temperature Sensor: "Predictive Thermowell Failure Detection", "Self-Diagnosing RTD Integrity", "Wireless Temperature Mesh", "Vibration-Compensated Measurements", "Fast Response Thin-Film Technology"

RULES:
- NO duplication with existing parameters
- Generate EXACTLY 10-15 parameters (not fewer!)
- Focus on innovations from past 12-24 months
- Use human-readable names with proper capitalization (e.g., "AI-Powered Drift Compensation")
- Each parameter MUST be SPECIFIC to {product_type} - avoid generic features
- Prioritize measurement-principle-specific advanced features over generic IoT capabilities
- Consider unique challenges for this product type and how innovations address them

OUTPUT (JSON):
{{
  "advanced_parameters": [
    "<parameter 1 specific to {product_type}>",
    "<parameter 2 specific to {product_type}>",
    "<parameter 3 specific to {product_type}>",
    ... (10-15 total parameters)
  ],
  "innovation_justification": "<brief explanation of why these innovations are particularly relevant for {product_type}>",
  "total_count": <10-15>
}}"""
}


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_product_type(product_type: str) -> str:
    """Normalize product type for cache keys."""
    return re.sub(r"[^a-z0-9]", "", product_type.lower()) if product_type else ""


def _get_azure_collection():
    """Get or create Azure Blob collection singleton."""
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is not None:
        return _COLLECTION_CACHE
    try:
        conn = get_azure_blob_connection()
        if hasattr(conn, 'get_collection'):
            _COLLECTION_CACHE = conn.get_collection("advanced_parameters")
        else:
            _COLLECTION_CACHE = None
    except Exception as exc:
        logger.warning("[AdvancedSpecsFunctions] Azure collection unavailable: %s", exc)
        _COLLECTION_CACHE = None
    return _COLLECTION_CACHE


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Robustly extract a JSON object or array from raw LLM output."""
    if not raw:
        return {}

    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    # Strategy 1: Full text parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Strategy 2: Find first JSON object {...}
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(cleaned[s:e + 1])
        except Exception:
            pass

    # Strategy 3: Find JSON array [...]
    s, e = cleaned.find("["), cleaned.rfind("]")
    if s != -1 and e > s:
        try:
            arr = json.loads(cleaned[s:e + 1])
            return {"parameters": arr} if isinstance(arr, list) else {}
        except Exception:
            pass

    # Strategy 4: Fallback line-by-line parsing
    lines = [l.strip(" -*.\t") for l in cleaned.splitlines() if l.strip() and len(l.strip()) > 3]
    skip = ("return", "task", "rules", "rules:")
    items = [l for l in lines if not l.lower().startswith(skip)]
    if items:
        return {"parameters": items}

    logger.warning("[AdvancedSpecsFunctions] Could not parse LLM response as JSON")
    return {}


def _build_result(product_type: str, specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build standard result dict."""
    return {
        "product_type": product_type,
        "vendor_specifications": [],
        "vendor_parameters": [],
        "unique_specifications": specs,
        "unique_parameters": specs,
        "total_vendors_searched": 0,
        "total_unique_specifications": len(specs),
        "total_unique_parameters": len(specs),
        "existing_specifications_filtered": 0,
        "discovery_successful": len(specs) > 0,
    }


# =============================================================================
# FUNCTION 1: GET EXISTING SCHEMA PARAMS
# =============================================================================

def get_existing_schema_params(product_type: str) -> set:
    """
    Load existing schema parameters to avoid duplicates.

    Args:
        product_type: Product type to load params for

    Returns:
        Set of existing parameter names
    """
    if not product_type:
        return set()

    normalized = _normalize_product_type(product_type)

    # Check cache first
    cached = _SCHEMA_PARAM_CACHE.get(normalized)
    if cached is not None:
        return cached

    try:
        from common.core.loading import load_requirements_schema

        schema = load_requirements_schema(product_type)
        if not schema:
            return set()

        params: set = set()
        for req_type in ["mandatory_requirements", "optional_requirements"]:
            for fields in schema.get(req_type, {}).values():
                if isinstance(fields, dict):
                    params.update(fields.keys())

        _SCHEMA_PARAM_CACHE.set(normalized, params)
        return params

    except Exception as exc:
        logger.warning("[AdvancedSpecsFunctions] Schema load failed for %s: %s", product_type, exc)
        return set()


# =============================================================================
# FUNCTION 2: CHECK CACHE (TWO-TIER)
# =============================================================================

def check_advanced_specs_cache(product_type: str) -> Optional[List[Dict[str, Any]]]:
    """
    Two-tier cache check: in-memory first, then Azure Blob.

    Args:
        product_type: Product type to look up

    Returns:
        Cached specs list or None for cache miss
    """
    if not product_type:
        return None

    normalized = _normalize_product_type(product_type)

    # In-memory first (fastest)
    in_mem = _SPEC_CACHE.get(normalized)
    if in_mem is not None:
        logger.info("[AdvancedSpecsFunctions] In-memory cache HIT (%d specs)", len(in_mem))
        return in_mem

    # Azure Blob fallback
    try:
        collection = _get_azure_collection()
        if collection is not None:
            doc = collection.find_one({"normalized_product_type": normalized})
            if doc:
                specs = doc.get("unique_specifications")
                if isinstance(specs, list) and specs:
                    logger.info("[AdvancedSpecsFunctions] Azure cache HIT (%d specs)", len(specs))
                    # Warm in-memory cache
                    _SPEC_CACHE.set(normalized, specs)
                    return specs
    except Exception as exc:
        logger.warning("[AdvancedSpecsFunctions] Azure cache check failed: %s", exc)

    logger.info("[AdvancedSpecsFunctions] Cache MISS - will invoke LLM")
    return None


# =============================================================================
# FUNCTION 3: LLM DISCOVERY
# =============================================================================

def discover_advanced_specs_llm(
    product_type: str,
    existing_params: set
) -> str:
    """
    Single LLM call with PARAMETER_DISCOVERY prompt.

    Args:
        product_type: Product type to discover params for
        existing_params: Set of existing parameters to exclude

    Returns:
        Raw LLM response string
    """
    logger.info("[AdvancedSpecsFunctions] Invoking LLM for: %s", product_type)

    try:
        prompt_text = _ADV_PARAMS_PROMPTS.get("PARAMETER_DISCOVERY", "")
        if not prompt_text:
            raise ValueError("PARAMETER_DISCOVERY prompt missing")

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | StrOutputParser()
        existing_list = sorted(list(existing_params))

        raw = chain.invoke({
            "product_type": product_type,
            "existing_parameters": json.dumps(existing_list),
        })

        logger.info("[AdvancedSpecsFunctions] LLM response length: %d chars", len(str(raw)))
        return raw

    except Exception as e:
        logger.error("[AdvancedSpecsFunctions] LLM call failed: %s", e)
        return ""


def _fallback_generic_specs(product_type: str) -> List[Dict[str, Any]]:
    """Fallback: use generic specification generation when LLM call fails."""
    logger.warning("[AdvancedSpecsFunctions] Using generic specs fallback for %s", product_type)

    try:
        prompt_text = _ADV_PARAMS_PROMPTS.get("GENERIC_SPECIFICATIONS", "")
        if not prompt_text:
            return []

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        prompt = ChatPromptTemplate.from_template(prompt_text)
        raw = (prompt | llm | StrOutputParser()).invoke({
            "product_type": product_type,
            "category": product_type,
        })

        params = json.loads(raw.strip().replace("```json", "").replace("```", ""))
        if isinstance(params, list):
            return [
                {"key": re.sub(r"\s+", "_", str(p).lower().strip()), "name": str(p)}
                for p in params[:15]
            ]
    except Exception as fe:
        logger.warning("[AdvancedSpecsFunctions] Generic fallback also failed: %s", fe)

    return []


# =============================================================================
# FUNCTION 4: PARSE RESPONSE
# =============================================================================

def parse_advanced_specs_response(
    raw_response: str,
    existing_params: set,
    product_type: str
) -> List[Dict[str, Any]]:
    """
    Parse LLM output into deduplicated list of {key, name} dicts.

    Args:
        raw_response: Raw LLM response
        existing_params: Set of existing params to filter out
        product_type: Product type (for fallback)

    Returns:
        List of unique spec dicts: [{"key": str, "name": str}, ...]
    """
    # LLM call failed or empty - attempt fallback
    if not raw_response:
        return _fallback_generic_specs(product_type)

    payload = _extract_json_object(raw_response)

    # Support both "parameters" and "advanced_parameters" keys
    if isinstance(payload, dict):
        raw_params = payload.get("parameters") or payload.get("advanced_parameters") or []
    else:
        raw_params = []

    logger.info("[AdvancedSpecsFunctions] Extracted %d raw params from LLM", len(raw_params))

    # Deduplicate against existing params
    existing_norm = {p.lower().replace("_", "") for p in existing_params}
    seen_norm: set = set()
    unique_specs: List[Dict[str, Any]] = []

    for name in raw_params:
        if not isinstance(name, str):
            continue
        human_name = name.strip()
        if not human_name:
            continue

        # Build deterministic snake_case key
        key = re.sub(r"[^a-z0-9 ]", "", human_name.lower())
        key = re.sub(r"\s+", "_", key).strip("_")
        norm = key.replace("_", "")

        if not key or norm in existing_norm or norm in seen_norm:
            continue

        unique_specs.append({"key": key, "name": human_name})
        seen_norm.add(norm)

    logger.info("[AdvancedSpecsFunctions] %d unique specs after deduplication", len(unique_specs))
    return unique_specs


# =============================================================================
# FUNCTION 5: PERSIST RESULTS
# =============================================================================

def persist_advanced_specs(
    product_type: str,
    specs: List[Dict[str, Any]],
    existing_params: set
) -> bool:
    """
    Save to Azure Blob + warm in-memory cache.

    Args:
        product_type: Product type
        specs: List of specs to persist
        existing_params: Existing params (for snapshot)

    Returns:
        True if persisted successfully
    """
    if not specs:
        logger.info("[AdvancedSpecsFunctions] No specs to persist - skipping")
        return False

    normalized = _normalize_product_type(product_type)
    now = datetime.utcnow()
    existing_list = sorted(list(existing_params))

    # Warm in-memory cache
    _SPEC_CACHE.set(normalized, specs)

    # Persist to Azure Blob
    try:
        collection = _get_azure_collection()
        if collection is not None:
            doc = {
                "product_type": product_type,
                "normalized_product_type": normalized,
                "unique_specifications": specs,
                "existing_parameters_snapshot": existing_list,
                "created_at": now,
                "updated_at": now,
            }
            collection.update_one(
                {"normalized_product_type": normalized},
                {"$set": doc},
                upsert=True,
            )
            logger.info("[AdvancedSpecsFunctions] Persisted %d specs to Azure", len(specs))
            return True
        else:
            logger.debug("[AdvancedSpecsFunctions] Azure unavailable - in-memory only")
            return True  # In-memory still warm
    except Exception as exc:
        logger.warning("[AdvancedSpecsFunctions] Azure persist failed: %s", exc)
        return True  # In-memory still warm


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def discover_advanced_specs(
    product_type: str,
    session_id: Optional[str] = None,
    existing_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main function to discover advanced specs.
    Replaces AdvancedSpecificationAgent.discover().

    Flow:
    1. get_existing_schema_params()
    2. check_advanced_specs_cache() - return if hit
    3. discover_advanced_specs_llm()
    4. parse_advanced_specs_response()
    5. persist_advanced_specs()
    6. Return result dict

    Args:
        product_type: Product type to discover params for
        session_id: Optional session identifier
        existing_schema: Optional current schema

    Returns:
        {
            "success": bool,
            "product_type": str,
            "unique_specifications": [{"key": str, "name": str}, ...],
            "total_unique_specifications": int,
            "discovery_successful": bool,
            "fallback_used": bool,
            "session_id": str | None
        }
    """
    logger.info("[AdvancedSpecsFunctions] Starting discovery for: %s", product_type)

    if not product_type:
        return {
            "success": False,
            "product_type": product_type,
            "unique_specifications": [],
            "total_unique_specifications": 0,
            "discovery_successful": False,
            "error": "product_type is required",
        }

    try:
        # Step 1: Load existing params
        existing_params = get_existing_schema_params(product_type)
        logger.info("[AdvancedSpecsFunctions] Loaded %d existing params", len(existing_params))

        # Step 2: Check cache
        cached_specs = check_advanced_specs_cache(product_type)
        if cached_specs is not None:
            result = _build_result(product_type, cached_specs)
            result["success"] = True
            result["fallback_used"] = False
            result["session_id"] = session_id
            result["cache_hit"] = True
            return result

        # Step 3: LLM discovery
        raw_response = discover_advanced_specs_llm(product_type, existing_params)

        # Step 4: Parse response
        specs = parse_advanced_specs_response(raw_response, existing_params, product_type)
        fallback_used = not raw_response and len(specs) > 0

        # Step 5: Persist results
        persist_advanced_specs(product_type, specs, existing_params)

        # Step 6: Build result
        result = _build_result(product_type, specs)
        result["success"] = True
        result["fallback_used"] = fallback_used
        result["session_id"] = session_id
        result["cache_hit"] = False

        logger.info("[AdvancedSpecsFunctions] Completed - %d specs (fallback=%s)",
                   len(specs), fallback_used)
        return result

    except Exception as e:
        logger.error("[AdvancedSpecsFunctions] Discovery failed: %s", e, exc_info=True)
        return {
            "success": False,
            "product_type": product_type,
            "unique_specifications": [],
            "total_unique_specifications": 0,
            "discovery_successful": False,
            "fallback_used": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def format_specs_for_display(
    specifications: List[Dict[str, Any]],
    max_items: Optional[int] = None
) -> str:
    """Format discovered specifications as a human-readable numbered list."""
    if not specifications:
        return "No specifications available."

    specs_to_show = specifications[:max_items] if max_items else specifications
    remaining = len(specifications) - len(specs_to_show)
    lines = []

    for i, spec in enumerate(specs_to_show, 1):
        name = spec.get("name", spec.get("key", "Unknown"))
        vendor = spec.get("vendor", "")
        description = spec.get("description", "")
        line = f"{i}. {name}"
        if vendor:
            line += f" ({vendor})"
        if description:
            line += f" - {description}"
        lines.append(line)

    if remaining > 0:
        lines.append(f"... and {remaining} more specifications")

    return "\n".join(lines)


def get_spec_keys(specifications: List[Dict[str, Any]]) -> List[str]:
    """Extract specification keys from a discovery result list."""
    return [s.get("key", "") for s in specifications if s.get("key")]


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

def discover_advanced_parameters(product_type: str) -> Dict[str, Any]:
    """
    Backward-compatible alias for discover_advanced_specs().

    This function preserves the old API from advanced_parameters.py:
        from advanced_parameters import discover_advanced_parameters
        result = discover_advanced_parameters("Pressure Transmitter")

    Args:
        product_type: Product type to discover parameters for

    Returns:
        Same shape as discover_advanced_specs()
    """
    return discover_advanced_specs(product_type=product_type)


# Alias class for code that used `AdvancedSpecificationAgent().discover()`
class AdvancedSpecificationAgent:
    """
    Backward-compatible wrapper class.

    Provides the .discover() method API that old code expects:
        agent = AdvancedSpecificationAgent()
        result = agent.discover(product_type="Pressure Transmitter")
    """

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Delegate to the simplified function."""
        return discover_advanced_specs(
            product_type=product_type,
            session_id=session_id,
            existing_schema=existing_schema,
            **kwargs
        )
