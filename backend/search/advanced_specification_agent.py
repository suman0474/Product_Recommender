"""
Advanced Parameters Deep Agent
================================

Step 2 of Product Search Workflow:
- Merges the discovery logic (advanced_parameters.py) and the wrapper (advanced_specification_agent.py)
  into a single LangGraph-powered Deep Agent.

Deep Agent Architecture (LangGraph):
  Each phase of the discovery pipeline runs as a discrete, observable graph node:
    1. load_existing     — Load existing schema parameters to avoid duplicates
    2. check_cache       — Check in-memory / Azure Blob cache for prior results
    3. call_llm          — Single LLM call to discover new advanced parameters
    4. parse_response    — Extract + normalise parameters from raw LLM response
    5. persist_results   — Save discovered parameters to Azure Blob + in-memory cache
    6. assemble_result   — Build the final typed result dict

  Conditional edges allow early exit from check_cache → assemble_result
  when a valid cache entry exists, skipping LLM calls entirely.

Backward Compatibility:
  - `AdvancedSpecificationAgent` is kept as an alias to `AdvancedSpecificationAgent`
  - `discover_advanced_parameters(product_type)` function preserved for callers in
    `advanced_specification_agent.py` (both are now in this one file)
"""

import json
import logging
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

# LangGraph
from langgraph.graph import StateGraph, START, END

# LangChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# backend-D imports
from common.config.azure_blob_config import get_azure_blob_connection
from common.config import AgenticConfig
from common.services.llm.fallback import create_llm_with_fallback

# Configure logging
logger = logging.getLogger(__name__)

# Debug flags
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

TASK: Identify up to 5 advanced parameters NOT in existing parameters.

ADVANCED FEATURE CATEGORIES:
- AI/ML: Predictive diagnostics, anomaly detection, self-learning
- Wireless/IoT: Bluetooth, LoRaWAN, 5G, mesh networking
- Cloud: Edge computing, remote monitoring, cloud analytics
- Diagnostics: Self-calibration, health monitoring, predictive maintenance
- Energy: Low-power modes, energy harvesting, solar power
- Security: Cybersecurity, encrypted protocols, secure boot
- Digital Twin: Integration, virtual commissioning
- AR/VR: Augmented reality support, QR code diagnostics

RULES:
- NO duplication with existing parameters
- Innovations from past 6 months only
- Human-readable names (not snake_case)
- Include series/model hints where applicable
- Maximum 5 parameters

IMPORTANT: Generate parameters SPECIFIC to the {product_type}. Do NOT use generic examples.
Consider the unique measurement principles, communication protocols, and industry applications for this exact product type.

OUTPUT (JSON):
{{
  "advanced_parameters": [
    "<parameter 1 specific to {product_type}>",
    "<parameter 2 specific to {product_type}>",
    "<parameter 3 specific to {product_type}>",
    "<parameter 4 specific to {product_type}>",
    "<parameter 5 specific to {product_type}>"
  ],
  "innovation_justification": "<brief explanation of why these are relevant for {product_type}>",
  "total_count": <1-5>
}}"""
}


# =============================================================================
# IN-MEMORY BOUNDED TTL CACHES  (thread-safe, LRU eviction)
# =============================================================================

IN_MEMORY_CACHE_TTL_MINUTES = 10
SCHEMA_CACHE_TTL_MINUTES = 30
_COLLECTION_CACHE = None  # module-level Azure Blob collection singleton


class _BoundedTTLCache:
    """Thread-safe cache with TTL expiration and LRU eviction."""

    def __init__(self, max_size: int = 500, ttl_minutes: int = 10, name: str = "Cache"):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl_minutes = ttl_minutes
        self.name = name
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if entry["expires_at"] <= datetime.utcnow():
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                while len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(minutes=self._ttl_minutes),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_SPEC_CACHE = _BoundedTTLCache(max_size=500, ttl_minutes=IN_MEMORY_CACHE_TTL_MINUTES, name="ADV_SPEC_CACHE")
_SCHEMA_PARAM_CACHE = _BoundedTTLCache(max_size=500, ttl_minutes=SCHEMA_CACHE_TTL_MINUTES, name="SCHEMA_PARAM_CACHE")


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_product_type(product_type: str) -> str:
    return re.sub(r"[^a-z0-9]", "", product_type.lower()) if product_type else ""


def _get_in_memory_specs(product_type: str) -> Optional[List[Dict[str, Any]]]:
    if not product_type:
        return None
    return _SPEC_CACHE.get(_normalize_product_type(product_type))


def _set_in_memory_specs(product_type: str, specs: List[Dict[str, Any]]) -> None:
    if not product_type:
        return
    _SPEC_CACHE.set(_normalize_product_type(product_type), specs)


def _get_azure_collection():
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
        logger.warning("[AdvancedParamsAgent] Azure collection unavailable: %s", exc)
        _COLLECTION_CACHE = None
    return _COLLECTION_CACHE


def _build_result(product_type: str, specs: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Robustly extract a JSON object or array from raw LLM output."""
    if not raw:
        return {}
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    # Try full text first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # First JSON object
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(cleaned[s: e + 1])
        except Exception:
            pass

    # JSON array
    s, e = cleaned.find("["), cleaned.rfind("]")
    if s != -1 and e > s:
        try:
            arr = json.loads(cleaned[s: e + 1])
            return {"parameters": arr} if isinstance(arr, list) else {}
        except Exception:
            pass

    # Fallback: newline-separated plain list
    lines = [l.strip(" -*.\t") for l in cleaned.splitlines() if l.strip() and len(l.strip()) > 3]
    skip = ("return", "task", "rules", "rules:")
    items = [l for l in lines if not l.lower().startswith(skip)]
    if items:
        return {"parameters": items}

    logger.warning("[AdvancedParamsAgent] Could not parse LLM response as JSON")
    return {}


def _get_existing_parameters(product_type: str) -> set:
    """Load existing schema parameters (cached) to avoid suggesting duplicates."""
    if not product_type:
        return set()
    normalized = _normalize_product_type(product_type)
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
        logger.warning("[AdvancedParamsAgent] Schema load failed for %s: %s", product_type, exc)
        return set()


def _convert_to_human_readable(specs: List[str]) -> Dict[str, str]:
    return {s: s.replace("_", " ").title() for s in specs}


# =============================================================================
# DEEP AGENT STATE
# =============================================================================

class AdvancedParamsAgentState(TypedDict, total=False):
    """
    LangGraph state for the Advanced Parameters Deep Agent.

    Carries all intermediate data between nodes so each node is a pure,
    testable function that only reads from and writes to this shared dict.
    """
    # --- Inputs ---
    product_type: str
    session_id: Optional[str]
    existing_schema: Optional[Dict[str, Any]]

    # --- Node 1: load_existing ---
    existing_params: set             # Known parameters in current schema

    # --- Node 2: check_cache ---
    cached_specs: Optional[List[Dict[str, Any]]]   # None = cache miss

    # --- Node 3: call_llm ---
    raw_llm_response: str

    # --- Node 4: parse_response ---
    unique_specifications: List[Dict[str, Any]]

    # --- Node 5: persist_results ---
    persisted: bool

    # --- Node 6: assemble_result ---
    result: Dict[str, Any]

    # --- Observability ---
    current_node: str
    error: Optional[str]
    fallback_used: bool
    start_time: float


# =============================================================================
# GRAPH NODE FUNCTIONS
# =============================================================================

def _node_load_existing(state: AdvancedParamsAgentState) -> dict:
    """Node 1 — Load current schema parameters to use as exclusion list for the LLM."""
    product_type = state.get("product_type", "")
    logger.info("[AdvancedParamsAgent] Node 1: load_existing — product_type='%s'", product_type)

    try:
        existing = _get_existing_parameters(product_type)
        logger.info("[AdvancedParamsAgent] ✓ Loaded %d existing parameters", len(existing))
        return {"existing_params": existing, "current_node": "load_existing"}
    except Exception as e:
        logger.warning("[AdvancedParamsAgent] load_existing failed (non-critical): %s", e)
        return {"existing_params": set(), "current_node": "load_existing"}


def _node_check_cache(state: AdvancedParamsAgentState) -> dict:
    """Node 2 — Check in-memory + Azure Blob cache for prior discovery results."""
    product_type = state.get("product_type", "")
    logger.info("[AdvancedParamsAgent] Node 2: check_cache")

    # In-memory first (fastest)
    in_mem = _get_in_memory_specs(product_type)
    if in_mem is not None:
        logger.info("[AdvancedParamsAgent] ✓ In-memory cache HIT (%d specs)", len(in_mem))
        return {"cached_specs": in_mem, "current_node": "check_cache"}

    # Azure Blob fallback
    try:
        collection = _get_azure_collection()
        if collection is not None:
            normalized = _normalize_product_type(product_type)
            doc = collection.find_one({"normalized_product_type": normalized})
            if doc:
                specs = doc.get("unique_specifications")
                if isinstance(specs, list) and specs:
                    logger.info("[AdvancedParamsAgent] ✓ Azure cache HIT (%d specs)", len(specs))
                    _set_in_memory_specs(product_type, specs)  # Warm in-memory
                    return {"cached_specs": specs, "current_node": "check_cache"}
    except Exception as exc:
        logger.warning("[AdvancedParamsAgent] Azure cache check failed: %s", exc)

    logger.info("[AdvancedParamsAgent] Cache MISS — will invoke LLM")
    return {"cached_specs": None, "current_node": "check_cache"}


def _node_call_llm(state: AdvancedParamsAgentState) -> dict:
    """Node 3 — Call LLM to discover new advanced parameters not in existing schema."""
    product_type = state.get("product_type", "")
    existing_params = state.get("existing_params", set())

    logger.info("[AdvancedParamsAgent] Node 3: call_llm")
    logger.info("=" * 60)
    logger.info("[AdvancedParamsAgent] LLM INVOKED for product: %s", product_type)
    logger.info("=" * 60)

    try:
        prompt_text = _ADV_PARAMS_PROMPTS.get("PARAMETER_DISCOVERY", "")
        if not prompt_text:
            raise ValueError("PARAMETER_DISCOVERY prompt section missing from advanced_parameters_prompts.txt")

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

        logger.info("[AdvancedParamsAgent] LLM response length: %d chars", len(str(raw)))
        logger.info("[AdvancedParamsAgent] LLM preview: %s", str(raw)[:300])
        return {"raw_llm_response": raw, "current_node": "call_llm"}

    except Exception as e:
        logger.error("[AdvancedParamsAgent] LLM call failed: %s", e)
        return {"raw_llm_response": "", "error": str(e), "current_node": "call_llm"}


def _node_parse_response(state: AdvancedParamsAgentState) -> dict:
    """Node 4 — Parse LLM output into a deduplicated list of {key, name} dicts."""
    raw = state.get("raw_llm_response", "")
    existing_params = state.get("existing_params", set())
    product_type = state.get("product_type", "")

    logger.info("[AdvancedParamsAgent] Node 4: parse_response")

    # LLM call failed or empty — attempt generic fallback
    if not raw:
        return _fallback_generic_specs(product_type, state)

    payload = _extract_json_object(raw)

    # Support both "parameters" and "advanced_parameters" keys from prompt variations
    if isinstance(payload, dict):
        raw_params = payload.get("parameters") or payload.get("advanced_parameters") or []
    else:
        raw_params = []

    logger.info("[AdvancedParamsAgent] Extracted %d raw params from LLM payload", len(raw_params))

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

    logger.info("[AdvancedParamsAgent] ✓ %d unique new specifications after deduplication", len(unique_specs))
    return {"unique_specifications": unique_specs, "current_node": "parse_response", "fallback_used": False}


def _fallback_generic_specs(product_type: str, state: AdvancedParamsAgentState) -> dict:
    """Fallback: use generic specification generation when LLM call fails."""
    logger.warning("[AdvancedParamsAgent] Using generic specs fallback for %s", product_type)
    try:
        prompt_text = _ADV_PARAMS_PROMPTS.get("GENERIC_SPECIFICATIONS", "")
        if not prompt_text:
            return {"unique_specifications": [], "current_node": "parse_response", "fallback_used": True}

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
            specs = [
                {"key": re.sub(r"\s+", "_", str(p).lower().strip()), "name": str(p)}
                for p in params[:15]
            ]
            return {"unique_specifications": specs, "current_node": "parse_response", "fallback_used": True}
    except Exception as fe:
        logger.warning("[AdvancedParamsAgent] Generic fallback also failed: %s", fe)

    return {"unique_specifications": [], "current_node": "parse_response", "fallback_used": True}


def _node_persist_results(state: AdvancedParamsAgentState) -> dict:
    """Node 5 — Persist discovered specs to Azure Blob + warm in-memory cache."""
    product_type = state.get("product_type", "")
    unique_specs = state.get("unique_specifications", [])
    existing_params = state.get("existing_params", set())

    logger.info("[AdvancedParamsAgent] Node 5: persist_results (%d specs)", len(unique_specs))

    if not unique_specs:
        logger.info("[AdvancedParamsAgent] No new specs to persist — skipping")
        return {"persisted": False, "current_node": "persist_results"}

    normalized = _normalize_product_type(product_type)
    now = datetime.utcnow()
    existing_list = sorted(list(existing_params))

    # Warm in-memory cache
    _set_in_memory_specs(product_type, unique_specs)

    # Persist to Azure Blob
    try:
        collection = _get_azure_collection()
        if collection is not None:
            doc = {
                "product_type": product_type,
                "normalized_product_type": normalized,
                "unique_specifications": unique_specs,
                "existing_parameters_snapshot": existing_list,
                "created_at": now,
                "updated_at": now,
            }
            collection.update_one(
                {"normalized_product_type": normalized},
                {"$set": doc},
                upsert=True,
            )
            logger.info("[AdvancedParamsAgent] ✓ Persisted %d specs to Azure Blob", len(unique_specs))
        else:
            logger.debug("[AdvancedParamsAgent] Azure collection not available — in-memory only")
    except Exception as exc:
        logger.warning("[AdvancedParamsAgent] Azure persist failed (in-memory still warm): %s", exc)

    return {"persisted": True, "current_node": "persist_results"}


def _node_assemble_result(state: AdvancedParamsAgentState) -> dict:
    """Node 6 — Build the final result dict, merging cached or freshly discovered specs."""
    product_type = state.get("product_type", "")

    # Prefer cached specs (cache-hit path), otherwise use freshly parsed
    specs = state.get("cached_specs") or state.get("unique_specifications") or []
    fallback_used = state.get("fallback_used", False)

    logger.info("[AdvancedParamsAgent] Node 6: assemble_result — %d specs", len(specs))

    result = _build_result(product_type, specs)
    result["fallback_used"] = fallback_used
    result["session_id"] = state.get("session_id")

    return {"result": result, "current_node": "assemble_result"}


# =============================================================================
# CONDITIONAL ROUTING
# =============================================================================

def _route_after_check_cache(state: AdvancedParamsAgentState) -> str:
    """Skip LLM pipeline entirely when cache has valid specs."""
    if state.get("cached_specs") is not None:
        logger.info("[AdvancedParamsAgent] Cache hit — routing to assemble_result")
        return "assemble_result"
    return "call_llm"


def _route_after_call_llm(state: AdvancedParamsAgentState) -> str:
    """Skip parse_response if LLM returned empty (error state)."""
    if not state.get("raw_llm_response"):
        logger.warning("[AdvancedParamsAgent] Empty LLM response — routing to assemble_result")
        return "assemble_result"
    return "parse_response"


# =============================================================================
# GRAPH BUILDER (lazy singleton)
# =============================================================================

_agent_graph = None


def _build_agent_graph():
    """Build (or return cached) the compiled LangGraph for advanced parameter discovery."""
    global _agent_graph
    if _agent_graph is not None:
        return _agent_graph

    graph = StateGraph(AdvancedParamsAgentState)

    graph.add_node("load_existing",     _node_load_existing)
    graph.add_node("check_cache",       _node_check_cache)
    graph.add_node("call_llm",          _node_call_llm)
    graph.add_node("parse_response",    _node_parse_response)
    graph.add_node("persist_results",   _node_persist_results)
    graph.add_node("assemble_result",   _node_assemble_result)

    graph.add_edge(START, "load_existing")
    graph.add_edge("load_existing", "check_cache")

    graph.add_conditional_edges(
        "check_cache",
        _route_after_check_cache,
        {"call_llm": "call_llm", "assemble_result": "assemble_result"},
    )

    graph.add_conditional_edges(
        "call_llm",
        _route_after_call_llm,
        {"parse_response": "parse_response", "assemble_result": "assemble_result"},
    )

    graph.add_edge("parse_response",  "persist_results")
    graph.add_edge("persist_results", "assemble_result")
    graph.add_edge("assemble_result", END)

    _agent_graph = graph.compile()
    logger.info("[AdvancedParamsAgent] LangGraph compiled successfully")
    return _agent_graph


# =============================================================================
# DEEP AGENT CLASS
# =============================================================================

class AdvancedSpecificationAgent:
    """
    Advanced Parameters Deep Agent — Step 2 of Product Search Workflow.

    Discovers the latest advanced specifications for a given product type using
    a LangGraph pipeline. Each phase runs as a discrete, observable graph node.

    Pipeline:
        load_existing → check_cache → [call_llm → parse_response → persist_results] → assemble_result

    Cache-hit path skips LLM entirely (load_existing → check_cache → assemble_result).

    This class merges the former `advanced_specification_agent.py` wrapper and the
    `advanced_parameters.py` core module into a single cohesive Deep Agent.
    """

    @timed_execution("ADVANCED_PARAMS", threshold_ms=20000)
    @debug_log("ADVANCED_PARAMS", log_args=True, log_result=False)
    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Discover advanced parameters for a product type via the LangGraph Deep Agent.

        Args:
            product_type: Product type to discover parameters for
            session_id:   Session identifier (for logging / tracking)
            existing_schema: Optional current schema (used to avoid duplicate fields)

        Returns:
            {
                "success": bool,
                "product_type": str,
                "unique_specifications": [{"key": str, "name": str}, ...],
                "total_unique_specifications": int,
                "existing_specifications_filtered": int,
                "vendors_searched": [],
                "discovery_successful": bool,
                "fallback_used": bool,
                "session_id": str | None
            }
        """
        logger.info("[AdvancedParamsAgent] Starting Deep Agent discovery")
        logger.info("[AdvancedParamsAgent] Product Type: %s", product_type)
        logger.info("[AdvancedParamsAgent] Session: %s", session_id or "N/A")

        if not product_type:
            return {
                "success": False,
                "product_type": product_type,
                "unique_specifications": [],
                "total_unique_specifications": 0,
                "existing_specifications_filtered": 0,
                "vendors_searched": [],
                "discovery_successful": False,
                "error": "product_type is required",
            }

        try:
            import time
            graph = _build_agent_graph()
            initial_state: AdvancedParamsAgentState = {
                "product_type": product_type.strip(),
                "session_id": session_id,
                "existing_schema": existing_schema,
                "start_time": time.time(),
                "fallback_used": False,
            }

            final_state = graph.invoke(initial_state)
            result = final_state.get("result", {})

            if not result:
                raise RuntimeError("Graph produced no result")

            result["success"] = True
            result.setdefault("vendors_searched", [])
            logger.info(
                "[AdvancedParamsAgent] ✓ Completed — %d specs (fallback=%s)",
                result.get("total_unique_specifications", 0),
                result.get("fallback_used", False),
            )
            return result

        except Exception as e:
            logger.error("[AdvancedParamsAgent] ✗ Deep Agent failed: %s", e, exc_info=True)
            return {
                "success": False,
                "product_type": product_type,
                "unique_specifications": [],
                "total_unique_specifications": 0,
                "existing_specifications_filtered": 0,
                "vendors_searched": [],
                "discovery_successful": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def discover_with_filtering(
        self,
        product_type: str,
        existing_schema: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Discover parameters with explicit schema-based post-filtering.

        Runs `.discover()` then removes any specs whose key already appears
        in the provided schema (both mandatory and optional sections).
        """
        result = self.discover(
            product_type=product_type,
            session_id=session_id,
            existing_schema=existing_schema,
        )
        if not result.get("success"):
            return result

        schema_keys: set = set()
        if existing_schema:
            for section in ("mandatory", "optional", "mandatory_requirements", "optional_requirements"):
                schema_keys.update(existing_schema.get(section, {}).keys())

        unique_specs = result.get("unique_specifications", [])
        filtered = [s for s in unique_specs if s.get("key", "").lower() not in schema_keys]
        extra_filtered = len(unique_specs) - len(filtered)

        if extra_filtered > 0:
            logger.info("[AdvancedParamsAgent] ℹ Post-filtered %d additional specs against provided schema", extra_filtered)

        result["unique_specifications"] = filtered
        result["total_unique_specifications"] = len(filtered)
        result["existing_specifications_filtered"] = (
            result.get("existing_specifications_filtered", 0) + extra_filtered
        )
        return result

    def format_for_display(
        self,
        specifications: List[Dict[str, Any]],
        max_items: Optional[int] = None,
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
                line += f" — {description}"
            lines.append(line)

        if remaining > 0:
            lines.append(f"... and {remaining} more specifications")

        return "\n".join(lines)

    def get_specification_keys(self, specifications: List[Dict[str, Any]]) -> List[str]:
        """Extract specification keys from a discovery result list."""
        return [s.get("key", "") for s in specifications if s.get("key")]

    def get_specifications_by_vendor(
        self, specifications: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group specifications by vendor name."""
        by_vendor: Dict[str, List] = {}
        for spec in specifications:
            vendor = spec.get("vendor", "Unknown")
            by_vendor.setdefault(vendor, []).append(spec)
        return by_vendor


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

#: Alias — callers that still import `AdvancedSpecificationAgent` work unchanged.
AdvancedSpecificationAgent = AdvancedSpecificationAgent


def discover_advanced_parameters(product_type: str) -> Dict[str, Any]:
    """
    Module-level function preserved for backward compatibility.

    Delegates to AdvancedSpecificationAgent.discover() and returns the
    same shape that the old `advanced_parameters.py` function returned.
    """
    agent = AdvancedSpecificationAgent()
    return agent.discover(product_type=product_type)
