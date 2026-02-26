"""
EnGenie Chat Orchestrator

Sequential flow: web search → LLM (grounded with web context) → response.
Handles domain validation and follow-up memory.
"""

import logging
from typing import Dict, Any

from .engenie_chat_memory import (
    add_to_history, is_follow_up_query, resolve_follow_up
)


try:
    from common.agentic.workflows.circuit_breaker import get_circuit_breaker, CircuitOpenError
    from common.rag_cache import get_rag_cache, cache_get, cache_set
    from common.agentic.workflows.rate_limiter import get_rate_limiter, acquire_for_service
    from common.agentic.workflows.fast_fail import should_fail_fast, check_and_set_fast_fail
    from common.rag_logger import set_trace_id, get_trace_id, OrchestratorLogger
    UTILITIES_AVAILABLE = True
except ImportError as e:
    UTILITIES_AVAILABLE = False

# Import AgenticConfig for timeouts and TemperaturePreset for consistent temperatures
try:
    from common.config.agentic_config import AgenticConfig, TemperaturePreset
    ORCHESTRATOR_TIMEOUT = AgenticConfig.ORCHESTRATOR_TIMEOUT_SECONDS
except ImportError:
    ORCHESTRATOR_TIMEOUT = 1200  # Default 20 minutes
    # Fallback temperature presets if config not available
    class TemperaturePreset:
        CONVERSATION = 0.7
        ANALYSIS = 0.2
        VALIDATION = 0.1

logger = logging.getLogger(__name__)

try:
    from common.prompts.prompt_loader import load_prompt
    _LLM_FALLBACK_SYSTEM_PROMPT = load_prompt("engenie_chat_prompts", "LLM_FALLBACK_SYSTEM_PROMPT")
except Exception:
    # Fallback to hardcoded if file not found
    _LLM_FALLBACK_SYSTEM_PROMPT = (
        "You are EnGenie, an expert industrial automation and instrumentation consultant.\n\n"
        "Respond to the user's question below."
    )

def query_llm_fallback(query: str, session_id: str, web_context: str = "") -> Dict[str, Any]:
    """
    Use LLM directly, optionally grounded with web search context.

    When web_context is provided, it is injected before the user question so the
    LLM can synthesize and cite real-time information rather than rely solely on
    training knowledge.

    Uses create_llm_with_fallback for automatic key rotation and OpenAI fallback
    on RESOURCE_EXHAUSTED errors.
    """
    try:
        from common.services.llm.fallback import create_llm_with_fallback, invoke_with_retry_fallback
        from langchain_core.messages import SystemMessage, HumanMessage

        chat_temperature = TemperaturePreset.CONVERSATION

        llm = create_llm_with_fallback(
            model=AgenticConfig.DEFAULT_MODEL,
            temperature=chat_temperature,
            skip_test=True
        )

        # Inject web context into the user message when available
        if web_context:
            human_content = (
                f"--- Web Search Results ---\n"
                f"{web_context}\n"
                f"--- End Web Search Results ---\n\n"
                f"User question: {query}"
            )
        else:
            human_content = query

        messages = [
            SystemMessage(content=_LLM_FALLBACK_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        response = invoke_with_retry_fallback(
            llm,
            messages,
            max_retries=3,
            fallback_to_openai=True,
            model=AgenticConfig.DEFAULT_MODEL,
            temperature=chat_temperature
        )

        return {
            "success": True,
            "source": "llm",
            "answer": response.content if hasattr(response, 'content') else str(response),
            "data": {}
        }
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] LLM error: {e}")
        return {
            "success": False,
            "source": "llm",
            "error": str(e),
            "answer": "I'm sorry, I couldn't process your request. Please try again."
        }


def merge_results(
    llm_result: Dict[str, Any],
    web_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the final response from the LLM result (which already incorporates
    web search context). Web result is used only for sources_used tracking.

    Args:
        llm_result: Result dict from query_llm_fallback
        web_result: Result dict from query_web_search (for metadata only)

    Returns:
        Response dict with keys: success, answer, source, sources_used, found_in_database
    """
    sources_used = []
    answer = ""

    if llm_result.get("success"):
        answer = llm_result.get("answer", "").strip()
        if answer:
            sources_used.append("llm")

    if web_result.get("success") and web_result.get("answer", "").strip():
        sources_used.append("web_search")

    if answer:
        logger.info(f"[ORCHESTRATOR] Final answer: {len(answer)} chars, sources: {sources_used}")
        return {
            "success": True,
            "answer": answer,
            "source": "llm",
            "found_in_database": False,
            "sources_used": sources_used,
        }

    logger.warning("[ORCHESTRATOR] LLM returned no answer — static fallback")
    return {
        "success": False,
        "answer": (
            "I wasn't able to process your request right now. "
            "Please try rephrasing your question and I'll do my best to help."
        ),
        "source": "fallback",
        "found_in_database": False,
        "sources_used": sources_used,
    }


def query_web_search(query: str, session_id: str) -> Dict[str, Any]:
    """
    Query web search (GoogleSerper) for current/external information.

    Returns same shape as query_llm_fallback for easy merging.
    """
    try:
        from Indexing.tools.web_search import serper_search
        raw = serper_search(query)
        if raw:
            return {
                "success": True,
                "source": "web_search",
                "answer": raw,
                "data": {}
            }
        return {
            "success": False,
            "source": "web_search",
            "error": "No results returned",
            "answer": ""
        }
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Web search error: {e}")
        return {
            "success": False,
            "source": "web_search",
            "error": str(e),
            "answer": ""
        }


def run_engenie_chat_query(
    query: str,
    session_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for EnGenie Chat queries.

    Sequential flow: web search → LLM (grounded with web context) → merge.
    Web results are injected into the LLM prompt so the LLM synthesizes a single
    coherent answer rather than producing two separate outputs.

    Handles:
    1. Domain validation (blocks out-of-domain before any LLM call)
    2. Memory resolution (follow-up detection)
    3. Web search (fast, non-blocking on failure)
    4. LLM call grounded with web context
    5. Response assembly

    Args:
        query: User query
        session_id: Session ID for memory tracking
        **kwargs: Accepted but ignored (streaming compat)

    Returns:
        Unified response dict
    """
    logger.info(f"[ORCHESTRATOR] Processing query: {query[:100]}...")

    from common.validators import validate_query_domain

    logger.info(f"[ORCHESTRATOR] Validating domain for session {session_id}")
    validation_result = validate_query_domain(
        query=query,
        session_id=session_id,
        context={},
        use_fast_path=True
    )

    if not validation_result.is_valid:
        logger.info(
            f"[ORCHESTRATOR] ⚠️ OUT_OF_DOMAIN blocked (LLM NOT called): {query[:50]}... "
            f"(confidence: {validation_result.confidence:.2f}, reasoning: {validation_result.reasoning[:80]})"
        )
        
        return {
            'success': True,
            'answer': validation_result.reject_message or (
                "I'm EnGenie, your industrial automation assistant. I can help with:\n\n"
                "• **Instrument Identification** - Finding the right products for your needs\n\n"
                "• **Product Search** - Searching for specific industrial instruments\n\n"
                "• **Standards & Compliance** - Questions about industrial standards (ISA, IEC, etc.)\n\n"
                "• **Technical Knowledge** - Industrial automation concepts and best practices\n\n"
                "Please ask a question related to industrial automation, instrumentation, or process control."
            ),
            'source': 'validation_blocked',
            'found_in_database': False,
            'awaiting_confirmation': False,
            'sources_used': [],
            'is_follow_up': False,
            'classification': {
                'primary_source': 'out_of_domain',
                'confidence': validation_result.confidence,
                'reasoning': validation_result.reasoning
            },
            'blocked_reason': 'out_of_domain',
        }

    logger.info(
        f"[ORCHESTRATOR] ✅ Valid query - proceeding with LLM: {validation_result.target_workflow} "
        f"(confidence: {validation_result.confidence:.2f})"
    )

    # Step 1: Memory resolution
    is_follow_up = is_follow_up_query(query, session_id)
    resolved_query = query

    if is_follow_up:
        logger.info(f"[ORCHESTRATOR] Detected follow-up query")
        resolved_query = resolve_follow_up(query, session_id)

    # Step 2: Web search on clean query (not the context blob), then LLM grounded with web context
    logger.info("[ORCHESTRATOR] Running web search...")
    web_result = query_web_search(query, session_id)
    web_context = web_result.get("answer", "").strip() if web_result.get("success") else ""
    if web_context:
        logger.info(f"[ORCHESTRATOR] Web search returned {len(web_context)} chars — injecting into LLM prompt")
    else:
        logger.info("[ORCHESTRATOR] Web search returned no results — LLM will use training knowledge only")

    # LLM gets resolved_query (with conversation history for follow-up continuity)
    logger.info("[ORCHESTRATOR] Running LLM with web context...")
    llm_result = query_llm_fallback(resolved_query, session_id, web_context=web_context)

    # Step 3: Build final response
    merged = merge_results(llm_result, web_result)

    # Step 4: Save to memory
    add_to_history(
        session_id=session_id,
        query=query,
        response=merged.get("answer", ""),
        sources_used=merged.get("sources_used", [])
    )

    merged["is_follow_up"] = is_follow_up

    return merged


# =============================================================================
# WORKFLOW REGISTRATION
# =============================================================================

def _register_workflow():
    """Register this workflow with the central registry."""
    try:
        from common.agentic.workflows.workflow_registry import get_workflow_registry, WorkflowMetadata
        
        registry = get_workflow_registry()
        
        # Register as "engenie_chat" (Primary)
        registry.register(WorkflowMetadata(
            name="engenie_chat",
            display_name="EnGenie Chat",
            description="Conversational knowledge assistant: web search feeds LLM context sequentially, producing a single grounded answer. Includes follow-up memory and domain validation.",
            keywords=[
                "question", "what is", "tell me", "explain", "compare", "difference",
                "standard", "certification", "vendor", "supplier", "specification",
                "how does", "why", "when", "which", "datasheet", "model",
                "help", "information", "about", "chat", "conversational"
            ],
            intents=[
                "question", "greeting", "chitchat", "confirm", "reject",
                "standards", "vendor_strategy", "grounded_chat", "comparison"
            ],
            capabilities=[
                "llm_primary",
                "web_search_sequential_grounding",
                "follow_up_detection",
                "conversation_memory",
                "domain_validation",
                "streaming"
            ],
            entry_function=run_engenie_chat_query,
            priority=50,  # Primary workflow
            tags=["core", "knowledge", "rag", "chat", "conversational"],
            min_confidence_threshold=0.35  # Flexible
        ))
        logger.info("[EnGenieChatOrchestrator] Registered as 'engenie_chat' (primary)")
        

        
    except ImportError as e:
        logger.debug(f"[EnGenieChatOrchestrator] Registry not available: {e}")
    except Exception as e:
        logger.warning(f"[EnGenieChatOrchestrator] Failed to register: {e}")

# Register on module load
_register_workflow()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'run_engenie_chat_query',
    'query_llm_fallback',
    'query_web_search',
    'merge_results'
]
