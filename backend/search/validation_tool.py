"""
Validation Tool - Backward Compatibility Re-exports
====================================================

This file is now a THIN RE-EXPORT layer for backward compatibility.
All implementation has been moved to validation_deep_agent.py.

DO NOT ADD NEW CODE HERE - add it to validation_deep_agent.py instead.

Re-exports:
- ValidationTool: Backward-compatible wrapper class
- ValidationDeepAgent: Pure LangGraph agent
- Session caching functions
- Request context functions
"""

# ═══════════════════════════════════════════════════════════════════════════
# RE-EXPORT EVERYTHING FROM validation_deep_agent.py
# ═══════════════════════════════════════════════════════════════════════════

from search.validation_deep_agent import (
    # Main classes
    ValidationTool,
    ValidationDeepAgent,

    # State definition
    ValidationDeepAgentState,

    # Session caching functions
    _get_session_enrichment,
    _cache_session_enrichment,
    clear_session_enrichment_cache,
    _get_session_context,
    _cache_session_context,
    clear_session_cache,

    # Request context functions
    set_request_context,
    get_request_session_id,
    get_request_workflow_thread_id,
    get_isolated_cache_key,
    clear_request_context,

    # Workflow creation
    create_validation_workflow,

    # Example usage
    example_usage,
)

# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main classes
    "ValidationTool",
    "ValidationDeepAgent",

    # State definition
    "ValidationDeepAgentState",

    # Session caching functions
    "_get_session_enrichment",
    "_cache_session_enrichment",
    "clear_session_enrichment_cache",
    "_get_session_context",
    "_cache_session_context",
    "clear_session_cache",

    # Request context functions
    "set_request_context",
    "get_request_session_id",
    "get_request_workflow_thread_id",
    "get_isolated_cache_key",
    "clear_request_context",

    # Workflow creation
    "create_validation_workflow",

    # Example usage
    "example_usage",
]


if __name__ == "__main__":
    # Run example from the deep agent module
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    example_usage()
