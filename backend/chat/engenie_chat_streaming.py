"""
EnGenie Chat Streaming Module

Provides streaming wrapper for EnGenie Chat queries with progress updates
for classification, web search, LLM execution, and completion.
"""

import logging
from typing import Dict, Any, Optional, Callable

from common.agentic.workflows.streaming_utils import ProgressEmitter, ProgressStep, with_streaming
from .engenie_chat_orchestrator import run_engenie_chat_query

logger = logging.getLogger(__name__)


# ============================================================================
# ENGENIE CHAT WORKFLOW STREAMING
# ============================================================================

@with_streaming(
    progress_steps=[
        ProgressStep('initialize', 'Processing your question...', 5),
        ProgressStep('classify', 'Classifying intent...', 20),
        ProgressStep('query', 'Querying LLM and web search...', 60),
        ProgressStep('finalize', 'Finalizing response...', 90),
        ProgressStep('complete', 'Query completed successfully', 100)
    ],
    workflow_name="engenie_chat",
    extract_result_metadata=lambda r: {
        'source': r.get('source', 'unknown'),
        'sources_used': r.get('sources_used', []),
        'found_in_database': r.get('found_in_database', False),
    }
)
def run_engenie_chat_query_stream(
    query: str,
    session_id: str = "default",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run EnGenie Chat query with streaming progress updates.

    Streaming wrapper around run_engenie_chat_query (web search → LLM sequential).
    The decorator handles progress emission; this function just executes the workflow.

    Args:
        query: User's question or query
        session_id: Session identifier for memory management
        progress_callback: Callback function to emit progress updates

    Returns:
        Query result with answer, sources, metadata
    """
    return run_engenie_chat_query(query=query, session_id=session_id)


# ============================================================================
# ENHANCED STREAMING WITH MANUAL PROGRESS CONTROL
# ============================================================================

def run_engenie_chat_query_stream_detailed(
    query: str,
    session_id: str = "default",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run EnGenie Chat query with detailed manual progress control.

    Provides granular progress updates at each stage of the workflow.
    Use when you need finer control over progress reporting than the
    decorator-based run_engenie_chat_query_stream provides.

    Args:
        query: User's question or query
        session_id: Session identifier
        progress_callback: Callback function to emit progress updates

    Returns:
        Query result with answer, sources, metadata
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        emitter.emit('initialize', 'Processing your question...', 5)
        emitter.emit('classify', 'Analyzing query intent...', 20)
        emitter.emit('query', 'Querying web search, then generating LLM response...', 40)

        result = run_engenie_chat_query(query=query, session_id=session_id)

        sources_used = result.get('sources_used', [])
        if sources_used:
            emitter.emit(
                'sources_found',
                f'Retrieved data from: {", ".join(sources_used)}',
                80,
                data={'sources': sources_used}
            )

        emitter.emit(
            'complete',
            'Query completed successfully',
            100,
            data={
                'source': result.get('source', 'unknown'),
                'found_in_database': result.get('found_in_database', False),
            }
        )

        return result

    except Exception as e:
        logger.error(f"[ENGENIE_CHAT_STREAM] Error: {e}", exc_info=True)
        emitter.error(f'Query failed: {str(e)}', error_details=e)
        return {
            'success': False,
            'error': str(e),
            'answer': 'Sorry, something went wrong. Please try again.',
            'source': 'error',
            'found_in_database': False,
            'sources_used': [],
        }


# Export both streaming versions
__all__ = [
    'run_engenie_chat_query_stream',
    'run_engenie_chat_query_stream_detailed'
]
