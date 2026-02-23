"""
EnGenie Chat Memory Module

Session-based conversation memory for EnGenie Chat RAG.
Handles follow-up resolution and query context.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Session storage with automatic cleanup
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_SESSION_TTL = 3600  # 1 hour


class ConversationEntry:
    """Single Q&A entry in conversation history."""
    
    def __init__(
        self,
        query: str,
        response: str,
        sources_used: List[str] = None,
        timestamp: datetime = None
    ):
        self.query = query
        self.response = response
        self.sources_used = sources_used or []
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "sources_used": self.sources_used,
            "timestamp": self.timestamp.isoformat()
        }


def get_session(session_id: str) -> Dict[str, Any]:
    """Get or create a session."""
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "history": [],
                "context": {}
            }
        else:
            _sessions[session_id]["last_accessed"] = time.time()
        return _sessions[session_id]


def add_to_history(
    session_id: str,
    query: str,
    response: str,
    sources_used: List[str] = None
) -> None:
    """Add a Q&A entry to session history."""
    session = get_session(session_id)
    entry = ConversationEntry(
        query=query,
        response=response,
        sources_used=sources_used
    )
    session["history"].append(entry)
    logger.info(f"[MEMORY] Added to session {session_id}: {len(session['history'])} entries")


def get_history(session_id: str, limit: int = 10) -> List[ConversationEntry]:
    """Get recent conversation history."""
    session = get_session(session_id)
    return session["history"][-limit:]


def is_follow_up_query(query: str, session_id: str) -> bool:
    """
    Detect if query is a follow-up to previous conversation.

    Follow-up indicators:
    - Pronouns referring to previous context (it, this, that, they)
    - Short queries (< 5 words) without an explicit standalone subject
    - Comparison phrases (what about, how about, compared to)

    Uses word-boundary matching for single-word pronouns to prevent false
    positives: "it" inside "specification" or "transmitters" must not trigger.
    """
    history = get_history(session_id)
    if not history:
        return False

    query_lower = query.lower().strip()

    # Multi-word phrases are specific enough; substring check is fine.
    MULTI_WORD_INDICATORS = [
        "what about", "how about", "compared to", "versus",
        "which one", "the other", "another"
    ]
    for indicator in MULTI_WORD_INDICATORS:
        if indicator in query_lower:
            return True

    # Single-word pronouns / connectors: require whole-word match to avoid
    # false positives like "it" ⊂ "transmitters", "and" ⊂ "standard".
    WORD_INDICATORS = [
        "it", "this", "that", "they", "them", "those",
        "also", "too", "same", "similar",
        "more", "less", "better", "worse",
    ]
    for indicator in WORD_INDICATORS:
        if re.search(r'\b' + re.escape(indicator) + r'\b', query_lower):
            return True

    # Short queries (<5 words) are often follow-ups *unless* they contain a
    # standalone subject: a known brand name, model number, or standard code.
    words = query_lower.split()
    if len(words) < 5:
        STANDALONE_SUBJECTS = [
            # Major vendor / brand names
            "rosemount", "yokogawa", "emerson", "honeywell", "siemens",
            "endress", "hauser", "abb", "fisher", "krohne", "vega",
            "ifm", "pepperl", "turck", "danfoss", "burkert",
            "micro motion", "magnetrol", "wika", "fluke", "omega",
            # Common model families
            "3051", "ejx", "644", "dvc", "5400", "5300",
            # Standards / safety bodies
            "iec", "iso", "api", "atex", "iecex", "sil",
            "isa", "asme", "nfpa", "ansi", "din",
        ]
        has_explicit_subject = any(kw in query_lower for kw in STANDALONE_SUBJECTS)
        if not has_explicit_subject:
            return True

    return False


def resolve_follow_up(query: str, session_id: str) -> str:
    """
    Resolve a follow-up query using conversation context.
    
    Returns the resolved query with context injected.
    """
    history = get_history(session_id, limit=3)
    if not history:
        return query
    
    # Get the last query topic
    last_entry = history[-1]
    last_query = last_entry.query
    
    # Build context from recent history
    context_parts = []
    for entry in history[-3:]:
        context_parts.append(f"Previous Q: {entry.query}")
        if entry.response:
            # Truncate long responses
            resp_summary = entry.response[:200] + "..." if len(entry.response) > 200 else entry.response
            context_parts.append(f"Previous A: {resp_summary}")
    
    context = "\n".join(context_parts)
    
    # Build resolved query
    resolved = f"""Based on this conversation context:
{context}

Current follow-up question: {query}

Please answer the current question considering the previous context."""
    
    logger.info(f"[MEMORY] Resolved follow-up query for session {session_id}")
    return resolved


def set_context(session_id: str, key: str, value: Any) -> None:
    """Set session context variable."""
    session = get_session(session_id)
    session["context"][key] = value


def get_context(session_id: str, key: str, default: Any = None) -> Any:
    """Get session context variable."""
    session = get_session(session_id)
    return session["context"].get(key, default)


def clear_session(session_id: str) -> None:
    """Clear a session."""
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info(f"[MEMORY] Cleared session {session_id}")


def cleanup_expired_sessions() -> int:
    """Remove expired sessions. Returns count removed."""
    now = time.time()
    removed = 0
    with _sessions_lock:
        expired = [
            sid for sid, session in _sessions.items()
            if now - session.get("last_accessed", 0) > _SESSION_TTL
        ]
        for sid in expired:
            del _sessions[sid]
            removed += 1
    if removed:
        logger.info(f"[MEMORY] Cleaned up {removed} expired sessions")
    return removed


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Get a summary of the session for debugging."""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "history_count": len(session["history"]),
        "context_keys": list(session["context"].keys()),
        "created_at": session.get("created_at"),
        "last_accessed": session.get("last_accessed")
    }
