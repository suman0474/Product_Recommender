"""
SessionOrchestrator - Manages active user sessions at application level

Thread-safe singleton that tracks:
- Which users are logged in
- When they were last active
- What workflows they're running
- Session lifecycle (create, heartbeat, end)

Design:
- Uses RLock (reentrant lock) for thread-safe access
- Holds locks <1ms per operation
- Background cleanup removes expired sessions

Usage:
    orchestrator = SessionOrchestrator.get_instance()
    session = orchestrator.create_session(user_id, main_thread_id)
    orchestrator.heartbeat(main_thread_id)
    orchestrator.end_session(main_thread_id)
"""

import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import logging

# Debug imports
from debug_flags import debug_log, timed_execution, is_debug_enabled

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WorkflowContext:
    """Represents an active workflow in a session"""
    workflow_id: str
    workflow_type: str  # e.g., "instrument_identifier", "solution", "product_search"
    parent_workflow_id: Optional[str] = None  # For child workflows
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

    def is_active(self, ttl_minutes: int = 60) -> bool:
        """Check if workflow has been active recently"""
        idle_time = (datetime.now() - self.last_activity).total_seconds()
        return idle_time < (ttl_minutes * 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "parent_workflow_id": self.parent_workflow_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SessionContext:
    """Represents an active user session"""
    user_id: str
    main_thread_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    workflow_threads: Dict[str, WorkflowContext] = field(default_factory=dict)
    active: bool = True
    is_saved: bool = False  # Saved workflow sessions persist longer
    request_count: int = 0
    zone: str = "default"  # Geographic zone (US-WEST, US-EAST, EU, ASIA)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        self.request_count += 1

    def is_expired(self, ttl_minutes: int = 30) -> bool:
        """Check if session has expired due to inactivity"""
        if self.is_saved:
            return False  # Saved sessions don't expire from inactivity

        idle_time = (datetime.now() - self.last_activity).total_seconds()
        ttl_seconds = ttl_minutes * 60
        return idle_time > ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "user_id": self.user_id,
            "main_thread_id": self.main_thread_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "workflow_count": len(self.workflow_threads),
            "active": self.active,
            "is_saved": self.is_saved,
            "request_count": self.request_count,
            "zone": self.zone
        }


# ============================================================================
# SINGLETON CLASS
# ============================================================================

class SessionOrchestrator:
    """
    Singleton that manages all active user sessions.

    Thread-safe through RLock. Can be safely accessed from multiple Flask workers.

    Architecture:
        - _sessions: main_thread_id -> SessionContext
        - _user_sessions: user_id -> [main_thread_ids]
        - _request_log: main_thread_id -> [request logs]

    Thread Safety:
        - All public methods use self._lock (RLock)
        - Lock hold time < 1ms (only during dict operations)
        - Workflow execution happens OUTSIDE of locks
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        """Initialize orchestrator (called once via singleton)"""
        # Thread-safe storage for sessions
        self._sessions: Dict[str, SessionContext] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> [main_thread_ids]
        self._request_log: Dict[str, List[Dict]] = {}  # main_thread_id -> [logs]

        # RLock for thread-safe access (reentrant - same thread can acquire multiple times)
        self._lock = threading.RLock()

        # Configuration
        self._default_session_ttl_minutes = 30
        self._max_sessions_per_user = 10
        self._max_request_log_entries = 100

        logger.info("[SESSION_ORCHESTRATOR] Initialized")

    @classmethod
    def get_instance(cls) -> 'SessionOrchestrator':
        """Get singleton instance (thread-safe double-check locking)"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = SessionOrchestrator()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing only)"""
        with cls._instance_lock:
            cls._instance = None

    # ========================================================================
    # CORE SESSION OPERATIONS
    # ========================================================================

    @debug_log("WORKFLOW", log_args=False)
    @timed_execution("WORKFLOW", threshold_ms=100)
    def get_or_create_session(
        self,
        user_id: str,
        main_thread_id: str,
        is_saved: bool = False,
        zone: str = "default",
        metadata: Dict[str, Any] = None,
        reuse_existing: bool = True
    ) -> SessionContext:
        """
        Get existing session or create new one (prevents duplicates).

        This method provides session deduplication at the backend level:
        1. Check if main_thread_id already exists -> return it
        2. Check if user has active sessions in same zone -> reuse if requested
        3. Otherwise create new session

        Args:
            user_id: The user's ID
            main_thread_id: Session ID from frontend
            is_saved: Whether this is a saved workflow session
            zone: Geographic zone for routing
            metadata: Additional session metadata
            reuse_existing: If True, reuse active session for same user/zone

        Returns:
            SessionContext: Existing or newly created session
        """
        with self._lock:
            # Check if this main_thread_id already exists
            if main_thread_id in self._sessions:
                logger.info(
                    f"[SESSION_ORCHESTRATOR] Reusing existing session: {main_thread_id}"
                )
                return self._sessions[main_thread_id]

            # Check if user has active sessions we can reuse
            if reuse_existing:
                existing_sessions = self.get_user_sessions(user_id)
                for session in existing_sessions:
                    if session.active and session.zone == zone and not session.is_expired():
                        logger.info(
                            f"[SESSION_ORCHESTRATOR] Reusing active session for user '{user_id}': "
                            f"{session.main_thread_id}"
                        )
                        # Update activity timestamp
                        session.update_activity()
                        return session

        # No existing session found - create new one
        return self.create_session(user_id, main_thread_id, is_saved, zone, metadata)

    @debug_log("WORKFLOW", log_args=False)
    def create_session(
        self,
        user_id: str,
        main_thread_id: str,
        is_saved: bool = False,
        zone: str = "default",
        metadata: Dict[str, Any] = None
    ) -> SessionContext:
        """
        Create new session (called on user login).

        Note: Consider using get_or_create_session() instead to prevent duplicates.

        Args:
            user_id: The user's ID (e.g., "user123")
            main_thread_id: Session ID from frontend (e.g., "main_user123_uuid_ts")
            is_saved: Whether this is a saved workflow session (persists longer)
            zone: Geographic zone for routing
            metadata: Additional session metadata

        Returns:
            SessionContext: The created session

        Thread Safety:
            - Lock held only during creation (~1ms)
            - Returns immediately after session stored
        """
        with self._lock:
            now = datetime.now()
            session = SessionContext(
                user_id=user_id,
                main_thread_id=main_thread_id,
                created_at=now,
                last_activity=now,
                is_saved=is_saved,
                zone=zone,
                metadata=metadata or {}
            )

            # Store session
            self._sessions[main_thread_id] = session

            # Update user index (track sessions per user)
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []

            # Enforce max sessions per user (cleanup old sessions)
            if len(self._user_sessions[user_id]) >= self._max_sessions_per_user:
                # Remove oldest session
                oldest_session_id = self._user_sessions[user_id][0]
                logger.info(
                    f"[SESSION_ORCHESTRATOR] Max sessions reached for user '{user_id}', "
                    f"removing oldest: {oldest_session_id}"
                )
                self._remove_session_internal(oldest_session_id)

            self._user_sessions[user_id].append(main_thread_id)

            # Initialize request log
            self._request_log[main_thread_id] = []

            logger.info(
                f"[SESSION_ORCHESTRATOR] Created session for user '{user_id}': "
                f"{main_thread_id} (saved={is_saved}, zone={zone})"
            )

            return session

    @debug_log("WORKFLOW")
    def get_session_context(self, main_thread_id: str) -> Optional[SessionContext]:
        """
        Retrieve session by main_thread_id

        Args:
            main_thread_id: Session ID from frontend

        Returns:
            SessionContext if found and active, None otherwise

        Use Case:
            Called by API endpoints to verify session is valid before processing request

        Example:
            session = orchestrator.get_session_context("main_user1_uuid_ts")
            if session is None:
                return error 400 "Session not found"
        """
        with self._lock:
            session = self._sessions.get(main_thread_id)
            if session and session.active:
                return session
            return None

    @debug_log("WORKFLOW")
    def heartbeat(self, main_thread_id: str) -> Dict[str, Any]:
        """
        Update session's last activity (called every 5 minutes from frontend)

        Args:
            main_thread_id: Session ID from frontend

        Returns:
            Dict with:
                - success: True if session was updated, False otherwise
                - status: "active", "not_found", "expired", or "inactive"
                - session_id: The session ID
                - message: Human-readable status message

        [FIX Feb 2026 #8] Changed from bool to dict for structured response.
        This helps debug session lifecycle issues and allows frontend to
        distinguish between different failure reasons.

        Use Case:
            Frontend calls /sessions/heartbeat every 5 minutes to keep session alive.
            Without heartbeat, session expires after 30 minutes of inactivity.
        """
        with self._lock:
            session = self._sessions.get(main_thread_id)

            if session is None:
                logger.warning(
                    f"[SESSION_ORCHESTRATOR] Heartbeat for non-existent session: "
                    f"{main_thread_id}"
                )
                return {
                    "success": False,
                    "status": "not_found",
                    "session_id": main_thread_id,
                    "message": "Session not found. Please create a new session."
                }

            if not session.active:
                logger.warning(
                    f"[SESSION_ORCHESTRATOR] Heartbeat for inactive session: "
                    f"{main_thread_id}"
                )
                return {
                    "success": False,
                    "status": "inactive",
                    "session_id": main_thread_id,
                    "message": "Session is inactive."
                }

            if session.is_expired():
                logger.warning(
                    f"[SESSION_ORCHESTRATOR] Heartbeat for expired session: "
                    f"{main_thread_id}"
                )
                return {
                    "success": False,
                    "status": "expired",
                    "session_id": main_thread_id,
                    "message": "Session has expired. Please create a new session."
                }

            # Session is valid - update activity
            session.update_activity()
            logger.debug(f"[SESSION_ORCHESTRATOR] Heartbeat: {main_thread_id}")
            return {
                "success": True,
                "status": "active",
                "session_id": main_thread_id,
                "last_activity": session.last_activity.isoformat(),
                "message": "Heartbeat received"
            }

    @debug_log("WORKFLOW")
    def end_session(self, main_thread_id: str) -> Optional[SessionContext]:
        """
        End session (called on user logout)

        Args:
            main_thread_id: Session ID from frontend

        Returns:
            The ended SessionContext if found, None otherwise

        Use Case:
            Called on logout to clean up session and prepare for next login.
            Also cleans up all associated workflows and request logs.
        """
        with self._lock:
            return self._remove_session_internal(main_thread_id)

    def _remove_session_internal(self, main_thread_id: str) -> Optional[SessionContext]:
        """Internal method to remove session (must be called with lock held)"""
        session = self._sessions.pop(main_thread_id, None)

        if session:
            # Clean up user index
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    mid for mid in self._user_sessions[session.user_id]
                    if mid != main_thread_id
                ]

                # Remove user from index if no more sessions
                if not self._user_sessions[session.user_id]:
                    del self._user_sessions[session.user_id]

            # Clean up request log
            if main_thread_id in self._request_log:
                del self._request_log[main_thread_id]

            logger.info(
                f"[SESSION_ORCHESTRATOR] Ended session for user '{session.user_id}': "
                f"{main_thread_id}"
            )
            return session
        else:
            logger.warning(
                f"[SESSION_ORCHESTRATOR] End called for non-existent session: "
                f"{main_thread_id}"
            )
            return None

    # ========================================================================
    # WORKFLOW TRACKING
    # ========================================================================

    def add_workflow_to_session(
        self,
        main_thread_id: str,
        workflow_id: str,
        workflow_type: str,
        parent_workflow_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[WorkflowContext]:
        """
        Add a workflow to a session (track which workflows are running)

        Args:
            main_thread_id: Session ID
            workflow_id: The workflow's thread ID
            workflow_type: Type of workflow (e.g., "instrument_identifier", "product_search")
            parent_workflow_id: Parent workflow ID (for child workflows)
            metadata: Additional workflow metadata

        Returns:
            WorkflowContext if added successfully, None if session not found
        """
        with self._lock:
            session = self._sessions.get(main_thread_id)
            if not session:
                logger.warning(
                    f"[SESSION_ORCHESTRATOR] Cannot add workflow - session not found: "
                    f"{main_thread_id}"
                )
                return None

            workflow = WorkflowContext(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                parent_workflow_id=parent_workflow_id,
                metadata=metadata or {}
            )

            session.workflow_threads[workflow_id] = workflow
            session.update_activity()

            logger.debug(
                f"[SESSION_ORCHESTRATOR] Added workflow '{workflow_type}' ({workflow_id}) "
                f"to session {main_thread_id}"
            )

            return workflow

    def update_workflow_activity(
        self,
        main_thread_id: str,
        workflow_id: str
    ) -> bool:
        """Update workflow's last activity timestamp"""
        with self._lock:
            session = self._sessions.get(main_thread_id)
            if session and workflow_id in session.workflow_threads:
                session.workflow_threads[workflow_id].update_activity()
                session.update_activity()
                return True
            return False

    def remove_workflow_from_session(
        self,
        main_thread_id: str,
        workflow_id: str
    ) -> Optional[WorkflowContext]:
        """
        Remove a workflow from a session

        Args:
            main_thread_id: Session ID
            workflow_id: The workflow's thread ID

        Returns:
            The removed WorkflowContext if found, None otherwise
        """
        with self._lock:
            session = self._sessions.get(main_thread_id)
            if not session:
                return None

            workflow = session.workflow_threads.pop(workflow_id, None)
            if workflow:
                logger.debug(
                    f"[SESSION_ORCHESTRATOR] Removed workflow {workflow_id} "
                    f"from session {main_thread_id}"
                )

            return workflow

    def get_workflows_for_session(
        self,
        main_thread_id: str,
        workflow_type: Optional[str] = None
    ) -> List[WorkflowContext]:
        """
        Get all workflows for a session, optionally filtered by type

        Args:
            main_thread_id: Session ID
            workflow_type: Optional filter by workflow type

        Returns:
            List of WorkflowContext objects
        """
        with self._lock:
            session = self._sessions.get(main_thread_id)
            if not session:
                return []

            workflows = list(session.workflow_threads.values())

            if workflow_type:
                workflows = [w for w in workflows if w.workflow_type == workflow_type]

            return workflows

    # ========================================================================
    # REQUEST LOGGING
    # ========================================================================

    def log_request(
        self,
        main_thread_id: str,
        action: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Log a request to this session (for debugging/auditing)

        Args:
            main_thread_id: Session ID
            action: What happened (e.g., "instrument_identifier", "product_search")
            metadata: Additional info about the request
        """
        with self._lock:
            if main_thread_id not in self._request_log:
                self._request_log[main_thread_id] = []

            log_entry = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }

            self._request_log[main_thread_id].append(log_entry)

            # Trim old entries if over limit
            if len(self._request_log[main_thread_id]) > self._max_request_log_entries:
                self._request_log[main_thread_id] = \
                    self._request_log[main_thread_id][-self._max_request_log_entries:]

    def get_request_log(self, main_thread_id: str) -> List[Dict]:
        """Get request log for a session (for debugging)"""
        with self._lock:
            return self._request_log.get(main_thread_id, []).copy()

    # ========================================================================
    # STATISTICS & MONITORING
    # ========================================================================

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active sessions (for monitoring)

        Returns:
            Dictionary with session stats

        Use Case:
            Called by /sessions/stats endpoint for admin monitoring

        Example Output:
            {
                "active_sessions": 42,
                "active_users": 38,
                "total_workflows": 150,
                "sessions": { ... }
            }
        """
        with self._lock:
            total_workflows = sum(
                len(s.workflow_threads) for s in self._sessions.values()
            )

            sessions_info = {}
            for mid, session in self._sessions.items():
                sessions_info[mid] = session.to_dict()

            return {
                "active_sessions": len(self._sessions),
                "active_users": len(self._user_sessions),
                "total_workflows": total_workflows,
                "sessions": sessions_info
            }

    def get_user_sessions(self, user_id: str) -> List[SessionContext]:
        """Get all sessions for a specific user"""
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            return [
                self._sessions[sid] for sid in session_ids
                if sid in self._sessions
            ]

    def get_active_session_count(self) -> int:
        """Get count of active sessions"""
        with self._lock:
            return len(self._sessions)

    def get_active_user_count(self) -> int:
        """Get count of active users"""
        with self._lock:
            return len(self._user_sessions)

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup_expired_sessions(self, ttl_minutes: int = None) -> int:
        """
        Background cleanup task - remove expired sessions

        Called by background task (every 1 minute)

        Args:
            ttl_minutes: Time to live for sessions (default from config)

        Returns:
            Number of sessions cleaned up
        """
        if ttl_minutes is None:
            ttl_minutes = self._default_session_ttl_minutes

        with self._lock:
            expired_sessions = []

            for main_thread_id, session in self._sessions.items():
                if session.is_expired(ttl_minutes):
                    expired_sessions.append(main_thread_id)

            # Clean them up
            for main_thread_id in expired_sessions:
                self._remove_session_internal(main_thread_id)

            if expired_sessions:
                logger.info(
                    f"[SESSION_ORCHESTRATOR] Cleaned up {len(expired_sessions)} expired sessions"
                )

            return len(expired_sessions)

    def cleanup_user_sessions(
        self,
        user_id: str,
        keep_latest: int = 1,
        force: bool = False
    ) -> int:
        """
        Cleanup old sessions for a specific user, keeping only N most recent.

        Args:
            user_id: User to cleanup sessions for
            keep_latest: Number of sessions to keep (default: 1)
            force: If True, cleanup even saved sessions

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            user_sessions = self.get_user_sessions(user_id)

            if len(user_sessions) <= keep_latest:
                return 0  # Nothing to cleanup

            # Sort by last activity (most recent first)
            user_sessions.sort(key=lambda s: s.last_activity, reverse=True)

            # Determine which to remove
            to_remove = []
            for session in user_sessions[keep_latest:]:
                if force or not session.is_saved:
                    to_remove.append(session.main_thread_id)

            # Remove them
            for main_thread_id in to_remove:
                self._remove_session_internal(main_thread_id)

            if to_remove:
                logger.info(
                    f"[SESSION_ORCHESTRATOR] Cleaned up {len(to_remove)} sessions for user '{user_id}' "
                    f"(kept {keep_latest} most recent)"
                )

            return len(to_remove)

    def cleanup_inactive_workflows(self, ttl_minutes: int = 60) -> int:
        """
        Remove workflows that have been inactive for too long

        Args:
            ttl_minutes: Time to live for workflows

        Returns:
            Number of workflows cleaned up
        """
        with self._lock:
            total_cleaned = 0

            for session in self._sessions.values():
                inactive_workflows = [
                    wid for wid, workflow in session.workflow_threads.items()
                    if not workflow.is_active(ttl_minutes)
                ]

                for wid in inactive_workflows:
                    del session.workflow_threads[wid]
                    total_cleaned += 1

            if total_cleaned:
                logger.info(
                    f"[SESSION_ORCHESTRATOR] Cleaned up {total_cleaned} inactive workflows"
                )

            return total_cleaned

    # ========================================================================
    # DEBUG / ADMIN METHODS
    # ========================================================================

    def clear_all(self):
        """Clear all sessions (for testing only)"""
        with self._lock:
            self._sessions.clear()
            self._user_sessions.clear()
            self._request_log.clear()
            logger.warning("[SESSION_ORCHESTRATOR] All sessions cleared!")

    def debug_dump(self) -> Dict[str, Any]:
        """Dump all internal state for debugging"""
        with self._lock:
            return {
                "sessions_count": len(self._sessions),
                "users_count": len(self._user_sessions),
                "sessions": {
                    mid: session.to_dict()
                    for mid, session in self._sessions.items()
                },
                "user_sessions": dict(self._user_sessions),
                "request_log_sizes": {
                    mid: len(logs)
                    for mid, logs in self._request_log.items()
                }
            }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_session_orchestrator() -> SessionOrchestrator:
    """Get the SessionOrchestrator singleton instance"""
    return SessionOrchestrator.get_instance()
