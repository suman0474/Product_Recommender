# agentic/deep_agent/deep_agentic_workflow.py
# =============================================================================
# DEEP AGENTIC WORKFLOW ORCHESTRATOR
# =============================================================================
#
# Complete orchestration of the Product Search Workflow with Deep Agent
# capabilities including:
# - Failure memory and learning from past errors
# - Adaptive prompt optimization
# - Automatic workflow progression
# - Session state management
# - User decision handling
#
# Workflow Steps:
# 1. VALIDATION - Product type detection & schema generation
# 2. ADVANCED PARAMETERS - Discover specifications from vendors
# 3. SALES AGENT - Requirements collection (optional)
# 4. VENDOR ANALYSIS - Parallel vendor matching
# 5. RANKING - Final product ranking
#
# =============================================================================

import logging
import time
import uuid
import threading
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Phases of the Deep Agentic Workflow"""
    INITIAL = "initial"
    VALIDATION = "validation"
    AWAIT_MISSING_FIELDS = "await_missing_fields"
    COLLECT_FIELDS = "collect_fields"
    AWAIT_ADVANCED_PARAMS = "await_advanced_params"
    ADVANCED_DISCOVERY = "advanced_discovery"
    AWAIT_ADVANCED_SELECTION = "await_advanced_selection"
    VENDOR_ANALYSIS = "vendor_analysis"
    RANKING = "ranking"
    COMPLETE = "complete"
    ERROR = "error"


class UserDecision(Enum):
    """Types of user decisions"""
    CONTINUE = "continue"
    ADD_FIELDS = "add_fields"
    SKIP = "skip"
    YES = "yes"
    NO = "no"
    SELECT_ALL = "select_all"
    SELECT_SPECIFIC = "select_specific"
    UNKNOWN = "unknown"


@dataclass
class WorkflowState:
    """Complete state of a workflow session"""
    session_id: str
    thread_id: str
    phase: WorkflowPhase
    user_input: str
    product_type: Optional[str] = None
    schema: Optional[Dict] = None
    provided_requirements: Dict = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    discovered_specs: List[Dict] = field(default_factory=list)
    selected_specs: List[Dict] = field(default_factory=list)
    vendor_analysis: Optional[Dict] = None
    ranking_result: Optional[Dict] = None
    steps_completed: List[str] = field(default_factory=list)
    awaiting_user_input: bool = False
    last_message: str = ""
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'phase': self.phase.value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowState':
        data = data.copy()
        data['phase'] = WorkflowPhase(data.get('phase', 'initial'))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# SESSION STATE MANAGER
# =============================================================================

class WorkflowSessionManager:
    """
    Manages workflow session states with persistence and automatic cleanup.

    Key features:
    - Thread-safe state management
    - Session lookup by thread_id OR session_id
    - Automatic expiration of old sessions
    - Persistence support (optional)
    """

    def __init__(self, ttl_seconds: int = 7200, persistence_path: Optional[str] = None):
        self._states: Dict[str, WorkflowState] = {}
        self._session_to_thread: Dict[str, str] = {}  # session_id -> thread_id mapping
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._persistence_path = persistence_path

        if persistence_path:
            self._load_from_disk()

        logger.info(f"[WORKFLOW_SESSION] Manager initialized (TTL={ttl_seconds}s)")

    def create_session(self, user_input: str, session_id: Optional[str] = None) -> WorkflowState:
        """Create a new workflow session."""
        with self._lock:
            thread_id = f"thread_{uuid.uuid4().hex[:12]}"
            session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

            state = WorkflowState(
                session_id=session_id,
                thread_id=thread_id,
                phase=WorkflowPhase.INITIAL,
                user_input=user_input
            )

            self._states[thread_id] = state
            self._session_to_thread[session_id] = thread_id

            logger.info(f"[WORKFLOW_SESSION] Created new session: thread={thread_id}, session={session_id}")

            return state

    def get_state(self, thread_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[WorkflowState]:
        """Get workflow state by thread_id or session_id."""
        with self._lock:
            # Try thread_id first
            if thread_id and thread_id in self._states:
                state = self._states[thread_id]
                logger.debug(f"[WORKFLOW_SESSION] Found state by thread_id: {thread_id}, phase={state.phase.value}")
                return state

            # Try session_id
            if session_id and session_id in self._session_to_thread:
                thread_id = self._session_to_thread[session_id]
                if thread_id in self._states:
                    state = self._states[thread_id]
                    logger.debug(f"[WORKFLOW_SESSION] Found state by session_id: {session_id}, phase={state.phase.value}")
                    return state

            logger.debug(f"[WORKFLOW_SESSION] No state found for thread={thread_id}, session={session_id}")
            return None

    def update_state(self, state: WorkflowState) -> None:
        """Update workflow state."""
        with self._lock:
            state.updated_at = datetime.now().isoformat()
            self._states[state.thread_id] = state
            self._session_to_thread[state.session_id] = state.thread_id

            logger.debug(f"[WORKFLOW_SESSION] Updated state: thread={state.thread_id}, phase={state.phase.value}")

            if self._persistence_path:
                self._save_to_disk()

    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        with self._lock:
            now = datetime.now()
            expired = []

            for thread_id, state in self._states.items():
                updated = datetime.fromisoformat(state.updated_at)
                if (now - updated).total_seconds() > self._ttl:
                    expired.append(thread_id)

            for thread_id in expired:
                state = self._states.pop(thread_id, None)
                if state:
                    self._session_to_thread.pop(state.session_id, None)

            if expired:
                logger.info(f"[WORKFLOW_SESSION] Cleaned up {len(expired)} expired sessions")

            return len(expired)

    def _save_to_disk(self):
        """Persist states to disk."""
        if not self._persistence_path:
            return
        try:
            from pathlib import Path
            Path(self._persistence_path).parent.mkdir(parents=True, exist_ok=True)

            data = {
                'states': {k: v.to_dict() for k, v in self._states.items()},
                'mappings': self._session_to_thread
            }
            with open(self._persistence_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"[WORKFLOW_SESSION] Failed to save: {e}")

    def _load_from_disk(self):
        """Load states from disk."""
        if not self._persistence_path:
            return
        try:
            from pathlib import Path
            if not Path(self._persistence_path).exists():
                return

            with open(self._persistence_path, 'r') as f:
                data = json.load(f)

            for thread_id, state_dict in data.get('states', {}).items():
                self._states[thread_id] = WorkflowState.from_dict(state_dict)

            self._session_to_thread = data.get('mappings', {})

            logger.info(f"[WORKFLOW_SESSION] Loaded {len(self._states)} sessions from disk")
        except Exception as e:
            logger.error(f"[WORKFLOW_SESSION] Failed to load: {e}")


# =============================================================================
# DEEP AGENTIC WORKFLOW ORCHESTRATOR
# =============================================================================

class DeepAgenticWorkflowOrchestrator:
    """
    Complete Deep Agentic Workflow Orchestrator.

    Orchestrates the 5-step product search workflow with:
    - Automatic phase progression
    - User decision handling
    - Deep Agent integration (failure memory, adaptive prompts)
    - Parallel processing where possible

    Usage:
        orchestrator = DeepAgenticWorkflowOrchestrator()
        result = orchestrator.process_request(
            user_input="Need a pressure transmitter...",
            session_id="search_123",
            user_decision="continue"
        )
    """

    def __init__(self,
                 enable_ppi: bool = True,
                 enable_deep_agent: bool = True,
                 max_workers: int = 5,
                 auto_progress: bool = False):
        """
        Initialize the orchestrator.

        Args:
            enable_ppi: Enable PPI workflow for schema generation
            enable_deep_agent: Enable deep agent for schema population
            max_workers: Max parallel workers for vendor analysis
            auto_progress: Auto-progress through phases (for testing)
        """
        self.enable_ppi = enable_ppi
        self.enable_deep_agent = enable_deep_agent
        self.max_workers = max_workers
        self.auto_progress = auto_progress

        # Session manager
        # Cosmos DB Session manager
        from common.infrastructure.state.session.cosmos_manager import get_cosmos_workflow_session_manager
        
        # Try to initialize Cosmos DB session manager
        self.session_manager = get_cosmos_workflow_session_manager()
        
        if self.session_manager:
            logger.info("[DEEP_WORKFLOW] Using Azure Cosmos DB for session persistence")
            # Wrap with adapter to handle Dict/Object conversion
            self.session_manager = self.CosmosSessionManagerAdapter(self.session_manager)
            
        # Fallback to file-based if Cosmos is not available
        if not self.session_manager:
            logger.warning("[DEEP_WORKFLOW] Cosmos DB not available, falling back to file storage")
            self.session_manager = WorkflowSessionManager(
                ttl_seconds=7200,
                persistence_path="data/workflow_sessions.json"
            )

        # Tools (lazy loaded)
        self._validation_tool = None
        self._advanced_params_tool = None
        self._vendor_analysis_tool = None # Deprecated
        self._ranking_tool = None         # Deprecated
        self._workflow_helper = None

        logger.info(
            f"[DEEP_WORKFLOW] Orchestrator initialized "
            f"(ppi={enable_ppi}, deep_agent={enable_deep_agent}, auto={auto_progress})"
        )

    @property
    def validation_tool(self):
        if self._validation_tool is None:
            from search.validation_tool import ValidationTool
            self._validation_tool = ValidationTool(enable_ppi=self.enable_ppi)
        return self._validation_tool

    @property
    def advanced_params_tool(self):
        if self._advanced_params_tool is None:
            from search.advanced_specification_agent import AdvancedSpecificationAgent
            self._advanced_params_tool = AdvancedSpecificationAgent()
        return self._advanced_params_tool

    @property
    def workflow_helper(self):
        """Workflow helper using tool-based architecture."""
        if self._workflow_helper is None:
            # Create a simple wrapper that delegates to the tool classes
            self._workflow_helper = self._create_workflow_helper()
        return self._workflow_helper

    def _create_workflow_helper(self):
        """Create a workflow helper using tool-based architecture."""
        from search.vendor_analysis_deep_agent import VendorAnalysisTool
        from search.ranking_tool import RankingTool

        class ToolBasedWorkflowHelper:
            """Wrapper that provides workflow helper interface using tools."""

            def __init__(self):
                self.vendor_tool = VendorAnalysisTool()
                self.ranking_tool = RankingTool(use_llm_ranking=True)

            def run_vendor_analysis_step(self, structured_requirements, product_type, session_id, schema=None):
                """Run vendor analysis using VendorAnalysisTool."""
                return self.vendor_tool.analyze(
                    structured_requirements=structured_requirements,
                    product_type=product_type,
                    session_id=session_id,
                    schema=schema
                )

            def run_ranking_step(self, vendor_analysis, session_id, structured_requirements=None):
                """Run ranking using RankingTool."""
                return self.ranking_tool.rank(
                    vendor_analysis=vendor_analysis,
                    session_id=session_id,
                    structured_requirements=structured_requirements
                )

        return ToolBasedWorkflowHelper()

    # =========================================================================
    # COSMOS DB ADAPTER (Fixes Object vs Dict mismatch)
    # =========================================================================
    class CosmosSessionManagerAdapter:
        """Adapts CosmosWorkflowSessionManager (Dict-based) to WorkflowSessionManager interface (Object-based)"""
        def __init__(self, manager):
            self.manager = manager
            
        def get_state(self, thread_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional['WorkflowState']:
            data = self.manager.get_state(thread_id, session_id)
            if data:
                return WorkflowState.from_dict(data)
            return None
            
        def create_session(self, user_input: str, session_id: Optional[str] = None) -> 'WorkflowState':
            data = self.manager.create_session(user_input, session_id)
            return WorkflowState.from_dict(data)
            
        def update_state(self, state: 'WorkflowState') -> None:
            self.manager.update_state(state.to_dict())
            
        def cleanup_expired(self) -> int:
            return 0  # Cosmos handles TTL automatically

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def process_request(self,
                        user_input: Optional[str] = None,
                        session_id: Optional[str] = None,
                        thread_id: Optional[str] = None,
                        main_thread_id: Optional[str] = None,  # ✅ ADD
                        zone: Optional[str] = None,  # ✅ ADD
                        user_decision: Optional[str] = None,
                        user_provided_fields: Optional[Dict] = None,
                        product_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a workflow request.

        ✅ UPDATED: Accepts UI-provided thread IDs

        This is the main entry point that handles both new requests and
        continuations of existing workflows.

        Args:
            user_input: User's input text (required for new sessions)
            session_id: Session identifier
            thread_id: Thread identifier (for resuming) - This is workflow_thread_id from UI
            main_thread_id: ✅ Main thread ID from UI (format: main_*)
            zone: ✅ Geographic zone (US-WEST, US-EAST, etc.)
            user_decision: User's decision (continue, add_fields, yes, no, etc.)
            user_provided_fields: Fields provided by user
            product_type_hint: Hint for expected product type

        Returns:
            Workflow result with current state and next action
        """
        start_time = time.time()

        logger.info(
            f"\n{'='*70}\n"
            f"[DEEP_WORKFLOW] Processing request\n"
            f"   Main Thread ID: {main_thread_id}\n"  # ✅ LOG
            f"   Workflow Thread ID: {thread_id}\n"  # ✅ LOG
            f"   Zone: {zone}\n"  # ✅ LOG
            f"   Session: {session_id}\n"
            f"   Decision: {user_decision}\n"
            f"   Input: {user_input[:50] if user_input else 'None'}...\n"
            f"{'='*70}"
        )

        # ✅ STORE THREAD CONTEXT
        self.main_thread_id = main_thread_id
        self.zone = zone or 'DEFAULT'

        try:
            # Get or create workflow state
            state = self._get_or_create_state(
                user_input=user_input,
                session_id=session_id,
                thread_id=thread_id,
                decision=user_decision
            )

            if not state:
                return self._error_response("Failed to create workflow state")

            # Parse user decision
            decision = self._parse_user_decision(user_decision)

            # Process based on current phase
            result = self._process_phase(
                state=state,
                decision=decision,
                user_provided_fields=user_provided_fields,
                product_type_hint=product_type_hint
            )

            # Update state
            self.session_manager.update_state(state)

            # Build response
            duration_ms = int((time.time() - start_time) * 1000)

            response = {
                "success": True,
                "session_id": state.session_id,
                "thread_id": state.thread_id,
                "current_phase": state.phase.value,
                "awaiting_user_input": state.awaiting_user_input,
                "sales_agent_response": state.last_message,
                "product_type": state.product_type,
                "schema": state.schema,
                "missing_fields": state.missing_fields,
                "steps_completed": state.steps_completed,
                "completed": state.phase == WorkflowPhase.COMPLETE,
                "duration_ms": duration_ms,
                **result
            }

            logger.info(
                f"[DEEP_WORKFLOW] Request processed in {duration_ms}ms, "
                f"phase={state.phase.value}, awaiting={state.awaiting_user_input}"
            )

            return response

        except Exception as e:
            logger.error(f"[DEEP_WORKFLOW] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(str(e))

    # =========================================================================
    # PHASE PROCESSING
    # =========================================================================

    def _process_phase(self,
                       state: WorkflowState,
                       decision: UserDecision,
                       user_provided_fields: Optional[Dict],
                       product_type_hint: Optional[str]) -> Dict[str, Any]:
        """Process the current phase and transition to next."""

        phase_handlers = {
            WorkflowPhase.INITIAL: self._handle_initial,
            WorkflowPhase.VALIDATION: self._handle_validation,
            WorkflowPhase.AWAIT_MISSING_FIELDS: self._handle_await_missing_fields,
            WorkflowPhase.COLLECT_FIELDS: self._handle_collect_fields,
            WorkflowPhase.AWAIT_ADVANCED_PARAMS: self._handle_await_advanced_params,
            WorkflowPhase.ADVANCED_DISCOVERY: self._handle_advanced_discovery,
            WorkflowPhase.AWAIT_ADVANCED_SELECTION: self._handle_await_advanced_selection,
            WorkflowPhase.VENDOR_ANALYSIS: self._handle_vendor_analysis,
            WorkflowPhase.RANKING: self._handle_ranking,
            WorkflowPhase.COMPLETE: self._handle_complete,
            WorkflowPhase.ERROR: self._handle_error
        }

        handler = phase_handlers.get(state.phase, self._handle_unknown)
        return handler(state, decision, user_provided_fields, product_type_hint)

    def _handle_initial(self, state: WorkflowState, decision: UserDecision,
                        fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle initial phase - start validation."""
        logger.info(f"[DEEP_WORKFLOW] Phase: INITIAL -> Starting validation")

        state.phase = WorkflowPhase.VALIDATION
        return self._handle_validation(state, decision, fields, hint)

    def _handle_validation(self, state: WorkflowState, decision: UserDecision,
                           fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle validation phase."""
        logger.info(f"[DEEP_WORKFLOW] Phase: VALIDATION")

        try:
            # Run validation
            result = self.validation_tool.validate(
                user_input=state.user_input,
                expected_product_type=hint,
                session_id=state.session_id
            )

            if not result.get('success'):
                state.phase = WorkflowPhase.ERROR
                state.error = result.get('error', 'Validation failed')
                state.last_message = f"Validation failed: {state.error}"
                return {"error": state.error}

            # Update state with validation results
            state.product_type = result['product_type']
            state.schema = self._normalize_schema(result['schema'])
            state.missing_fields = result['missing_fields']
            state.provided_requirements = result['provided_requirements']
            state.steps_completed.append('validation')

            # Determine next phase
            if state.missing_fields:
                state.phase = WorkflowPhase.AWAIT_MISSING_FIELDS
                state.awaiting_user_input = True
                state.last_message = (
                    f"I've analyzed your requirements for **{state.product_type}**.\n\n"
                    f"The following fields are missing: {', '.join(state.missing_fields[:5])}"
                    f"{'...' if len(state.missing_fields) > 5 else ''}.\n\n"
                    f"Would you like to:\n"
                    f"• **Add** the missing details\n"
                    f"• **Continue** anyway with available information"
                )
            else:
                state.phase = WorkflowPhase.AWAIT_ADVANCED_PARAMS
                state.awaiting_user_input = True
                state.last_message = (
                    f"All required fields are provided for **{state.product_type}**.\n\n"
                    f"Would you like to discover advanced specifications from top vendors?\n"
                    f"• **Yes** - Discover advanced specs\n"
                    f"• **No** - Proceed to product search"
                )

            return {
                "validation_result": {
                    "productType": state.product_type,
                    "detectedSchema": state.schema,
                    "providedRequirements": state.provided_requirements,
                    "missingFields": state.missing_fields,
                    "ppiWorkflowUsed": result.get('ppi_workflow_used', False)
                }
            }

        except Exception as e:
            logger.error(f"[DEEP_WORKFLOW] Validation error: {e}")
            state.phase = WorkflowPhase.ERROR
            state.error = str(e)
            return {"error": str(e)}

    def _handle_await_missing_fields(self, state: WorkflowState, decision: UserDecision,
                                     fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle user decision on missing fields.

        IMPORTANT: The HITL message asks:
        "do you want to add missing values... tell me YES or NO"
        - YES = continue WITHOUT adding (proceed to advanced discovery)
        - NO = user wants TO ADD missing values (collect fields)
        """
        logger.info(f"[DEEP_WORKFLOW] Phase: AWAIT_MISSING_FIELDS, decision={decision.value}")

        if decision == UserDecision.UNKNOWN:
            # Still waiting for decision
            state.awaiting_user_input = True
            return {}

        # ╔══════════════════════════════════════════════════════════════════════════╗
        # ║  FIX: Align with HITL message semantics                                   ║
        # ║  HITL asks: "shall i continue without adding missing specs? YES or NO"   ║
        # ║  YES = continue without adding → proceed to advanced discovery           ║
        # ║  NO = user wants to add → collect fields                                 ║
        # ╚══════════════════════════════════════════════════════════════════════════╝
        if decision in [UserDecision.YES, UserDecision.CONTINUE, UserDecision.SKIP]:
            # YES = User wants to CONTINUE without filling fields → Advanced Discovery
            logger.info(f"[DEEP_WORKFLOW] User said YES/CONTINUE - proceeding to Advanced Params Discovery")
            state.phase = WorkflowPhase.ADVANCED_DISCOVERY
            state.awaiting_user_input = False
            return self._handle_advanced_discovery(state, decision, fields, hint)

        if decision in [UserDecision.NO, UserDecision.ADD_FIELDS]:
            # NO = User wants TO ADD missing fields → Collect Fields
            logger.info(f"[DEEP_WORKFLOW] User said NO/ADD_FIELDS - collecting missing fields")
            state.phase = WorkflowPhase.COLLECT_FIELDS
            state.awaiting_user_input = True
            state.last_message = (
                f"Please provide values for these fields:\n\n" +
                "\n".join([f"• **{f}**" for f in state.missing_fields[:10]])
            )
            return {}

        # Unknown decision - ask again
        state.awaiting_user_input = True
        state.last_message = (
            "I didn't understand. Please respond with:\n"
            "• **Add** - to provide missing fields\n"
            "• **Continue** - to proceed without them"
        )
        return {}

    def _handle_collect_fields(self, state: WorkflowState, decision: UserDecision,
                               fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle field collection from user."""
        logger.info(f"[DEEP_WORKFLOW] Phase: COLLECT_FIELDS")

        if not fields:
            state.awaiting_user_input = True
            return {}

        # Merge provided fields
        if 'mandatory' in state.provided_requirements:
            state.provided_requirements['mandatory'].update(fields)
        else:
            state.provided_requirements.update(fields)

        # Remove filled fields from missing list
        for key in fields.keys():
            if key in state.missing_fields:
                state.missing_fields.remove(key)

        if state.missing_fields:
            state.phase = WorkflowPhase.AWAIT_MISSING_FIELDS
            state.awaiting_user_input = True
            state.last_message = (
                f"Thanks! Fields updated.\n\n"
                f"Still missing: {', '.join(state.missing_fields)}.\n\n"
                f"Would you like to add these or continue?"
            )
        else:
            state.phase = WorkflowPhase.AWAIT_ADVANCED_PARAMS
            state.awaiting_user_input = True
            state.steps_completed.append('field_collection')
            state.last_message = (
                f"All fields provided for **{state.product_type}**.\n\n"
                f"Would you like to discover advanced specifications?"
            )

        return {}

    def _handle_await_advanced_params(self, state: WorkflowState, decision: UserDecision,
                                      fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle user decision on advanced params."""
        logger.info(f"[DEEP_WORKFLOW] Phase: AWAIT_ADVANCED_PARAMS, decision={decision.value}")

        if decision == UserDecision.UNKNOWN:
            state.awaiting_user_input = True
            return {}

        if decision in [UserDecision.YES, UserDecision.CONTINUE]:
            # Run advanced discovery
            state.phase = WorkflowPhase.ADVANCED_DISCOVERY
            state.awaiting_user_input = False
            return self._handle_advanced_discovery(state, decision, fields, hint)

        if decision in [UserDecision.NO, UserDecision.SKIP]:
            # Skip to vendor analysis
            logger.info(f"[DEEP_WORKFLOW] Skipping advanced params, proceeding to vendor analysis")
            state.phase = WorkflowPhase.VENDOR_ANALYSIS
            state.awaiting_user_input = False
            return self._handle_vendor_analysis(state, decision, fields, hint)

        state.awaiting_user_input = True
        state.last_message = "Would you like to discover advanced specifications? (Yes/No)"
        return {}

    def _handle_advanced_discovery(self, state: WorkflowState, decision: UserDecision,
                                   fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle advanced parameters discovery."""
        logger.info(f"[DEEP_WORKFLOW] Phase: ADVANCED_DISCOVERY")

        try:
            result = self.advanced_params_tool.discover(
                product_type=state.product_type,
                session_id=state.session_id,
                existing_schema=state.schema
            )

            state.discovered_specs = result.get('unique_specifications', [])
            state.steps_completed.append('advanced_parameters')

            if state.discovered_specs:
                state.phase = WorkflowPhase.AWAIT_ADVANCED_SELECTION
                state.awaiting_user_input = True

                specs_display = "\n".join([
                    f"• {s.get('name', s.get('key', 'Unknown'))}"
                    for s in state.discovered_specs[:10]
                ])
                if len(state.discovered_specs) > 10:
                    specs_display += f"\n• ... and {len(state.discovered_specs) - 10} more"

                state.last_message = (
                    f"Discovered {len(state.discovered_specs)} advanced specifications:\n\n"
                    f"{specs_display}\n\n"
                    f"Would you like to:\n"
                    f"• **All** - Include all specifications\n"
                    f"• **Skip** - Proceed without them\n"
                    f"• Or specify which ones you want"
                )

                return {
                    "advanced_parameters_result": {
                        "discovered_specifications": state.discovered_specs,
                        "total_discovered": len(state.discovered_specs)
                    }
                }
            else:
                # No specs found, proceed to vendor analysis
                state.phase = WorkflowPhase.VENDOR_ANALYSIS
                state.awaiting_user_input = False
                state.last_message = "No additional specs found. Proceeding to product search..."
                return self._handle_vendor_analysis(state, decision, fields, hint)

        except Exception as e:
            # =================================================================
            # FIX: IMPROVED ERROR HANDLING FOR ADVANCED PARAMETERS FAILURE
            # - Log detailed error information for debugging
            # - Mark step as failed (not completed)
            # - Notify user that this step was skipped due to error
            # - Continue to vendor analysis with available data
            # =================================================================
            error_msg = str(e)
            error_type = type(e).__name__

            logger.error(
                f"[DEEP_WORKFLOW] Advanced discovery FAILED: {error_type}: {error_msg}",
                exc_info=True
            )

            # Check for common error types
            if "KeyError" in error_type:
                logger.error(
                    "[DEEP_WORKFLOW] KeyError suggests missing prompts or configuration. "
                    "Check prompts_library/ prompt files."
                )
            elif "timeout" in error_msg.lower():
                logger.error(
                    "[DEEP_WORKFLOW] Timeout error - LLM took too long to respond. "
                    "Consider increasing timeout or simplifying query."
                )

            # Mark this step as skipped (not completed) due to error
            state.steps_completed.append('advanced_parameters_skipped')
            state.error = f"Advanced parameters discovery failed: {error_msg[:100]}"

            # Log warning that we're proceeding without advanced params
            logger.warning(
                "[DEEP_WORKFLOW] Proceeding to vendor analysis WITHOUT advanced parameters. "
                "User may get fewer/less relevant results."
            )

            # Update message to inform user about the issue
            state.last_message = (
                f"Advanced specifications discovery encountered an issue and was skipped. "
                f"Proceeding with standard product search..."
            )

            # Continue to vendor analysis even on error
            state.phase = WorkflowPhase.VENDOR_ANALYSIS
            return self._handle_vendor_analysis(state, decision, fields, hint)

    def _handle_await_advanced_selection(self, state: WorkflowState, decision: UserDecision,
                                         fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle user selection of advanced specs."""
        logger.info(f"[DEEP_WORKFLOW] Phase: AWAIT_ADVANCED_SELECTION, decision={decision.value}")

        if decision == UserDecision.UNKNOWN:
            state.awaiting_user_input = True
            return {}

        if decision == UserDecision.SELECT_ALL or decision == UserDecision.YES:
            state.selected_specs = state.discovered_specs
            logger.info(f"[DEEP_WORKFLOW] User selected ALL {len(state.selected_specs)} specs")
        elif decision in [UserDecision.SKIP, UserDecision.NO]:
            state.selected_specs = []
            logger.info(f"[DEEP_WORKFLOW] User skipped advanced specs")
        else:
            # For now, treat as skip - could implement specific selection
            state.selected_specs = []

        # Proceed to vendor analysis
        state.phase = WorkflowPhase.VENDOR_ANALYSIS
        state.awaiting_user_input = False
        return self._handle_vendor_analysis(state, decision, fields, hint)

    def _handle_vendor_analysis(self, state: WorkflowState, decision: UserDecision,
                                fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle vendor analysis."""
        logger.info(f"[DEEP_WORKFLOW] Phase: VENDOR_ANALYSIS")

        state.last_message = f"Analyzing vendors for **{state.product_type}**..."

        try:
            # Build requirements for analysis
            requirements = {
                "product_type": state.product_type,
                "mandatory": state.provided_requirements.get('mandatory', {}),
                "optional": state.provided_requirements.get('optional', {}),
                "advanced": state.selected_specs
            }
            # Run vendor analysis via workflow helper
            result = self.workflow_helper.run_vendor_analysis_step(
                structured_requirements=requirements,
                product_type=state.product_type,
                session_id=state.session_id,
                schema=state.schema
            )

            state.vendor_analysis = result
            state.steps_completed.append('vendor_analysis')

            # Proceed to ranking
            state.phase = WorkflowPhase.RANKING
            return self._handle_ranking(state, decision, fields, hint)

        except Exception as e:
            logger.error(f"[DEEP_WORKFLOW] Vendor analysis error: {e}")
            # Create minimal result
            state.vendor_analysis = {
                "success": False,
                "vendor_matches": [],
                "error": str(e)
            }
            state.phase = WorkflowPhase.RANKING
            return self._handle_ranking(state, decision, fields, hint)

    def _handle_ranking(self, state: WorkflowState, decision: UserDecision,
                        fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle product ranking."""
        logger.info(f"[DEEP_WORKFLOW] Phase: RANKING")

        try:
            # Run ranking via workflow helper
            result = self.workflow_helper.run_ranking_step(
                vendor_analysis=state.vendor_analysis,
                session_id=state.session_id,
                structured_requirements=state.provided_requirements
            )

            state.ranking_result = result
            state.steps_completed.append('ranking')
            state.phase = WorkflowPhase.COMPLETE

            return self._handle_complete(state, decision, fields, hint)

        except Exception as e:
            logger.error(f"[DEEP_WORKFLOW] Ranking error: {e}")
            state.ranking_result = {
                "success": False,
                "overall_ranking": [],
                "error": str(e)
            }
            state.phase = WorkflowPhase.COMPLETE
            return self._handle_complete(state, decision, fields, hint)

    def _handle_complete(self, state: WorkflowState, decision: UserDecision,
                         fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle workflow completion."""
        logger.info(f"[DEEP_WORKFLOW] Phase: COMPLETE")

        state.awaiting_user_input = False

        # Build final message
        ranking = state.ranking_result or {}
        ranked_products = ranking.get('overall_ranking', [])

        if ranked_products:
            state.last_message = (
                f"Product search complete for **{state.product_type}**!\n\n"
                f"Found {len(ranked_products)} matching products from top vendors.\n\n"
                f"Top recommendation: **{ranked_products[0].get('product_name', 'N/A')}** "
                f"by {ranked_products[0].get('vendor', 'N/A')}"
            )
        else:
            state.last_message = (
                f"Product search complete for **{state.product_type}**.\n\n"
                f"No exact matches found. Please try adjusting your requirements."
            )

        return {
            "ready_for_vendor_search": True,
            "final_requirements": {
                "productType": state.product_type,
                "mandatoryRequirements": state.provided_requirements.get('mandatory', {}),
                "optionalRequirements": state.provided_requirements.get('optional', {}),
                "advancedParameters": state.selected_specs
            },
            "analysisResult": {
                "vendorAnalysis": state.vendor_analysis,
                "overallRanking": {
                    "rankedProducts": ranked_products
                }
            }
        }

    def _handle_error(self, state: WorkflowState, decision: UserDecision,
                      fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle error state."""
        state.awaiting_user_input = False
        state.last_message = f"An error occurred: {state.error}"
        return {"error": state.error}

    def _handle_unknown(self, state: WorkflowState, decision: UserDecision,
                        fields: Optional[Dict], hint: Optional[str]) -> Dict:
        """Handle unknown phase."""
        logger.warning(f"[DEEP_WORKFLOW] Unknown phase: {state.phase}")
        state.phase = WorkflowPhase.ERROR
        state.error = f"Unknown workflow phase: {state.phase}"
        return {"error": state.error}

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_or_create_state(self,
                             user_input: Optional[str],
                             session_id: Optional[str],
                             thread_id: Optional[str],
                             decision: Optional[str] = None) -> Optional[WorkflowState]:
        """Get existing state or create new one."""
        # Try to find existing state
        state = self.session_manager.get_state(thread_id=thread_id, session_id=session_id)

        if state:
            logger.info(f"[DEEP_WORKFLOW] Resuming session: {state.thread_id}, phase={state.phase.value}")
            return state

        # Need to create new state
        if not user_input:
            # FIX: If we have a decision but no input (e.g. backend restart + button click),
            # synthesize input to prevent crash
            if decision:
                user_input = f"User selected: {decision}"
                logger.warning(f"[DEEP_WORKFLOW] Creating session from decision '{decision}' (input was empty)")
            else:
                logger.error("[DEEP_WORKFLOW] Cannot create session without user_input or decision")
                return None

        state = self.session_manager.create_session(user_input, session_id)
        logger.info(f"[DEEP_WORKFLOW] Created new session: {state.thread_id}")
        return state

    def _parse_user_decision(self, decision: Optional[str]) -> UserDecision:
        """Parse user decision string into enum."""
        if not decision:
            return UserDecision.UNKNOWN

        decision = decision.lower().strip()

        if any(w in decision for w in ['continue', 'anyway', 'proceed']):
            return UserDecision.CONTINUE
        if any(w in decision for w in ['add', 'fill', 'provide', 'missing']):
            return UserDecision.ADD_FIELDS
        if any(w in decision for w in ['skip', 'no thanks']):
            return UserDecision.SKIP
        if decision in ['yes', 'y', 'ok', 'sure', 'discover']:
            return UserDecision.YES
        if decision in ['no', 'n', 'nope']:
            return UserDecision.NO
        if any(w in decision for w in ['all', 'everything', 'include all']):
            return UserDecision.SELECT_ALL

        return UserDecision.UNKNOWN

    def _normalize_schema(self, schema: Optional[Dict]) -> Dict:
        """Normalize schema format for frontend."""
        if not schema:
            return {"mandatoryRequirements": {}, "optionalRequirements": {}}

        # Handle nested schema
        if "schema" in schema and isinstance(schema["schema"], dict):
            schema = schema["schema"]

        mandatory = (
            schema.get("mandatoryRequirements") or
            schema.get("mandatory_requirements") or
            schema.get("mandatory") or
            {}
        )

        optional = (
            schema.get("optionalRequirements") or
            schema.get("optional_requirements") or
            schema.get("optional") or
            {}
        )

        return {
            "mandatoryRequirements": mandatory,
            "optionalRequirements": optional
        }

    def _error_response(self, error: str) -> Dict:
        """Create error response."""
        return {
            "success": False,
            "error": error,
            "awaiting_user_input": False,
            "completed": False
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_global_orchestrator: Optional[DeepAgenticWorkflowOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_deep_agentic_orchestrator() -> DeepAgenticWorkflowOrchestrator:
    """Get or create the global orchestrator instance."""
    global _global_orchestrator

    with _orchestrator_lock:
        if _global_orchestrator is None:
            _global_orchestrator = DeepAgenticWorkflowOrchestrator()
        return _global_orchestrator


def reset_orchestrator():
    """Reset the global orchestrator (for testing)."""
    global _global_orchestrator
    with _orchestrator_lock:
        _global_orchestrator = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DeepAgenticWorkflowOrchestrator",
    "WorkflowSessionManager",
    "WorkflowState",
    "WorkflowPhase",
    "UserDecision",
    "get_deep_agentic_orchestrator",
    "reset_orchestrator"
]
