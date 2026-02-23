# solution/workflow.py
# =============================================================================
# SOLUTION DEEP AGENT WORKFLOW - LangGraph Integration
# =============================================================================
#
# Connects all Solution Deep Agent components into a LangGraph StateGraph:
#
# Phase 1: Flash Personality Planning
# Phase 2: Semantic Intent Classification
# Phase 3: Context Loading (Memory + Personal)
# Phase 4: Solution Analysis
# Phase 5: Deep Identification (Instruments + Accessories)
# Phase 6: Parallel 3-Source Specification Enrichment
# Phase 7: Sample Input Generation (Isolated)
# Phase 8: Flash Response Composition
#
# Modification path:
#   modification_node
#     ├─► (requirement change) ──► load_context  (re-identification)
#     └─► (BOM edit)           ──► apply_standards ──► compose_response
#
# =============================================================================

import ast
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable

from langgraph.graph import StateGraph, END

from .solution_deep_agent import (
    SolutionDeepAgentState,
    create_solution_deep_agent_state,
    mark_phase_complete,
    add_system_message,
)
from .intent_analyzer import SolutionIntentClassifier
from .context_manager import SolutionContextManager
from .identification_agents import (
    InstrumentIdentificationAgent,
    AccessoryIdentificationAgent,
    identify_instruments_and_accessories_parallel,
)
from .flash_personality import FlashPersonality, ExecutionStrategy
from common.agentic.workflows.specification_utils import build_sample_input
from .orchestration import (
    OrchestrationContext,
    get_current_context,
    get_orchestration_logger,
    get_solution_orchestrator,
)

from common.agentic.deep_agent.memory import DeepAgentMemory
from common.infrastructure.state.checkpointing.local import compile_with_checkpointing
from common.infrastructure.state.context.lock_monitor import with_workflow_lock

from common.services.llm.fallback import create_llm_with_fallback
from common.config import AgenticConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .prompts import (
    SOLUTION_INTENT_PROMPT,
    MODIFICATION_PROCESSING_PROMPT,
    CLARIFICATION_PROMPT,
    RESET_CONFIRMATION_PROMPT,
)
from taxonomy_rag.normalization_agent import TaxonomyNormalizationAgent

logger = logging.getLogger(__name__)

# =============================================================================
# SESSION MEMORY REGISTRY
# DeepAgentMemory contains threading.Lock and is not msgpack-serializable by
# LangGraph's MemorySaver. It is kept here, keyed by session_id, and never
# stored inside the LangGraph state dict.
# =============================================================================
_SESSION_MEMORIES: Dict[str, DeepAgentMemory] = {}


def _get_session_memory(session_id: str) -> DeepAgentMemory:
    """Get or create the DeepAgentMemory for a session."""
    if session_id not in _SESSION_MEMORIES:
        _SESSION_MEMORIES[session_id] = DeepAgentMemory()
    return _SESSION_MEMORIES[session_id]


def _cleanup_session_memory(session_id: str) -> None:
    """Remove memory for a session after workflow completes."""
    _SESSION_MEMORIES.pop(session_id, None)


# =============================================================================
# PROMPT LOADING
# =============================================================================

from common.prompts import SOLUTION_DEEP_AGENT_PROMPTS

_DEEP_AGENT_PROMPTS = SOLUTION_DEEP_AGENT_PROMPTS

def _get_prompts() -> Dict[str, str]:
    """Lazy-load deep agent prompts."""
    return _DEEP_AGENT_PROMPTS


# =============================================================================
# NODE 1: FLASH PERSONALITY PLANNING
# =============================================================================

def personality_plan_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 1: Flash Personality Planning.

    Analyzes input complexity and creates an execution plan
    that determines strategy, tone, and which phases to run.
    """
    logger.info("[SolutionDeepAgent] Phase 1: Flash Personality Planning...")

    try:
        # Build context from conversation history
        ctx_manager = SolutionContextManager()
        ctx_manager.load_history(state.get("conversation_history", []))
        conversation_context = ctx_manager.get_enriched_context()

        flash = FlashPersonality()
        plan = flash.plan(
            user_input=state["user_input"],
            conversation_context=conversation_context,
            personal_context=state.get("personal_context"),
        )

        state["personality_plan"] = {
            "strategy": plan.strategy.value,
            "tone": plan.tone.value,
            "phases_to_run": plan.phases_to_run,
            "skip_enrichment": plan.skip_enrichment,
            "parallel_identification": plan.parallel_identification,
            "max_enrichment_items": plan.max_enrichment_items,
            "context_depth": plan.context_depth,
            "confidence": plan.confidence,
            "reasoning": plan.reasoning,
        }
        state["execution_strategy"] = plan.strategy.value
        state["response_tone"] = plan.tone.value

        add_system_message(state, f"Flash plan: {plan.strategy.value}/{plan.tone.value}")

    except Exception as e:
        logger.warning(f"[SolutionDeepAgent] Planning failed, using defaults: {e}")
        state["personality_plan"] = {"strategy": "full", "tone": "professional"}
        state["execution_strategy"] = "full"
        state["response_tone"] = "professional"

    mark_phase_complete(state, "personality_plan")
    return state


# =============================================================================
# NODE 2: SEMANTIC INTENT CLASSIFICATION
# =============================================================================

def semantic_intent_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 2: Semantic Intent Classification.

    Uses embedding similarity to determine if the input is a solution request.
    Also uses an LLM to refine the intent into specific action categories:
    - REQUIREMENTS (standard flow)
    - MODIFICATION (update existing items)
    - CLARIFICATION (ambiguity resolution)
    - RESET (clear session)
    - CHAT/ROUTER (redirect)
    """
    logger.info("[SolutionDeepAgent] Phase 2: Semantic Intent Classification...")

    # Guard: empty or whitespace-only input
    user_input = state.get("user_input", "")
    if not user_input or not str(user_input).strip():
        logger.warning("[SolutionDeepAgent] Empty user input — skipping intent classification")
        state["is_solution_workflow"] = False
        state["intent_classification"] = {"refined_type": "error", "is_solution": False}
        state["response"] = "Please provide a valid input."
        state["response_data"] = {
            "workflow": "solution",
            "type": "error",
            "message": "Empty input received.",
            "awaiting_selection": False,
        }
        return state

    try:
        # 1. Base Classifier (Embeddings + Keywords)
        classifier = SolutionIntentClassifier()
        result = classifier.classify(
            user_input=state["user_input"],
            conversation_history=state.get("conversation_history", []),
        )

        state["intent_classification"] = {
            "is_solution": result.is_solution,
            "confidence": result.confidence,
            "method": result.method,
            "intent_type": result.intent_type,
            "domain": result.domain,
            "industry": result.industry,
            "solution_indicators": result.solution_indicators,
        }
        state["is_solution_workflow"] = result.is_solution
        state["intent_confidence"] = result.confidence
        state["intent_method"] = result.method

        if result.domain:
            state["provided_requirements"]["domain"] = result.domain
        if result.industry:
            state["provided_requirements"]["industry"] = result.industry

        # 2. LLM Refinement (Deep Intent Check)
        # We run this to distinguish between REQUIREMENTS vs MODIFICATION vs CLARIFICATION vs RESET
        # especially when we have conversation history.
        try:
            llm = create_llm_with_fallback(
                model=AgenticConfig.FLASH_MODEL,
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            
            # Format history for context
            ctx_manager = SolutionContextManager()
            ctx_manager.load_history(state.get("conversation_history", []))
            conversation_context = ctx_manager.get_enriched_context()

            prompt = ChatPromptTemplate.from_template(SOLUTION_INTENT_PROMPT)
            chain = prompt | llm | JsonOutputParser()
            
            llm_intent = chain.invoke({
                "user_input": state["user_input"],
                "conversation_context": conversation_context or "No previous context."
            })
            
            # Refined intent takes precedence for granular actions
            refined_type = llm_intent.get("type", "requirements").lower()
            state["intent_classification"]["refined_type"] = refined_type
            state["intent_classification"]["reasoning"] = llm_intent.get("reasoning", "")
            
            # Update main intent flags based on refined type
            if refined_type == "modification":
                state["is_modification"] = True
                state["is_solution_workflow"] = True
                # [GAP 8] Propagate flag into workflow_data so standards_application_node
                # can distinguish a modification run from a full identification run.
                if "workflow_data" not in state or not isinstance(state.get("workflow_data"), dict):
                    state["workflow_data"] = {}
                state["workflow_data"]["is_modification"] = True
            elif refined_type == "concise_bom":
                state["is_solution_workflow"] = True
            elif refined_type == "clarification":
                state["clarification_needed"] = True
                state["is_solution_workflow"] = True
            elif refined_type == "reset":
                state["is_solution_workflow"] = True  # Handle inside workflow
            elif refined_type in ("router_needed", "invalid_input"):
                state["is_solution_workflow"] = False

            logger.info(f"[SolutionDeepAgent] Refined Intent: {refined_type}")

        except Exception as llm_err:
            logger.warning(f"[SolutionDeepAgent] LLM intent refinement failed: {llm_err}")
            state["intent_classification"]["refined_type"] = result.intent_type

        add_system_message(
            state,
            f"Intent: {state['intent_classification'].get('refined_type')} "
            f"(conf={result.confidence:.2f})"
        )

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Intent classification failed: {e}")
        state["is_solution_workflow"] = True  # Default to solution
        state["error"] = str(e)

    mark_phase_complete(state, "classify_intent")
    return state


# =============================================================================
# NODE 3: CONTEXT LOADING
# =============================================================================

def load_context_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 3: Load Context (Memory + Personal).

    Loads conversation memory, personal context, and active thread analysis.
    Populates the DeepAgentMemory with relevant data.
    """
    logger.info("[SolutionDeepAgent] Phase 3: Loading Context...")

    try:
        ctx_manager = SolutionContextManager()

        # Load conversation history
        ctx_manager.load_history(state.get("conversation_history", []))
        # Add current user input
        ctx_manager.add_message("user", state["user_input"])

        # Load personal context
        user_id = state.get("user_id", "")
        if user_id:
            ctx_manager.load_personal_context(user_id)

        # Analyze active thread
        thread_context = ctx_manager.analyze_active_thread()
        state["active_thread_context"] = thread_context

        # Get enriched context for downstream nodes
        enriched_context = ctx_manager.get_enriched_context()

        # Store extracted entities in provided_requirements
        entities = ctx_manager.get_extracted_entities()
        if entities["specifications"]:
            state["provided_requirements"].update(entities["specifications"])
        if entities["safety_requirements"]:
            state["provided_requirements"].update(entities["safety_requirements"])
        if entities["vendors"]:
            state["provided_requirements"]["preferred_vendors"] = entities["vendors"]

        # Store personal context
        state["personal_context"] = ctx_manager.get_personal_context()

        # Initialize memory in the session registry (NOT in state — avoids
        # LangGraph msgpack serialization error on DeepAgentMemory with Lock)
        memory = _get_session_memory(state.get("session_id", ""))

        # Inject taxonomy into memory
        try:
            from taxonomy_rag import inject_taxonomy_into_memory
            inject_taxonomy_into_memory(memory)
        except Exception as tax_err:
            logger.warning(f"[SolutionDeepAgent] Failed to inject taxonomy: {tax_err}")

        # Store user context in memory
        from common.agentic.deep_agent.memory import UserContextMemory
        memory.store_user_context(UserContextMemory(
            raw_input=state["user_input"],
            domain=state.get("provided_requirements", {}).get("domain", ""),
            process_type="",
            industry=state.get("provided_requirements", {}).get("industry", ""),
            safety_requirements=entities.get("safety_requirements", {}),
            environmental_conditions={},
            constraint_keywords=[],
            key_parameters=entities.get("specifications", {}),
        ))

        # Store enriched context for use by identification
        state["instrument_context"] = enriched_context

        add_system_message(
            state,
            f"Context loaded: {thread_context.get('message_count', 0)} messages, "
            f"{len(entities.get('specifications', {}))} specs, "
            f"{len(entities.get('vendors', []))} vendors"
        )

    except Exception as e:
        logger.warning(f"[SolutionDeepAgent] Context loading partially failed: {e}")
        state["instrument_context"] = state["user_input"]

    mark_phase_complete(state, "load_context")
    return state


# =============================================================================
# NODE 4: SOLUTION ANALYSIS
# =============================================================================

def solution_analysis_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 4: Analyze Solution Context.

    Uses the SOLUTION_ANALYSIS_DEEP prompt with conversation context
    to extract domain, process, safety, and parameter information.
    """
    logger.info("[SolutionDeepAgent] Phase 4: Solution Analysis...")

    try:
        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=180,  # Solution analysis can be complex — allow 3 minutes
        )

        # Use the deep solution analysis prompt
        prompts = _get_prompts()
        prompt_text = prompts.get("SOLUTION_ANALYSIS_DEEP", "")

        # Fallback to original prompt if deep version not found
        if not prompt_text:
            from common.prompts import SOLUTION_DEEP_AGENT_PROMPTS
            prompt_text = SOLUTION_DEEP_AGENT_PROMPTS.get("SOLUTION_DESIGN", "")

        prompt = ChatPromptTemplate.from_template(prompt_text)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        # Build invoke params based on available template variables
        invoke_params = {"solution_description": state["user_input"]}

        # Add context if the prompt supports it
        if "{conversation_context}" in prompt_text:
            invoke_params["conversation_context"] = state.get("instrument_context", "")
        if "{personal_context}" in prompt_text:
            invoke_params["personal_context"] = str(state.get("personal_context", {}))

        result = chain.invoke(invoke_params)

        state["solution_analysis"] = result
        state["solution_name"] = result.get("solution_name", "Industrial Solution")

        # Extract context for instrument identification
        context = result.get("context_for_instruments", state["user_input"])
        # Prepend enriched conversation context
        existing_context = state.get("instrument_context", "")
        if existing_context:
            state["instrument_context"] = f"{context}\n\n{existing_context}"
        else:
            state["instrument_context"] = context

        # Extract safety requirements
        safety = result.get("safety_requirements", {})
        if safety.get("sil_level") or safety.get("sil_rating"):
            state["provided_requirements"]["sil_level"] = (
                safety.get("sil_level") or safety.get("sil_rating")
            )
        if safety.get("hazardous_area"):
            state["provided_requirements"]["hazardous_area"] = True

        add_system_message(
            state,
            f"Solution: {state['solution_name']} "
            f"({result.get('industry', 'Unknown')})"
        )

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Solution analysis failed: {e}")
        state["solution_analysis"] = {}
        state["solution_name"] = "Solution"
        state["instrument_context"] = state["user_input"]
        state["error"] = str(e)

    mark_phase_complete(state, "analyze_solution")
    return state


# =============================================================================
# NODE 4b: REASONING CHAIN
# =============================================================================

def reasoning_chain_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 4b: Chain-of-Thought Reasoning.

    Performs step-by-step reasoning before identification:
    - Decomposes the solution into measurement/control/safety points
    - Decides which items need standards deep agent calls
    - Plans parallel enrichment batches
    - Identifies cross-over risks
    """
    logger.info("[SolutionDeepAgent] Phase 4b: Reasoning Chain...")

    try:
        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        prompts = _get_prompts()
        prompt_text = prompts.get("REASONING_CHAIN", "")

        if not prompt_text:
            logger.debug("[SolutionDeepAgent] REASONING_CHAIN prompt not found, skipping")
            mark_phase_complete(state, "reasoning_chain")
            return state

        plan = state.get("personality_plan", {})
        prompt = ChatPromptTemplate.from_template(prompt_text)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "solution_description": state["user_input"],
            "solution_analysis": json.dumps(state.get("solution_analysis", {})),
            "conversation_context": state.get("instrument_context", ""),
            "execution_strategy": plan.get("strategy", "full"),
            "max_parallel": plan.get("max_enrichment_items", 5),
        })

        # Store reasoning output for downstream nodes
        state["solution_analysis"]["reasoning_chain"] = result

        # Extract standards domains from reasoning for enrichment phase
        domain_reasoning = result.get("domain_reasoning", {})
        if domain_reasoning.get("standards_files_to_consult"):
            state["provided_requirements"]["standards_files"] = (
                domain_reasoning["standards_files_to_consult"]
            )
        if domain_reasoning.get("sil_required"):
            state["standards_detected"] = True

        # Extract orchestration plan
        orch_plan = result.get("orchestration_plan", {})
        if orch_plan.get("standards_domains"):
            state["provided_requirements"]["standards_domains"] = (
                orch_plan["standards_domains"]
            )

        add_system_message(
            state,
            f"Reasoning: ~{result.get('decomposition', {}).get('estimated_instruments', '?')} instruments, "
            f"standards={'yes' if domain_reasoning.get('sil_required') else 'maybe'}"
        )

    except Exception as e:
        logger.warning(f"[SolutionDeepAgent] Reasoning chain failed (non-blocking): {e}")

    mark_phase_complete(state, "reasoning_chain")
    return state


# =============================================================================
# NODE 5: DEEP IDENTIFICATION
# =============================================================================

def deep_identification_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 5: Deep Identification of Instruments & Accessories.

    Uses deep agent identification with:
    - Memory-aware prompts
    - Conversation context enrichment
    - Adaptive prompt optimization
    - Failure learning
    """
    logger.info("[SolutionDeepAgent] Phase 5: Deep Identification...")

    tools_called = list(state.get("tools_called") or [])
    tool_results_summary = dict(state.get("tool_results_summary") or {})
    quality_flags = list(state.get("quality_flags") or [])

    try:
        context = state.get("instrument_context", state["user_input"])
        memory = _SESSION_MEMORIES.get(state.get("session_id", ""))

        # Run identification
        inst_result, acc_result = identify_instruments_and_accessories_parallel(
            requirements=context,
            context=json.dumps(state.get("solution_analysis", {})),
            conversation_context=state.get("active_thread_context", {}).get("context_summary", ""),
            memory=memory,
        )

        # Normalize items with context awareness
        norm_agent = TaxonomyNormalizationAgent(memory=memory)

        # Process instruments
        instruments = []
        if inst_result.get("success"):
            instruments = inst_result.get("instruments", [])
            
            # Normalize instruments
            try:
                instruments = norm_agent.normalize_with_context(
                    items=instruments,
                    user_input=state["user_input"],
                    history=state.get("conversation_history", []),
                    item_type="instrument"
                )
            except Exception as e:
                logger.warning(f"[SolutionDeepAgent] Instrument normalization failed: {e}")

            solution_analysis = state.get("solution_analysis", {})

            # Deduplicate instruments: keep unique by (category, product_name).
            seen_instruments: dict = {}
            deduped_instruments = []
            for inst in instruments:
                cat = (inst.get("category") or "").strip().lower()
                name = (inst.get("product_name") or inst.get("name") or "").strip().lower()
                dedup_key = f"{cat}||{name}"
                if dedup_key not in seen_instruments:
                    seen_instruments[dedup_key] = True
                    deduped_instruments.append(inst)
                else:
                    logger.info(f"[SolutionDeepAgent] Deduplicating instrument: {name} ({cat})")
            instruments = deduped_instruments

            for inst in instruments:
                sample_input = inst.get("sample_input", "")
                safety = solution_analysis.get("safety_requirements", {})
                if safety.get("sil_level"):
                    sample_input += f" with {safety['sil_level']} rating"
                key_params = solution_analysis.get("key_parameters", {})
                if key_params.get("temperature_range"):
                    sample_input += f" for {key_params['temperature_range']} temperature"
                inst["sample_input"] = sample_input
                inst["solution_purpose"] = f"Required for {state.get('solution_name', 'solution')}"

        state["identified_instruments"] = instruments

        # Process accessories
        accessories = []
        if acc_result.get("success"):
            accessories = acc_result.get("accessories", [])
            
            try:
                accessories = norm_agent.normalize_with_context(
                    items=accessories,
                    user_input=state["user_input"],
                    history=state.get("conversation_history", []),
                    item_type="accessory"
                )
            except Exception as e:
                logger.warning(f"[SolutionDeepAgent] Accessory normalization failed: {e}")

            # Deduplicate accessories: the LLM generates one entry per instrument for
            # shared accessory types (e.g. "Cable Gland" x11 for 11 instruments).
            # Merge duplicates by (category, accessory_name) key — keep first occurrence,
            # accumulate quantity, and collect all related_instrument references.
            seen_accessories: dict = {}
            deduped_accessories = []
            for acc in accessories:
                cat = (acc.get("category") or "").strip().lower()
                name = (acc.get("accessory_name") or acc.get("name") or "").strip().lower()
                dedup_key = f"{cat}||{name}"
                if dedup_key and dedup_key not in seen_accessories:
                    seen_accessories[dedup_key] = len(deduped_accessories)
                    deduped_accessories.append(dict(acc))
                else:
                    # Merge into the existing entry: accumulate quantity and related_instrument
                    existing = deduped_accessories[seen_accessories[dedup_key]]
                    existing["quantity"] = (existing.get("quantity") or 1) + (acc.get("quantity") or 1)
                    # Collect all parent instruments this accessory serves
                    existing_related = existing.get("related_instrument", "")
                    new_related = acc.get("related_instrument", "")
                    if new_related and new_related not in existing_related:
                        existing["related_instrument"] = f"{existing_related}, {new_related}" if existing_related else new_related
            accessories = deduped_accessories
            logger.info(
                f"[SolutionDeepAgent] Accessories after deduplication: {len(accessories)} "
                f"(from {len(acc_result.get('accessories', []))} raw)"
            )

            for acc in accessories:
                acc_sample = f"{acc.get('category', 'Accessory')} for {acc.get('related_instrument', 'instruments')}"
                safety = state.get("solution_analysis", {}).get("safety_requirements", {})
                if safety.get("hazardous_area"):
                    acc_sample += " (explosion-proof required)"
                acc["sample_input"] = acc_sample

        state["identified_accessories"] = accessories

        # Build unified item list
        all_items = []
        item_number = 1

        # Get thread IDs
        workflow_thread_id = state.get("workflow_thread_id")
        main_thread_id = state.get("main_thread_id")

        # Generate thread IDs for items
        thread_manager = None
        try:
            from common.infrastructure.state.execution.thread_manager import HierarchicalThreadManager
            thread_manager = HierarchicalThreadManager
        except ImportError:
            pass

        # Add instruments
        for inst in instruments:
            inst_name = inst.get("product_name", "Unknown_Instrument")
            item_thread_id = None

            if thread_manager and workflow_thread_id:
                try:
                    sanitized = inst_name.replace(" ", "_").replace("/", "_")
                    item_thread_id = thread_manager.generate_item_thread_id(
                        workflow_thread_id=workflow_thread_id,
                        item_type="instrument",
                        item_name=sanitized,
                        item_number=item_number,
                    )
                except Exception:
                    pass

            all_items.append({
                "number": item_number,
                "type": "instrument",
                "name": inst_name,
                "category": inst.get("category", "Instrument"),
                "quantity": inst.get("quantity", 1),
                "specifications": inst.get("specifications", {}),
                "sample_input": inst.get("sample_input", ""),
                "purpose": inst.get("solution_purpose", ""),
                "strategy": inst.get("strategy", ""),
                "item_thread_id": item_thread_id,
                "workflow_thread_id": workflow_thread_id,
                "main_thread_id": main_thread_id,
            })
            item_number += 1

        # Add accessories
        for acc in accessories:
            raw_category = acc.get("category", "Accessory")
            acc_name = acc.get("accessory_name", "Unknown Accessory")

            # Smart category: extract product type from name if category is generic
            if raw_category.lower() in ["accessories", "accessory", ""]:
                if " for " in acc_name:
                    smart_category = acc_name.split(" for ")[0].strip() or raw_category
                else:
                    smart_category = acc_name
            else:
                smart_category = raw_category

            item_thread_id = None
            if thread_manager and workflow_thread_id:
                try:
                    sanitized = acc_name.replace(" ", "_").replace("/", "_")
                    item_thread_id = thread_manager.generate_item_thread_id(
                        workflow_thread_id=workflow_thread_id,
                        item_type="accessory",
                        item_name=sanitized,
                        item_number=item_number,
                    )
                except Exception:
                    pass

            all_items.append({
                "number": item_number,
                "type": "accessory",
                "name": acc_name,
                "category": smart_category,
                "quantity": acc.get("quantity", 1),
                "sample_input": acc.get("sample_input", ""),
                "purpose": f"Supports {acc.get('related_instrument', 'instruments')}",
                "related_instrument": acc.get("related_instrument", ""),
                "item_thread_id": item_thread_id,
                "workflow_thread_id": workflow_thread_id,
                "main_thread_id": main_thread_id,
            })
            item_number += 1

        # Fallback if nothing found
        if not all_items:
            analysis = state.get("solution_analysis", {})
            industry = analysis.get("industry", "Industrial")
            process_type = analysis.get("process_type", "process")
            all_items = [{
                "number": 1,
                "type": "instrument",
                "name": f"{industry} Instrument",
                "category": "Process Instrument",
                "quantity": 1,
                "specifications": {},
                "sample_input": f"Industrial instrument for {process_type} in {industry}",
                "purpose": f"General instrumentation for {state.get('solution_name', 'solution')}",
            }]

        state["all_items"] = all_items
        state["total_items"] = len(all_items)

        # Tool audit
        tools_called.append("identify_instruments")
        tools_called.append("identify_accessories")
        tool_results_summary["identify_instruments"] = {
            "count": len(instruments),
            "success": inst_result.get("success", False),
            "attempt": inst_result.get("attempt", 1),
        }
        tool_results_summary["identify_accessories"] = {
            "count": len(accessories),
            "success": acc_result.get("success", False),
            "method": acc_result.get("method", "unknown"),
        }

        add_system_message(
            state,
            f"Identified {len(instruments)} instruments + {len(accessories)} accessories"
        )

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Identification failed: {e}")
        state["error"] = str(e)
        quality_flags.append(f"identification_error: {str(e)[:80]}")

    state["tools_called"] = tools_called
    state["tool_results_summary"] = tool_results_summary
    state["quality_flags"] = quality_flags
    mark_phase_complete(state, "identify_items")
    return state


# =============================================================================
# NODE 6: STANDARDS-BASED SPECIFICATION ENRICHMENT
# =============================================================================

def parallel_enrichment_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 6: Standards-Based Specification Enrichment (Simplified).

    For each identified instrument/accessory:
    1. Retrieve relevant standards content from common/standards/ RAG (greedy, no domain filter)
    2. Call LLM with SOLUTION_STANDARDS_EXTRACTION prompt to extract ALL applicable specs
    3. LLM tags each extracted value with [STANDARDS]
    4. Merge into item's specifications dict — existing values (user/inferred) take precedence
    """
    logger.info("[SolutionDeepAgent] Phase 6: Standards-Based Specification Enrichment...")

    all_items = state.get("all_items", [])
    plan = state.get("personality_plan", {})
    tools_called = list(state.get("tools_called") or [])
    tool_results_summary = dict(state.get("tool_results_summary") or {})
    quality_flags = list(state.get("quality_flags") or [])

    if not all_items:
        logger.warning("[SolutionDeepAgent] No items to enrich")
        state["tools_called"] = tools_called
        state["tool_results_summary"] = tool_results_summary
        state["quality_flags"] = quality_flags
        mark_phase_complete(state, "enrich_specs")
        return state

    # Check if enrichment should be skipped (FAST strategy)
    if plan.get("skip_enrichment", False):
        logger.info("[SolutionDeepAgent] Skipping enrichment (FAST strategy)")
        add_system_message(state, "Enrichment skipped (fast mode)")
        state["tools_called"] = tools_called
        state["tool_results_summary"] = tool_results_summary
        state["quality_flags"] = quality_flags
        mark_phase_complete(state, "enrich_specs")
        return state

    try:
        from common.rag.standards import enrich_identified_items_with_standards
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from common.prompts import STANDARDS_DEEP_AGENT_PROMPTS

        extraction_prompt_text = STANDARDS_DEEP_AGENT_PROMPTS.get("SOLUTION_STANDARDS_EXTRACTION", "")

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=180,
        )

        def _enrich_single_item_with_standards(item: dict) -> dict:
            """
            For a single item:
            1. Retrieve relevant standards content via RAG (greedy, top_k=6)
            2. Call LLM with SOLUTION_STANDARDS_EXTRACTION to extract specs tagged [STANDARDS]
            3. Merge into item specs — existing values take precedence
            """
            try:
                item_desc = (
                    f"{item.get('name', '')} ({item.get('category', '')})\n"
                    f"Specifications: {json.dumps(item.get('specifications', {}))}\n"
                    f"Context: {item.get('sample_input', '')}"
                )

                # Step 1: Retrieve standards content
                enriched_list = enrich_identified_items_with_standards(
                    items=[item.copy()],
                    product_type=item.get("category"),
                    top_k=6,
                    max_workers=1,
                )
                standards_info = (
                    enriched_list[0].get("standards_info", {}) if enriched_list else {}
                )
                standards_content = json.dumps(standards_info, indent=2) if standards_info else ""

                if not standards_content or not extraction_prompt_text:
                    logger.debug(
                        f"[SolutionDeepAgent] No standards content for '{item.get('name')}'; skipping LLM extraction"
                    )
                    return item

                # Step 2: LLM extracts and tags specs with [STANDARDS]
                prompt = ChatPromptTemplate.from_template(extraction_prompt_text)
                chain = prompt | llm | JsonOutputParser()
                result = chain.invoke({
                    "item_description": item_desc,
                    "standards_content": standards_content,
                })

                extracted_specs = result.get("specifications", {})

                # Step 3: Merge — existing specs take precedence
                merged_specs = dict(item.get("specifications") or {})
                for key, val in extracted_specs.items():
                    if key not in merged_specs:
                        # Guarantee the [STANDARDS] tag for the UI
                        val_str = str(val).replace("[STANDARDS]", "").strip()
                        merged_specs[key] = f"{val_str} [STANDARDS]"

                enriched_item = dict(item)
                enriched_item["specifications"] = merged_specs
                enriched_item["standards_applied"] = True
                return enriched_item

            except Exception as exc:
                logger.warning(
                    f"[SolutionDeepAgent] Standards enrichment failed for "
                    f"'{item.get('name', 'unknown')}': {exc}"
                )
                return item

        # ── Run enrichment via identity-aware orchestrator ──────────────────────
        start_ts = time.time()
        enriched_items = list(all_items)  # preserve order slot-by-slot

        # Resolve orchestration context from state
        orch_ctx_data = state.get("orchestration_ctx") or {}
        if orch_ctx_data:
            root_orch_ctx = OrchestrationContext.from_dict(orch_ctx_data)
        else:
            root_orch_ctx = OrchestrationContext.root(
                session_id=state.get("session_id", "default")
            )

        orchestrator = get_solution_orchestrator()

        results = orchestrator.run_parallel(
            fn=_enrich_single_item_with_standards,
            items=all_items,
            root_ctx=root_orch_ctx,
            label_fn=lambda i: f"enrich:{i.get('category', i.get('name', '?'))}",
            timeout_seconds=180.0,
        )

        # Rebuild list in original order — zip preserves submission order
        for idx, (iid, enriched) in enumerate(results.items()):
            if isinstance(enriched, dict) and not enriched.get("_orchestration_error"):
                enriched_items[idx] = enriched
            else:
                quality_flags.append(
                    f"enrichment_thread_error: {str(enriched.get('error', '?'))[:80]}"
                )

        state["all_items"] = enriched_items
        processing_ms = int((time.time() - start_ts) * 1000)

        standards_applied_count = sum(1 for i in enriched_items if i.get("standards_applied"))
        tools_called.append("standards_enrichment")
        tool_results_summary["standards_enrichment"] = {
            "items_enriched": standards_applied_count,
            "total_items": len(enriched_items),
            "processing_time_ms": processing_ms,
        }
        add_system_message(
            state,
            f"Standards enrichment: {standards_applied_count}/{len(enriched_items)} items enriched in {processing_ms}ms"
        )
        logger.info(
            "[SolutionDeepAgent] Standards enrichment complete: %d/%d items in %dms",
            standards_applied_count, len(enriched_items), processing_ms,
        )

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Standards enrichment node failed: {e}")
        import traceback
        traceback.print_exc()
        quality_flags.append(f"standards_enrichment_error: {str(e)[:80]}")

    state["tools_called"] = tools_called
    state["tool_results_summary"] = tool_results_summary
    state["quality_flags"] = quality_flags
    mark_phase_complete(state, "enrich_specs")
    return state


# =============================================================================
# NODE 7: SAMPLE INPUT GENERATION
# =============================================================================

def sample_input_generation_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 7: Generate Isolated Sample Inputs.

    Each item gets its sample_input generated using ONLY its own specifications.
    Includes cross-over validation.
    """
    logger.info("[SolutionDeepAgent] Phase 7: Sample Input Generation...")

    try:
        all_items = state.get("all_items", [])

        # Generate sample inputs with natural language prose
        for item in all_items:
            existing_sample = item.get("sample_input", "").strip()
            new_spec_sample = build_sample_input(item).strip()
            
            if existing_sample and new_spec_sample:
                # Clean up trailing periods to prevent double punctuation
                existing_clean = existing_sample.rstrip(".")
                item["sample_input"] = f"{existing_clean}. {new_spec_sample}"
            else:
                item["sample_input"] = new_spec_sample or existing_sample

        # Attach parent provenance to every item so the frontend can forward
        # these IDs when it dispatches a child product-search for that item.
        parent_session_id = state.get("session_id", "")
        parent_instance_id = state.get("workflow_thread_id", "")
        for item in all_items:
            item["parent_session_id"] = parent_session_id
            item["parent_instance_id"] = parent_instance_id
            item["parent_workflow"] = "solution_deep_agent"

        state["all_items"] = all_items

        add_system_message(
            state,
            f"Sample inputs generated for {len(all_items)} items "
            f"(isolation: structural via orchestration context)"
        )

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Sample input generation failed: {e}")

    mark_phase_complete(state, "generate_samples")
    return state


# =============================================================================
# NODE 8: FLASH RESPONSE COMPOSITION
# =============================================================================

def flash_response_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 8: Flash Response Composition.

    Uses the Flash personality to compose the final response
    with appropriate tone and formatting.
    Also fetches generic images for items.
    """
    logger.info("[SolutionDeepAgent] Phase 8: Flash Response Composition...")

    try:
        flash = FlashPersonality()
        plan_dict = state.get("personality_plan", {})

        # Reconstruct ExecutionPlan from dict
        from .flash_personality import ExecutionPlan, ExecutionStrategy, ResponseTone
        plan = ExecutionPlan(
            strategy=ExecutionStrategy(plan_dict.get("strategy", "full")),
            tone=ResponseTone(plan_dict.get("tone", "professional")),
            phases_to_run=plan_dict.get("phases_to_run", []),
            skip_enrichment=plan_dict.get("skip_enrichment", False),
            parallel_identification=plan_dict.get("parallel_identification", True),
            max_enrichment_items=plan_dict.get("max_enrichment_items", 10),
            context_depth=plan_dict.get("context_depth", "moderate"),
            confidence=plan_dict.get("confidence", 0.5),
            reasoning=plan_dict.get("reasoning", ""),
        )

        # Compose response with personality
        response = flash.compose_response(
            items=state.get("all_items", []),
            solution_name=state.get("solution_name", "Solution"),
            solution_analysis=state.get("solution_analysis", {}),
            plan=plan,
            total_items=state.get("total_items", 0),
        )

        state["response"] = response

        # Fetch generic images
        try:
            from common.services.azure.image_utils import fetch_generic_images_batch

            product_types_map = {}
            all_items = state.get("all_items", [])
            for idx, item in enumerate(all_items):
                product_type = (item.get("name") or item.get("category", "")).strip()
                if product_type:
                    if product_type not in product_types_map:
                        product_types_map[product_type] = []
                    product_types_map[product_type].append(idx)

            if product_types_map:
                image_results = fetch_generic_images_batch(
                    list(product_types_map.keys()), max_parallel_cache_checks=5
                )
                images_attached = 0
                for product_type, indices in product_types_map.items():
                    image_data = image_results.get(product_type)
                    if image_data and image_data.get("url"):
                        for idx in indices:
                            all_items[idx]["imageUrl"] = image_data["url"]
                            all_items[idx]["image_url"] = image_data["url"]
                            images_attached += 1
                logger.info(f"[SolutionDeepAgent] Images: {images_attached}/{len(all_items)}")

        except Exception as img_err:
            logger.warning(f"[SolutionDeepAgent] Image fetch failed: {img_err}")

        # Build response_data
        solution_name = state.get("solution_name", "Solution")
        modification_diff = state.get("modification_diff", {})
        workflow_data = state.get("workflow_data") or {}
        intent_cls    = state.get("intent_classification") or {}
        total_items   = state.get("total_items", 0)

        # [GAP 4] Build agent_path execution trace
        agent_path = [
            {"phase": phase}
            for phase in (state.get("phases_completed") or [])
        ]

        # [GAP 4] Emit full router sub-fields (matches reference file schema)
        router_category = {
            "routed_to":          workflow_data.get("routed_to", ""),
            "target_page":        workflow_data.get("router_target_page", ""),
            "requires_routing":   workflow_data.get("router_requires_routing", False),
            "url":                workflow_data.get("router_url", ""),
            "intent_type":        intent_cls.get("intent_type", ""),
            "confidence":         intent_cls.get("confidence", 0.0),
        }

        response_data: Dict[str, Any] = {
            # ── Core fields ──────────────────────────────────────────────────
            "workflow":            "solution",
            "solution_name":      solution_name,
            "solution_analysis":  state.get("solution_analysis", {}),
            "project_name":       solution_name,
            "items":              state.get("all_items", []),
            "total_items":        total_items,
            "awaiting_selection": True,
            "instructions": (
                f"Reply with item number (1-{total_items}) to get product recommendations"
            ),
            "intent_classification": intent_cls,
            "crossover_validation":  state.get("crossover_validation", {}),
            "execution_strategy":    state.get("execution_strategy", "full"),
            "processing_time_ms":    state.get("processing_time_ms", 0),
            # ── Quality audit ────────────────────────────────────────────────
            "identification_quality_score": state.get("identification_quality_score", 0),
            "enrichment_quality_score":     state.get("enrichment_quality_score", 0),
            "quality_flags":    state.get("quality_flags", []),
            "tools_called":     state.get("tools_called", []),
            "tool_results_summary": state.get("tool_results_summary", {}),
            "phases_completed": state.get("phases_completed", []),
            # ── [GAP 4] Extended fields expected by the reference file ────────
            "open_dashboard":       workflow_data.get("open_dashboard", False),
            "sample_input":         workflow_data.get("sample_input", ""),
            "standards_applied":    workflow_data.get("standards_applied", False),
            "awaiting_confirmation": workflow_data.get(
                "awaiting_single_item_confirmation", False
            ),
            # [GAP 7] Clarification context (why clarification was requested)
            "clarification_context": state.get("clarification_context") or {},
            # [GAP 4] Agent execution path trace
            "agent_path":           agent_path,
            # [GAP 4] Full router sub-fields
            "router_category":      router_category,
            "router_target_page":   workflow_data.get("router_target_page", ""),
            "router_requires_routing": workflow_data.get("router_requires_routing", False),
            "router_url":           workflow_data.get("router_url", ""),
        }

        # Include modification changes if this was a modification pass
        if modification_diff.get("changes"):
            response_data["changes_made"]         = modification_diff["changes"]
            response_data["modification_summary"] = modification_diff.get("summary", "")
            response_data["modification_type"]    = modification_diff.get("operation_type", "")

        # Bubble up any non-fatal warnings
        if state.get("quality_flags"):
            warnings = [
                f for f in state["quality_flags"]
                if any(kw in f for kw in ("error", "failed", "warning"))
            ]
            if warnings:
                response_data["warnings"] = warnings

        state["response_data"] = response_data

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Response composition failed: {e}")
        # Fallback
        lines = [f"**{state.get('solution_name', 'Solution')} - Identified Items**\n"]
        for item in state.get("all_items", []):
            lines.append(f"{item.get('number', '?')}. {item.get('name', 'Unknown')} ({item.get('category', '')})")
        lines.append(f"\nReply with an item number to search for products.")
        state["response"] = "\n".join(lines)
        state["response_data"] = {
            "workflow": "solution",
            "items": state.get("all_items", []),
            "total_items": state.get("total_items", 0),
            "awaiting_selection": True,
        }

    state["current_phase"] = "complete"
    mark_phase_complete(state, "compose_response")

    logger.info(
        f"[SolutionDeepAgent] Complete: {state.get('total_items', 0)} items, "
        f"{state.get('processing_time_ms', 0)}ms"
    )

    return state


# =============================================================================
# STANDARDS CALL PLANNER (helper for parallel_enrichment_node)
# =============================================================================

def _plan_standards_calls(
    state: SolutionDeepAgentState,
    items: list,
) -> dict:
    """
    Use STANDARDS_DEEP_AGENT_CALL prompt to decide which standards deep agent
    calls to make, for which items, targeting which documents.

    Returns the call plan dict or empty dict if prompt unavailable.
    """
    try:
        prompts = _get_prompts()
        prompt_text = prompts.get("STANDARDS_DEEP_AGENT_CALL", "")
        if not prompt_text:
            return {}

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=30,
        )

        solution_analysis = state.get("solution_analysis", {})
        reasoning_chain = solution_analysis.get("reasoning_chain", {})

        # Build simplified item list for the prompt (avoid token overflow)
        simplified_items = [
            {
                "name": item.get("name", ""),
                "category": item.get("category", ""),
                "type": item.get("type", "instrument"),
                "specifications": {
                    k: v for k, v in list(item.get("specifications", {}).items())[:5]
                },
            }
            for item in items[:20]
        ]

        prompt = ChatPromptTemplate.from_template(prompt_text)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "identified_items": json.dumps(simplified_items, indent=2),
            "domain": solution_analysis.get("industry", "General Industrial"),
            "safety_requirements": json.dumps(
                solution_analysis.get("safety_requirements", {})
            ),
            "reasoning_chain": json.dumps(reasoning_chain),
        })

        return result

    except Exception as e:
        logger.debug(f"[SolutionDeepAgent] Standards call planning failed (non-blocking): {e}")
        return {}


# =============================================================================
# NODE R1: REFLECT AFTER IDENTIFICATION
# =============================================================================

def reflect_identification_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Reflection: Evaluate identification quality and decide next step.

    Algorithmic scoring (0-100) based on:
    - Item count (0 = 0pts, <3 = 30pts, <8 = 60pts, ≥8 = 85pts)
    - Instrument count bonus (up to +15pts)
    - Fallback item penalty (-20pts)

    Decisions:
    - "enrich"      — proceed to parallel enrichment (default)
    - "skip_enrich" — skip enrichment (FAST strategy)
    - "error"       — hard error with 0 items, route to compose_response
    """
    logger.info("[SolutionDeepAgent] Reflecting on identification quality...")

    all_items = state.get("all_items", [])
    identified_instruments = state.get("identified_instruments", [])
    total_items = len(all_items)
    error = state.get("error")

    tools_called = list(state.get("tools_called") or [])
    tool_results_summary = dict(state.get("tool_results_summary") or {})
    quality_flags = list(state.get("quality_flags") or [])

    tools_called.append("reflect_identification")

    # Compute score
    if total_items == 0:
        score = 0
        quality_flags.append("identification_zero_items")
    elif total_items < 3:
        score = 30
        quality_flags.append(f"identification_low_items: {total_items}")
    elif total_items < 8:
        score = 60
    else:
        score = 85

    # Bonus: real instruments found
    if identified_instruments:
        score = min(score + min(len(identified_instruments) * 3, 15), 100)

    # Penalty: only fallback generic item returned
    if total_items == 1:
        fallback_name = (all_items[0].get("name") or "").lower()
        if "instrument" in fallback_name and not identified_instruments:
            score = max(score - 20, 0)
            quality_flags.append("identification_used_fallback")

    # Decide routing
    plan = state.get("personality_plan", {})
    if total_items == 0 and error:
        decision = "error"
    elif plan.get("skip_enrichment", False):
        decision = "skip_enrich"
        logger.info("[SolutionDeepAgent] reflect_identification: skip_enrich (FAST plan)")
    else:
        decision = "enrich"

    tool_results_summary["reflect_identification"] = {
        "total_items": total_items,
        "instrument_count": len(identified_instruments),
        "score": score,
        "decision": decision,
    }

    logger.info(
        "[SolutionDeepAgent] reflect_identification: score=%d items=%d decision=%s",
        score, total_items, decision,
    )

    state["identification_quality_score"] = score
    state["tools_called"] = tools_called
    state["tool_results_summary"] = tool_results_summary
    state["quality_flags"] = quality_flags
    state["_reflect_id_decision"] = decision
    mark_phase_complete(state, "reflect_identification")
    return state


def route_after_identification(state: SolutionDeepAgentState) -> str:
    """Conditional edge: route based on identification reflection decision."""
    return state.get("_reflect_id_decision", "enrich")


# =============================================================================
# NODE R2: REFLECT AFTER ENRICHMENT
# =============================================================================

def reflect_enrichment_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Reflection: Evaluate enrichment quality (non-blocking — always routes to generate_samples).

    Algorithmic scoring (0-100) based on:
    - Average spec count per item (threshold bands)
    - Standards enrichment bonus (+10pts if standards_detected and specs populated)
    - Crossover issues penalty (-5pts per issue)

    Always proceeds to generate_samples regardless of score.
    Low scores are flagged in quality_flags for response_data audit.
    """
    logger.info("[SolutionDeepAgent] Reflecting on enrichment quality...")

    all_items = state.get("all_items", [])

    tools_called = list(state.get("tools_called") or [])
    tool_results_summary = dict(state.get("tool_results_summary") or {})
    quality_flags = list(state.get("quality_flags") or [])

    tools_called.append("reflect_enrichment")

    # Compute average specs per item
    spec_counts = [
        len(item.get("specifications", {}))
        for item in all_items
        if isinstance(item.get("specifications"), dict)
    ]
    avg_specs = sum(spec_counts) / len(spec_counts) if spec_counts else 0.0

    if not all_items:
        score = 0
    elif avg_specs >= 8:
        score = 90
    elif avg_specs >= 5:
        score = 75
    elif avg_specs >= 3:
        score = 55
    elif avg_specs >= 1:
        score = 35
    else:
        score = 10
        quality_flags.append("enrichment_no_specs")

    # Bonus: standards-enriched specs present
    if state.get("standards_detected") and any(
        item.get("specifications", {}).get("applicable_standards")
        or item.get("specifications", {}).get("certifications")
        for item in all_items
    ):
        score = min(score + 10, 100)

    if score < 40:
        quality_flags.append(f"enrichment_quality_low: {score}")

    total_specs = sum(len(item.get("specifications", {})) for item in all_items)
    tool_results_summary["reflect_enrichment"] = {
        "total_items": len(all_items),
        "avg_specs_per_item": round(avg_specs, 1),
        "total_specs": total_specs,
        "isolation": "structural_via_orchestration_context",
        "score": score,
    }

    logger.info(
        "[SolutionDeepAgent] reflect_enrichment: score=%d avg_specs=%.1f",
        score, avg_specs,
    )

    state["enrichment_quality_score"] = score
    state["tools_called"] = tools_called
    state["tool_results_summary"] = tool_results_summary
    state["quality_flags"] = quality_flags
    mark_phase_complete(state, "reflect_enrichment")
    return state


# =============================================================================
# NODE 5b: TAXONOMY NORMALIZATION (LLM-BASED)
# =============================================================================

def taxonomy_normalization_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Phase 5b (runs AFTER standards enrichment): LLM-Based Taxonomy Normalization.

    Uses an LLM call with the full taxonomy list to map each item name to the
    closest canonical organizational taxonomy name. This handles cases where
    item names don't directly match aliases (e.g., abbreviations, synonyms,
    vendor-specific terms).

    Each item in all_items receives:
      canonical_name  — matched taxonomy name, or original if no match
      taxonomy_matched — True if a taxonomy entry was found
    """
    logger.info("[SolutionDeepAgent] Phase 5b: LLM-Based Taxonomy Normalization...")

    tools_called = list(state.get("tools_called") or [])
    tool_results_summary = dict(state.get("tool_results_summary") or {})
    quality_flags = list(state.get("quality_flags") or [])

    all_items = state.get("all_items", [])

    try:
        from taxonomy_rag.loader import load_taxonomy
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        prompts = _get_prompts()
        norm_prompt_text = prompts.get("SOLUTION_TAXONOMY_NORMALIZATION", "")

        if not norm_prompt_text:
            logger.warning(
                "[SolutionDeepAgent] SOLUTION_TAXONOMY_NORMALIZATION prompt not found; "
                "falling back to passthrough"
            )
            for item in all_items:
                item.setdefault("canonical_name", item.get("name", ""))
                item.setdefault("taxonomy_matched", False)
            state["taxonomy_normalization_applied"] = False
            mark_phase_complete(state, "taxonomy_normalization")
            return state

        # Load taxonomy
        taxonomy = load_taxonomy()
        instruments_taxonomy = "\n".join(
            "- {name}{aliases}".format(
                name=item["name"],
                aliases=(
                    f" (aliases: {', '.join(item['aliases'])})"
                    if item.get("aliases") else ""
                ),
            )
            for item in taxonomy.get("instruments", [])
        )
        accessories_taxonomy = "\n".join(
            "- {name}{aliases}".format(
                name=item["name"],
                aliases=(
                    f" (aliases: {', '.join(item['aliases'])})"
                    if item.get("aliases") else ""
                ),
            )
            for item in taxonomy.get("accessories", [])
        )

        if not all_items:
            mark_phase_complete(state, "taxonomy_normalization")
            return state

        # Build items list for the LLM
        items_list = "\n".join(
            f"{i + 1}. {item.get('name', 'Unknown')} [{item.get('type', 'instrument')}]"
            for i, item in enumerate(all_items)
        )

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=60,
        )

        prompt = ChatPromptTemplate.from_template(norm_prompt_text)
        chain = prompt | llm | JsonOutputParser()

        result = chain.invoke({
            "items_list": items_list,
            "instruments_taxonomy": instruments_taxonomy or "No instruments taxonomy available.",
            "accessories_taxonomy": accessories_taxonomy or "No accessories taxonomy available.",
        })

        # Build lookup from original_name → normalization entry
        normalized_map: Dict[str, Any] = {
            entry["original_name"]: entry
            for entry in result.get("normalized", [])
            if "original_name" in entry
        }

        # Apply results to all_items
        for item in all_items:
            raw_name = item.get("name", "")
            entry = normalized_map.get(raw_name)
            if entry:
                item["canonical_name"] = entry.get("canonical_name", raw_name)
                item["taxonomy_matched"] = bool(entry.get("taxonomy_matched", False))
            else:
                item["canonical_name"] = raw_name
                item["taxonomy_matched"] = False

        state["all_items"] = all_items
        state["taxonomy_normalization_applied"] = True

        total_matched = sum(1 for i in all_items if i.get("taxonomy_matched"))
        tools_called.append("taxonomy_normalization")
        tool_results_summary["taxonomy_normalization"] = {
            "total_items": len(all_items),
            "matched": total_matched,
            "unmatched": len(all_items) - total_matched,
        }

        add_system_message(
            state,
            f"Taxonomy normalization: {total_matched}/{len(all_items)} items mapped to canonical names"
        )
        logger.info(
            "[SolutionDeepAgent] Taxonomy normalization complete: %d/%d items matched",
            total_matched, len(all_items),
        )

    except Exception as e:
        logger.warning(
            f"[SolutionDeepAgent] Taxonomy normalization failed (non-blocking): {e}"
        )
        quality_flags.append(f"taxonomy_norm_error: {str(e)[:80]}")
        # Graceful passthrough — items keep their original names
        for item in all_items:
            item.setdefault("canonical_name", item.get("name", ""))
            item.setdefault("taxonomy_matched", False)
        state["taxonomy_normalization_applied"] = False

    state["tools_called"] = tools_called
    state["tool_results_summary"] = tool_results_summary
    state["quality_flags"] = quality_flags
    mark_phase_complete(state, "taxonomy_normalization")
    return state


# =============================================================================
# NODE 2b: MODIFICATION
# =============================================================================

# ─── OPERATION-TYPE CLASSIFIER PROMPT ─────────────────────────────────────────
# Used inside modification_node to distinguish *requirement changes* (which need
# the full identification pipeline to re-run) from *BOM edits* (in-place CRUD).
_MOD_OP_TYPE_PROMPT = """You are an expert in industrial instrumentation systems.
Classify whether the user's modification request is a REQUIREMENT CHANGE or a BOM EDIT.

A REQUIREMENT CHANGE changes the underlying process parameters or project scope —
examples: "change the pressure to 100 psi", "use SIL 2 safety level", "the fluid is now
corrosive", "the temperature range changed", "redesign for offshore environment".
These always need the instrument list to be re-generated from scratch.

A BOM EDIT directly adds, removes, or updates specific items already in the list —
examples: "add 2 pressure transmitters", "remove the flow meter", "change quantity to 5",
"rename item 3", "add a junction box as accessory".
These can be applied in-place without re-running identification.

USER REQUEST:
{modification_request}

EXISTING ITEMS SUMMARY:
{items_summary}

Output ONLY valid JSON:
{{"operation_type": "REQUIREMENT_CHANGE" | "BOM_EDIT", "confidence": 0.0-1.0, "reasoning": "..."}}"""


def modification_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Handle modification of existing instruments and accessories.

    Enhancements (iterative):
    - [GAP 2] Operation-type classification: distinguishes REQUIREMENT_CHANGE
      (re-triggers identification) from BOM_EDIT (in-place CRUD).
      For requirement changes, calls update_requirements() from modification_agent
      to compute a delta and populate state["user_input"] with the merged context,
      then sets state["_needs_reidentification"] = True for graph routing.
    - Auto-loads instruments/accessories from session memory when state is empty
    - Validates that there is something to modify before calling LLM
    - Enriches modification request with conversation history context
    - Preserves existing standards fields (standards_specs, applicable_standards,
      standards_summary, standards_source) on unchanged items
    - Flags newly added items with '_newly_added' so the standards node
      only processes them (cleaned before saving)
    """
    logger.info("[SolutionDeepAgent] Processing Modification...")
    state["_needs_reidentification"] = False  # default: BOM edit path

    try:
        current_instruments = state.get("identified_instruments", [])
        current_accessories = state.get("identified_accessories", [])
        session_id = state.get("session_id", "default")

        # ── Auto-load from session memory when state is empty ────────────────
        # Scan conversation_history for the most recent assistant message
        # that stored instruments/accessories in its metadata.
        if not current_instruments and not current_accessories:
            logger.info(
                "[SolutionDeepAgent] No instruments/accessories in state — "
                "scanning conversation history for previous results..."
            )
            try:
                for msg in reversed(state.get("conversation_history", [])):
                    meta = msg.get("metadata", {})
                    hist_instruments = meta.get("identified_instruments", [])
                    hist_accessories = meta.get("identified_accessories", [])
                    if hist_instruments or hist_accessories:
                        current_instruments = hist_instruments
                        current_accessories = hist_accessories
                        state["identified_instruments"] = current_instruments
                        state["identified_accessories"] = current_accessories
                        logger.info(
                            f"[SolutionDeepAgent] Loaded {len(current_instruments)} instruments "
                            f"and {len(current_accessories)} accessories from conversation history"
                        )
                        break
            except Exception as mem_err:
                logger.warning(f"[SolutionDeepAgent] History instrument load failed: {mem_err}")

        # ── Validate that there is something to modify ────────────────────────
        if not current_instruments and not current_accessories:
            logger.warning("[SolutionDeepAgent] No existing instruments/accessories to modify")
            state["error"] = "Nothing to modify"
            state["response"] = (
                "You don't have any instruments or accessories yet. "
                "Please provide requirements first."
            )
            state["response_data"] = {
                "workflow": "solution",
                "type": "error",
                "message": state["response"],
                "awaiting_selection": False,
            }
            return state

        # ── Enrich modification request with conversation history ─────────────
        modification_request = state.get("user_input", "")
        try:
            ctx_manager = SolutionContextManager()
            ctx_manager.load_history(state.get("conversation_history", []))
            conversation_context = ctx_manager.get_enriched_context()
            if conversation_context:
                enriched_request = (
                    f"Recent Conversation:\n{conversation_context}\n\n"
                    f"Current User Message: {modification_request}"
                )
                logger.info("[SolutionDeepAgent] Enriched modification request with history")
            else:
                enriched_request = modification_request
        except Exception:
            enriched_request = modification_request

        all_items = state.get("all_items", [])

        # ── [GAP 2] Operation-type classification ──────────────────────────────
        # Quickly classify whether this is a requirement change (needs re-ID) or
        # an in-place BOM edit before invoking the heavier modification LLM.
        operation_type = "BOM_EDIT"  # safe default
        try:
            op_llm = create_llm_with_fallback(
                model=AgenticConfig.FLASH_MODEL,
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            items_summary = "; ".join(
                f"{item.get('number', idx+1)}. {item.get('name', item.get('category', 'Item'))}"
                for idx, item in enumerate(all_items[:20])
            ) or "No items yet"

            op_prompt = ChatPromptTemplate.from_template(_MOD_OP_TYPE_PROMPT)
            op_chain = op_prompt | op_llm | JsonOutputParser()
            op_result = op_chain.invoke({
                "modification_request": enriched_request,
                "items_summary": items_summary,
            })
            operation_type = op_result.get("operation_type", "BOM_EDIT")
            op_confidence = op_result.get("confidence", 0.0)
            op_reasoning = op_result.get("reasoning", "")
            logger.info(
                f"[SolutionDeepAgent] Modification op-type: {operation_type} "
                f"(confidence={op_confidence:.2f}) — {op_reasoning[:80]}"
            )
        except Exception as op_err:
            logger.warning(
                f"[SolutionDeepAgent] Op-type classification failed ({op_err}); "
                f"defaulting to BOM_EDIT"
            )

        # ── [GAP 2] Requirement-change path: call update_requirements() ─────────
        if operation_type == "REQUIREMENT_CHANGE":
            logger.info(
                "[SolutionDeepAgent] Requirement change detected — "
                "calling update_requirements() for delta extraction"
            )
            try:
                # Inlined logic from modification_agent.update_requirements
                prompts = _get_prompts()
                mod_prompt_text = prompts.get("MODIFICATION_PROMPT", "")
                
                if not mod_prompt_text:
                    raise ValueError("MODIFICATION_PROMPT not found in prompts")

                # Derive the "original requirements" from the earliest user turn
                original_requirements = ""
                for msg in state.get("conversation_history", []):
                    if msg.get("role") == "user":
                        original_requirements = msg.get("content", "")
                        break
                if not original_requirements:
                    original_requirements = modification_request

                # Step 1: Fetch user documents
                user_documents_context = _fetch_user_documents(session_id)

                # Step 2: Extract delta via LLM (text output)
                llm_delta = create_llm_with_fallback(
                    model=AgenticConfig.FLASH_MODEL,
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                
                mod_prompt = ChatPromptTemplate.from_template(mod_prompt_text)
                mod_chain = mod_prompt | llm_delta | StrOutputParser()

                raw_analysis = mod_chain.invoke({
                    "original_requirements": original_requirements,
                    "modification_request": enriched_request,
                    "user_documents_context": user_documents_context,
                })

                # Parse fields from the plain-text LLM output
                op_type = _extract_field(raw_analysis, "OPERATION_TYPE")
                changes_raw = _extract_block(raw_analysis, "CHANGES_DETECTED")
                updated_reqs_raw = _extract_block(raw_analysis, "UPDATED_REQUIREMENTS")
                re_id_needed = "YES" in _extract_field(raw_analysis, "RE_IDENTIFICATION_NEEDED").upper()

                changes_detected = [
                    line.lstrip("- ").strip()
                    for line in changes_raw.splitlines()
                    if line.strip().startswith("-")
                ]
                
                req_result = {
                    "success": True,
                    "operation_type": op_type or "MIXED",
                    "changes_detected": changes_detected,
                    "updated_requirements": updated_reqs_raw or modification_request,
                    "re_identification": re_id_needed,
                    "raw_analysis": raw_analysis
                }

                if req_result.get("success"):
                    updated_reqs = req_result.get("updated_requirements", modification_request)
                    changes_detected = req_result.get("changes_detected", [])
                    logger.info(
                        f"[SolutionDeepAgent] Requirement delta extracted: "
                        f"{len(changes_detected)} changes → triggering re-identification"
                    )
                    # Update user_input with merged requirements for re-identification
                    state["user_input"] = updated_reqs
                    state["_needs_reidentification"] = True
                    state["modification_diff"] = {
                        "changes": changes_detected,
                        "summary": req_result.get("raw_analysis", "")[:200],
                        "operation_type": req_result.get("operation_type", "MIXED"),
                    }
                    add_system_message(
                        state,
                        f"Requirements updated ({req_result.get('operation_type')}): "
                        f"{len(changes_detected)} changes; re-identification triggered"
                    )
                    return state
                else:
                    # update_requirements failed — fall back to BOM_EDIT path
                    logger.warning(
                        f"[SolutionDeepAgent] update_requirements() failed: "
                        f"{req_result.get('error')} — falling back to BOM_EDIT"
                    )
            except Exception as ureq_err:
                logger.warning(
                    f"[SolutionDeepAgent] update_requirements import/call error: "
                    f"{ureq_err} — falling back to BOM_EDIT"
                )

        # Prepare context for LLM
        current_state_payload = {
            "instruments": current_instruments,
            "accessories": current_accessories,
            "all_items_summary": [
                f"{item.get('number')}. {item.get('name')} (Qty: {item.get('quantity')})"
                for item in all_items
            ]
        }

        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        prompt = ChatPromptTemplate.from_template(MODIFICATION_PROCESSING_PROMPT)
        # [GAP 9] Use StrOutputParser first so we can manually clean code-block
        # fences (``` json) and apply ast.literal_eval fallback before JSON parse.
        str_chain = prompt | llm | StrOutputParser()
        raw_mod_response = str_chain.invoke({
            "current_state": json.dumps(current_state_payload, indent=2),
            "modification_request": enriched_request
        })

        # Strip markdown code fences
        cleaned_mod = raw_mod_response.strip()
        if cleaned_mod.startswith("```json"):
            cleaned_mod = cleaned_mod[7:]
        elif cleaned_mod.startswith("```"):
            cleaned_mod = cleaned_mod[3:]
        if cleaned_mod.endswith("```"):
            cleaned_mod = cleaned_mod[:-3]
        cleaned_mod = cleaned_mod.strip()

        # Parse JSON (with ast.literal_eval fallback for edge-case quoting)
        result = {}
        try:
            result = json.loads(cleaned_mod)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(cleaned_mod)
            except Exception as parse_err:
                logger.warning(
                    f"[SolutionDeepAgent] Modification LLM parse failed: {parse_err}; "
                    f"treating as no-op"
                )

        if result.get("success"):
            new_instruments = result.get("instruments", [])
            new_accessories = result.get("accessories", [])

            # ── Standards-fields preservation & _newly_added flagging ──────────
            # Fields the Deep Agent already computed — must survive modification
            standards_fields = [
                "standards_specs",
                "applicable_standards",
                "standards_summary",
                "standards_source",
            ]

            # Build look-up maps keyed by category (lower-cased)
            orig_inst_by_cat = {
                (inst.get("category") or "").strip().lower(): inst
                for inst in current_instruments
                if inst.get("category")
            }
            orig_acc_by_cat = {
                (acc.get("category") or "").strip().lower(): acc
                for acc in current_accessories
                if acc.get("category")
            }

            for inst in new_instruments:
                cat = (inst.get("category") or "").strip().lower()
                original = orig_inst_by_cat.get(cat)
                if original:
                    for field in standards_fields:
                        if field in original and field not in inst:
                            inst[field] = original[field]
                    inst["_newly_added"] = False
                else:
                    inst["_newly_added"] = True

            for acc in new_accessories:
                cat = (acc.get("category") or "").strip().lower()
                original = orig_acc_by_cat.get(cat)
                if original:
                    for field in standards_fields:
                        if field in original and field not in acc:
                            acc[field] = original[field]
                    acc["_newly_added"] = False
                else:
                    acc["_newly_added"] = True

            # Ensure default fields
            for item in new_instruments:
                item.setdefault("strategy", "")
                item.setdefault("quantity", 1)
                item.setdefault("specified_vendors", [])
                item.setdefault("specified_model_families", [])
                item.setdefault("user_specified_standards", [])
                item.setdefault("skip_standards", False)

            for item in new_accessories:
                item.setdefault("strategy", "")
                item.setdefault("quantity", 1)
                item.setdefault("specified_vendors", [])
                item.setdefault("specified_model_families", [])
                item.setdefault("user_specified_standards", [])
                item.setdefault("skip_standards", False)

            # ── Clean up _newly_added before saving (don't leak to API) ────────
            for inst in new_instruments:
                inst.pop("_newly_added", None)
            for acc in new_accessories:
                acc.pop("_newly_added", None)

            # Update state
            state["identified_instruments"] = new_instruments
            state["identified_accessories"] = new_accessories

            # Rebuild all_items
            updated_all_items = []
            idx = 1
            for inst in new_instruments:
                item = inst.copy()
                item["number"] = idx
                item["type"] = "instrument"
                updated_all_items.append(item)
                idx += 1
            for acc in new_accessories:
                item = acc.copy()
                item["number"] = idx
                item["type"] = "accessory"
                updated_all_items.append(item)
                idx += 1

            state["all_items"] = updated_all_items
            state["total_items"] = len(updated_all_items)
            state["modification_diff"] = {
                "changes": result.get("changes_made", []),
                "summary": result.get("summary", "")
            }

            add_system_message(state, f"Modified: {result.get('summary')}")
            logger.info(
                f"[SolutionDeepAgent] Modification applied — "
                f"{len(new_instruments)} instruments, {len(new_accessories)} accessories"
            )
        else:
            logger.warning("[SolutionDeepAgent] Modification LLM returned success=false")

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Modification node failed: {e}")
        state["error"] = str(e)

    return state


# =============================================================================
# NODE 2b-ii: STANDARDS APPLICATION  (GAP 1)
# =============================================================================

def standards_application_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Post-modification standards enrichment node.

    Integrates locally-implemented standards RAG with LLM-generated web knowledge
    in parallel threads, applied ONLY to items flagged as '_newly_added' by
    modification_node.  Existing items keep their pre-computed standards data.

    Two enrichment sources run concurrently per item:
      Thread A — Document RAG (enrich_identified_items_with_standards):
                 Queries Pinecone with domain-routed source filtering.
                 Returns applicable_standards, certifications, safety_requirements,
                 communication_protocols, and environmental_requirements from
                 uploaded standards documents.
      Thread B — LLM Knowledge (generate_llm_specs):
                 Uses Gemini Flash with iterative generation (30+ specs).
                 Covers performance, electrical, physical, environmental,
                 and compliance categories using web-trained knowledge.

    Merge strategy (precedence: user > RAG > LLM):
      - User-specified values are NEVER overwritten.
      - Standards-RAG values are added as "<value> [STANDARD]" when the key
        does not already exist (case-insensitive match).
      - LLM values fill remaining gaps without the [STANDARD] suffix.

    This node fires only after a BOM-edit modification.  It skips:
      - Items without '_newly_added' == True (unchanged items already have
        their standards data from the original identification pipeline).
      - Items whose 'standards_specs' is already populated.
      - Full pipeline runs (identification path uses parallel_enrichment_node).

    Non-fatal: any per-item failure appends to quality_flags and continues.
    """
    logger.info("[SolutionDeepAgent] Standards Application Node (GAP 1)...")

    instruments = state.get("identified_instruments", [])
    accessories = state.get("identified_accessories", [])
    quality_flags = list(state.get("quality_flags") or [])
    tools_called = list(state.get("tools_called") or [])
    standards_applied_count = 0

    if not instruments and not accessories:
        logger.info("[SolutionDeepAgent] apply_standards: nothing to enrich")
        return state

    # ── Determine which items need standards ────────────────────────────────────
    # _newly_added may have been cleaned by modification_node before reaching here.
    # Re-evaluate: items that have NO standards_specs are treated as needing them.
    def _needs_standards(item: Dict[str, Any]) -> bool:
        if item.get("skip_standards"):
            return False
        existing = item.get("standards_specs")
        if existing and len(existing) > 0:
            return False  # already enriched
        return True

    items_to_enrich_inst = [i for i in instruments if _needs_standards(i)]
    items_to_enrich_acc  = [a for a in accessories  if _needs_standards(a)]

    if not items_to_enrich_inst and not items_to_enrich_acc:
        logger.info(
            "[SolutionDeepAgent] apply_standards: all items already have standards data"
        )
        return state

    logger.info(
        f"[SolutionDeepAgent] apply_standards: enriching "
        f"{len(items_to_enrich_inst)} instruments + {len(items_to_enrich_acc)} accessories"
    )

    # ── Per-item parallel enrichment: Doc RAG + LLM ─────────────────────────────
    def _enrich_item_with_standards(
        item: Dict[str, Any],
        is_accessory: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs Document-RAG and LLM knowledge generation in parallel for a single item,
        then merges results into the item's specifications.
        """
        category   = item.get("category", "")
        item_name  = item.get("name", category) or category
        item_label = f"{'Accessory' if is_accessory else 'Instrument'}: {item_name}"

        if not category:
            logger.debug(f"[apply_standards] Skipping {item_label} — no category")
            return item

        doc_rag_result  = {}
        llm_spec_result = {}

        # ── Parallel: Thread A = Doc RAG, Thread B = LLM knowledge ────────────
        with ThreadPoolExecutor(max_workers=2) as executor:

            # Thread A: Document RAG (Pinecone, domain-routed)
            def _run_doc_rag():
                try:
                    from common.rag.standards import enrich_identified_items_with_standards
                    enriched = enrich_identified_items_with_standards(
                        items=[item.copy()],
                        product_type=category,
                        top_k=4,
                        max_workers=1,
                    )
                    if enriched:
                        return enriched[0].get("standards_info", {})
                    return {}
                except Exception as e:
                    logger.warning(f"[apply_standards] Doc RAG failed for '{category}': {e}")
                    return {}

            # Thread B: LLM knowledge generation (iterative, 30+ specs)
            def _run_llm_specs():
                try:
                    from common.standards.generation.llm_generator import generate_llm_specs
                    result = generate_llm_specs(
                        product_type=category,
                        category=item.get("type", "instrument"),
                        context=item.get("sample_input") or item.get("description", ""),
                        min_specs=20,  # lower minimum in modification context
                    )
                    return result.get("specifications", {})
                except Exception as e:
                    logger.warning(f"[apply_standards] LLM specs failed for '{category}': {e}")
                    return {}

            fut_rag = executor.submit(_run_doc_rag)
            fut_llm = executor.submit(_run_llm_specs)

            doc_rag_result  = fut_rag.result()
            llm_spec_result = fut_llm.result()

        # ── Merge into item specs (user > RAG > LLM) ──────────────────────────
        original_specs: Dict[str, Any] = item.get("specifications", {}) or {}
        merged_specs = dict(original_specs)  # start with user values (highest priority)

        # Build case-insensitive key set from existing (user-specified) values
        def _key_exists(key: str, existing: Dict) -> bool:
            k_lower = key.strip().lower()
            return any(k_lower == ek.strip().lower() for ek in existing)

        # 1. Layer in RAG standards-sourced fields (add [STANDARD] tag)
        rag_applicable = doc_rag_result.get("applicable_standards", [])
        rag_certifications = doc_rag_result.get("certifications", [])
        rag_safety = doc_rag_result.get("safety_requirements", {})
        rag_comms = doc_rag_result.get("communication_protocols", [])
        rag_env = doc_rag_result.get("environmental_requirements", {})

        if rag_applicable and not _key_exists("Applicable Standards", merged_specs):
            merged_specs["Applicable Standards"] = (
                ", ".join(rag_applicable[:5]) + " [STANDARD]"
            )
        if rag_certifications and not _key_exists("Certifications", merged_specs):
            merged_specs["Certifications"] = (
                ", ".join(rag_certifications[:3]) + " [STANDARD]"
            )
        if rag_comms and not _key_exists("Communication Protocols", merged_specs):
            merged_specs["Communication Protocols"] = (
                ", ".join(rag_comms[:3]) + " [STANDARD]"
            )
        for skey, sval in rag_safety.items():
            if not _key_exists(skey, merged_specs):
                merged_specs[skey] = f"{sval} [STANDARD]"
        for ekey, eval_ in rag_env.items():
            if not _key_exists(ekey, merged_specs):
                merged_specs[ekey] = f"{eval_} [STANDARD]"

        # 2. Layer in LLM-generated specs as fill-in (no [STANDARD] tag)
        for lkey, lval in llm_spec_result.items():
            if not _key_exists(lkey, merged_specs):
                raw_value = lval.get("value", lval) if isinstance(lval, dict) else lval
                if raw_value and str(raw_value).lower() not in ("null", "none", "n/a", ""):
                    merged_specs[lkey] = raw_value

        item["specifications"] = merged_specs

        # Persist structured standards fields for downstream use
        if doc_rag_result.get("applicable_standards"):
            item.setdefault("standards_specs", {})
            item["applicable_standards"] = rag_applicable
            item["standards_summary"] = (
                "Applicable Standards: " + ", ".join(rag_applicable[:3])
            )
            item["standards_source"] = "document_rag"
            if rag_certifications:
                item["standards_specs"]["certifications"] = rag_certifications
            if rag_applicable:
                item["standards_specs"]["applicable_standards"] = rag_applicable

        return item

    try:
        # ── Resolve root orchestration context from state ───────────────────────
        orch_ctx_data = state.get("orchestration_ctx") or {}
        if orch_ctx_data:
            root_orch_ctx = OrchestrationContext.from_dict(orch_ctx_data)
        else:
            root_orch_ctx = OrchestrationContext.root(
                session_id=state.get("session_id", "default")
            )

        orchestrator = get_solution_orchestrator()
        _node_log = get_orchestration_logger(logger)

        # ── Wrapper to bind is_accessory into closure ───────────────────────────
        def _make_inst_task(is_acc: bool):
            def _task(item: Dict[str, Any]) -> Dict[str, Any]:
                _log = get_orchestration_logger(logger)
                _log.debug(f"[apply_standards] Starting enrichment for '{item.get('category', '?')}'")
                return _enrich_item_with_standards(item.copy(), is_acc)
            return _task

        # ── Instruments: each runs in isolated child context ────────────────────
        inst_results = orchestrator.run_parallel(
            fn=_make_inst_task(False),
            items=items_to_enrich_inst,
            root_ctx=root_orch_ctx,
            label_fn=lambda i: f"apply_std:inst:{i.get('category', '?')}",
            timeout_seconds=120.0,
        )

        # ── Accessories: same pattern ────────────────────────────────────────────
        acc_results = orchestrator.run_parallel(
            fn=_make_inst_task(True),
            items=items_to_enrich_acc,
            root_ctx=root_orch_ctx,
            label_fn=lambda a: f"apply_std:acc:{a.get('category', '?')}",
            timeout_seconds=120.0,
        )

        # ── Map instance_id results back to category for quick lookup ────────────
        # items_to_enrich_inst list order matches the submission order in orchestrator,
        # so we zip with the keys (which are ordered dicts in Python 3.7+).
        enriched_inst_map: Dict[str, Dict] = {}
        for (iid, enriched), orig_inst in zip(inst_results.items(), items_to_enrich_inst):
            cat = (orig_inst.get("category") or "").strip().lower()
            if isinstance(enriched, dict) and not enriched.get("error"):
                enriched_inst_map[cat] = enriched
                standards_applied_count += 1
            else:
                quality_flags.append(
                    f"apply_standards_inst_failed:{cat}:{str(enriched.get('error','?'))[:60]}"
                )

        enriched_acc_map: Dict[str, Dict] = {}
        for (iid, enriched), orig_acc in zip(acc_results.items(), items_to_enrich_acc):
            cat = (orig_acc.get("category") or "").strip().lower()
            if isinstance(enriched, dict) and not enriched.get("error"):
                enriched_acc_map[cat] = enriched
                standards_applied_count += 1
            else:
                quality_flags.append(
                    f"apply_standards_acc_failed:{cat}:{str(enriched.get('error','?'))[:60]}"
                )

        # ── Write enriched items back, preserving order ─────────────────────────
        updated_instruments = []
        for inst in instruments:
            cat = (inst.get("category") or "").strip().lower()
            if cat in enriched_inst_map:
                enriched = enriched_inst_map[cat]
                enriched["number"] = inst.get("number", enriched.get("number"))
                enriched.pop("_newly_added", None)
                updated_instruments.append(enriched)
            else:
                updated_instruments.append(inst)

        updated_accessories = []
        for acc in accessories:
            cat = (acc.get("category") or "").strip().lower()
            if cat in enriched_acc_map:
                enriched = enriched_acc_map[cat]
                enriched["number"] = acc.get("number", enriched.get("number"))
                enriched.pop("_newly_added", None)
                updated_accessories.append(enriched)
            else:
                updated_accessories.append(acc)

        state["identified_instruments"] = updated_instruments
        state["identified_accessories"] = updated_accessories

        # ── Rebuild all_items ──────────────────────────────────────────────────
        updated_all = []
        idx = 1
        for inst in updated_instruments:
            item = inst.copy()
            item["number"] = idx
            item["type"] = "instrument"
            updated_all.append(item)
            idx += 1
        for acc in updated_accessories:
            item = acc.copy()
            item["number"] = idx
            item["type"] = "accessory"
            updated_all.append(item)
            idx += 1

        state["all_items"] = updated_all
        state["total_items"] = len(updated_all)

        tools_called.append("standards_application_node")
        add_system_message(
            state,
            f"Standards applied to {standards_applied_count} items "
            f"({len(items_to_enrich_inst)} instruments, {len(items_to_enrich_acc)} accessories)"
        )
        _node_log.info(
            f"[SolutionDeepAgent] apply_standards complete: "
            f"{standards_applied_count} items enriched"
        )

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] apply_standards node failed: {e}", exc_info=True)
        quality_flags.append(f"apply_standards_error: {str(e)[:100]}")
        # Non-fatal: continue without standards enrichment

    state["quality_flags"] = quality_flags
    state["tools_called"] = tools_called
    return state


# =============================================================================
# NODE 2b-iii: CONCISE BOM  (GAP 3)
# =============================================================================

def concise_bom_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Score and filter an existing BOM without re-running identification.

    [GAP 3] Wires concise_bom() from modification_agent into the graph.

    The user wants to trim or prioritize the identified list.
    Each item gets a relevance_score (0-100) and a keep=True/False flag.
    Items with keep=False are removed from the active BOM.
    Standards enrichment is applied ONLY to items with keep=True
    (handled internally by modification_agent.concise_bom()).

    After scoring, state is rebuilt to contain only kept items so that
    flash_response_node renders the concised BOM as the final response.

    Non-fatal: on any failure, current BOM is preserved unchanged.
    """
    logger.info("[SolutionDeepAgent] Concise BOM Node (GAP 3)...")

    instruments = state.get("identified_instruments", [])
    accessories = state.get("identified_accessories", [])
    quality_flags = list(state.get("quality_flags") or [])
    tools_called  = list(state.get("tools_called") or [])

    if not instruments and not accessories:
        logger.warning("[SolutionDeepAgent] concise_bom: no BOM to concise")
        state["response"] = (
            "There are no instruments or accessories to trim. "
            "Please provide requirements first."
        )
        state["response_data"] = {
            "workflow": "solution",
            "type": "error",
            "message": state["response"],
            "awaiting_selection": False,
        }
        return state

    try:
        # Recover the original first-turn requirements from conversation history
        original_requirements = ""
        for msg in state.get("conversation_history", []):
            if msg.get("role") == "user":
                original_requirements = msg.get("content", "")
                break
        if not original_requirements:
            original_requirements = state.get("user_input", "")

        # Inlined logic from modification_agent.concise_bom
        prompts = _get_prompts()
        concise_prompt_text = prompts.get("BOM_CONCISENESS_PROMPT", "")
        
        if not concise_prompt_text:
            raise ValueError("BOM_CONCISENESS_PROMPT not found in prompts")

        # Step 1: Fetch user documents
        user_documents_context = _fetch_user_documents(state.get("session_id", "default"))

        # Build current BOM context
        current_bom = json.dumps(
            {"instruments": instruments, "accessories": accessories},
            indent=2,
        )

        llm_concise = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        concise_prompt = ChatPromptTemplate.from_template(concise_prompt_text)
        concise_parser = JsonOutputParser()
        concise_chain = concise_prompt | llm_concise | concise_parser

        result = concise_chain.invoke({
            "current_bom": current_bom,
            "conciseness_request": state.get("user_input", ""),
            "original_requirements": original_requirements or "Not provided",
            "user_documents_context": user_documents_context,
        })

        if result.get("success"):
            scored_inst_map = {
                (i.get("category") or "").strip().lower(): i 
                for i in result.get("scored_instruments", [])
            }
            scored_acc_map = {
                (a.get("category") or "").strip().lower(): a 
                for a in result.get("scored_accessories", [])
            }

            # Merge scores back into originals and filter
            kept_instruments = []
            for inst in instruments:
                cat = (inst.get("category") or "").strip().lower()
                score_info = scored_inst_map.get(cat, {})
                inst["relevance_score"] = score_info.get("relevance_score", 100)
                inst["keep"] = score_info.get("keep", True)
                if inst["keep"]:
                    kept_instruments.append(inst)

            kept_accessories = []
            for acc in accessories:
                cat = (acc.get("category") or "").strip().lower()
                score_info = scored_acc_map.get(cat, {})
                acc["relevance_score"] = score_info.get("relevance_score", 100)
                acc["keep"] = score_info.get("keep", True)
                if acc["keep"]:
                    kept_accessories.append(acc)

            state["identified_instruments"] = kept_instruments
            state["identified_accessories"] = kept_accessories

            # Rebuild all_items with kept items only
            updated_all = []
            idx = 1
            for inst in kept_instruments:
                item = inst.copy()
                item["number"] = idx
                item["type"] = "instrument"
                updated_all.append(item)
                idx += 1
            for acc in kept_accessories:
                item = acc.copy()
                item["number"] = idx
                item["type"] = "accessory"
                updated_all.append(item)
                idx += 1

            state["all_items"] = updated_all
            state["total_items"] = len(updated_all)

            retained = result.get("retained_count", len(kept_instruments + kept_accessories))
            removed  = result.get("removed_count", 0)
            summary  = result.get("conciseness_summary", "")

            state["modification_diff"] = {
                "changes": [f"Retained {retained} items, removed {removed}"],
                "summary": summary,
                "operation_type": "CONCISE_BOM",
                "retained_count": retained,
                "removed_count":  removed,
            }

            tools_called.append("concise_bom")
            add_system_message(
                state,
                f"BOM concised: {retained} kept, {removed} removed. {summary[:80]}"
            )
            logger.info(
                f"[SolutionDeepAgent] concise_bom complete: "
                f"{retained} kept, {removed} removed"
            )
        else:
            err = result.get("error", "Unknown error")
            logger.warning(f"[SolutionDeepAgent] concise_bom() returned failure: {err}")
            quality_flags.append(f"concise_bom_failed: {err[:80]}")

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] concise_bom_node failed: {e}", exc_info=True)
        quality_flags.append(f"concise_bom_node_error: {str(e)[:100]}")
        # Non-fatal: BOM is unchanged

    state["quality_flags"] = quality_flags
    state["tools_called"]  = tools_called
    return state


# =============================================================================
# NODE 2c: CLARIFICATION
# =============================================================================

def clarification_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Handle clarification requests/questions.

    [GAP 7] Also populates state["clarification_context"] dict with
    missing_information and reason, which the frontend uses to display
    *why* clarification is being requested (not just what to ask).
    """
    logger.info("[SolutionDeepAgent] Processing Clarification...")

    try:
        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Context
        ctx_manager = SolutionContextManager()
        ctx_manager.load_history(state.get("conversation_history", []))
        context = ctx_manager.get_enriched_context()

        prompt = ChatPromptTemplate.from_template(CLARIFICATION_PROMPT)
        chain = prompt | llm | JsonOutputParser()

        result = chain.invoke({
            "user_input": state["user_input"],
            "conversation_context": context
        })

        missing_info = result.get("missing_information", "")
        reasoning   = result.get("reasoning", result.get("reason", ""))

        state["clarification_questions"] = result.get("clarification_questions", [])
        state["response"] = result.get("message", "Could you clarify that?")

        # [GAP 7] Populate clarification_context for frontend to display WHY
        state["clarification_context"] = {
            "missing_information": missing_info,
            "reason": reasoning,
        }

        state["response_data"] = {
            "workflow": "solution",
            "type": "clarification",
            "questions": state["clarification_questions"],
            "missing_info": missing_info,
            "clarification_context": state["clarification_context"],
            "awaiting_selection": False,
        }

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Clarification node failed: {e}")
        state["error"] = str(e)

    return state


# =============================================================================
# NODE 2d: RESET
# =============================================================================

def reset_node(state: SolutionDeepAgentState) -> SolutionDeepAgentState:
    """
    Handle session reset.

    Handles three sub-cases:
    - reset_confirmed : user explicitly confirms → clears session
    - reset_cancelled : user declines         → preserves data
    - pending         : user said 'reset' but hasn't confirmed yet → ask
    """
    logger.info("[SolutionDeepAgent] Processing Reset...")

    try:
        user_input_lower = state.get("user_input", "").lower()

        # ── Confirmed reset ───────────────────────────────────────────────────
        if any(kw in user_input_lower for kw in ("yes", "confirm", "sure", "go ahead", "proceed")):
            session_id = state.get("session_id", "")
            _cleanup_session_memory(session_id)

            state["reset_confirmed"] = True
            state["identified_instruments"] = []
            state["identified_accessories"] = []
            state["all_items"] = []
            state["total_items"] = 0
            state["response"] = (
                "Session reset. How can I help you regarding your instrumentation needs?"
            )
            state["response_data"] = {
                "workflow": "solution",
                "type": "reset_confirmed",
                "message": "Session cleared. Ready for a new request.",
                "awaiting_selection": False,
            }
            logger.info(f"[SolutionDeepAgent] Session reset confirmed for: {session_id}")

        # ── Cancelled reset ───────────────────────────────────────────────────
        elif any(kw in user_input_lower for kw in ("no", "cancel", "keep", "don't", "nevermind", "never mind")):
            state["response"] = (
                "Okay, keeping your current data intact. What would you like to do next?"
            )
            state["response_data"] = {
                "workflow": "solution",
                "type": "reset_cancelled",
                "message": "Reset cancelled. Your instruments and accessories are preserved.",
                "awaiting_selection": False,
            }
            logger.info("[SolutionDeepAgent] Reset cancelled — data preserved")

        # ── Ask for confirmation ──────────────────────────────────────────────
        else:
            llm = create_llm_with_fallback(
                model=AgenticConfig.FLASH_MODEL,
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            prompt = ChatPromptTemplate.from_template(RESET_CONFIRMATION_PROMPT)
            chain = prompt | llm | JsonOutputParser()
            result = chain.invoke({"user_input": state["user_input"]})

            state["response"] = result.get("message", "Are you sure you want to reset the session?")
            state["response_data"] = {
                "workflow": "solution",
                "type": "reset_confirmation",
                "message": result.get("message"),
                "awaiting_selection": False,
            }

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Reset node failed: {e}")
        state["error"] = str(e)

    return state


# =============================================================================
# WORKFLOW CREATION
# =============================================================================

def create_solution_deep_agent_workflow() -> StateGraph:
    """
    Create the Solution Deep Agent Workflow.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │              SOLUTION DEEP AGENT WORKFLOW                │
    ├─────────────────────────────────────────────────────────┤
    │  Phase 1: personality_plan (Flash Planning)             │
    │     ↓                                                   │
    │  Phase 2: classify_intent (Semantic + Embeddings)       │
    │     ↓ (conditional: is_solution?)                       │
    │  Phase 3: load_context (Memory + Personal + Thread)     │
    │     ↓                                                   │
    │  Phase 4: analyze_solution (Deep Context Extraction)    │
    │     ↓                                                   │
    │  Phase 4b: reasoning_chain (CoT: decompose, plan,       │
    │            standards decisions, cross-over risks)       │
    │     ↓                                                   │
    │  Phase 5: identify_items (Deep Agent ID, Parallel)      │
    │     ↓                                                   │
    │  Phase 5b: taxonomy_normalization ← ORCHESTRATOR BRIDGE │
    │            (RAG: raw terms → canonical DB terms)        │
    │     ↓                                                   │
    │  Phase 6: enrich_specs (Standards Call Planning +       │
    │           Parallel 3-Source Enrichment with merge)      │
    │     ↓                                                   │
    │  Phase 7: generate_samples (Isolated Sample Inputs)     │
    │     ↓                                                   │
    │  Phase 8: compose_response (Flash Personality Response) │
    │     ↓                                                   │
    │  [END - Awaiting User Selection]                        │
    └─────────────────────────────────────────────────────────┘
    """
    workflow = StateGraph(SolutionDeepAgentState)

    # Add nodes
    workflow.add_node("personality_plan", personality_plan_node)
    workflow.add_node("classify_intent", semantic_intent_node)
    
    # New interaction nodes
    workflow.add_node("modification", modification_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("reset", reset_node)
    workflow.add_node("concise_bom", concise_bom_node)  # GAP 3

    workflow.add_node("load_context", load_context_node)
    workflow.add_node("analyze_solution", solution_analysis_node)
    workflow.add_node("reasoning_chain", reasoning_chain_node)
    workflow.add_node("identify_items", deep_identification_node)
    workflow.add_node("reflect_identification", reflect_identification_node)
    workflow.add_node("taxonomy_normalization", taxonomy_normalization_node)
    workflow.add_node("enrich_specs", parallel_enrichment_node)
    workflow.add_node("reflect_enrichment", reflect_enrichment_node)
    workflow.add_node("generate_samples", sample_input_generation_node)
    workflow.add_node("apply_standards", standards_application_node)  # GAP 1
    workflow.add_node("compose_response", flash_response_node)

    # Entry point
    workflow.set_entry_point("personality_plan")

    # Edges
    workflow.add_edge("personality_plan", "classify_intent")

    # Conditional: Route based on refined intent
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue_after_intent,
        {
            "load_context":  "load_context",
            "modification":  "modification",
            "clarification": "clarification",
            "reset":         "reset",
            "concise_bom":   "concise_bom",  # GAP 3
            "end":           END,
        },
    )

    # Modification path → conditional routing based on operation type:
    #   REQUIREMENT_CHANGE  → re-trigger full identification (load_context)
    #   BOM_EDIT (default)  → targeted standards application for new items only
    def _route_after_modification(state: SolutionDeepAgentState) -> str:
        if state.get("_needs_reidentification"):
            logger.info("[SolutionDeepAgent] Requirement change → re-routing to load_context")
            return "load_context"
        return "apply_standards"

    workflow.add_conditional_edges(
        "modification",
        _route_after_modification,
        {
            "load_context":   "load_context",
            "apply_standards": "apply_standards",
        },
    )
    
    # Clarification/Reset/ConciseBOM path → end (return response directly)
    workflow.add_edge("clarification", END)
    workflow.add_edge("reset", END)
    workflow.add_edge("concise_bom", "compose_response")  # GAP 3

    workflow.add_edge("load_context", "analyze_solution")
    workflow.add_edge("analyze_solution", "reasoning_chain")
    workflow.add_edge("reasoning_chain", "identify_items")

    # Quality gate after identification
    workflow.add_edge("identify_items", "reflect_identification")
    workflow.add_conditional_edges(
        "reflect_identification",
        route_after_identification,
        {
            # enrich path → standards enrichment first, then taxonomy normalization
            "enrich":       "enrich_specs",
            # fast path → skip standards, go directly to taxonomy normalization
            "skip_enrich":  "taxonomy_normalization",
            # hard error → bypass everything
            "error":        "compose_response",
        },
    )

    # Quality gate after standards enrichment (non-blocking — always proceeds)
    workflow.add_edge("enrich_specs", "reflect_enrichment")
    # Taxonomy normalization runs AFTER standards enrichment
    workflow.add_edge("reflect_enrichment", "taxonomy_normalization")
    workflow.add_edge("taxonomy_normalization", "generate_samples")

    workflow.add_edge("generate_samples", "compose_response")

    # apply_standards (modification BOM-edit path) → compose_response directly
    # (no sample generation needed for a BOM edit — items already have samples)
    workflow.add_edge("apply_standards", "compose_response")
    workflow.add_edge("compose_response", END)

    return workflow


# =============================================================================
# ROUTING
# =============================================================================

def should_continue_after_intent(state: SolutionDeepAgentState) -> str:
    """
    Conditional edge: continue if solution intent, otherwise call the router
    agent to determine which workflow the input actually belongs to and return
    a rerouting response.
    """
    if state.get("is_solution_workflow", True):
        # Route based on refined intent
        refined = state.get("intent_classification", {}).get("refined_type", "requirements")

        if refined == "modification":
            return "modification"
        if refined == "clarification":
            return "clarification"
        if refined == "reset":
            return "reset"
        if refined == "concise_bom":  # [GAP 3]
            return "concise_bom"

        return "load_context"

    # Input is not a solution request — delegate to the router agent to find
    # the correct target workflow (search, chat, out_of_domain, …)
    user_input = state.get("user_input", "")
    if not isinstance(user_input, str):
        user_input = str(user_input)

    intent_type = state.get("intent_classification", {}).get("intent_type", "unknown")
    session_id = state.get("session_id", "default")
    refined_internal = state.get("intent_classification", {}).get("refined_type", "requirements")

    # [GAP FIX] If the LLM already identified this as an invalid/gibberish input,
    # skip the router entirely and go straight to out_of_domain.
    if refined_internal == "invalid_input":
        from common.agentic.agents.routing.intent_classifier import OUT_OF_DOMAIN_MESSAGE
        logger.info(f"[SolutionDeepAgent] LLM explicitly marked as invalid_input. Rerouting directly to out_of_domain.")
        
        reroute_target = "out_of_domain"
        nested_result = {
            "response": OUT_OF_DOMAIN_MESSAGE,
            "response_text": OUT_OF_DOMAIN_MESSAGE,
            "response_data": {
                "intent": "invalid_input",
                "routed_to": reroute_target
            }
        }
    else:
        reroute_target = "chat"  # safe default

        try:
            from common.agentic.agents.routing.intent_classifier import IntentClassificationRoutingAgent
            router = IntentClassificationRoutingAgent()

            # Bypass workflow lock by prefixing with 'main_', which tells the
            # IntentClassificationRoutingAgent that this is a free-switching session
            # and it shouldn't force us back into the 'solution' workflow.
            routing_result = router.classify(
                query=user_input,
                session_id=f"main_reroute_{session_id}",
            )

            # Dynamically invoke the respective workflow directly using the router
            nested_result = router.invoke_target_workflow(
                query=user_input,
                routing_result=routing_result,
                session_id=session_id
            )

            # Always extract .value from the WorkflowType enum — prevents
            # "WorkflowType.ENGENIE_CHAT" strings reaching the frontend
            _wf = getattr(routing_result, "target_workflow", None)
            reroute_target = _wf.value if hasattr(_wf, "value") else (str(_wf) if _wf else "chat")
            if not reroute_target or reroute_target.startswith("WorkflowType"):
                reroute_target = "chat"
        except Exception as e:
            logger.warning(
                f"[SolutionDeepAgent] Router agent call failed ({e}); "
                f"falling back to intent_type mapping"
            )
            nested_result = {}
            # Fallback: map SolutionIntentClassifier intent_type to a workflow name
            reroute_target = {
                "comparison": "search",
            }.get(intent_type, "chat")

    logger.info(
        f"[SolutionDeepAgent] Not a solution request "
        f"(intent={intent_type}, refined={refined_internal}) — rerouting to '{reroute_target}'"
    )

    state["response"] = nested_result.get("response", nested_result.get("response_text", 
        f"Your query has been identified as a {reroute_target} request. "
        f"Routing you to the appropriate workflow."
    ))

    # [GAP 4] Persist router fields into workflow_data so flash_response_node
    # can surface them in the router_category / router_* response_data fields.
    if not isinstance(state.get("workflow_data"), dict):
        state["workflow_data"] = {}
    state["workflow_data"].update({
        "routed_to":               reroute_target,
        "router_target_page":      reroute_target,
        "router_requires_routing": True,
        "router_url":              f"/{reroute_target}",
    })

    # Preserve all the nested result data, but inject our redirect metadata
    response_data = nested_result.get("response_data", nested_result) if nested_result else {}
    if not isinstance(response_data, dict):
        response_data = {}
        
    response_data.update({
        "workflow": "solution",
        "is_solution": False,
        "routed_to": reroute_target,
        "should_redirect": True,
        "intent_type": intent_type,
        "awaiting_selection": False,
        "router_target_page":      reroute_target,
        "router_requires_routing": True,
        "router_url":              f"/{reroute_target}",
        "raw_nested_result": nested_result # Keep reference
    })
    
    state["response_data"] = response_data
    return "end"



@with_workflow_lock(session_id_param="session_id", timeout=60.0)
def run_solution_deep_agent(
    user_input: str,
    session_id: str = "default",
    user_id: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    personal_context: Optional[Dict[str, Any]] = None,
    checkpointing_backend: str = "memory",
) -> Dict[str, Any]:
    """
    Run the Solution Deep Agent workflow.

    This is the primary entry point for solution analysis. It replaces
    the old run_solution_workflow with deep agent capabilities.

    Args:
        user_input: User's solution description
        session_id: Session identifier
        user_id: User identifier for personal context
        conversation_history: Previous messages for context continuity
        personal_context: User preferences and configurations
        checkpointing_backend: Backend for state persistence

    Returns:
        Dict with success, response, response_data
    """
    try:
        logger.info(f"[SolutionDeepAgent] Starting for session: {session_id}")
        user_input_str = str(user_input) if not isinstance(user_input, str) else user_input
        logger.info(f"[SolutionDeepAgent] Input: {user_input_str[:100]}...")

        initial_state = create_solution_deep_agent_state(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history,
            personal_context=personal_context,
        )

        workflow = create_solution_deep_agent_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)

        result = compiled.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )

        processing_time = result.get("processing_time_ms", 0)
        logger.info(
            f"[SolutionDeepAgent] Complete: {result.get('total_items', 0)} items "
            f"in {processing_time}ms"
        )

        return {
            "success": True,
            "response": result.get("response", ""),
            "response_data": result.get("response_data", {}),
            "error": result.get("error"),
        }

    except TimeoutError as e:
        logger.error(f"[SolutionDeepAgent] Lock timeout: {e}")
        return {
            "success": False,
            "error": "Another workflow is running. Please try again.",
        }
    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "response": f"Error analyzing solution: {str(e)}",
            "response_data": {
                "workflow": "solution",
                "error": str(e),
                "awaiting_selection": False,
            },
        }
    finally:
        # Always release memory from the registry once the workflow finishes
        _cleanup_session_memory(session_id)


def run_solution_deep_agent_stream(
    user_input: str,
    session_id: str = "default",
    user_id: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    personal_context: Optional[Dict[str, Any]] = None,
    checkpointing_backend: str = "memory",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run the Solution Deep Agent with streaming progress updates.

    Args:
        user_input: User's solution description
        session_id: Session identifier
        user_id: User identifier
        conversation_history: Previous messages
        personal_context: User preferences
        checkpointing_backend: Backend for state persistence
        progress_callback: Callback for progress updates

    Returns:
        Workflow result
    """
    try:
        from common.utils.streaming import ProgressEmitter
        emitter = ProgressEmitter(progress_callback)
    except ImportError:
        emitter = None

    try:
        if emitter:
            emitter.emit("initialize", "Planning solution analysis...", 5)
            emitter.emit("classify_intent", "Classifying request type...", 15)
            emitter.emit("load_context", "Loading conversation context...", 25)
            emitter.emit("analyze_solution", "Analyzing solution requirements...", 35)
            emitter.emit("identify_items", "Identifying instruments and accessories...", 55)
            emitter.emit("enrich_specs", "Enriching with technical specifications...", 75)
            emitter.emit("generate_samples", "Generating search queries...", 85)

        result = run_solution_deep_agent(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history,
            personal_context=personal_context,
            checkpointing_backend=checkpointing_backend,
        )

        if emitter:
            if result.get("success"):
                emitter.emit(
                    "complete",
                    "Solution analysis complete!",
                    100,
                    data={
                        "item_count": result.get("response_data", {}).get("total_items", 0),
                        "solution_name": result.get("response_data", {}).get("solution_name", ""),
                    },
                )
            else:
                emitter.error(result.get("error", "Unknown error"))

        return result

    except Exception as e:
        logger.error(f"[SolutionDeepAgent] Stream failed: {e}", exc_info=True)
        if emitter:
            emitter.error(str(e))
        return {"success": False, "error": str(e)}


# =============================================================================
# WORKFLOW REGISTRATION
# =============================================================================

def _register_workflow():
    """Register the Solution Deep Agent with the central registry."""
    try:
        from common.workflows.base.registry import (
            WorkflowMetadata, RetryPolicy, RetryStrategy, get_workflow_registry
        )

        # Note: We register with a different name to coexist during migration
        get_workflow_registry().register(WorkflowMetadata(
            name="solution_deep_agent",
            display_name="Solution Deep Agent",
            description=(
                "Deep agent architecture for complex multi-instrument systems. "
                "Features semantic intent classification, conversation memory, "
                "parallel 3-source specification enrichment, Flash personality, "
                "and cross-over validation."
            ),
            keywords=[
                "system", "design", "complete", "profiling", "package",
                "comprehensive", "multiple instruments", "reactor",
                "distillation", "heating circuit", "safety system",
            ],
            intents=["solution", "system", "systems", "complex_system", "design"],
            capabilities=[
                "multi_instrument", "parallel_identification",
                "standards_enrichment", "solution_analysis",
                "accessory_matching", "conversation_memory",
                "semantic_intent", "flash_personality",
            ],
            entry_function=run_solution_deep_agent,
            priority=110,  # Higher than old solution workflow (100)
            tags=["core", "complex", "engineering", "deep_agent"],
            retry_policy=RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                base_delay_ms=1000,
            ),
            min_confidence_threshold=0.6,
        ))
        logger.info("[SolutionDeepAgent] Registered with WorkflowRegistry")
    except ImportError:
        logger.debug("[SolutionDeepAgent] Registry not available")
    except Exception as e:
        logger.warning(f"[SolutionDeepAgent] Registration failed: {e}")


# Register on module load
_register_workflow()

# =============================================================================
# INTERNAL HELPERS (Inlined from modification_agent)
# =============================================================================

def _fetch_user_documents(session_id: str) -> str:
    """Fetch user documents from Azure Blob for additional context."""
    try:
        from common.config.azure_blob_config import get_azure_blob_connection
        blob_conn = get_azure_blob_connection()
        documents_collection = blob_conn["collections"]["documents"]
        user_docs = documents_collection.find({"session_id": session_id})

        if user_docs:
            docs_summary = []
            for doc in user_docs[:3]:
                name = doc.get("filename", doc.get("name", "Unknown Document"))
                summary = (
                    f"Document: {name}\n"
                    f"Type: {doc.get('document_type', 'General')}\n"
                    f"Description: {doc.get('description', '')}"
                )
                docs_summary.append(summary)

            if docs_summary:
                return "\n---\n".join(docs_summary)
    except Exception as e:
        logger.warning(f"[SolutionDeepAgent] Azure Blob document fetch failed: {e}")

    return "No user specific documents found."


def _extract_field(text: str, label: str) -> str:
    """Extract a single-line field value from structured plain-text LLM output."""
    for line in text.splitlines():
        if line.strip().upper().startswith(label.upper() + ":"):
            return line.split(":", 1)[-1].strip()
    return ""


def _extract_block(text: str, label: str) -> str:
    """Extract a multi-line block from plain-text output."""
    lines = text.splitlines()
    collecting = False
    block = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith(label.upper() + ":"):
            collecting = True
            remainder = line.split(":", 1)[-1].strip()
            if remainder:
                block.append(remainder)
            continue
        if collecting:
            # Stop at next uppercase label
            if stripped and stripped.endswith(":") and stripped[:-1].isupper():
                break
            block.append(line)
    return "\n".join(block).strip()
