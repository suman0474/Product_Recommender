# solution_N/identification_agents.py
# =============================================================================
# DEEP AGENT IDENTIFICATION - Instruments & Accessories
# =============================================================================
#
# Replaces tool-based identification (identify_instruments_tool,
# identify_accessories_tool) with deep agent functions that leverage:
# - Memory-aware identification (reads/writes DeepAgentMemory)
# - Adaptive prompts via AdaptivePromptEngine
# - Failure learning via SchemaFailureMemory
# - Conversation context for enriched understanding
# - Parallel execution of instrument + accessory identification
#
# =============================================================================

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.services.llm.fallback import create_llm_with_fallback
from common.config import AgenticConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from common.prompts import SOLUTION_DEEP_AGENT_PROMPTS

from common.agentic.deep_agent.memory import (
    DeepAgentMemory,
    IdentifiedItemMemory,
)
from taxonomy_rag.context_manager import TaxonomyContextManager

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

_SOLUTION_PROMPTS = SOLUTION_DEEP_AGENT_PROMPTS
_INSTRUMENT_PROMPT = _SOLUTION_PROMPTS.get("INSTRUMENT_IDENTIFICATION", "")
_ACCESSORIES_PROMPT = _SOLUTION_PROMPTS.get("ACCESSORIES_IDENTIFICATION", "")


# =============================================================================
# ACCESSORY KEYWORD FALLBACK (preserved from tool)
# =============================================================================

ACCESSORY_KEYWORDS = {
    "thermowell": "Thermowell / Protective Sleeve",
    "extension wire": "Thermocouple Extension Wire",
    "shield": "Thermocouple Protection Shield",
    "head": "Connection Head / Terminal Block",
    "impulse line": "Impulse Line / Tubing",
    "isolation valve": "Isolation Valve / Manifold",
    "snubber": "Snubber / Pulsation Dampener",
    "straightener": "Flow Straightener",
    "orifice": "Orifice Plate / Flow Restriction",
    "cable gland": "Cable Gland / Connector",
    "junction box": "Junction Box / Terminal Enclosure",
    "mounting bracket": "Mounting Bracket / Support",
    "gasket": "Gasket / Sealing Material",
    "power supply": "Power Supply Unit",
    "signal cable": "Signal Cable / Wiring",
    "connector": "Connector / Adapter",
    "conduit": "Conduit / Cable Protection",
    "terminal strip": "Terminal Strip / Connection Board",
    "calibration": "Calibration Kit / Standard",
    # Missing entries (gap vs original tool — added)
    "equalizing": "Equalizing Connection",
    "tap": "Pressure Tap / Sensing Point",
    "fastener": "Fastener / Bolt Set",
    "strain relief": "Strain Relief / Cable Management",
    "protective cap": "Protective Cap / Cover",
    "spare parts": "Spare Parts / Replacement Kit",
}


# =============================================================================
# INSTRUMENT IDENTIFICATION AGENT
# =============================================================================

class InstrumentIdentificationAgent:
    """
    Deep Agent for instrument identification.

    Replaces identify_instruments_tool with:
    - Memory-aware identification
    - Adaptive prompt optimization
    - Failure learning
    - Context-enriched prompts
    """

    def __init__(self, memory: Optional[DeepAgentMemory] = None):
        self.memory = memory
        self._adaptive_engine = None
        self._failure_memory = None
        self._init_learning()

    def _init_learning(self):
        """Initialize adaptive prompt engine and failure memory."""
        try:
            from common.agentic.deep_agent.agents.adaptive_prompt_engine import get_adaptive_prompt_engine
            self._adaptive_engine = get_adaptive_prompt_engine()
        except ImportError:
            logger.debug("[InstrumentAgent] Adaptive prompt engine not available")

        try:
            from common.infrastructure.caching import get_schema_failure_memory
            self._failure_memory = get_schema_failure_memory()
        except ImportError:
            logger.debug("[InstrumentAgent] Failure memory not available")

    def _get_optimized_prompt(self, base_prompt: str, product_type: str = "solution") -> str:
        """Get an optimized prompt using adaptive engine."""
        if self._adaptive_engine is None:
            return base_prompt

        try:
            optimization = self._adaptive_engine.optimize_prompt(
                base_prompt=base_prompt,
                product_type=product_type,
            )
            if optimization and optimization.strategy.value != "standard":
                logger.info(
                    f"[InstrumentAgent] Using {optimization.strategy.value} prompt "
                    f"(confidence={optimization.confidence:.2f})"
                )
                return optimization.modifications[0] if optimization.modifications else base_prompt
        except Exception as e:
            logger.debug(f"[InstrumentAgent] Prompt optimization skipped: {e}")

        return base_prompt

    def _get_taxonomy_guidance(self, item_type: str = "instruments", query: str = "") -> str:
        """
        Get taxonomy guidance string using RAG retrieval.
        Falls back to memory dump if RAG fails or returns no results.
        """
        guidance_items = []

        # 1. Try RAG Retrieval
        try:
            from taxonomy_rag import get_taxonomy_rag
            rag = get_taxonomy_rag()
            # Map item_type to singular for RAG metadata filter
            filter_type = "instrument" if "instrument" in item_type else "accessory"
            
            results = rag.retrieve(query=query, top_k=5, item_type=filter_type)
            if results:
                logger.info(f"[InstrumentAgent] RAG retrieved {len(results)} taxonomy terms")
                for item in results:
                    text = f"- {item['name']}"
                    if item.get('aliases'):
                        text += f" (Aliases: {', '.join(item['aliases'])})"
                    guidance_items.append(text)
        except Exception as e:
             logger.warning(f"[InstrumentAgent] Taxonomy RAG failed: {e}")

        # 2. Fallback to Memory if RAG empty (for small taxonomies this is fine)
        if not guidance_items and self.memory:
            taxonomy = self.memory.get_taxonomy()
            if taxonomy and item_type in taxonomy:
                items = taxonomy.get(item_type, [])
                for item in items:
                    text = f"- {item['name']}"
                    if item.get('aliases'):
                        text += f" (Aliases: {', '.join(item['aliases'])})"
                    guidance_items.append(text)
        
        if not guidance_items:
            return ""

        return "\n".join(guidance_items)

    def identify(
        self,
        requirements: str,
        context: str = "",
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Identify instruments from requirements using deep agent approach.

        Args:
            requirements: Process requirements or solution description
            context: Additional context from solution analysis
            conversation_context: Enriched context from conversation history

        Returns:
            Dict with success, instruments, project_name, summary
        """
        last_error = None
        start_time = time.time()

        # Context Awareness: Resolve anaphora (e.g. "it")
        ctx_manager = TaxonomyContextManager()
        # If we have memory, we can try to load history/active items if available
        # Ideally, we'd pass history into identify(), but conversation_context is a string here.
        # We can extract recent entities from memory if needed.
        if self.memory:
             # Populate active entities from memory if possible
             # For now, just rely on heuristics or if active items are passed in context
             pass

        resolved_requirements = ctx_manager.resolve_contextual_references(requirements)
        if resolved_requirements != requirements:
            logger.info(f"[InstrumentAgent] Resolved requirements: '{requirements}' -> '{resolved_requirements}'")

        # Build enriched requirements with context
        enriched_requirements = resolved_requirements
        if context:
            enriched_requirements = f"{resolved_requirements}\n\nAdditional Context:\n{context}"
        if conversation_context:
            enriched_requirements = f"{enriched_requirements}\n\n{conversation_context}"

        # Inject Taxonomy Guidance
        taxonomy_guidance = self._get_taxonomy_guidance("instruments", query=requirements)
        if taxonomy_guidance:
            enriched_requirements += (
                f"\n\n[TAXONOMY RULE] Use these STANDARD NAMES for items if applicable:\n"
                f"{taxonomy_guidance}\n"
                f"Map user terms (e.g., 'PT', 'Pressure Sensor') to the standard name (e.g., 'Pressure Transmitter')."
            )

        # Get optimized prompt
        prompt_text = self._get_optimized_prompt(_INSTRUMENT_PROMPT, "solution_instruments")

        # 3 retry attempts with progressive refinement
        for attempt in range(1, 4):
            try:
                logger.info(f"[InstrumentAgent] Attempt {attempt}/3...")

                llm = create_llm_with_fallback(
                    model=AgenticConfig.FLASH_MODEL,
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    timeout=180,
                )

                prompt = ChatPromptTemplate.from_template(prompt_text)
                parser = JsonOutputParser()
                chain = prompt | llm | parser

                result = chain.invoke({"requirements": enriched_requirements})

                if not isinstance(result, dict):
                    last_error = f"Invalid response type: {type(result)}"
                    continue

                instruments = result.get("instruments", [])

                if instruments and isinstance(instruments, list):
                    # Store in memory
                    if self.memory:
                        for inst in instruments:
                            self.memory.store_identified_item(IdentifiedItemMemory(
                                item_id=f"inst_{inst.get('category', 'unknown')}_{time.time_ns()}",
                                item_type="instrument",
                                product_type=inst.get("category", ""),
                                category=inst.get("category", ""),
                                quantity=int(inst.get("quantity", 1)),
                                user_specifications=inst.get("specifications", {}),
                                relevant_documents=[],
                            ))

                    # Record success in failure memory
                    if self._failure_memory:
                        try:
                            self._failure_memory.record_success(
                                product_type="solution_instruments",
                                spec_count=len(instruments),
                            )
                        except Exception:
                            pass

                    elapsed_ms = int((time.time() - start_time) * 1000)
                    logger.info(
                        f"[InstrumentAgent] Identified {len(instruments)} instruments "
                        f"in {elapsed_ms}ms (attempt {attempt})"
                    )

                    return {
                        "success": True,
                        "project_name": result.get("project_name", "Solution"),
                        "instruments": instruments,
                        "instrument_count": len(instruments),
                        "summary": result.get("summary", ""),
                        "elapsed_ms": elapsed_ms,
                        "attempt": attempt,
                    }
                else:
                    last_error = "Empty instruments list"
                    continue

            except Exception as e:
                last_error = "Extraction failed"

                # Record failure for learning
                if self._failure_memory and attempt == 3:
                    try:
                        from common.infrastructure.caching import FailureType
                        self._failure_memory.record_failure(
                            product_type="solution_instruments",
                            failure_type=FailureType.EXTRACTION_ERROR,
                            error_message=str(e),
                        )
                    except Exception:
                        pass

                continue

        logger.error(f"[InstrumentAgent] All 3 attempts failed: {last_error}")
        return {
            "success": False,
            "instruments": [],
            "error": f"Identification failed after 3 attempts: {last_error}",
        }


# =============================================================================
# ACCESSORY IDENTIFICATION AGENT
# =============================================================================

class AccessoryIdentificationAgent:
    """
    Deep Agent for accessory identification.

    Replaces identify_accessories_tool with:
    - Memory-aware identification
    - Keyword fallback preserved
    - Context-enriched from solution analysis
    """

    def __init__(self, memory: Optional[DeepAgentMemory] = None):
        self.memory = memory

    def _get_taxonomy_guidance(self, item_type: str = "accessories", query: str = "") -> str:
        """
        Get taxonomy guidance string using RAG retrieval.
        Falls back to memory dump if RAG fails or returns no results.
        """
        guidance_items = []

        # 1. Try RAG Retrieval
        try:
            from taxonomy_rag import get_taxonomy_rag
            rag = get_taxonomy_rag()
             # Map item_type to singular for RAG metadata filter
            filter_type = "instrument" if "instrument" in item_type else "accessory"
            
            results = rag.retrieve(query=query, top_k=5, item_type=filter_type)
            if results:
                logger.info(f"[AccessoryAgent] RAG retrieved {len(results)} taxonomy terms")
                for item in results:
                    text = f"- {item['name']}"
                    if item.get('aliases'):
                        text += f" (Aliases: {', '.join(item['aliases'])})"
                    guidance_items.append(text)
        except Exception as e:
             logger.warning(f"[AccessoryAgent] Taxonomy RAG failed: {e}")

        # 2. Fallback to Memory
        if not guidance_items and self.memory:
            taxonomy = self.memory.get_taxonomy()
            if taxonomy and item_type in taxonomy:
                items = taxonomy.get(item_type, [])
                for item in items:
                    text = f"- {item['name']}"
                    if item.get('aliases'):
                        text += f" (Aliases: {', '.join(item['aliases'])})"
                    guidance_items.append(text)

        if not guidance_items:
            return ""
            
        return "\n".join(guidance_items)

    def identify(
        self,
        instruments: List[Dict[str, Any]],
        process_context: str = "",
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Identify accessories for the given instruments.

        Args:
            instruments: List of identified instruments
            process_context: Process/solution context
            conversation_context: Enriched conversation context

        Returns:
            Dict with success, accessories, method
        """
        if not instruments:
            return {
                "success": False,
                "accessories": [],
                "error": "No instruments provided",
                "method": "validation_failure",
            }

        context = process_context or "General industrial process"
        if conversation_context:
            context = f"{context}\n\n{conversation_context}"

        # Inject Taxonomy Guidance
        taxonomy_guidance = self._get_taxonomy_guidance("accessories", query=context)
        if taxonomy_guidance:
            context += (
                f"\n\n[TAXONOMY RULE] Use these STANDARD NAMES for accessories if applicable:\n"
                f"{taxonomy_guidance}\n"
                f"Map generic terms to these standard names."
            )

        last_error = None

        # 3 retry attempts
        for attempt in range(1, 4):
            try:
                logger.info(f"[AccessoryAgent] Attempt {attempt}/3...")

                llm = create_llm_with_fallback(
                    model=AgenticConfig.FLASH_MODEL,
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    timeout=180,
                )

                prompt = ChatPromptTemplate.from_template(_ACCESSORIES_PROMPT)
                parser = JsonOutputParser()
                chain = prompt | llm | parser

                result = chain.invoke({
                    "instruments": json.dumps(instruments, indent=2),
                    "process_context": context,
                })

                if not isinstance(result, dict):
                    last_error = f"Invalid response type: {type(result)}"
                    continue

                accessories = result.get("accessories", [])

                if accessories and isinstance(accessories, list):
                    # Store in memory
                    if self.memory:
                        for acc in accessories:
                            self.memory.store_identified_item(IdentifiedItemMemory(
                                item_id=f"acc_{acc.get('category', 'unknown')}_{time.time_ns()}",
                                item_type="accessory",
                                product_type=acc.get("category", ""),
                                category=acc.get("category", ""),
                                quantity=int(acc.get("quantity", 1)),
                                user_specifications=acc.get("specifications", {}),
                                relevant_documents=[],
                            ))

                    logger.info(f"[AccessoryAgent] Identified {len(accessories)} accessories (attempt {attempt})")
                    return {
                        "success": True,
                        "accessories": accessories,
                        "accessory_count": len(accessories),
                        "summary": result.get("summary", ""),
                        "method": "llm_identification",
                    }
                else:
                    last_error = "Empty accessories list"
                    continue

            except Exception as e:
                last_error = "Extraction failed"
                continue

        # Fallback to keyword extraction
        logger.warning(f"[AccessoryAgent] LLM failed, using keyword fallback")
        fallback = self._keyword_fallback(instruments, context)

        if fallback:
            return {
                "success": True,
                "accessories": fallback,
                "accessory_count": len(fallback),
                "method": "keyword_fallback",
                "llm_failure_reason": last_error,
            }

        return {
            "success": False,
            "accessories": [],
            "error": f"Identification failed: {last_error}",
            "method": "none",
        }

    @staticmethod
    def _keyword_fallback(
        instruments: List[Dict[str, Any]],
        context: str,
    ) -> List[Dict[str, Any]]:
        """Extract accessories using keyword matching when LLM fails."""
        combined_text = f"{context} {json.dumps(instruments)}".lower()
        matched = set()

        for keyword, accessory_type in ACCESSORY_KEYWORDS.items():
            if keyword in combined_text:
                matched.add(accessory_type)

        accessories = []
        for idx, acc_type in enumerate(sorted(matched), 1):
            accessories.append({
                "number": idx,
                "name": acc_type,
                "category": acc_type.split("/")[0].strip(),
                "quantity": 1,
                "specifications": {"extraction_method": "keyword_fallback"},
                "source_method": "keyword_extraction",
                "confidence": "low",
            })

        logger.info(f"[AccessoryAgent] Keyword fallback: {len(accessories)} accessories")
        return accessories


# =============================================================================
# PARALLEL IDENTIFICATION
# =============================================================================

def identify_instruments_and_accessories_parallel(
    requirements: str,
    context: str = "",
    conversation_context: str = "",
    memory: Optional[DeepAgentMemory] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run instrument and accessory identification in parallel.

    Phase 1: Identify instruments (required first for accessories)
    Phase 2: Identify accessories using instrument results

    Note: Accessories depend on instruments, so true parallelism is limited.
    However, we use a shared LLM instance to reduce overhead.

    Args:
        requirements: Process requirements
        context: Solution analysis context
        conversation_context: Enriched conversation context
        memory: Deep agent memory for persistence

    Returns:
        Tuple of (instrument_result, accessory_result)
    """
    instrument_agent = InstrumentIdentificationAgent(memory=memory)
    accessory_agent = AccessoryIdentificationAgent(memory=memory)

    # Phase 1: Identify instruments
    inst_result = instrument_agent.identify(
        requirements=requirements,
        context=context,
        conversation_context=conversation_context,
    )

    # Phase 2: Identify accessories based on instruments
    instruments = inst_result.get("instruments", [])
    acc_result = accessory_agent.identify(
        instruments=instruments,
        process_context=context,
        conversation_context=conversation_context,
    )

    return (inst_result, acc_result)
