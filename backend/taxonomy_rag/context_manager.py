import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TaxonomyContextManager:
    """
    Manages conversation context for Taxonomy Normalization.
    Handles history tracking, anaphora resolution, and context filtering.
    """

    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.active_entities: Dict[str, List[Dict[str, Any]]] = {
            "instruments": [],
            "accessories": []
        }

    def load_history(self, history: List[Dict[str, str]]) -> None:
        """Load conversation history (last N turns)."""
        if not history:
            return
        # Keep only user and assistant messages to reduce noise
        filtered = [
            msg for msg in history 
            if msg.get("role") in ("user", "assistant")
        ]
        self.history = filtered[-self.max_history:]

    def set_active_entities(self, instruments: List[Dict[str, Any]], accessories: List[Dict[str, Any]]) -> None:
        """
        Update the list of currently identified items in the solution.
        These serve as the primary context for resolving references like "it" or "the meter".
        """
        self.active_entities["instruments"] = instruments or []
        self.active_entities["accessories"] = accessories or []

    def resolve_contextual_references(self, user_input: str) -> str:
        """
        Enrich user input by resolving anaphora (pronouns) using active entities.
        Example: "Change it to 500C" -> "Change the Pressure Transmitter to 500C"
        """
        lower_input = user_input.lower()
        
        # 1. Check for specific pronouns/references
        anaphora_triggers = [
            "it", "that", "this", 
            "the instrument", "the meter", "the sensor", "the transmitter", 
            "the device", "the unit"
        ]
        
        # 2. Add variable shorthand context triggers
        variable_triggers = {
            "temperature": "temperature",
            "temp": "temperature",
            "flow": "flow",
            "pressure": "pressure",
            "level": "level"
        }
        
        # First, see if they used a generic variable word and we have a specific matching instrument active
        for var, var_type in variable_triggers.items():
            if re.search(r'\b' + re.escape(var) + r'\b', lower_input):
                for inst in reversed(self.active_entities["instruments"]):
                    canonical = (inst.get("canonical_name") or inst.get("name") or "").lower()
                    if var_type in canonical:
                        mapped_name = inst.get('canonical_name') or inst.get('name')
                        logger.info(f"[TaxonomyContext] Resolved variable '{var}' to active entity: '{mapped_name}'")
                        if lower_input.strip() == var:
                            # If the input is EXACTLY the variable (e.g., the raw extracted item name), just replace it
                            return mapped_name
                        return f"{user_input} (referring to {mapped_name})"

        # Fallback to standard anaphora
        has_trigger = False
        exact_trigger = False
        for t in anaphora_triggers:
            if re.search(r'\b' + re.escape(t) + r'\b', lower_input):
                has_trigger = True
                if lower_input.strip() == t:
                    exact_trigger = True
                break
        
        if not has_trigger:
            return user_input

        # 3. Try to find the most recent/relevant entity
        # Strategy: Logic favors the last mentioned instrument if available
        # In a real scenario, this would use an LLM or dependency parsing. 
        # For now, we use a heuristic based on active items.
        
        latest_instrument = None
        if self.active_entities["instruments"]:
            # Assume the last added instrument is the most relevant topic
            latest_instrument = self.active_entities["instruments"][-1]
            
        if latest_instrument:
            canonical = latest_instrument.get("canonical_name") or latest_instrument.get("name")
            if canonical:
                logger.info(f"[TaxonomyContext] Resolved context: '{latest_instrument.get('name')}' for input '{user_input}'")
                if exact_trigger:
                    return canonical
                return f"{user_input} (referring to {canonical})"

        return user_input

    def filter_relevant_context(self) -> str:
        """
        Summarize history into a string relevant for taxonomy tasks.
        Excludes chatter, focuses on product modifications.
        """
        context_str = ""
        for msg in self.history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Simple heuristic: ignore very short acknowledgments
            if len(content.split()) > 2:
                context_str += f"{role.upper()}: {content}\n"
        return context_str.strip()
