import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TaxonomyNormalizationAgent:
    def __init__(self, memory=None):
        self.memory = memory
        self._rag = None

    def _get_rag(self):
        if self._rag is None:
            try:
                from taxonomy_rag.rag import get_taxonomy_rag
                self._rag = get_taxonomy_rag()
            except Exception as e:
                logger.warning(f"[TaxonomyNorm] Could not load TaxonomyRAG: {e}")
        return self._rag

    def normalize_with_context(
        self, 
        items: List[Dict[str, Any]], 
        user_input: str,
        history: List[Dict[str, str]],
        item_type: str = "instrument"
    ) -> List[Dict[str, Any]]:
        """
        Normalize items taking conversation context into account.
        Resolves anaphora (e.g., 'it') and prioritizes active solution entities.
        """
        if not items:
            return items

        from taxonomy_rag.context_manager import TaxonomyContextManager
        
        # Initialize context manager
        ctx = TaxonomyContextManager()
        ctx.load_history(history)
        
        # We don't have the full list of prior active entities here easily without
        # passing them in, but we can infer from history or rely on the RAG agent's
        # injected memory if we extended the memory interface.
        # For now, we focus on enriching the term resolution with user input context.
        
        # Enhance resolution logic
        alias_map = self._get_alias_map()
        normalized = []

        for item in items:
            raw_name = (item.get("product_type") or item.get("product_name") or item.get("name") or item.get("category") or "")
            short_description = item.get("short_description", "") 
            
            # Combine raw name with rich context if available
            enrichment_base = f"{raw_name} {short_description}".strip()

            # 1. Resolve context on the name itself if it looks generic
            # e.g. "it", "the meter" -> resolved name
            resolved_name_from_context = ctx.resolve_contextual_references(raw_name)
            
            logger.info(f"[TaxonomyNorm Context] Raw Name: '{raw_name}' | Short Description: '{short_description}'")
            logger.info(f"[TaxonomyNorm Context] Resolved Name: '{resolved_name_from_context}' | Dispatching to Taxonomy...")

            # 2. Try standard resolution with potentially enriched name - passing the rich block to RAG
            resolution = self._resolve_term(
                raw_name=resolved_name_from_context, 
                item_type=item_type, 
                alias_map=alias_map,
                context_description=short_description
            )

            # 3. Fallback: If no match and we have a strong context hint from user input
            # (Logic: if the user input explicitly mentions a recognized term that matches this item)
            if not resolution["taxonomy_matched"]:
                # Try resolving against user input directly if the item name was vague
                enriched_input_term = ctx.resolve_contextual_references(user_input)
                # Heuristic: Check if the user input contains a taxonomy term
                # This is expensive to check against all aliases, so we rely on RAG with the full sentence
                pass

            enriched_item = dict(item)
            enriched_item["canonical_name"] = resolution["canonical_name"]
            
            # Explicitly perform renaming if a taxonomy match was successful
            if resolution["taxonomy_matched"]:
                if "product_type" in enriched_item:
                    enriched_item["product_type"] = resolution["canonical_name"]
                elif "product_name" in enriched_item:
                    enriched_item["product_name"] = resolution["canonical_name"]
                if "name" in enriched_item:
                    enriched_item["name"] = resolution["canonical_name"]
                    
            enriched_item["taxonomy_matched"] = resolution["taxonomy_matched"]
            enriched_item["match_source"] = resolution["match_source"]
            
            # Metadata for debugging
            if resolved_name_from_context != raw_name:
                enriched_item["context_resolved_from"] = raw_name
                enriched_item["context_resolved_to"] = resolved_name_from_context

            normalized.append(enriched_item)
            
        return normalized

    def _get_alias_map(self) -> Dict[str, str]:
        # Comprehensive alias map covering common user terms and abbreviations
        # Reference: taxonomy_rag_prompts.txt [ALIAS_MAP_REFERENCE] section
        alias_map: Dict[str, str] = {
            # TEMPERATURE
            "temperature": "Temperature Transmitter",
            "temp": "Temperature Transmitter",
            "temp sensor": "Temperature Transmitter",
            "temperature sensor": "Temperature Transmitter",
            "temperature probe": "Temperature Transmitter",
            "tt": "Temperature Transmitter",
            "ti": "Temperature Indicator",
            "te": "Temperature Element",
            "rtd": "Resistance Temperature Detector",
            "pt100": "Resistance Temperature Detector",
            "pt1000": "Resistance Temperature Detector",
            "tc": "Thermocouple",
            "thermocouple": "Thermocouple",
            "type k": "Thermocouple",
            "type j": "Thermocouple",
            "type t": "Thermocouple",

            # PRESSURE
            "pressure": "Pressure Transmitter",
            "press": "Pressure Transmitter",
            "pt": "Pressure Transmitter",
            "pi": "Pressure Indicator",
            "pg": "Pressure Gauge",
            "pressure gauge": "Pressure Gauge",
            "pdt": "Differential Pressure Transmitter",
            "dp": "Differential Pressure Transmitter",
            "dp transmitter": "Differential Pressure Transmitter",
            "differential pressure": "Differential Pressure Transmitter",
            "ps": "Pressure Switch",
            "pressure switch": "Pressure Switch",
            "psv": "Pressure Safety Valve",
            "prv": "Pressure Safety Valve",
            "pressure relief": "Pressure Safety Valve",

            # FLOW
            "flow": "Flow Meter",
            "flow meter": "Flow Meter",
            "ft": "Flow Transmitter",
            "fe": "Flow Element",
            "mag meter": "Magnetic Flow Meter",
            "magnetic meter": "Magnetic Flow Meter",
            "magnetic flow": "Magnetic Flow Meter",
            "coriolis": "Coriolis Flow Meter",
            "coriolis meter": "Coriolis Flow Meter",
            "mass flow": "Coriolis Flow Meter",
            "vortex": "Vortex Flow Meter",
            "vortex meter": "Vortex Flow Meter",
            "ultrasonic flow": "Ultrasonic Flow Meter",
            "clamp on": "Ultrasonic Flow Meter",
            "turbine meter": "Turbine Flow Meter",
            "orifice": "Orifice Plate",
            "orifice plate": "Orifice Plate",

            # LEVEL
            "level": "Level Transmitter",
            "lt": "Level Transmitter",
            "li": "Level Indicator",
            "ls": "Level Switch",
            "level switch": "Level Switch",
            "radar": "Radar Level Transmitter",
            "radar level": "Radar Level Transmitter",
            "80 ghz": "Radar Level Transmitter",
            "gwr": "Guided Wave Radar",
            "guided wave": "Guided Wave Radar",
            "ultrasonic level": "Ultrasonic Level Transmitter",
            "hydrostatic": "Hydrostatic Level Transmitter",
            "submersible": "Hydrostatic Level Transmitter",
            "float": "Float Level Switch",
            "float switch": "Float Level Switch",
            "displacer": "Displacer Level Transmitter",

            # ANALYTICAL
            "ph": "pH Analyzer",
            "ph meter": "pH Analyzer",
            "ph analyzer": "pH Analyzer",
            "conductivity": "Conductivity Analyzer",
            "do": "Dissolved Oxygen Analyzer",
            "dissolved oxygen": "Dissolved Oxygen Analyzer",
            "orp": "ORP Analyzer",
            "turbidity": "Turbidity Analyzer",
            "gas analyzer": "Gas Analyzer",

            # VALVES
            "cv": "Control Valve",
            "control valve": "Control Valve",
            "valve": "Control Valve",
            "modulating valve": "Control Valve",
            "positioner": "Valve Positioner",
            "valve positioner": "Valve Positioner",
            "actuator": "Valve Actuator",
            "ball valve": "Ball Valve",
            "globe valve": "Globe Valve",
            "butterfly valve": "Butterfly Valve",
            "solenoid": "Solenoid Valve",
            "solenoid valve": "Solenoid Valve",

            # ACCESSORIES
            "thermowell": "Thermowell",
            "tw": "Thermowell",
            "well": "Thermowell",
            "protection tube": "Thermowell",
            "manifold": "Manifold",
            "3 valve manifold": "3-Valve Manifold",
            "5 valve manifold": "5-Valve Manifold",
            "cable gland": "Cable Gland",
            "cable fitting": "Cable Gland",
            "jb": "Junction Box",
            "junction box": "Junction Box",
            "j-box": "Junction Box",
            "terminal head": "Terminal Head",
            "connection head": "Terminal Head",
            "gasket": "Gasket",
            "spiral wound": "Spiral Wound Gasket",
            "ring joint": "Ring Joint Gasket",
            "impulse line": "Impulse Line",
            "sensing line": "Impulse Line",
            "tubing": "Impulse Line",
            "mounting bracket": "Mounting Bracket",
            "bracket": "Mounting Bracket",
            "air filter regulator": "Air Filter Regulator",
            "afr": "Air Filter Regulator",
            "limit switch": "Limit Switch",
        }
        
        if not self.memory:
            return alias_map

        try:
            taxonomy = self.memory.get_taxonomy()
            if not taxonomy:
                return alias_map

            for section in ("instruments", "accessories"):
                for item in taxonomy.get(section, []):
                    canonical = item.get("name", "")
                    if not canonical:
                        continue
                    # Don't overwrite explicit logic
                    if canonical.lower() not in alias_map:
                        alias_map[canonical.lower()] = canonical
                    for alias in item.get("aliases", []):
                        if alias.lower() not in alias_map:
                            alias_map[alias.lower()] = canonical
        except Exception as e:
            logger.debug(f"[TaxonomyNorm] Could not build alias map: {e}")

        return alias_map

    def _resolve_term(self, raw_name: str, item_type: str, alias_map: Dict[str, str], context_description: str = "") -> Dict[str, Any]:
        raw_lower = raw_name.strip().lower()

        if raw_lower in alias_map:
            canonical = alias_map[raw_lower]
            logger.info(f"[TaxonomyNorm] Alias match: '{raw_name}' → '{canonical}'")
            return {
                "canonical_name": canonical,
                "taxonomy_matched": True,
                "match_source": "alias_map",
            }

        rag = self._get_rag()
        if rag:
            try:
                # Combine raw name and context description for a richer query
                query = f"{raw_name} {context_description}".strip()
                results = rag.retrieve(query=query, top_k=1, item_type=item_type)
                if results and results[0].get("score", 0) >= 0.70:
                    canonical = results[0]["name"]
                    logger.info(
                        f"[TaxonomyNorm] RAG match: '{raw_name}' → '{canonical}' "
                        f"(score={results[0]['score']:.2f})"
                    )
                    return {
                        "canonical_name": canonical,
                        "taxonomy_matched": True,
                        "match_source": "rag_retrieval",
                    }
            except Exception as e:
                logger.debug(f"[TaxonomyNorm] RAG retrieval error for '{raw_name}': {e}")

        logger.debug(f"[TaxonomyNorm] No match for '{raw_name}', using as-is")
        return {
            "canonical_name": raw_name,
            "taxonomy_matched": False,
            "match_source": "passthrough",
        }

    def normalize(self, items: List[Dict[str, Any]], item_type: str = "instrument") -> List[Dict[str, Any]]:
        if not items:
            return items

        alias_map = self._get_alias_map()
        normalized = []

        for item in items:
            raw_name = (item.get("product_type") or item.get("product_name") or item.get("name") or item.get("category") or "")
            short_description = item.get("short_description", "")
            
            logger.info(f"[TaxonomyNorm Standard] Processing Item: '{raw_name}'")
            logger.info(f"[TaxonomyNorm Standard] Extracted Short Desc: '{short_description}'")
            
            resolution = self._resolve_term(
                raw_name=raw_name, 
                item_type=item_type, 
                alias_map=alias_map,
                context_description=short_description
            )

            enriched_item = dict(item)
            enriched_item["canonical_name"] = resolution["canonical_name"]
            
            # Explicitly perform renaming if a taxonomy match was successful
            if resolution["taxonomy_matched"]:
                if "product_type" in enriched_item:
                    enriched_item["product_type"] = resolution["canonical_name"]
                elif "product_name" in enriched_item:
                    enriched_item["product_name"] = resolution["canonical_name"]
                if "name" in enriched_item:
                    enriched_item["name"] = resolution["canonical_name"]
                    
            enriched_item["taxonomy_matched"] = resolution["taxonomy_matched"]
            enriched_item["match_source"] = resolution["match_source"]

            normalized.append(enriched_item)

        matched = sum(1 for i in normalized if i.get("taxonomy_matched"))
        logger.info(
            f"[TaxonomyNorm] Normalized {len(normalized)} {item_type}(s): "
            f"{matched}/{len(normalized)} matched taxonomy"
        )
        return normalized

    def normalize_all(self, instruments: List[Dict[str, Any]], accessories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "standardized_instruments": self.normalize(instruments, "instrument"),
            "standardized_accessories": self.normalize(accessories, "accessory"),
        }

    def _get_reverse_alias_map(self) -> Dict[str, str]:
        """
        Builds a map from Canonical Name -> Shortest Alias (Database Code).
        Example: "Pressure Transmitter" -> "PT"
        """
        reverse_map: Dict[str, str] = {}
        if not self.memory:
            return reverse_map

        try:
            taxonomy = self.memory.get_taxonomy()
            if not taxonomy:
                return reverse_map

            for section in ("instruments", "accessories"):
                for item in taxonomy.get(section, []):
                    canonical = item.get("name", "")
                    aliases = item.get("aliases", [])
                    if not canonical or not aliases:
                        continue
                    
                    # Heuristic: The "Database Code" (e.g. PT) is often the shortest alias
                    # or the one that is all-caps. We'll pick the shortest one.
                    shortest_alias = min(aliases, key=len)
                    reverse_map[canonical.lower()] = shortest_alias
                    
        except Exception as e:
            logger.debug(f"[TaxonomyNorm] Could not build reverse alias map: {e}")

        return reverse_map

    def reverse_normalize(self, items: List[Dict[str, Any]], item_type: str = "instrument") -> List[Dict[str, Any]]:
        """
        Map canonical names back to their database codes/aliases.
        Example: "Pressure Transmitter" -> "PT"
        """
        if not items:
            return items

        reverse_map = self._get_reverse_alias_map()
        normalized = []

        for item in items:
            # We look for the canonical name in the item, or fallback to name
            name_to_reverse = (item.get("canonical_name") or item.get("product_type") or item.get("product_name") or item.get("name") or "").strip()
            name_lower = name_to_reverse.lower()
            
            reversed_code = name_to_reverse # Default to original if no match
            match_source = "passthrough"
            
            # 1. Exact canonical match
            if name_lower in reverse_map:
                reversed_code = reverse_map[name_lower]
                match_source = "reverse_map"
                logger.info(f"[TaxonomyNorm] Reverse match: '{name_to_reverse}' → '{reversed_code}'")
            
            enriched_item = dict(item)
            enriched_item["reverse_name"] = reversed_code
            enriched_item["reverse_match_source"] = match_source
            
            normalized.append(enriched_item)
            
        return normalized

    def reverse_normalize_all(self, instruments: List[Dict[str, Any]], accessories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "reverse_instruments": self.reverse_normalize(instruments, "instrument"),
            "reverse_accessories": self.reverse_normalize(accessories, "accessory"),
        }

    def batch_normalize_solution_items(
        self,
        instruments: List[Dict[str, Any]],
        accessories: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        user_input: str = ""
    ) -> Dict[str, Any]:
        """
        Batch normalize all instruments and accessories from Solution Deep Agent.
        
        This is the main entry point for processing identified items from the
        Solution workflow before they are passed to the Search workflow.
        
        Args:
            instruments: List of identified instruments with 'product_type', 'quantity', etc.
            accessories: List of identified accessories
            conversation_history: Optional conversation context for anaphora resolution
            user_input: Original user input for context enrichment
            
        Returns:
            {
                "standardized_instruments": [...],  # Normalized instruments
                "standardized_accessories": [...],  # Normalized accessories
                "normalization_stats": {
                    "total_items": 30,
                    "instruments_processed": 10,
                    "accessories_processed": 20,
                    "instruments_matched": 8,
                    "accessories_matched": 18,
                    "match_rate": 0.867
                }
            }
        """
        logger.info(
            f"[TaxonomyNorm] Batch normalization: "
            f"{len(instruments)} instruments, {len(accessories)} accessories"
        )
        
        # If context is provided, use context-aware normalization
        if conversation_history and user_input:
            standardized_instruments = self.normalize_with_context(
                instruments, user_input, conversation_history, "instrument"
            )
            standardized_accessories = self.normalize_with_context(
                accessories, user_input, conversation_history, "accessory"
            )
        else:
            # Standard normalization
            standardized_instruments = self.normalize(instruments, "instrument")
            standardized_accessories = self.normalize(accessories, "accessory")
        
        # Calculate statistics
        total_items = len(instruments) + len(accessories)
        instruments_matched = sum(
            1 for item in standardized_instruments 
            if item.get("taxonomy_matched", False)
        )
        accessories_matched = sum(
            1 for item in standardized_accessories 
            if item.get("taxonomy_matched", False)
        )
        
        total_matched = instruments_matched + accessories_matched
        match_rate = total_matched / total_items if total_items > 0 else 0.0
        
        stats = {
            "total_items": total_items,
            "instruments_processed": len(instruments),
            "accessories_processed": len(accessories),
            "instruments_matched": instruments_matched,
            "accessories_matched": accessories_matched,
            "total_matched": total_matched,
            "match_rate": round(match_rate, 3)
        }
        
        logger.info(
            f"[TaxonomyNorm] Batch complete: {total_matched}/{total_items} matched "
            f"({stats['match_rate']*100:.1f}%)"
        )
        
        return {
            "standardized_instruments": standardized_instruments,
            "standardized_accessories": standardized_accessories,
            "normalization_stats": stats
        }
