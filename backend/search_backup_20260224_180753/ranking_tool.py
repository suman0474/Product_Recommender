"""
Ranking Tool for Product Search Workflow
=========================================

Step 5 (Final) of Product Search Workflow:
- Takes vendor analysis results
- Uses LLM with get_ranking_prompt for intelligent ranking
- Generates detailed key strengths and concerns for each product
- Returns final ordered ranking with parameter-by-parameter analysis

Ranking Prompt Analysis (from prompts.py get_ranking_prompt):
============================================================
The ranking prompt instructs the LLM to:

1. **Step-by-step ranking process:**
   - Review all vendor analysis results and identify common patterns
   - Extract ALL mandatory and optional parameter matches for each product
   - Identify any limitations or concerns mentioned in the vendor analysis
   - Calculate comparative scores based on requirement fulfillment
   - Rank products from best to worst match

2. **Key Strengths extraction:**
   For each parameter that matches requirements:
   - [Friendly Parameter Name](User Requirement)
   - Product provides "[Product Specification]"
   - Holistic explanation: why it matches, justification, impact, interactions

3. **Concerns extraction:**
   For each parameter that does not match:
   - Holistic explanation: why it doesn't meet requirement
   - Limitation, potential impact, interactions with other parameters

4. **Critical requirements:**
   - Extract EVERY limitation from vendor analysis "Key Limitations" section
   - Include EVERY parameter in either strengths or concerns
   - Preserve limitations for buyer decision-making
   - Verification checklist before finalizing

This tool integrates:
- get_ranking_prompt from prompts.py for LLM-powered ranking
- invoke_ranking_chain for LLM execution
- Fallback to score-based ranking if LLM fails
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union

# Configure logging
logger = logging.getLogger(__name__)

# Debug flags for ranking tool debugging
try:
    from debug_flags import debug_log, timed_execution, is_debug_enabled
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    def debug_log(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def timed_execution(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def is_debug_enabled(module):
        return False


class RankingTool:
    """
    Ranking Tool - Step 5 (Final) of Product Search Workflow

    Responsibilities:
    1. Process vendor analysis results
    2. Apply get_ranking_prompt for LLM-powered comparative ranking
    3. Extract detailed key strengths (parameter matches with explanations)
    4. Extract concerns (unmatched requirements, limitations)
    5. Return ordered ranking with detailed analysis for each product

    The ranking uses a step-by-step approach as defined in get_ranking_prompt:
    - Reviews vendor analysis results
    - Extracts mandatory/optional parameter matches
    - Identifies limitations and concerns
    - Calculates comparative scores
    - Ranks products from best to worst
    """

    def __init__(self, use_llm_ranking: bool = True):
        """
        Initialize the ranking tool.

        Args:
            use_llm_ranking: If True, use LLM with get_ranking_prompt for intelligent ranking.
                           If False, use simple score-based ranking (faster but less detailed).
        """
        self.use_llm_ranking = use_llm_ranking
        logger.info("[RankingTool] Initialized (LLM ranking: %s)", use_llm_ranking)

    @timed_execution("RANKING_TOOL", threshold_ms=20000)
    @debug_log("RANKING_TOOL", log_args=True, log_result=False)
    def rank(
        self,
        vendor_analysis: Dict[str, Any],
        session_id: Optional[str] = None,
        structured_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Rank products based on vendor analysis using LLM-powered ranking prompt.

        The ranking prompt (get_ranking_prompt) instructs the LLM to:
        1. Review all vendor analysis and identify patterns
        2. Extract ALL parameter matches for each product
        3. Identify limitations/concerns from vendor analysis
        4. Calculate comparative scores
        5. Rank products with detailed key strengths and concerns

        Args:
            vendor_analysis: Results from VendorAnalysisTool containing vendor_matches
            session_id: Session tracking ID
            structured_requirements: Optional user requirements for context

        Returns:
            {
                'success': bool,
                'overall_ranking': list of ranked products with:
                    - rank: int
                    - product_name: str
                    - vendor: str
                    - overall_score/match_score: int
                    - requirements_match: bool
                    - key_strengths: list/str (detailed parameter matches)
                    - concerns: list/str (limitations and unmatched requirements)
                'top_product': dict (highest ranked product)
                'total_ranked': int
                'ranking_summary': str
            }
        """
        logger.info("[RankingTool] Starting product ranking")
        logger.info("[RankingTool] Session: %s", session_id or "N/A")

        result = {
            "success": False,
            "session_id": session_id
        }

        try:
            # Step 1: Validate input
            logger.info("[RankingTool] Step 1: Validating vendor analysis input")
            vendor_matches = vendor_analysis.get('vendor_matches', [])

            if not vendor_matches:
                logger.warning("[RankingTool] No vendor matches to rank")
                result['success'] = True
                result['overall_ranking'] = []
                result['top_product'] = None
                result['total_ranked'] = 0
                result['ranking_summary'] = "No products available to rank"
                return result

            logger.info("[RankingTool] Found %d vendor matches to rank", len(vendor_matches))

            # Step 2: Generate ranking using LLM or fallback
            logger.info("[RankingTool] Step 2: Generating product ranking")

            if self.use_llm_ranking:
                overall_ranking = self._rank_with_llm(vendor_analysis)
            else:
                overall_ranking = self._rank_by_score(vendor_analysis)

            if not overall_ranking:
                # Fallback to score-based ranking
                logger.warning("[RankingTool] LLM ranking failed, using score-based fallback")
                overall_ranking = self._rank_by_score(vendor_analysis)

            # Step 3: Process and normalize ranking results
            logger.info("[RankingTool] Step 3: Processing ranking results")

            # Ensure proper sorting by score
            overall_ranking = sorted(
                overall_ranking,
                key=lambda x: x.get('overall_score', x.get('match_score', x.get('matchScore', 0))),
                reverse=True
            )

            # Add/update rank numbers
            for i, product in enumerate(overall_ranking):
                product['rank'] = i + 1

            # Get top product
            top_product = overall_ranking[0] if overall_ranking else None

            # Build result
            result['success'] = True
            result['overall_ranking'] = overall_ranking
            result['top_product'] = top_product
            result['total_ranked'] = len(overall_ranking)

            # Generate summary
            if top_product:
                top_name = top_product.get('product_name', top_product.get('productName', 'Unknown'))
                top_vendor = top_product.get('vendor', 'Unknown')
                top_score = top_product.get('overall_score', top_product.get('match_score', top_product.get('matchScore', 0)))
                result['ranking_summary'] = (
                    f"Ranked {len(overall_ranking)} products. "
                    f"Top recommendation: {top_name} by {top_vendor} "
                    f"with {top_score}% match score"
                )
            else:
                result['ranking_summary'] = "No products ranked"

            logger.info("[RankingTool] Ranking complete: %s", result['ranking_summary'])

            return result

        except Exception as e:
            logger.error("[RankingTool] Ranking failed: %s", str(e), exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            return result

    def _normalize_reasoning(self, field_data: Union[str, List[Any]]) -> str:
        """Normalize complex LLM output into clean Markdown string."""
        if isinstance(field_data, str):
            return field_data
        
        if isinstance(field_data, list):
            items = []
            for item in field_data:
                # Handle Dictionary format (seen in logs: {'parameter':..., 'input':...})
                if isinstance(item, dict):
                    # Construct readable string from dict keys
                    param = item.get('parameter', '')
                    spec = item.get('input_value', item.get('product_specification', ''))
                    text = item.get('holistic_explanation', item.get('explanation', item.get('limitation', '')))
                    # Fallback for simple "limitation" dicts
                    if not param and not spec and text:
                        items.append(f"- {text}")
                    elif param:
                        items.append(f"**{param}**: {spec} - {text}")
                    else:
                        items.append(str(item))
                else:
                    items.append(str(item))
            return "\n\n".join(items)
        
        return str(field_data)

    def _rank_with_llm(self, vendor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM with get_ranking_prompt for intelligent ranking.

        This method:
        1. Sets up LangChain components
        2. Invokes invoke_ranking_chain with get_ranking_prompt
        3. Parses the LLM response into ranked products
        4. Each product includes detailed key_strengths and concerns

        Args:
            vendor_analysis: Vendor analysis results

        Returns:
            List of ranked products with detailed analysis
        """
        try:
            from core.chaining import (
                setup_langchain_components,
                invoke_ranking_chain,
                to_dict_if_pydantic
            )

            logger.info("[RankingTool] Using LLM-powered ranking with get_ranking_prompt")

            # Setup LangChain components
            components = setup_langchain_components()
            format_instructions = components.get('ranking_format_instructions', '')

            # Prepare vendor analysis for prompt
            vendor_analysis_str = json.dumps(vendor_analysis, indent=2, default=str)

            # Invoke ranking chain (uses get_ranking_prompt internally)
            logger.info("[RankingTool] Invoking ranking chain...")
            ranking_result = invoke_ranking_chain(
                components,
                vendor_analysis_str,
                format_instructions
            )

            if not ranking_result:
                logger.warning("[RankingTool] LLM ranking returned empty result")
                return []

            # Convert to dict if Pydantic
            ranking_dict = to_dict_if_pydantic(ranking_result)

            # Extract ranked products
            ranked_products = ranking_dict.get('ranked_products', ranking_dict.get('rankedProducts', []))

            logger.info("[RankingTool] LLM ranking produced %d ranked products", len(ranked_products))

            # Normalize field names for consistency
            normalized_products = []
            
            # Create lookup dictionaries for pricing info, description, and standards compliance from original vendor analysis
            pricing_lookup = {}
            description_lookup = {}
            standards_lookup = {}
            matched_reqs_lookup = {}
            
            if vendor_analysis and 'vendor_matches' in vendor_analysis:
                for match in vendor_analysis['vendor_matches']:
                    key = (match.get('vendor', ''), match.get('productName', match.get('product_name', '')))
                    pricing_lookup[key] = {
                        'pricing_url': match.get('pricing_url', ''),
                        'pricing_source': match.get('pricing_source', '')
                    }
                    description_lookup[key] = match.get('product_description', match.get('productDescription', ''))
                    standards_lookup[key] = match.get('standards_compliance', match.get('standardsCompliance', {}))
                    matched_reqs_lookup[key] = {
                        'matched_requirements': match.get('matched_requirements', match.get('matchedRequirements', {})),
                        'unmatched_requirements': match.get('unmatched_requirements', match.get('unmatchedRequirements', []))
                    }

            for product in ranked_products:
                p_vendor = product.get('vendor', '')
                p_name = product.get('product_name', product.get('productName', ''))
                
                # Retrieve pricing info, description, standards, and requirements
                pricing_info = pricing_lookup.get((p_vendor, p_name), {})
                product_desc = description_lookup.get((p_vendor, p_name), '')
                standards_info = standards_lookup.get((p_vendor, p_name), {})
                reqs_info = matched_reqs_lookup.get((p_vendor, p_name), {})
                
                normalized = {
                    'productName': p_name,
                    'vendor': p_vendor,
                    'modelFamily': product.get('model_family', product.get('modelFamily', '')),
                    'overallScore': product.get('overall_score', product.get('overallScore', product.get('match_score', product.get('matchScore', 0)))),
                    'matchScore': product.get('match_score', product.get('matchScore', product.get('overall_score', product.get('overallScore', 0)))),
                    'requirementsMatch': product.get('requirements_match', product.get('requirementsMatch', False)),
                    'keyStrengths': self._normalize_reasoning(product.get('key_strengths', product.get('keyStrengths', []))),
                    'concerns': self._normalize_reasoning(product.get('concerns', product.get('limitations', []))),
                    'productDescription': product_desc,
                    'standardsCompliance': standards_info,
                    'matchedRequirements': reqs_info.get('matched_requirements', {}),
                    'unmatchedRequirements': reqs_info.get('unmatched_requirements', []),
                    'pricing_url': pricing_info.get('pricing_url', ''),
                    'pricing_source': pricing_info.get('pricing_source', '')
                }
                normalized_products.append(normalized)

            return normalized_products

        except Exception as e:
            logger.error("[RankingTool] LLM ranking failed: %s", str(e), exc_info=True)
            return []

    def _rank_by_score(self, vendor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback: Simple score-based ranking without LLM.

        This is faster but produces less detailed key_strengths and concerns.
        Uses the reasoning and limitations directly from vendor analysis.

        Args:
            vendor_analysis: Vendor analysis results

        Returns:
            List of ranked products sorted by score
        """
        try:
            from core.chaining import get_final_ranking, to_dict_if_pydantic

            logger.info("[RankingTool] Using score-based ranking (fallback)")

            # Use existing get_final_ranking function
            ranking_result = get_final_ranking(vendor_analysis)

            if not ranking_result:
                return []

            ranking_dict = to_dict_if_pydantic(ranking_result)

            # Handle both 'ranked_products' and 'rankedProducts' keys
            ranked_products = ranking_dict.get('ranked_products', ranking_dict.get('rankedProducts', []))

            # Create lookup dictionaries for pricing info, description, and standards compliance from original vendor analysis
            pricing_lookup = {}
            description_lookup = {}
            standards_lookup = {}
            matched_reqs_lookup = {}
            
            if vendor_analysis and 'vendor_matches' in vendor_analysis:
                for match in vendor_analysis['vendor_matches']:
                    key = (match.get('vendor', ''), match.get('productName', match.get('product_name', '')))
                    pricing_lookup[key] = {
                        'pricing_url': match.get('pricing_url', ''),
                        'pricing_source': match.get('pricing_source', '')
                    }
                    description_lookup[key] = match.get('product_description', match.get('productDescription', ''))
                    standards_lookup[key] = match.get('standards_compliance', match.get('standardsCompliance', {}))
                    matched_reqs_lookup[key] = {
                        'matched_requirements': match.get('matched_requirements', match.get('matchedRequirements', {})),
                        'unmatched_requirements': match.get('unmatched_requirements', match.get('unmatchedRequirements', []))
                    }

            # Normalize field names
            normalized_products = []
            for product in ranked_products:
                # Compute requirements_match from score if not already set (≥80% = exact match)
                score = product.get('match_score', product.get('matchScore', product.get('overall_score', 0)))
                computed_match = score >= 80
                
                p_vendor = product.get('vendor', '')
                p_name = product.get('product_name', product.get('productName', ''))
                
                # Retrieve pricing info, description, standards, and requirements
                pricing_info = pricing_lookup.get((p_vendor, p_name), {})
                product_desc = description_lookup.get((p_vendor, p_name), '')
                standards_info = standards_lookup.get((p_vendor, p_name), {})
                reqs_info = matched_reqs_lookup.get((p_vendor, p_name), {})

                normalized = {
                    'productName': p_name,
                    'vendor': p_vendor,
                    'modelFamily': product.get('model_family', product.get('modelFamily', '')),
                    'overallScore': product.get('overall_score', product.get('overallScore', product.get('match_score', 0))),
                    'matchScore': product.get('match_score', product.get('matchScore', product.get('overall_score', 0))),
                    'requirementsMatch': product.get('requirements_match', product.get('requirementsMatch', computed_match)),
                    'keyStrengths': product.get('key_strengths', product.get('keyStrengths', product.get('reasoning', ''))),
                    'concerns': product.get('concerns', product.get('limitations', '')),
                    'productDescription': product_desc,
                    'standardsCompliance': standards_info,
                    'matchedRequirements': reqs_info.get('matched_requirements', {}),
                    'unmatchedRequirements': reqs_info.get('unmatched_requirements', []),
                    'pricing_url': pricing_info.get('pricing_url', ''),
                    'pricing_source': pricing_info.get('pricing_source', '')
                }
                normalized_products.append(normalized)


            return normalized_products

        except Exception as e:
            logger.error("[RankingTool] Score-based ranking failed: %s", str(e), exc_info=True)
            return []

    def generate_comparison_table(
        self,
        overall_ranking: List[Dict[str, Any]],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comparison table for ranked products.

        Args:
            overall_ranking: List of ranked products
            requirements: User requirements for comparison columns

        Returns:
            Comparison table structure for UI display
        """
        if not overall_ranking:
            return {"columns": [], "rows": []}

        # Extract all unique parameters from requirements
        all_params = set()

        # From mandatory requirements
        mandatory = requirements.get('mandatoryRequirements', requirements.get('mandatory', {}))
        all_params.update(mandatory.keys())

        # From optional requirements
        optional = requirements.get('optionalRequirements', requirements.get('optional', {}))
        all_params.update(optional.keys())

        # Build columns
        columns = [
            {"key": "rank", "label": "Rank"},
            {"key": "vendor", "label": "Vendor"},
            {"key": "product_name", "label": "Product"},
            {"key": "match_score", "label": "Match %"},
        ]

        # Add parameter columns
        for param in sorted(all_params):
            columns.append({
                "key": param,
                "label": self._format_field_name(param)
            })

        # Build rows
        rows = []
        for product in overall_ranking:
            row = {
                "rank": product.get('rank', 0),
                "vendor": product.get('vendor', 'Unknown'),
                "product_name": product.get('product_name', product.get('productName', 'Unknown')),
                "match_score": product.get('match_score', product.get('matchScore', 0)),
            }

            # Add parameter values from key strengths
            key_strengths = product.get('keyStrengths', product.get('key_strengths', []))
            for strength in key_strengths:
                if isinstance(strength, dict):
                    param_name = strength.get('parameter', strength.get('name', ''))
                    param_value = strength.get('specification', strength.get('value', ''))
                    # Match to column
                    for param in all_params:
                        if param.lower() in param_name.lower():
                            row[param] = param_value
                            break

            rows.append(row)

        return {
            "columns": columns,
            "rows": rows
        }

    def _format_field_name(self, field: str) -> str:
        """Convert camelCase or snake_case to Title Case."""
        import re
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field)
        words = words.replace('_', ' ')
        return words.title()


# Convenience function
def rank_products(
    vendor_analysis: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to rank products.

    Args:
        vendor_analysis: Results from vendor analysis
        session_id: Session tracking ID

    Returns:
        Ranking result
    """
    tool = RankingTool()
    return tool.rank(
        vendor_analysis=vendor_analysis,
        session_id=session_id
    )
