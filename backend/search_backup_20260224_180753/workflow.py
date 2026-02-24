# search/workflow.py
"""
Product Search Workflow - Simplified Architecture (Feb 2026)
============================================================

This file provides backward compatibility for old imports.

The workflow is now implemented using:
- Simplified functions in validation_functions.py and advanced_specs_functions.py
- Single LangGraph orchestrator in search_workflow.py
- VendorAnalysisDeepAgent (kept as deep agent - justified complexity)

For the main workflow functions, import from search module:
    from search import run_product_search_workflow

For individual functions:
    from search.validation_functions import run_validation, load_schema
    from search.advanced_specs_functions import discover_advanced_specs
    from search.vendor_analysis_deep_agent import VendorAnalysisDeepAgent
    from search.ranking_tool import RankingTool
"""

import logging

logger = logging.getLogger(__name__)

# Re-export workflow functions for backward compatibility
from . import (
    run_product_search_workflow,
    run_validation_only,
    run_advanced_params_only,
    run_analysis_only,
    process_from_solution_workflow,
    get_schema_only,
    validate_with_schema,
)

__all__ = [
    "run_product_search_workflow",
    "run_validation_only",
    "run_advanced_params_only",
    "run_analysis_only",
    "process_from_solution_workflow",
    "get_schema_only",
    "validate_with_schema",
]
