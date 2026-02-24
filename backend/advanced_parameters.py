# advanced_parameters.py
#
# Backward-compatibility shim.
# The implementation lives in search/advanced_specification_agent.py (upstream).
# This file re-exports the public API so that any existing callers using:
#
#     from advanced_parameters import discover_advanced_parameters
#
# continue to work without modification.

from search.advanced_specification_agent import (
    AdvancedSpecificationAgent,
    discover_advanced_parameters,
)

def discover_advanced_specs(product_type: str, session_id: str = None, **kwargs):
    """Convenience wrapper around AdvancedSpecificationAgent.discover()."""
    return AdvancedSpecificationAgent().discover(
        product_type=product_type,
        session_id=session_id,
    )

__all__ = [
    "AdvancedSpecificationAgent",
    "discover_advanced_parameters",
    "discover_advanced_specs",
]
