"""
Unified Standards Module

This module consolidates all standards-related functionality:
- RAG: Query and retrieve from standards documents
- Generation: Generate comprehensive specifications using standards
- Shared: Common utilities, cache, keywords, enrichment
"""

__version__ = "1.0.0"

# =============================================================================
# Import RAG functionality
# =============================================================================

from common.rag.standards import (
    # Workflow
    StandardsRAGState,
    create_standards_rag_state,
    create_standards_rag_workflow,
    get_standards_rag_workflow,
    run_standards_rag_workflow,
    # Chat
    StandardsChatAgent,
    create_standards_chat_agent,
    get_standards_chat_agent,
    # Memory
    StandardsRAGMemory,
    standards_rag_memory,
    get_standards_rag_memory,
    resolve_standards_follow_up,
    add_to_standards_memory,
    clear_standards_memory,
    # Enrichment
    enrich_identified_items_with_standards,
    validate_items_against_domain_standards,
    is_standards_related_question,
    route_standards_question,
)

# =============================================================================
# Import Generation functionality
# =============================================================================

from .generation import (
    StandardsDeepAgentState,
    ConsolidatedSpecs,
    WorkerResult,
    StandardConstraint,
    run_standards_deep_agent,
    run_standards_deep_agent_batch,
    get_standards_deep_agent_workflow,
    STANDARD_DOMAINS,
    STANDARD_FILES,
    StandardsSpecification,
    StandardsMapping,
)

# =============================================================================
# Import Shared utilities
# =============================================================================

from .shared import (
    # Cache
    get_cached_standards,
    cache_standards,
    clear_standards_cache,
    get_cache_key,
    # Constants
    MIN_STANDARDS_SPECS_COUNT,
    MAX_SPECS_PER_ITEM,
    MAX_SPECS_PER_DOMAIN,
    CHUNK_SIZE,
    DEEP_AGENT_LLM_MODEL,
    DEFAULT_TOP_K,
    STANDARDS_DOCX_DIR,
    # Keywords
    StandardsDomain,
    DOMAIN_KEYWORDS,
    DOMAIN_TO_DOCUMENTS,
    FIELD_GROUPS,
    # Detector
    classify_domain,
    get_routed_documents,
    detect_standards_indicators,
    # Enrichment
    ParallelSchemaEnricher,
    enrich_schema_parallel,
    enrich_schema_async,
    enrich_items_parallel,
    run_3_source_enrichment,
    is_valid_spec_value,
    normalize_category,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # Version
    '__version__',

    # RAG - Workflow
    'StandardsRAGState',
    'create_standards_rag_state',
    'create_standards_rag_workflow',
    'get_standards_rag_workflow',
    'run_standards_rag_workflow',

    # RAG - Chat
    'StandardsChatAgent',
    'create_standards_chat_agent',
    'get_standards_chat_agent',

    # RAG - Memory
    'StandardsRAGMemory',
    'standards_rag_memory',
    'get_standards_rag_memory',
    'resolve_standards_follow_up',
    'add_to_standards_memory',
    'clear_standards_memory',

    # RAG - Enrichment
    'enrich_identified_items_with_standards',
    'validate_items_against_domain_standards',
    'is_standards_related_question',
    'route_standards_question',

    # Generation - Deep Agent
    'StandardsDeepAgentState',
    'ConsolidatedSpecs',
    'WorkerResult',
    'StandardConstraint',
    'run_standards_deep_agent',
    'run_standards_deep_agent_batch',
    'get_standards_deep_agent_workflow',
    'STANDARD_DOMAINS',
    'STANDARD_FILES',

    # Generation - Integration
    'StandardsSpecification',
    'StandardsMapping',

    # Shared - Cache
    'get_cached_standards',
    'cache_standards',
    'clear_standards_cache',
    'get_cache_key',

    # Shared - Constants
    'MIN_STANDARDS_SPECS_COUNT',
    'MAX_SPECS_PER_ITEM',
    'MAX_SPECS_PER_DOMAIN',
    'CHUNK_SIZE',
    'DEEP_AGENT_LLM_MODEL',
    'DEFAULT_TOP_K',
    'STANDARDS_DOCX_DIR',

    # Shared - Keywords
    'StandardsDomain',
    'DOMAIN_KEYWORDS',
    'DOMAIN_TO_DOCUMENTS',
    'FIELD_GROUPS',

    # Shared - Detector
    'classify_domain',
    'get_routed_documents',
    'detect_standards_indicators',

    # Shared - Enrichment
    'ParallelSchemaEnricher',
    'enrich_schema_parallel',
    'enrich_schema_async',
    'enrich_items_parallel',
    'run_3_source_enrichment',
    'is_valid_spec_value',
    'normalize_category',
]
