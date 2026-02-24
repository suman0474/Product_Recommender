# agentic/api.py
# Flask API Endpoints for Agentic Workflow
#
# ARCHITECTURE PRINCIPLE:
# - All workflow execution MUST go through API endpoints
# - Direct workflow function calls ONLY allowed within endpoint view functions
# - Orchestration code (router, chainers) MUST use internal_api.api_client
# - This ensures complete decoupling between workflows
#
# This module exposes LangGraph workflows as REST API endpoints that can be
# called by the UI or internally by other workflows through the api_client.


import json
import logging
import uuid
import threading
import time as time_module
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, session
from functools import wraps

# Import rate limiting
# Import rate limiting
from common.infrastructure.rate_limiter import get_limiter, RateLimitConfig

# Import consolidated decorators and utilities
from common.utils.auth_decorators import login_required
from .utils import (
    api_response, 
    handle_errors,
    convert_keys_to_camel_case,
    clean_empty_values,
    map_provided_to_schema,
    get_missing_mandatory_fields,
    friendly_field_name
)


from common.workflows.base.workflow import run_workflow
# Import from solution (aliased for backward compatibility)
import importlib
_solution_workflow = importlib.import_module("solution.workflow")
run_solution_workflow = _solution_workflow.run_solution_deep_agent
# Comparison logic consolidated into search
# Indexing import moved to local scope in run_indexing endpoint

# Streaming endpoints are in api_streaming.py

# Import internal API client for workflow orchestration
from .internal import api_client
from common.agentic.models import create_initial_state, IntentType, WorkflowType
import importlib as _importlib

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION CONTEXT - Session/Workflow/Task Isolation
# ═══════════════════════════════════════════════════════════════════════════
from common.infrastructure.context import (
    ExecutionContext,
    execution_context,
    set_context,
    get_context
)

# Lazy loaders — avoids module-level import of hyphenated package
_solution_id_agents = None

def _get_id_agents():
    global _solution_id_agents
    if _solution_id_agents is None:
        _solution_id_agents = _importlib.import_module("solution.identification_agents")
    return _solution_id_agents


logger = logging.getLogger(__name__)

# Create Blueprint
agentic_bp = Blueprint('agentic', __name__, url_prefix='/api/agentic')


# ============================================================================
# SERVER-SIDE WORKFLOW STATE STORAGE (BOUNDED WITH AUTO-CLEANUP)
# Replaces Flask session to fix concurrent tab issues (cookie overwrite)
# Phase 4 Optimization: Bounded memory with automatic cleanup
# ============================================================================
from ..caching import get_workflow_state_manager

# Get bounded state manager (auto-starts cleanup thread)
_state_manager = get_workflow_state_manager(
    max_states=10000,      # Max 10,000 concurrent states
    ttl_seconds=3600       # 1 hour TTL
)


def get_workflow_state(thread_id: str) -> Dict[str, Any]:
    """Get workflow state for a thread (thread-safe, bounded)."""
    state = _state_manager.get(thread_id)
    if state:
        logger.debug(f"[WORKFLOW_STATE] Retrieved state for {thread_id}: phase={state.get('phase')}")
    return state


def set_workflow_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Save workflow state for a thread (thread-safe, bounded with LRU eviction)."""
    _state_manager.set(thread_id, state)
    logger.debug(f"[WORKFLOW_STATE] Saved state for {thread_id}: phase={state.get('phase')}")


def cleanup_expired_workflow_states() -> int:
    """
    Manual cleanup trigger (automatic cleanup runs in background).

    Returns count of manually triggered cleanup attempts.
    Note: Automatic cleanup happens every 5 minutes in background.
    """
    # Trigger immediate cleanup if needed
    stats = _state_manager.get_stats()
    usage = stats.get("usage_percent", 0)

    if usage > 80:
        logger.info(f"[WORKFLOW_STATE] Usage at {usage}%, triggering cleanup...")
        _state_manager._cleanup_expired_states()

    return 1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_session_id() -> str:
    """Get or create session ID"""
    if 'agentic_session_id' not in session:
        session['agentic_session_id'] = str(uuid.uuid4())
    return session['agentic_session_id']


def create_execution_context_from_request(
    data: Dict[str, Any],
    workflow_type: str
) -> ExecutionContext:
    """
    Create ExecutionContext from API request data.

    Extracts all IDs from request and creates proper hierarchy for
    session/workflow/task isolation.

    Args:
        data: Request JSON data
        workflow_type: Type of workflow being invoked (product_search, solution, etc.)

    Returns:
        ExecutionContext ready for workflow execution
    """
    # Extract user info
    user_id = data.get('user_id') or session.get('user_id') or 'anonymous'
    zone_str = data.get('zone') or 'DEFAULT'

    # Extract or generate session_id (main_thread_id)
    session_id = (
        data.get('main_thread_id') or
        data.get('session_id') or
        data.get('search_session_id')
    )

    if not session_id:
        # Generate using HierarchicalThreadManager if available
        try:
            from common.infrastructure.state.execution.thread_manager import (
                HierarchicalThreadManager, ThreadZone
            )
            try:
                zone = ThreadZone(zone_str)
            except ValueError:
                zone = ThreadZone.DEFAULT

            session_id = HierarchicalThreadManager.generate_main_thread_id(
                user_id=user_id,
                zone=zone
            )
        except ImportError:
            # Fallback format
            session_id = f"main_{user_id}_{zone_str}_{uuid.uuid4().hex[:8]}_{int(time_module.time()*1000)}"

    # Extract or generate workflow_id
    workflow_id = (
        data.get('workflow_thread_id') or
        data.get('thread_id') or
        ""  # Will be auto-generated by ExecutionContext
    )

    # Extract parent workflow ID if this is a nested workflow
    parent_workflow_id = data.get('parent_workflow_thread_id')

    # Create context
    ctx = ExecutionContext(
        session_id=session_id,
        user_id=user_id,
        zone=zone_str,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        parent_workflow_id=parent_workflow_id,
        correlation_id=data.get('correlation_id') or uuid.uuid4().hex[:12]
    )

    logger.debug(f"[Context] Created from request: {ctx.to_log_context()}")
    return ctx


# Note: api_response and handle_errors are imported from .api_utils
# The api_response function supports tags parameter for backward compatibility


# Rate limit decorator helpers
def get_rate_limit_decorator(limit_type):
    """
    Get a rate limit decorator for the specified type.

    Args:
        limit_type: Type of limit ('agentic_workflow', 'agentic_tool', etc.)

    Returns:
        Decorator function or no-op if limiter not available
    """
    limiter = get_limiter()
    if not limiter:
        return lambda f: f  # No-op if limiter not available

    limits = RateLimitConfig.LIMITS.get(limit_type, RateLimitConfig.DEFAULT_LIMITS)
    return limiter.limit(limits)


# Convenience decorator functions
workflow_limited = lambda f: get_rate_limit_decorator('agentic_workflow')(f) if get_limiter() else f
tool_limited = lambda f: get_rate_limit_decorator('agentic_tool')(f) if get_limiter() else f
session_limited = lambda f: get_rate_limit_decorator('session_management')(f) if get_limiter() else f
health_limited = lambda f: get_rate_limit_decorator('health')(f) if get_limiter() else f


# ============================================================================
# ROUTER ENDPOINTS
# ============================================================================

@agentic_bp.route('/classify-route', methods=['POST'])
@login_required
@handle_errors
def classify_route():
    """
    Classify Query and Route to Workflow
    ---
    tags:
      - LangChain Agents
    summary: Classify user input and route to appropriate workflow
    description: |
      Uses IntentClassificationRoutingAgent to classify user queries from the UI textarea
      and determine which workflow to route to:
      - solution: Complex systems requiring multiple instruments
      - instrument_identifier: Single product requirements
      - engenie_chat: Questions about products/standards (EnGenie Chat)
      - out_of_domain: Unrelated queries (rejected with helpful message)
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: User query from UI textarea
              example: "I need a pressure transmitter 0-100 PSI"
            context:
              type: object
              description: Optional context (current_step, conversation history)
    responses:
      200:
        description: Workflow routing decision
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                target_workflow:
                  type: string
                  enum: [solution, instrument_identifier, engenie_chat, out_of_domain]
                intent:
                  type: string
                  description: Raw intent from classify_intent_tool
                confidence:
                  type: number
                reasoning:
                  type: string
                is_solution:
                  type: boolean
                reject_message:
                  type: string
                  description: Message for out-of-domain queries
    """
    from ...agentic.agents.routing.intent_classifier import IntentClassificationRoutingAgent
    
    data = request.get_json()
    query = data.get('query', '').strip()
    context = data.get('context') or {}
    # CRITICAL: Get session_id for workflow state isolation between users
    session_id = data.get('session_id') or data.get('search_session_id') or 'default'
    
    # Accept workflow hint from frontend (will be validated by agent)
    workflow_hint = data.get('workflow_hint')
    if workflow_hint:
        context['workflow_hint'] = workflow_hint
    
    if not query:
        return api_response(False, error="query is required", status_code=400)
    
    logger.info(f"[CLASSIFY_ROUTE] Query: {query[:100]}... (session: {session_id[:16]}, hint: {workflow_hint})")
    
    agent = IntentClassificationRoutingAgent()
    result = agent.classify(query, session_id=session_id, context=context)
    
    return api_response(True, data=result.to_dict())


@agentic_bp.route('/product-info-decision', methods=['POST'])
@login_required
@handle_errors
def product_info_decision():
    """
    Get Product Info Page Routing Decision
    ---
    tags:
      - LangChain Agents
    summary: Determine if query should route to EnGenie Chat
    description: |
      Uses ProductInfoIntentAgent to make detailed routing decisions for
      the EnGenie Chat in the frontend.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: User query to analyze
              example: "Show me Yokogawa pressure transmitter models"
    responses:
      200:
        description: Routing decision
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                should_route:
                  type: boolean
                confidence:
                  type: number
                data_source:
                  type: string
                sources:
                  type: array
                  items:
                    type: string
                reasoning:
                  type: string
    """
    from chat.engenie_chat_intent_agent import get_engenie_chat_route_decision
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return api_response(False, error="query is required", status_code=400)
    
    logger.info(f"[ENGENIE_CHAT_DECISION] Analyzing query: {query[:100]}...")
    
    result = get_engenie_chat_route_decision(query)
    
    return api_response(True, data=result)





@agentic_bp.route('/validate-product-input', methods=['POST'])
@login_required
@handle_errors
def validate_product_input():
    """
    Validation API - Step 1: Detect Product Type from User Input

    Matches main.py /validate-product-type implementation.
    Uses session management and same helper functions as main.py.

    Request:
        {
            "user_input": "I need a pressure transmitter with 0-100 bar range",
            "search_session_id": "optional_session_id"
        }

    Response:
        {
            "success": true,
            "data": {
                "productType": "pressure transmitter",
                "confidence": 0.9,
                "reasoning": "Detected from user input analysis",
                "normalizedInput": "...",
                "sessionId": "session_id"
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[VALIDATE] Product Type Detection API Called")

    user_input = data.get('user_input') or data.get('message')

    if not user_input:
        logger.error("[VALIDATE] No user_input provided")
        return api_response(False, error="user_input is required", status_code=400)

    search_session_id = data.get('search_session_id', get_session_id())

    logger.info(f"[VALIDATE] Session {search_session_id}: Detecting product type")
    logger.info(f"[VALIDATE] User input: {user_input[:100]}...")

    try:
        # Import from main.py's loading module (same as main.py uses)
        from common.core.loading import load_requirements_schema

        # Load initial generic schema for product type detection
        initial_schema = load_requirements_schema()

        # Get the validation components from main app if available
        # This ensures we use the same LLM chain as main.py
        try:
            from main import components
            if not components:
                raise Exception("Backend components not ready")
        except:
            # Fallback: use our own LLM if main components not available
            from common.services.llm.fallback import create_llm_with_fallback
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            import os
            import json

            detection_prompt = ChatPromptTemplate.from_template("""
You are an expert validator. Extract the product type from user input.

User Input: {user_input}

Return JSON with 'product_type' field containing the detected product type in lowercase.
""")

            llm = create_llm_with_fallback(model="gemini-2.5-flash", temperature=0.1)
            parser = JsonOutputParser()
            chain = detection_prompt | llm | parser

            detection_result = chain.invoke({"user_input": user_input})
            components = {'validation_chain': chain, 'validation_format_instructions': ''}

        # Add session context to prevent cross-contamination (same as main.py)
        session_isolated_input = f"[Session: {search_session_id}] - Product type detection. User input: {user_input}"

        # Use validation chain to detect product type (same as main.py)
        if hasattr(components.get('validation_chain'), 'invoke'):
            detection_result = components['validation_chain'].invoke({
                "user_input": session_isolated_input,
                "schema": json.dumps(initial_schema, indent=2),
                "format_instructions": components.get('validation_format_instructions', '')
            })
        else:
            # Fallback
            detection_result = {'product_type': 'unknown'}

        detected_type = detection_result.get('product_type', 'UnknownProduct')

        logger.info(f"[VALIDATE] Detected product type: {detected_type}")

        # Store in session for later use (same as main.py)
        session[f'product_type_{search_session_id}'] = detected_type
        session[f'log_user_query_{search_session_id}'] = user_input

        response_data = {
            "productType": detected_type,
            "confidence": 0.9,
            "reasoning": "Detected from user input analysis",
            "normalizedInput": user_input,
            "sessionId": search_session_id
        }

        return api_response(True, data=response_data)

    except Exception as e:
        logger.error(f"[VALIDATE] Product type detection failed: {e}")
        import traceback
        logger.error(f"[VALIDATE] Traceback: {traceback.format_exc()}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/get-product-schema', methods=['POST'])
@login_required
@handle_errors
def get_product_schema():
    """
    Schema Get API - Step 2: Get Schema and Map with User Input

    Matches main.py /validate implementation pattern.
    Uses same helper functions: convert_keys_to_camel_case, map_provided_to_schema, clean_empty_values.

    Request:
        {
            "product_type": "pressure transmitter",
            "user_input": "I need a pressure transmitter with 0-100 bar range",
            "search_session_id": "optional_session_id"
        }

    Response:
        {
            "success": true,
            "data": {
                "productType": "pressure transmitter",
                "detectedSchema": {
                    "mandatoryRequirements": {...},
                    "optionalRequirements": {...}
                },
                "providedRequirements": {...},
                "missingMandatory": ["outputSignal"],
                "validationAlert": {
                    "message": "...",
                    "canContinue": true,
                    "missingFields": [...]
                }
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[SCHEMA_GET] Schema Retrieval and Mapping API Called")

    product_type = data.get('product_type')
    user_input = data.get('user_input') or data.get('message')
    search_session_id = data.get('search_session_id', get_session_id())

    if not product_type:
        logger.error("[SCHEMA_GET] No product_type provided")
        return api_response(False, error="product_type is required", status_code=400)

    if not user_input:
        logger.error("[SCHEMA_GET] No user_input provided")
        return api_response(False, error="user_input is required", status_code=400)

    logger.info(f"[SCHEMA_GET] Product type: {product_type}")
    logger.info(f"[SCHEMA_GET] User input: {user_input[:100]}...")

    try:
        # Import helper functions from main.py
        import sys
        import re
        import copy
        import json

        # Helper functions moved to api_utils.py

        # Load schema (same as main.py)
        from common.core.loading import load_requirements_schema, build_requirements_schema_from_web

        logger.info("[SCHEMA_GET] Loading schema...")
        specific_schema = load_requirements_schema(product_type)

        if not specific_schema or (not specific_schema.get("mandatory_requirements") and not specific_schema.get("optional_requirements")):
            logger.warning(f"[SCHEMA_GET] Schema not found, building from web for {product_type}")
            try:
                specific_schema = build_requirements_schema_from_web(product_type)
            except Exception as build_error:
                logger.error(f"[SCHEMA_GET] Web schema build failed: {build_error}")
                specific_schema = {
                    "mandatory_requirements": {},
                    "optional_requirements": {}
                }

        # Get validation components (same pattern as main.py)
        try:
            from main import components
            if not components:
                raise Exception("Components not ready")
        except:
            # Fallback if main components not available
            components = None

        # Add session context (same as main.py)
        session_isolated_input = f"[Session: {search_session_id}] - Schema validation. User input: {user_input}"

        # Validate using chain if available
        if components and hasattr(components.get('validation_chain'), 'invoke'):
            validation_result = components['validation_chain'].invoke({
                "user_input": session_isolated_input,
                "schema": json.dumps(specific_schema, indent=2),
                "format_instructions": components.get('validation_format_instructions', '')
            })
        else:
            # Fallback: basic extraction
            validation_result = {
                "product_type": product_type,
                "provided_requirements": {}
            }

        # Clean and map (same as main.py)
        cleaned_provided_reqs = clean_empty_values(validation_result.get("provided_requirements", {}))

        mapped_provided_reqs = map_provided_to_schema(
            convert_keys_to_camel_case(specific_schema),
            convert_keys_to_camel_case(cleaned_provided_reqs)
        )

        # Build response (same structure as main.py)
        response_data = {
            "productType": validation_result.get("product_type", product_type),
            "detectedSchema": convert_keys_to_camel_case(specific_schema),
            "providedRequirements": mapped_provided_reqs
        }

        # Missing fields identification moved to api_utils.py

        missing_mandatory_fields = get_missing_mandatory_fields(
            mapped_provided_reqs, response_data["detectedSchema"]
        )

        if missing_mandatory_fields:
            missing_fields_friendly = [friendly_field_name(f) for f in missing_mandatory_fields]
            missing_fields_str = ", ".join(missing_fields_friendly)

            response_data["validationAlert"] = {
                "message": f"Please provide the following required information: {missing_fields_str}",
                "canContinue": True,
                "missingFields": missing_mandatory_fields
            }
            response_data["missingMandatory"] = missing_mandatory_fields

        logger.info(f"[SCHEMA_GET] Extracted {len(cleaned_provided_reqs)} requirements")
        logger.info(f"[SCHEMA_GET] Missing mandatory: {len(missing_mandatory_fields)}")

        # Store in session (same as main.py)
        session[f'product_type_{search_session_id}'] = response_data["productType"]

        return api_response(True, data=response_data)

    except Exception as e:
        logger.error(f"[SCHEMA_GET] Schema retrieval and mapping failed: {e}")
        import traceback
        logger.error(f"[SCHEMA_GET] Traceback: {traceback.format_exc()}")
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# WORKFLOW ENDPOINTS
# ============================================================================


@agentic_bp.route('/chat', methods=['POST'])
@login_required
@handle_errors
def chat():
    """
    Main Chat Endpoint for Agentic Workflow
    ---
    tags:
      - Agentic Workflows
    summary: Process user message through agentic workflow
    description: |
      Main entry point for conversational AI workflows.
      Supports multiple workflow types including procurement and instrument identification.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: User message to process
              example: "I need pressure transmitters for a crude oil refinery"
            session_id:
              type: string
              description: Optional session ID for conversation continuity
              example: "abc123-session"
            workflow_type:
              type: string
              enum: [procurement, instrument_identification]
              default: procurement
              description: Type of workflow to run
    responses:
      200:
        description: Successful response from workflow
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                response:
                  type: string
                  description: Agent response text
                intent:
                  type: string
                  description: Classified intent
                product_type:
                  type: string
                  description: Detected product type
                requires_user_input:
                  type: boolean
                current_step:
                  type: string
      400:
        description: Bad request - missing required fields
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()
    workflow_type = data.get('workflow_type', 'procurement')

    # Run workflow
    result = run_workflow(
        user_input=message,
        session_id=session_id,
        workflow_type=workflow_type
    )

    return api_response(True, data=result)


@agentic_bp.route('/identify', methods=['POST'])
@login_required
@handle_errors
def identify_instruments():
    """
    Identify instruments from process requirements

    Request Body:
    {
        "requirements": "process description or requirements text"
    }

    Response:
    {
        "success": true,
        "data": {
            "project_name": "...",
            "instruments": [...],
            "accessories": [...],
            "summary": "..."
        }
    }
    """
    data = request.get_json()

    if not data or 'requirements' not in data:
        return api_response(False, error="Requirements are required", status_code=400)

    requirements = data['requirements']
    session_id = data.get('session_id') or get_session_id()

    # Run instrument identification workflow
    result = run_workflow(
        user_input=requirements,
        session_id=session_id,
        workflow_type='instrument_identification'
    )

    # Check for identification failure
    if result.get('identification_failed'):
        error_msg = result.get('identification_error', 'Failed to identify instruments')
        is_rate_limit = any(x in str(error_msg) for x in ['RESOURCE_EXHAUSTED', 'quota', '429'])
        return api_response(
            False, 
            error=error_msg, 
            status_code=503 if is_rate_limit else 500,
            retryable=is_rate_limit
        )

    return api_response(True, data=result)


@agentic_bp.route('/analyze', methods=['POST'])
@login_required
@handle_errors
def analyze_requirements():
    """
    Analyze requirements and run full procurement workflow

    Request Body:
    {
        "requirements": "technical requirements",
        "vendor_filter": ["optional", "vendor", "list"]
    }

    Response:
    {
        "success": true,
        "data": {
            "response": "...",
            "ranked_products": [...],
            "vendor_analysis": {...}
        }
    }
    """
    data = request.get_json()

    if not data or 'requirements' not in data:
        return api_response(False, error="Requirements are required", status_code=400)

    requirements = data['requirements']
    vendor_filter = data.get('vendor_filter')
    session_id = data.get('session_id') or get_session_id()

    # Store vendor filter in session if provided
    if vendor_filter:
        session['csv_vendor_filter'] = {
            'vendor_names': vendor_filter
        }

    # Run procurement workflow
    result = run_workflow(
        user_input=requirements,
        session_id=session_id,
        workflow_type='procurement'
    )

    return api_response(True, data=result)



# ============================================================================
# REMOVED DUPLICATE: run_analysis_endpoint() - Use run_product_analysis() at line ~2670 instead
# REMOVED DUPLICATE: solution_workflow() - Use solution_workflow_endpoint() at line ~1430 instead
# These duplicates were removed during code cleanup on 2026-02-16
# ============================================================================


# ============================================================================
# INDEX RAG ENDPOINT
# ============================================================================

@agentic_bp.route('/index-rag', methods=['POST'])
@login_required
@handle_errors
def index_rag_search():
    """
    Index RAG Product Search
    ---
    tags:
      - Index RAG
    summary: Search products using Index RAG with parallel indexing
    description: |
      Runs the Index RAG workflow which:
      1. Classifies user intent with Flash LLM
      2. Applies hierarchical metadata filter (Product → Vendor → Model)
      3. Runs parallel indexing (Database + LLM Web Search)
      4. Structures output with LLM
      
      This is a 4-node workflow with embedded metadata filtering.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: Product search query
              example: "I need a pressure transmitter from Yokogawa"
            product_type:
              type: string
              description: Optional explicit product type
              example: "pressure_transmitter"
            vendors:
              type: array
              items:
                type: string
              description: Optional vendor filter
              example: ["yokogawa", "emerson"]
            top_k:
              type: integer
              description: Max results per source (default 7)
              example: 7
            enable_web_search:
              type: boolean
              description: Enable LLM web search thread (default true)
              example: true
            session_id:
              type: string
              description: Optional session ID
    responses:
      200:
        description: Index RAG search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                output:
                  type: object
                  properties:
                    summary:
                      type: string
                    recommended_products:
                      type: array
                    total_found:
                      type: integer
                stats:
                  type: object
                  properties:
                    database_results:
                      type: integer
                    web_results:
                      type: integer
                    merged_results:
                      type: integer
                filters:
                  type: object
                metadata:
                  type: object
      400:
        description: Bad request - missing query
    """
    data = request.get_json()

    if not data or 'query' not in data:
        return api_response(False, error="query is required", status_code=400)

    query = data['query']
    product_type = data.get('product_type')
    vendors = data.get('vendors')
    top_k = data.get('top_k', 7)
    enable_web_search = data.get('enable_web_search', True)
    session_id = data.get('session_id') or get_session_id()

    logger.info("=" * 60)
    logger.info("[INDEX_RAG] Index RAG Search API Called")
    logger.info(f"[INDEX_RAG] Query: {query[:100]}...")
    logger.info(f"[INDEX_RAG] Product Type: {product_type}")
    logger.info(f"[INDEX_RAG] Vendors: {vendors}")
    logger.info(f"[INDEX_RAG] top_k: {top_k}, web_search: {enable_web_search}")

    try:
        from common.rag.index import run_index_rag_workflow

        result = run_index_rag_workflow(
            query=query,
            requirements={
                "product_type": product_type,
                "vendors": vendors
            } if product_type or vendors else None,
            session_id=session_id,
            top_k=top_k,
            enable_web_search=enable_web_search
        )

        if not result.get('success'):
            logger.error(f"[INDEX_RAG] Workflow failed: {result.get('error')}")
            return api_response(False, error=result.get('error', 'Index RAG failed'), status_code=500)

        stats = result.get('stats', {})
        logger.info(f"[INDEX_RAG] Success: {stats.get('filtered_results', 0)} results "
                   f"(JSON: {stats.get('json_count', 0)}, PDF: {stats.get('pdf_count', 0)}, Web: {stats.get('web_results', 0)})")
        logger.info(f"[INDEX_RAG] Processing time: {result.get('metadata', {}).get('processing_time_ms')}ms")
        
        if result.get('is_follow_up'):
            logger.info(f"[INDEX_RAG] Follow-up resolved: '{query}' -> '{result.get('resolved_query')}'")


        return api_response(True, data=result)

    except ImportError as ie:
        logger.error(f"[INDEX_RAG] Import error: {ie}")
        return api_response(False, error=f"Index RAG module not available: {ie}", status_code=500)
    except Exception as e:
        logger.error(f"[INDEX_RAG] Failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/compare', methods=['POST'])
@login_required
@handle_errors
def comparison_workflow():
    """
    Comparative Analysis Workflow
    ---
    tags:
      - Enhanced Workflows
    summary: Run vendor/product comparison workflow
    description: |
      Compares vendors and products based on text request.
      Returns ranked products with scoring breakdown.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: Comparison request
              example: "Compare Honeywell ST800 vs Emerson 3051S for pressure measurement"
            session_id:
              type: string
    responses:
      200:
        description: Ranked comparison results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                ranked_products:
                  type: array
                formatted_output:
                  type: string
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()

    from .search import run_product_search_with_comparison
    result = run_product_search_with_comparison(
        sample_input=message,
        product_type="instrument",
        session_id=session_id,
        auto_compare=True
    )

    return api_response(True, data=result)


@agentic_bp.route('/compare-from-spec', methods=['POST'])
@login_required
@handle_errors
def compare_from_spec():
    """
    Compare Vendors from SpecObject
    ---
    tags:
      - Enhanced Workflows
    summary: Run comparison from finalized specification
    description: |
      Triggered by [COMPARE VENDORS] button in UI.
      Takes a SpecObject from instrument detail capture and runs multi-level comparison.
      Supports vendor, series, and model level comparisons.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - spec_object
          properties:
            spec_object:
              type: object
              required:
                - product_type
              properties:
                product_type:
                  type: string
                  example: "pressure transmitter"
                category:
                  type: string
                  example: "Process Instrumentation"
                specifications:
                  type: object
                  example: {"range": "0-500 psi", "accuracy": "0.04%"}
                required_certifications:
                  type: array
                  items:
                    type: string
                  example: ["SIL2", "ATEX"]
                source_workflow:
                  type: string
                  example: "instrument_detail"
            comparison_type:
              type: string
              enum: [vendor, series, model, full]
              default: full
            session_id:
              type: string
    responses:
      200:
        description: Multi-level comparison results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                vendor_ranking:
                  type: array
                series_comparisons:
                  type: object
                top_recommendation:
                  type: object
                  properties:
                    vendor:
                      type: string
                    series:
                      type: string
                    model:
                      type: string
                    overall_score:
                      type: integer
                formatted_output:
                  type: string
    """
    data = request.get_json()

    if not data or 'spec_object' not in data:
        return api_response(False, error="spec_object is required", status_code=400)

    spec_object = data['spec_object']
    comparison_type = data.get('comparison_type', 'full')
    session_id = data.get('session_id') or get_session_id()
    user_id = data.get('user_id')

    from .search import trigger_comparison_from_product_search
    # Wrap spec_object as search_result for consolidated function
    result = trigger_comparison_from_product_search(
        search_result={"spec_object": spec_object, "ranked_results": spec_object.get("candidates", [])},
        session_id=session_id
    )

    return api_response(True, data=result)


@agentic_bp.route('/instrument-identifier', methods=['POST'])
@login_required
@handle_errors
def instrument_identifier():
    """
    Instrument Identifier Endpoint (List Generator)

    ✅ UPDATED: Expects UI-provided thread IDs

    Identifies instruments and accessories from requirements and returns a selection list.

    Request:
        {
            "message": "I need instruments for crude oil refinery...",
            "main_thread_id": "main_user123_US_WEST_...",      # ✅ From UI
            "workflow_thread_id": "instrument_identifier_...",  # ✅ From UI
            "session_id": "abc-123-def",
            "zone": "US-WEST"
        }

    Response includes:
        {
            "success": true,
            "data": {
                "response": "...",
                "response_data": {
                    "items": [...],
                    "thread_info": {
                        "main_thread_id": "main_user123_US_WEST_...",
                        "workflow_thread_id": "instrument_identifier_...",
                        "zone": "US-WEST"
                    }
                }
            }
        }
    """
    data = request.get_json()
    if not data:
         return api_response(False, error="No data provided", status_code=400)

    # Extract optional thread IDs from UI request (if provided)
    # If not provided, workflows will generate them automatically
    main_thread_id = data.get('main_thread_id')
    workflow_thread_id = data.get('workflow_thread_id')
    zone = data.get('zone')  # Optional - backend will detect if not provided
    session_id = data.get('session_id') or get_session_id()
    user_id = data.get('user_id') or session.get('user_id')

    # Frontend sends 'message', check for it, fallback to 'requirements'
    message = data.get('message') or data.get('requirements')
    if not message:
        return api_response(False, error="Message or requirements is required", status_code=400)

    # Run workflow
    result = run_workflow(
        user_input=message,
        session_id=session_id,
        workflow_type='instrument_identification'
    )

    # ✅ INCLUDE THREAD INFO IN RESPONSE
    if result.get('response_data') and result.get('thread_info'):
        result['response_data']['thread_info'] = result.get('thread_info')

    return api_response(True, data=result)


@agentic_bp.route('/solution', methods=['POST'])
@login_required
@handle_errors
def solution_workflow_endpoint():
    """
    Run Solution Deep Agent Workflow (Complex Engineering Challenges)
    ---
    tags:
      - Agentic Workflows
    summary: Run the solution deep agent for complex engineering requests
    """
    data = request.get_json()
    if not data:
        return api_response(False, error="No data provided", status_code=400)
    
    # Extract session ID
    session_id = data.get('session_id') or data.get('search_session_id') or get_session_id()
    user_id = session.get('user_id') or data.get('user_id', '')

    # Get the message/requirements
    message = data.get('user_input') or data.get('query') or data.get('message') or data.get('requirements')
    if not message:
        return api_response(False, error="Message or requirements is required", status_code=400)

    # Optional: Load conversation history if needed
    history = data.get('conversation_history', [])

    # Log solution workflow invocation
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[SOLUTION_API] Invoking solution workflow for session: {session_id}")
    logger.info(f"[SOLUTION_API] Input preview: {message[:100]}...")

    # Run workflow with all available context
    result = run_solution_workflow(
        user_input=message,
        session_id=session_id,
        user_id=str(user_id),
        conversation_history=history
    )

    # Clean up result for API response
    if isinstance(result, dict):
        logger.info(f"[SOLUTION_API] Solution workflow complete, items: {result.get('response_data', {}).get('total_items', 0)}")
        if "success" not in result:
             # Wrap raw state in success response
             return api_response(True, data={
                "success": True,
                "response": result.get("response", ""),
                "data": result,
                "response_data": result.get("response_data", {})
            })
        return api_response(True, data=result)

    return api_response(True, data={"success": True, "response": str(result)})



@agentic_bp.route('/potential-product-index', methods=['POST'])
@login_required
@handle_errors
def potential_product_index():
    """
    Potential Product Index Workflow
    ---
    tags:
      - Enhanced Workflows
    summary: Discover and index new product types
    description: |
      Triggered when no schema exists for a product type.
      Discovers vendors/models via LLM and generates schema.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - product_type
          properties:
            product_type:
              type: string
              description: Product type to index
              example: "differential pressure transmitter"
            session_id:
              type: string
    responses:
      200:
        description: Discovered vendors and generated schema
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                product_type:
                  type: string
                discovered_vendors:
                  type: array
                vendor_model_families:
                  type: object
                generated_schema:
                  type: object
                schema_saved:
                  type: boolean
    """
    data = request.get_json()

    if not data or 'product_type' not in data:
        return api_response(False, error="product_type is required", status_code=400)

    product_type = data['product_type']
    session_id = data.get('session_id') or get_session_id()

    result = run_potential_product_indexing_workflow(
        product_type=product_type,
        session_id=session_id
    )

    return api_response(True, data=result)



# ============================================================================
# TOOL ENDPOINTS
# ============================================================================

@agentic_bp.route('/tools/classify-intent', methods=['POST'])
@login_required
@handle_errors
def classify_intent_endpoint():
    """
    Test Intent Classification Tool
    ---
    tags:
      - LangChain Tools
    summary: Test classify_intent_tool directly
    description: |
      Directly invoke the LangChain classify_intent_tool.
      This tool classifies user input into intent categories for workflow routing.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - user_input
          properties:
            user_input:
              type: string
              description: User message to classify
              example: "I need pressure transmitters"
            current_step:
              type: string
              description: Current workflow step
              example: "start"
            context:
              type: string
              description: Conversation context
              example: "New conversation"
    responses:
      200:
        description: Intent classification result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                intent:
                  type: string
                  example: "requirements"
                confidence:
                  type: number
                  example: 0.95
                next_step:
                  type: string
    """
    from common.tools.intent_tools import classify_intent_tool

    data = request.get_json()
    if not data or 'user_input' not in data:
        return api_response(False, error="user_input is required", status_code=400)

    result = classify_intent_tool.invoke({
        "user_input": data['user_input'],
        "current_step": data.get('current_step'),
        "context": data.get('context')
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/validate-requirements', methods=['POST'])
@login_required
@handle_errors
def validate_requirements_endpoint():
    """
    Test Requirements Validation Tool
    ---
    tags:
      - LangChain Tools
    summary: Test validate_requirements_tool directly
    description: |
      Directly invoke the LangChain validate_requirements_tool.
      Validates user requirements against product schema.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - user_input
          properties:
            user_input:
              type: string
              description: Requirements text
              example: "Need 4-20mA pressure transmitter, 0-500 psi range"
            product_type:
              type: string
              description: Product type
              example: "pressure transmitter"
    responses:
      200:
        description: Validation result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                is_valid:
                  type: boolean
                missing_fields:
                  type: array
                  items:
                    type: string
    """
    from common.tools.schema_tools import validate_requirements_tool, load_schema_tool

    data = request.get_json()
    if not data or 'user_input' not in data:
        return api_response(False, error="user_input is required", status_code=400)

    product_type = data.get('product_type', '')

    # Load schema
    schema_result = load_schema_tool.invoke({"product_type": product_type})
    schema = schema_result.get("schema", {})

    # Validate
    result = validate_requirements_tool.invoke({
        "user_input": data['user_input'],
        "product_type": product_type,
        "schema": schema
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/search-vendors', methods=['POST'])
@login_required
@handle_errors
def search_vendors_endpoint():
    """
    Test Vendor Search Tool
    ---
    tags:
      - LangChain Tools
    summary: Test search_vendors_tool directly
    description: |
      Directly invoke the LangChain search_vendors_tool.
      Searches MongoDB for vendors offering specific products.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - product_type
          properties:
            product_type:
              type: string
              description: Product type to search for
              example: "pressure transmitter"
            requirements:
              type: object
              description: Optional requirements filter
    responses:
      200:
        description: Vendor search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                vendors:
                  type: array
                  items:
                    type: string
                vendor_count:
                  type: integer
    """
    # DEPRECATED: search_tools moved into search deep agent workflow
    # Use /api/agentic/product-search endpoint instead
    return api_response(False, error="This endpoint is deprecated. Use /api/agentic/product-search instead.", status_code=410)


@agentic_bp.route('/tools/analyze-match', methods=['POST'])
@login_required
@handle_errors
def analyze_match_endpoint():
    """
    Test Vendor Match Analysis Tool
    ---
    tags:
      - LangChain Tools
    summary: Test analyze_vendor_match_tool directly
    description: |
      Directly invoke the LangChain analyze_vendor_match_tool.
      Performs detailed parameter-by-parameter analysis of vendor products.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - requirements
          properties:
            vendor:
              type: string
              description: Vendor name
              example: "Honeywell"
            requirements:
              type: object
              description: User requirements
              example: {"outputSignal": "4-20mA", "range": "0-500 psi"}
            pdf_content:
              type: string
              description: Optional PDF datasheet content
            product_data:
              type: object
              description: Optional product JSON data
    responses:
      200:
        description: Analysis result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                match_score:
                  type: integer
                  example: 85
                matched_requirements:
                  type: object
    """
    from common.tools.analysis_tools import analyze_vendor_match_tool

    data = request.get_json()
    if not data or 'vendor' not in data or 'requirements' not in data:
        return api_response(False, error="vendor and requirements are required", status_code=400)

    result = analyze_vendor_match_tool.invoke({
        "vendor": data['vendor'],
        "requirements": data['requirements'],
        "pdf_content": data.get('pdf_content'),
        "product_data": data.get('product_data')
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/rank-products', methods=['POST'])
@login_required
@handle_errors
def rank_products_endpoint():
    """
    Test Product Ranking Tool
    ---
    tags:
      - LangChain Tools
    summary: Test rank_products_tool directly
    description: |
      Directly invoke the LangChain rank_products_tool.
      Ranks products based on analysis results using weighted criteria.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor_matches
          properties:
            vendor_matches:
              type: array
              description: Array of vendor analysis results
              items:
                type: object
            requirements:
              type: object
              description: Original requirements
    responses:
      200:
        description: Ranked products
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                ranked_products:
                  type: array
                  items:
                    type: object
                top_pick:
                  type: object
    """
    from tools.ranking_tools import rank_products_tool

    data = request.get_json()
    if not data or 'vendor_matches' not in data:
        return api_response(False, error="vendor_matches is required", status_code=400)

    result = rank_products_tool.invoke({
        "vendor_matches": data['vendor_matches'],
        "requirements": data.get('requirements', {})
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/search-images', methods=['POST'])
@login_required
@handle_errors
def search_images_endpoint():
    """
    Test Product Image Search Tool
    ---
    tags:
      - LangChain Tools
    summary: Test search_product_images_tool directly
    description: |
      Directly invoke the LangChain search_product_images_tool.
      Searches for product images using multi-tier fallback (Google CSE → Serper → SerpAPI).
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - product_name
            - product_type
          properties:
            vendor:
              type: string
              description: Vendor name
              example: "Honeywell"
            product_name:
              type: string
              description: Product model name
              example: "ST800"
            product_type:
              type: string
              description: Product type
              example: "pressure transmitter"
            model_family:
              type: string
              description: Optional model family
    responses:
      200:
        description: Image search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                images:
                  type: array
                  items:
                    type: object
    """
    # DEPRECATED: search_tools moved into search deep agent workflow
    return api_response(False, error="This endpoint is deprecated. Use /api/agentic/product-search instead.", status_code=410)



@agentic_bp.route('/tools/search-pdfs', methods=['POST'])
@login_required
@handle_errors
def search_pdfs_endpoint():
    """
    Test PDF Datasheet Search Tool
    ---
    tags:
      - LangChain Tools
    summary: Test search_pdf_datasheets_tool directly
    description: |
      Directly invoke the LangChain search_pdf_datasheets_tool.
      Searches for PDF datasheets using multi-tier fallback.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - product_type
          properties:
            vendor:
              type: string
              description: Vendor name
              example: "Emerson"
            product_type:
              type: string
              description: Product type
              example: "pressure transmitter"
            model_family:
              type: string
              description: Optional model family
              example: "3051S"
    responses:
      200:
        description: PDF search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                pdfs:
                  type: array
                  items:
                    type: object
                    properties:
                      url:
                        type: string
                      title:
                        type: string
    """
    # DEPRECATED: search_tools moved into search deep agent workflow
    return api_response(False, error="This endpoint is deprecated. Use /api/agentic/product-search instead.", status_code=410)



@agentic_bp.route('/tools/sales-interact', methods=['POST'])
@login_required
@handle_errors
def sales_interact_endpoint():
    """
    Test Sales Agent Interaction Tool
    ---
    tags:
      - LangChain Tools
    summary: Test sales_agent_interact_tool directly
    description: |
      Directly invoke the LangChain sales_agent_interact_tool.
      Handles conversational state and user interaction for the product search workflow.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - step
            - user_message
          properties:
            step:
              type: string
              description: Current workflow step
              example: "initialInput"
            user_message:
              type: string
              description: User message
              example: "I need 4-20mA pressure transmitters"
            product_type:
              type: string
              description: Detected product type
            data_context:
              type: object
              description: Context data for the step
    responses:
      200:
        description: Sales interaction result
    """
    # DEPRECATED: sales_agent_tool migrated to inline Flask API in main.py
    # Use /api/sales-agent endpoints instead
    return api_response(False, error="This endpoint is deprecated. Use /api/sales-agent endpoints instead.", status_code=410)


@agentic_bp.route('/tools/identify-instruments', methods=['POST'])
@login_required
@handle_errors
def identify_instruments_endpoint():
    """
    Test Instrument Identification Tool
    ---
    tags:
      - LangChain Tools
    summary: Test identify_instruments_tool directly
    description: |
      Directly invoke the LangChain identify_instruments_tool.
      Identifies instruments needed from process requirements.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - requirements
          properties:
            requirements:
              type: string
              description: Process requirements
              example: "I need to measure pressure and flow in my water treatment plant"
    responses:
      200:
        description: Instrument identification result
    """
    data = request.get_json()
    if not data or 'requirements' not in data:
        return api_response(False, error="requirements is required", status_code=400)

    agent = _get_id_agents().InstrumentIdentificationAgent()
    result = agent.identify(
        requirements=data['requirements']
    )

    return api_response(True, data=result)


# ============================================================================
# SESSION ENDPOINTS
# ============================================================================

@agentic_bp.route('/session', methods=['GET'])
@login_required
@handle_errors
def get_session():
    """Get current session info"""
    return api_response(True, data={
        "session_id": get_session_id(),
        "vendor_filter": session.get('csv_vendor_filter')
    })


@agentic_bp.route('/session', methods=['DELETE'])
@login_required
@handle_errors
def clear_session():
    """Clear current session"""
    session.pop('agentic_session_id', None)
    session.pop('csv_vendor_filter', None)
    return api_response(True, data={"message": "Session cleared"})


@agentic_bp.route('/session/vendor-filter', methods=['POST'])
@login_required
@handle_errors
def set_vendor_filter():
    """
    Set vendor filter from CSV upload

    Request Body:
    {
        "vendor_names": ["Vendor1", "Vendor2"]
    }
    """
    data = request.get_json()
    if not data or 'vendor_names' not in data:
        return api_response(False, error="vendor_names is required", status_code=400)

    session['csv_vendor_filter'] = {
        'vendor_names': data['vendor_names']
    }

    return api_response(True, data={
        "message": "Vendor filter set",
        "vendor_count": len(data['vendor_names'])
    })




# ============================================================================
# VALIDATION TOOL WRAPPER
# ============================================================================

@agentic_bp.route('/validate', methods=['POST'])
@handle_errors
@login_required
def agentic_validate():
    """
    Validation Tool Wrapper API

    Standalone validation endpoint for agentic workflows.
    Detects product type, loads/generates schema, and validates requirements.

    Request Body:
        {
            "user_input": str,      # Required: User's requirements description
            "message": str,         # Alternative to user_input
            "product_type": str,    # Optional: Expected product type
            "session_id": str,      # Optional: Session tracking ID
            "enable_ppi": bool      # Optional: Enable PPI workflow (default: True)
        }

    Returns:
        {
            "success": bool,
            "data": {
                "product_type": str,
                "detected_schema": dict,
                "provided_requirements": dict,
                "ppi_workflow_used": bool,
                "is_valid": bool,
                "missing_fields": list
            }
        }
    """
    try:
        from search import run_validation_only

        data = request.get_json()

        # Accept both 'user_input' and 'message'
        user_input = data.get('user_input') or data.get('message')
        if not user_input:
            return api_response(False, error="user_input or message is required", status_code=400)

        expected_product_type = data.get('product_type')
        session_id = data.get('session_id', 'default')
        enable_ppi = data.get('enable_ppi', True)

        logger.info(f"[VALIDATION_TOOL] Starting validation for session: {session_id}")
        logger.info(f"[VALIDATION_TOOL] User input: {user_input[:100]}...")

        # Use the new search deep agent functional API for validation
        validation_result = run_validation_only(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id,
            enable_ppi=enable_ppi
        )

        product_type = validation_result.get("product_type", "")
        schema = validation_result.get("detected_schema", {})
        requirements = validation_result.get("provided_requirements", {})
        missing_fields = validation_result.get("missing_fields", [])
        ppi_used = validation_result.get("ppi_workflow_used", False)
        is_valid = validation_result.get("is_valid", False)

        if missing_fields:
            logger.info(f"[VALIDATION_TOOL] Missing Fields: {missing_fields}")
        else:
            logger.info(f"[VALIDATION_TOOL] All mandatory fields provided")

        # =====================================================================
        # FIELD DESCRIPTIONS FOR ON-HOVER TOOLTIPS
        # Extracts descriptions from template specifications for frontend display
        # =====================================================================
        field_descriptions = {}
        try:
            # Try to get template specifications (60+ specs per product type)
            try:
                from common.agentic.deep_agent.specifications.templates.templates import (
                    get_all_specs_for_product_type
                )
                template_specs = get_all_specs_for_product_type(product_type)
                if template_specs:
                    for spec_key, spec_def in template_specs.items():
                        field_descriptions[spec_key] = spec_def.description
                    logger.info(f"[VALIDATION_TOOL] Loaded {len(field_descriptions)} field descriptions from templates")
            except ImportError:
                pass

            # Generate descriptions for schema fields not in templates
            # Extract all leaf field keys from schema
            def extract_keys(obj, prefix=""):
                keys = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, dict) and "value" not in value:
                            keys.extend(extract_keys(value, full_key))
                        else:
                            keys.append(full_key)
                return keys
            
            def prettify_field_name(field_name: str) -> str:
                """Convert field_name or fieldName to human-readable description"""
                import re
                # Handle camelCase: split on capital letters
                words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field_name)
                # Handle snake_case: replace underscores with spaces
                words = words.replace('_', ' ').replace('-', ' ')
                # Capitalize first letter of each word
                words = ' '.join(word.capitalize() for word in words.split())
                return f"Specification for {words}"
            
            all_keys = extract_keys(schema.get("mandatoryRequirements", {}))
            all_keys.extend(extract_keys(schema.get("optionalRequirements", {})))
            
            # For fields not in templates, create a description from the field name
            for field_key in all_keys:
                field_name = field_key.split(".")[-1]
                # Check if we already have a template description (match by field name)
                if field_name not in field_descriptions and field_key not in field_descriptions:
                    # Generate a readable description from the field name
                    field_descriptions[field_key] = prettify_field_name(field_name)
                elif field_name in field_descriptions:
                    # Copy template description to full key path
                    field_descriptions[field_key] = field_descriptions[field_name]
                
        except Exception as desc_err:
            logger.warning(f"[VALIDATION_TOOL] Field description extraction failed: {desc_err}")

        result = {
            "productType": product_type,
            "detectedSchema": schema,
            "providedRequirements": requirements,
            "ppiWorkflowUsed": ppi_used,
            "isValid": is_valid,
            "missingFields": missing_fields,
            "sessionId": session_id,
            "fieldDescriptions": field_descriptions,
            "fieldDescriptionsCount": len(field_descriptions)
        }

        logger.info(f"[VALIDATION_TOOL] Validation complete with {len(field_descriptions)} field descriptions")

        return api_response(True, data=result)

    except Exception as e:
        logger.error(f"[VALIDATION_TOOL] Validation failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# ADVANCED PARAMETERS TOOL WRAPPER
# ============================================================================

@agentic_bp.route('/advanced-parameters', methods=['POST'])
@handle_errors
@login_required
def agentic_advanced_parameters():
    """
    Advanced Parameters Discovery Tool Wrapper API

    Standalone advanced parameters discovery endpoint for agentic workflows.
    Discovers latest advanced specifications with series numbers from top vendors.

    Request Body:
        {
            "product_type": str,    # Required: Product type to discover parameters for
            "session_id": str       # Optional: Session tracking ID
        }

    Returns:
        {
            "success": bool,
            "data": {
                "product_type": str,
                "unique_specifications": [
                    {
                        "key": str,
                        "name": str
                    }
                ],
                "total_unique_specifications": int,
                "existing_specifications_filtered": int,
                "vendor_specifications": list
            }
        }
    """
    try:
        # Use AdvancedSpecificationAgent from upstream search module
        from search.advanced_specification_agent import AdvancedSpecificationAgent
        discover_advanced_specs = lambda product_type, session_id="default": AdvancedSpecificationAgent().discover(product_type=product_type, session_id=session_id)

        data = request.get_json()

        # Validate input
        product_type = data.get('product_type', '').strip()
        if not product_type:
            return api_response(False, error="product_type is required", status_code=400)

        session_id = data.get('session_id', 'default')

        logger.info(f"[ADVANCED_PARAMS_TOOL] Starting discovery for: {product_type}")
        logger.info(f"[ADVANCED_PARAMS_TOOL] Session: {session_id}")

        # Run discovery using simplified function
        result = discover_advanced_specs(
            product_type=product_type,
            session_id=session_id
        )

        # Log results
        unique_count = len(result.get('unique_specifications', []))
        filtered_count = result.get('existing_specifications_filtered', 0)

        logger.info(
            f"[ADVANCED_PARAMS_TOOL] Discovery complete: "
            f"{unique_count} new specifications, "
            f"{filtered_count} existing filtered"
        )

        return api_response(True, data=result)

    except Exception as e:
        logger.error(f"[ADVANCED_PARAMS_TOOL] Discovery failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# PRODUCT SEARCH WORKFLOW
# ============================================================================


@agentic_bp.route('/product-search', methods=['POST'])
@handle_errors
@login_required
def product_search():
    """
    Product Search Deep Agentic Workflow

    ✅ UPDATED: Expects UI-provided thread IDs

    This endpoint implements a STATEFUL workflow with Deep Agent capabilities:

    Flow (handled by DeepAgenticWorkflowOrchestrator):
    1. VALIDATION - Product type detection & schema generation (with failure memory)
    2. AWAIT_MISSING_FIELDS - User decision on missing fields
    3. ADVANCED_DISCOVERY - Discover specifications from vendors
    4. VENDOR_ANALYSIS - Parallel vendor matching
    5. RANKING - Final product ranking

    Features:
    - Session state management with persistence
    - User decision parsing and handling
    - Failure memory and learning from past errors
    - Adaptive prompt optimization
    - Automatic phase progression
    - UI-managed thread ID system

    Request Body:
        {
            "user_input": str,          # Required on first call
            "message": str,             # Alternative to user_input
            "main_thread_id": str,      # ✅ Required: Main thread ID from UI
            "workflow_thread_id": str,  # ✅ Required: Product search sub-thread ID from UI
            "zone": str,                # ✅ Optional: Geographic zone (US-WEST, etc.)
            "thread_id": str,           # DEPRECATED: Use workflow_thread_id instead
            "user_decision": str,       # User's choice: "add_fields", "continue", "yes", "no"
            "user_provided_fields": {},  # Fields provided by user
            "product_type": str,        # Optional: Product type hint
            "session_id": str,          # Session tracking ID
            "item_number": int,         # Optional: Item number from parent workflow
            "item_name": str,           # Optional: Item name
            "item_thread_id": str,      # Optional: Item thread ID
            "parent_workflow_thread_id": str  # Optional: Parent workflow thread ID
        }

    Returns:
        {
            "success": bool,
            "data": {
                "thread_id": str,               # For resuming conversation
                "session_id": str,              # Session identifier
                "awaiting_user_input": bool,    # True if workflow is paused
                "current_phase": str,           # Current workflow phase
                "sales_agent_response": str,    # Message to display to user
                "schema": dict,                 # Schema for left sidebar display
                "missing_fields": list,         # Missing mandatory fields
                "validation_result": dict,
                "available_advanced_params": list,
                "completed": bool,              # True if workflow is complete
                "thread_info": {                # ✅ Thread context
                    "main_thread_id": str,
                    "workflow_thread_id": str,
                    "zone": str
                }
            }
        }
    """
    try:
        # Tool-based search workflow
        from search import run_product_search_workflow

        data = request.get_json()
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        # ═══════════════════════════════════════════════════════════════════
        # CREATE EXECUTION CONTEXT for proper session/workflow isolation
        # ═══════════════════════════════════════════════════════════════════
        ctx = create_execution_context_from_request(data, workflow_type="product_search")

        # Extract legacy IDs for backward compatibility in response
        main_thread_id = ctx.session_id
        workflow_thread_id = ctx.workflow_id
        zone = ctx.zone
        session_id = data.get('session_id') or data.get('search_session_id') or ctx.workflow_id

        # Extract search parameters
        user_input = data.get('user_input') or data.get('message', '')
        product_type_hint = data.get('product_type', '')
        provided_requirements = data.get('user_provided_fields') or data.get('provided_requirements')
        user_decision = data.get('user_decision')
        current_phase = data.get('current_phase')
        auto_mode = data.get('auto_mode', False)
        source_workflow = data.get('source_workflow', 'direct')

        # Optional item context (from solution workflow BOM enrichment)
        item_thread_id = data.get('item_thread_id')
        parent_workflow_thread_id = data.get('parent_workflow_thread_id')

        logger.info(
            f"\n{'='*60}\n"
            f"[PRODUCT_SEARCH] Tool-Based Workflow Request\n"
            f"   Context: {ctx.to_log_context()}\n"
            f"   Session: {session_id}\n"
            f"   Input: {user_input[:50] if user_input else 'None'}...\n"
            f"{'='*60}"
        )

        # Run tool-based search workflow with ExecutionContext
        with execution_context(ctx):
            result = run_product_search_workflow(
                user_input=user_input,
                ctx=ctx,  # Pass ExecutionContext (preferred)
                session_id=session_id,  # Legacy parameter for backward compatibility
                expected_product_type=product_type_hint,
                user_provided_fields=provided_requirements,
                enable_ppi=True,
                auto_mode=auto_mode,
                user_decision=user_decision,
                current_phase=current_phase,
                source_workflow=source_workflow
            )

        # Map tool-based workflow result to expected API response shape.
        # run_product_search_workflow returns {"success", "response", "response_data", "error"}.
        response_data = result.get('response_data', {})

        # Extract data from tool-based workflow
        ranked_products_list = response_data.get('ranked_products', [])
        vendor_matches_dict = response_data.get('vendor_matches', {})
        product_type = response_data.get('product_type', product_type_hint)
        schema = response_data.get('schema', {})
        missing_fields = response_data.get('missing_fields', [])
        advanced_params = response_data.get('advanced_parameters', [])
        awaiting = response_data.get('awaiting_user_input', False)
        current_phase = response_data.get('current_phase', 'completed')

        # Convert vendor_matches to list format if it's a dict, otherwise use directly
        vendor_matches = []
        if isinstance(vendor_matches_dict, dict):
            for vendor_name, vendor_data in vendor_matches_dict.items():
                if isinstance(vendor_data, dict):
                    vendor_matches.append({
                        'vendor': vendor_name,
                        **vendor_data
                    })
                elif isinstance(vendor_data, list):
                    vendor_matches.extend(vendor_data)
        elif isinstance(vendor_matches_dict, list):
            vendor_matches = vendor_matches_dict

        mapped = {
            # Core identification
            'session_id': session_id,
            'thread_id': workflow_thread_id,
            'current_phase': current_phase,
            'product_type': product_type,

            # Workflow status
            'completed': response_data.get('completed', not awaiting),
            'awaiting_user_input': awaiting,

            # Content
            'sales_agent_response': result.get('response', ''),
            # FIX: Always return schema when available (needed for sidebar display)
            'schema': schema if schema else {},
            'missing_fields': missing_fields,
            'available_advanced_params': advanced_params,
            'provided_requirements': response_data.get('provided_requirements', {}),

            # Vendor / ranking results
            'vendorAnalysis': {
                'vendorMatches': vendor_matches,
                'totalMatches': len(vendor_matches),
            },
            'overallRanking': {
                'overall_ranking': ranked_products_list
            } if ranked_products_list else {},
            # Expose ranked products list for frontend consumption.
            # Inject productType (required by RankedProduct TypeScript interface).
            'ranked_products': [
                dict(p, productType=p.get('productType', product_type))
                for p in ranked_products_list
            ],
            'topRecommendation': ranked_products_list[0] if ranked_products_list else None,

            # Thread context (with ExecutionContext info)
            'thread_info': {
                'main_thread_id': main_thread_id,
                'workflow_thread_id': workflow_thread_id,
                'zone': zone,
                'item_thread_id': item_thread_id,
                'parent_workflow_thread_id': parent_workflow_thread_id,
            },

            # ExecutionContext info for tracing/debugging
            'context': {
                'session_id': ctx.session_id,
                'workflow_id': ctx.workflow_id,
                'instance_id': ctx.instance_id,
                'correlation_id': ctx.correlation_id,
            },
        }

        # Surface any error from the workflow
        if result.get('error'):
            mapped['error'] = result['error']
            mapped['success'] = False

        return api_response(result.get('success', True), data=mapped)

    except ImportError as e:
        logger.error(f"[PRODUCT_SEARCH] Import error - Search workflow not available: {e}")
        return api_response(
            False,
            error="Search workflow module not available. Please check installation.",
            status_code=500
        )

    except Exception as e:
        logger.error(f"[PRODUCT_SEARCH] Workflow failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)

# DEPRECATED: Old manual phase handling code removed
# Now using DeepAgenticWorkflowOrchestrator for all workflow management


@agentic_bp.route('/run-analysis', methods=['POST'])
@handle_errors
@login_required
def run_product_analysis():
    """
    Run Final Product Analysis (Steps 4-5: Vendor Analysis + Ranking)

    This endpoint executes the actual product search after requirements are collected.
    It calls workflow.run_analysis_only() to:
    - Step 4: Run vendor analysis (parallel PDF/JSON matching)
    - Step 5: Rank products with scores

    Request Body:
        {
            "structured_requirements": {
                "productType": str,
                "mandatoryRequirements": dict,
                "optionalRequirements": dict,
                "selectedAdvancedParams": dict  # Optional
            },
            "product_type": str,
            "schema": dict,  # Optional
            "session_id": str  # Optional
        }

    Returns:
        {
            "success": bool,
            "data": {
                "vendorAnalysis": {
                    "vendorMatches": [...],
                    "totalMatches": int
                },
                "overallRanking": {
                    "rankedProducts": [...]
                },
                "topRecommendation": {...},
                "analysisResult": {...}  # Complete result for RightPanel
            }
        }
    """
    try:
        from search import run_analysis_only

        data = request.get_json()
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        # Extract parameters
        structured_requirements = data.get('structured_requirements')
        product_type = data.get('product_type')
        schema = data.get('schema')
        session_id = data.get('session_id') or data.get('search_session_id') or f"analysis_{uuid.uuid4().hex[:8]}"

        # Validate required fields
        if not structured_requirements:
            return api_response(False, error="structured_requirements is required", status_code=400)

        if not product_type:
            return api_response(False, error="product_type is required", status_code=400)

        logger.info(f"[RUN_ANALYSIS] Starting final analysis")
        logger.info(f"[RUN_ANALYSIS] Product Type: {product_type}")
        logger.info(f"[RUN_ANALYSIS] Session: {session_id}")

        # Run Search Deep Agent functional API for analysis step
        analysis_result = run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id
        )

        if not analysis_result.get('success'):
            logger.error(f"[RUN_ANALYSIS] Analysis failed: {analysis_result.get('error')}")
            return api_response(False, error=analysis_result.get('error', 'Analysis failed'), status_code=500)

        logger.info(f"[RUN_ANALYSIS] Analysis complete")
        logger.info(f"[RUN_ANALYSIS] Products ranked: {analysis_result.get('totalRanked', 0)}")
        logger.info(f"[RUN_ANALYSIS] Exact matches: {analysis_result.get('exactMatchCount', 0)}")
        logger.info(f"[RUN_ANALYSIS] Approximate matches: {analysis_result.get('approximateMatchCount', 0)}")

        # =====================================================================
        # PRIMARY IMAGE FETCHING - Fetch generic images for ranked products
        # This is the primary layer; frontend will handle fallback for missing
        # =====================================================================
        try:
            from common.services.azure.image_utils import fetch_generic_product_image
            
            ranked_products = analysis_result.get('overall_ranking', [])
            images_fetched = 0
            
            # Fetch images for top 20 products to avoid rate limiting
            for product in ranked_products[:20]:
                # Skip if product already has an image
                if product.get('top_image') or product.get('topImage'):
                    continue
                
                # Use product_type for generic image lookup
                pt = product.get('productType') or product.get('product_type') or product_type
                if not pt:
                    continue
                
                try:
                    image_result = fetch_generic_product_image(pt)
                    if image_result and image_result.get('url'):
                        product['top_image'] = {
                            'url': image_result['url'],
                            'source': image_result.get('source', 'generic_cache'),
                            'product_type': pt
                        }
                        images_fetched += 1
                        logger.debug(f"[RUN_ANALYSIS] Image fetched for {product.get('productName', 'Unknown')}")
                except Exception as img_err:
                    logger.warning(f"[RUN_ANALYSIS] Image fetch failed for {pt}: {img_err}")
            
            logger.info(f"[RUN_ANALYSIS] Images fetched: {images_fetched}/{min(20, len(ranked_products))}")
            
        except ImportError as e:
            logger.warning(f"[RUN_ANALYSIS] Could not import generic_image_utils: {e}")
        except Exception as e:
            logger.warning(f"[RUN_ANALYSIS] Image fetching failed: {e}")

        # Flatten vendor_matches (dict keyed by vendor) into a list for the frontend
        raw_vendor_matches = analysis_result.get("vendor_matches", {})
        if isinstance(raw_vendor_matches, dict):
            vendor_matches_list = []
            for match_list in raw_vendor_matches.values():
                if isinstance(match_list, list):
                    vendor_matches_list.extend(match_list)
                elif isinstance(match_list, dict):
                    vendor_matches_list.append(match_list)
        elif isinstance(raw_vendor_matches, list):
            vendor_matches_list = raw_vendor_matches
        else:
            vendor_matches_list = []

        top_recommendation = analysis_result.get("top_product")
        if not top_recommendation and analysis_result.get("overall_ranking"):
            top_recommendation = analysis_result["overall_ranking"][0]

        # Map to expected response format (camelCase for frontend)
        final_result = {
            "vendorAnalysis": {
                "vendorMatches": vendor_matches_list,
                "totalMatches": len(vendor_matches_list)
            },
            "overallRanking": {
                "rankedProducts": analysis_result.get("overall_ranking", [])
            },
            "topRecommendation": top_recommendation,
            "analysisResult": analysis_result  # Pass full result for RightPanel
        }

        return api_response(True, data=final_result)

    except Exception as e:
        logger.error(f"[RUN_ANALYSIS] Failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# SALES AGENT TOOL WRAPPER API
# ============================================================================


@agentic_bp.route('/sales-agent', methods=['POST'])
@handle_errors
@login_required
def agentic_sales_agent():
    """
    Sales Agent Tool Wrapper API

    Provides conversational AI interface for product requirements collection.
    Handles step-by-step workflow with LLM-powered responses.

    Request Body:
        {
            "step": "initialInput",  # Current workflow step
            "user_message": "I need a pressure transmitter",
            "data_context": {  # Context data for current step
                "productType": "Pressure Transmitter",
                "availableParameters": [...],
                "selectedParameters": {...}
            },
            "session_id": "session_123",  # Session identifier
            "intent": "workflow",  # "workflow" or "knowledgeQuestion"
            "save_immediately": false  # Skip greeting if true
        }

    Response:
        {
            "success": true,
            "data": {
                "content": "AI-generated response message",
                "nextStep": "awaitAdditionalAndLatestSpecs",
                "maintainWorkflow": true,
                "dataContext": {...},  # Updated context
                "discoveredParameters": [...]  # Optional
            }
        }

    Workflow Steps:
        - greeting: Welcome message
        - initialInput: Initial product requirements
        - awaitMissingInfo: Collect missing mandatory fields
        - awaitAdditionalAndLatestSpecs: Additional specifications
        - awaitAdvancedSpecs: Advanced parameter specifications
        - showSummary: Display requirements summary
        - finalAnalysis: Complete analysis
    """
    try:
        from search.sales_agent_tool import SalesAgentTool

        data = request.get_json()
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        step = data.get('step', 'initialInput')
        user_message = data.get('user_message', '')
        data_context = data.get('data_context', {})
        session_id = data.get('session_id') or get_session_id()
        intent = data.get('intent', 'workflow')
        save_immediately = data.get('save_immediately', False)

        sales_agent = SalesAgentTool()
        result = sales_agent.process_step(
            step=step,
            user_message=user_message,
            data_context=data_context,
            session_id=session_id,
            intent=intent,
            save_immediately=save_immediately
        )

        return api_response(True, data=result)

    except Exception as e:
        logger.error(f"[SALES_AGENT] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)



# ============================================================================
# INDEXING AGENT WRAPPER API
# ============================================================================


@agentic_bp.route('/run-indexing', methods=['POST'])
@handle_errors
@login_required
def run_indexing_agent():
    """
    Indexing Agent Workflow Wrapper API
    ---
    tags:
      - Indexing
    summary: Run Deep Indexing Agent for Schema Generation
    description: |
      Manually triggers the Deep Agent Indexing workflow (formerly PPI) to:
      1. Discover vendors
      2. Download PDFs
      3. Extract specifications
      4. Generate product schema
    
    Request:
        {
            "product_type": "pressure transmitter"
        }
    
    Response:
        {
            "success": true,
            "data": {
                "schema": {...},
                "quality_score": 0.95,
                "execution_time_seconds": 45.2
            }
        }
    """
    try:
        from Indexing import run_potential_product_indexing_workflow

        data = request.get_json()
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        product_type = data.get('product_type')
        if not product_type:
            return api_response(False, error="product_type is required", status_code=400)
        
        logger.info(f"[INDEXING] Starting manual indexing for: {product_type}")

        result = run_potential_product_indexing_workflow(product_type=product_type)
        
        if result.get('success'):
            return api_response(True, data=result)
        else:
            return api_response(False, error=result.get('error', 'Indexing failed'), status_code=500)

    except ImportError as e:
        logger.error(f"[INDEXING] Import error: {e}")
        return api_response(False, error="Indexing module not available", status_code=500)
    except Exception as e:
        logger.error(f"[INDEXING] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# STANDARDS RAG API
# ============================================================================


@agentic_bp.route('/standards-query', methods=['POST'])
@login_required
@handle_errors
def standards_query():
    """
    Standards RAG Query API - Query engineering standards documentation.

    Queries the Standards RAG knowledge base for information about:
    - Applicable standards (ISO, IEC, API, ANSI, ISA)
    - Required certifications (SIL, ATEX, CE, etc.)
    - Safety requirements
    - Calibration standards
    - Environmental requirements
    - Communication protocols

    Request:
        {
            "question": "What are the SIL requirements for pressure transmitters?",
            "top_k": 5,
            "session_id": "optional_session_id"
        }

    Response:
        {
            "success": true,
            "data": {
                "answer": "According to IEC 61508...",
                "citations": [...],
                "confidence": 0.85,
                "sources_used": ["standards_doc.docx", ...],
                "metadata": {
                    "processing_time_ms": 1234,
                    "documents_retrieved": 5
                }
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[STANDARDS-RAG] Standards Query API Called")

    question = data.get('question')
    if not question:
        logger.error("[STANDARDS-RAG] No question provided")
        return api_response(False, error="question is required", status_code=400)

    top_k = data.get('top_k', 5)
    session_id = data.get('session_id')

    logger.info(f"[STANDARDS-RAG] Question: {question[:100]}...")
    logger.info(f"[STANDARDS-RAG] top_k: {top_k}")

    try:
        from common.rag.standards import run_standards_rag_workflow

        # Run the Standards RAG workflow
        result = run_standards_rag_workflow(
            question=question,
            session_id=session_id,
            top_k=top_k
        )

        if result.get('status') == 'success':
            logger.info("[STANDARDS-RAG] Query successful")
            return api_response(True, data={
                "answer": result['final_response'].get('answer', ''),
                "citations": result['final_response'].get('citations', []),
                "confidence": result['final_response'].get('confidence', 0.0),
                "sources_used": result['final_response'].get('sources_used', []),
                "metadata": result['final_response'].get('metadata', {}),
                "sessionId": session_id
            })
        else:
            logger.warning(f"[STANDARDS-RAG] Query failed: {result.get('error')}")
            return api_response(False, error=result.get('error', 'Standards query failed'))

    except Exception as e:
        logger.error(f"[STANDARDS-RAG] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/standards-enrich', methods=['POST'])
@login_required
@handle_errors
def standards_enrich():
    """
    Standards Enrichment API - Enrich a product schema with standards information.

    Takes a product type and optionally a schema, and returns the schema
    enriched with applicable standards from the Standards RAG knowledge base.

    Request:
        {
            "product_type": "pressure transmitter",
            "schema": { ... }  // Optional - will use default if not provided
        }

    Response:
        {
            "success": true,
            "data": {
                "product_type": "pressure transmitter",
                "applicable_standards": ["IEC 61508", "ISO 10849", ...],
                "certifications": ["SIL2", "ATEX Zone 1", ...],
                "safety_requirements": { ... },
                "calibration_standards": { ... },
                "environmental_requirements": { ... },
                "communication_protocols": ["HART", "4-20MA", ...],
                "confidence": 0.85,
                "sources": ["standards_doc.docx", ...]
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[STANDARDS-ENRICH] Standards Enrichment API Called")

    product_type = data.get('product_type')
    if not product_type:
        logger.error("[STANDARDS-ENRICH] No product_type provided")
        return api_response(False, error="product_type is required", status_code=400)

    schema = data.get('schema', {})

    logger.info(f"[STANDARDS-ENRICH] Product type: {product_type}")

    try:
        from common.tools.standards_enrichment_tool import get_applicable_standards, enrich_schema_with_standards

        if schema:
            # Enrich provided schema
            logger.info("[STANDARDS-ENRICH] Enriching provided schema")
            enriched_schema = enrich_schema_with_standards(product_type, schema)
            return api_response(True, data={
                "product_type": product_type,
                "enriched_schema": enriched_schema,
                "standards_added": 'standards' in enriched_schema
            })
        else:
            # Just get applicable standards
            logger.info("[STANDARDS-ENRICH] Getting applicable standards")
            standards_info = get_applicable_standards(product_type)
            return api_response(True, data={
                "product_type": product_type,
                "applicable_standards": standards_info.get('applicable_standards', []),
                "certifications": standards_info.get('certifications', []),
                "safety_requirements": standards_info.get('safety_requirements', {}),
                "calibration_standards": standards_info.get('calibration_standards', {}),
                "environmental_requirements": standards_info.get('environmental_requirements', {}),
                "communication_protocols": standards_info.get('communication_protocols', []),
                "confidence": standards_info.get('confidence', 0.0),
                "sources": standards_info.get('sources', [])
            })

    except Exception as e:
        logger.error(f"[STANDARDS-ENRICH] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/standards-validate', methods=['POST'])
@login_required
@handle_errors
def standards_validate():
    """
    Standards Validation API - Validate requirements against applicable standards.

    Checks if user requirements comply with engineering standards and provides
    recommendations for missing or incomplete specifications.

    Request:
        {
            "product_type": "pressure transmitter",
            "requirements": {
                "outputSignal": "4-20mA HART",
                "pressureRange": "0-100 bar"
            }
        }

    Response:
        {
            "success": true,
            "data": {
                "is_compliant": false,
                "compliance_issues": [...],
                "recommendations": [...],
                "applicable_standards": [...],
                "required_certifications": [...]
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[STANDARDS-VALIDATE] Standards Validation API Called")

    product_type = data.get('product_type')
    requirements = data.get('requirements')

    if not product_type:
        return api_response(False, error="product_type is required", status_code=400)
    if not requirements:
        return api_response(False, error="requirements is required", status_code=400)

    logger.info(f"[STANDARDS-VALIDATE] Product type: {product_type}")

    try:
        from common.tools.standards_enrichment_tool import validate_requirements_against_standards

        validation_result = validate_requirements_against_standards(product_type, requirements)

        return api_response(True, data={
            "product_type": product_type,
            "is_compliant": validation_result.get('is_compliant', False),
            "compliance_issues": validation_result.get('compliance_issues', []),
            "recommendations": validation_result.get('recommendations', []),
            "applicable_standards": validation_result.get('applicable_standards', []),
            "required_certifications": validation_result.get('required_certifications', []),
            "confidence": validation_result.get('confidence', 0.0)
        })

    except Exception as e:
        logger.error(f"[STANDARDS-VALIDATE] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)



# ============================================================================
# DEEP AGENT TEST ENDPOINT
# ============================================================================


@agentic_bp.route('/test-deep-agent', methods=['POST'])
@handle_errors
def test_deep_agent_schema_population():
    """
    Test Deep Agent Schema Population
    
    This endpoint tests the Deep Agent integration by running the 
    instrument identifier workflow with verbose logging.
    
    Request:
        {
            "user_input": "I need a pressure transmitter for crude oil storage",
            "run_full_workflow": false  // if true, runs instrument_identifier_workflow
        }
    
    Response:
        {
            "success": true,
            "data": {
                "items_enriched": 2,
                "schemas_populated": 2,
                "total_fields_populated": 15,
                "items": [...]
            }
        }
    """
    data = request.get_json() or {}
    
    print("\n" + "=" * 80)
    print("[TEST] DEEP AGENT SCHEMA POPULATION TEST")
    print("=" * 80)
    
    user_input = data.get('user_input', 'I need a pressure transmitter for crude oil storage with SIL2 requirements')
    run_full_workflow = data.get('run_full_workflow', False)
    
    print(f"[TEST] User input: {user_input[:80]}...")
    print(f"[TEST] Run full workflow: {run_full_workflow}")
    
    try:
        if run_full_workflow:
            # Run the full instrument identifier workflow using base run_workflow
            print("\n[TEST] Running FULL Instrument Identifier Workflow...")
            result = run_workflow(
                user_input=user_input,
                workflow_type='instrument_identification',
                session_id=f"test_{uuid.uuid4().hex[:8]}"
            )
            
            response_data = result.get('response_data', {})
            items = response_data.get('items', [])
            
            # Analyze enrichment results
            schemas_populated = sum(1 for item in items if item.get('schema_populated', False))
            total_fields = 0
            for item in items:
                schema = item.get('schema', {})
                pop_info = schema.get('_deep_agent_population', {})
                total_fields += pop_info.get('fields_populated', 0)
            
            return api_response(True, data={
                "test_type": "full_workflow",
                "workflow": "instrument_identifier",
                "items_total": len(items),
                "items_enriched": sum(1 for item in items if item.get('enrichment_status') == 'success'),
                "schemas_populated": schemas_populated,
                "total_fields_populated": total_fields,
                "response": result.get('response', ''),
                "items": items
            })
        
        else:
            # Run only the Deep Agent integration directly
            from common.agentic.deep_agent_integration import integrate_deep_agent_specifications
            
            # Create test items
            test_items = [
                {
                    "number": 1,
                    "type": "instrument",
                    "name": "Pressure Transmitter",
                    "category": "Pressure Measurement",
                    "quantity": 1,
                    "sample_input": user_input
                }
            ]
            
            # Check if user mentions temperature
            if 'temperature' in user_input.lower():
                test_items.append({
                    "number": 2,
                    "type": "instrument",
                    "name": "Temperature Sensor",
                    "category": "Temperature Measurement",
                    "quantity": 1,
                    "sample_input": user_input
                })
            
            print(f"\n[TEST] Test items: {len(test_items)}")
            for item in test_items:
                print(f"  - {item['name']} ({item['type']})")
            
            print("\n[TEST] Running Deep Agent integration (specs-only mode)...")
            
            # Run Deep Agent with schema population DISABLED
            # This matches the production workflow behavior - just extract specs
            enriched = integrate_deep_agent_specifications(
                all_items=test_items,
                user_input=user_input,
                solution_context=None,
                domain=None,
                enable_schema_population=False  # Match production behavior
            )
            
            # Analyze results
            results_summary = []
            total_fields = 0
            schemas_populated = 0
            
            print("\n[TEST] RESULTS:")
            print("-" * 60)
            
            for item in enriched:
                status = item.get('enrichment_status', 'unknown')
                schema_pop = item.get('schema_populated', False)
                if schema_pop:
                    schemas_populated += 1
                
                schema = item.get('schema', {})
                pop_info = schema.get('_deep_agent_population', {})
                fields = pop_info.get('fields_populated', 0)
                total_fields += fields
                
                standards = item.get('applicable_standards', [])
                certs = item.get('certifications', [])
                
                item_info = {
                    "name": item.get('name'),
                    "enrichment_status": status,
                    "schema_populated": schema_pop,
                    "fields_populated": fields,
                    "standards_count": len(standards),
                    "certifications_count": len(certs),
                    "standards": [s.get('code', s) if isinstance(s, dict) else s for s in standards[:5]],
                    "certifications": certs[:5]
                }
                results_summary.append(item_info)
                
                print(f"\n  📋 {item.get('name')}")
                print(f"     Status: {status}")
                print(f"     Schema Populated: {'✅' if schema_pop else '❌'}")
                print(f"     Fields: {fields}, Standards: {len(standards)}, Certs: {len(certs)}")
            
            print("\n" + "-" * 60)
            print(f"[TEST] SUMMARY:")
            print(f"  Items: {len(enriched)}")
            print(f"  Schemas populated: {schemas_populated}")
            print(f"  Total fields: {total_fields}")
            print("=" * 80 + "\n")
            
            return api_response(True, data={
                "test_type": "direct_integration",
                "user_input": user_input,
                "items_total": len(enriched),
                "items_enriched": sum(1 for item in enriched if item.get('enrichment_status') == 'success'),
                "schemas_populated": schemas_populated,
                "total_fields_populated": total_fields,
                "items_summary": results_summary,
                "full_items": enriched
            })
    
    except Exception as e:
        print(f"\n[TEST] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# THREAD MANAGEMENT ENDPOINTS (Hierarchical Thread System)
# ============================================================================

@agentic_bp.route('/threads/create-main', methods=['POST'])
@login_required
@handle_errors
def create_main_thread():
    """
    Create Main Thread with Zone Detection
    ---
    tags:
      - Thread Management
    summary: Create a new main thread with auto-detected or specified zone
    description: |
      Creates a new main thread ID for a user session. The zone can be:
      - Auto-detected from client IP (default)
      - Explicitly specified in request body
      - Specified via X-Thread-Zone header
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            user_id:
              type: string
              description: User identifier (optional, uses session user_id if not provided)
            zone:
              type: string
              description: Explicit zone override (US-WEST, US-EAST, EU-CENTRAL, etc.)
    responses:
      200:
        description: Main thread created successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                main_thread_id:
                  type: string
                zone:
                  type: string
                user_id:
                  type: string
                created_at:
                  type: string
    """
    from ..state.execution.thread_manager import HierarchicalThreadManager, ThreadZone
    from common.utils.zone_detector import resolve_zone, ThreadZone as ZoneEnum
    from datetime import datetime

    data = request.get_json() or {}

    # Get user_id from request, session, or default
    user_id = data.get('user_id') or session.get('user_id', 'anonymous')

    # Resolve zone: explicit > header > IP-detected > default
    explicit_zone = data.get('zone')
    if explicit_zone:
        zone = ThreadZone.from_string(explicit_zone)
    else:
        zone = resolve_zone(request)

    # Generate main thread ID
    main_thread_id = HierarchicalThreadManager.generate_main_thread_id(
        user_id=user_id,
        zone=zone
    )

    logger.info(f"[THREAD_API] Created main thread: {main_thread_id} for user {user_id} in zone {zone.value}")

    return api_response(True, data={
        "main_thread_id": main_thread_id,
        "zone": zone.value,
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    })


@agentic_bp.route('/threads/<thread_id>/tree', methods=['GET'])
@login_required
@handle_errors
def get_thread_tree(thread_id: str):
    """
    Get Thread Tree Structure
    ---
    tags:
      - Thread Management
    summary: Get the complete thread tree for a main thread
    description: |
      Returns the hierarchical tree structure starting from the specified thread ID.
      Includes all workflow threads and item sub-threads.
    parameters:
      - in: path
        name: thread_id
        required: true
        type: string
        description: Main thread ID or any thread ID in the hierarchy
    responses:
      200:
        description: Thread tree retrieved successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                main_thread_id:
                  type: string
                zone:
                  type: string
                workflow_threads:
                  type: array
                item_threads:
                  type: array
    """
    from ..state.execution.thread_manager import HierarchicalThreadManager
    from ..state.checkpointing.local import get_checkpointer

    # Parse thread ID to get hierarchy info
    info = HierarchicalThreadManager.parse_thread_id(thread_id)

    # Get main thread ID (either the thread itself or its parent)
    main_thread_id = info.main_thread_id or thread_id

    # Try to get tree from Azure Blob Storage if available
    try:
        from ..state.checkpointing.azure import get_azure_blob_checkpointer
        checkpointer = get_azure_blob_checkpointer()
        tree = checkpointer.get_thread_tree(main_thread_id)

        return api_response(True, data=tree)
    except Exception as e:
        logger.warning(f"[THREAD_API] Azure checkpointer not available: {e}")

        # Return basic parsed info
        return api_response(True, data={
            "main_thread_id": main_thread_id,
            "zone": info.zone,
            "parsed_info": info.to_dict(),
            "note": "Full tree not available (Azure Blob Storage not configured)"
        })


@agentic_bp.route('/threads/<item_thread_id>/state', methods=['GET'])
@login_required
@handle_errors
def get_item_state(item_thread_id: str):
    """
    Get Item State
    ---
    tags:
      - Thread Management
    summary: Get full persistent state for an item sub-thread
    description: |
      Retrieves the complete state for an identified instrument or accessory item.
      Includes specifications, search results, and status.
    parameters:
      - in: path
        name: item_thread_id
        required: true
        type: string
        description: Item thread ID
      - in: query
        name: zone
        type: string
        description: Zone for storage lookup (auto-detected if not provided)
    responses:
      200:
        description: Item state retrieved successfully
      404:
        description: Item state not found
    """
    from ..state.execution.thread_manager import HierarchicalThreadManager
    from common.utils.zone_detector import resolve_zone

    # Get zone from query param or detect
    zone = request.args.get('zone')
    if not zone:
        zone = resolve_zone(request).value

    # Try to get state from Azure Blob Storage
    try:
        from ..state.checkpointing.azure import get_azure_blob_checkpointer
        checkpointer = get_azure_blob_checkpointer()
        state = checkpointer.get_item_state(item_thread_id, zone)

        if state:
            return api_response(True, data=state)
        else:
            return api_response(False, error="Item state not found", status_code=404)
    except Exception as e:
        logger.error(f"[THREAD_API] Failed to get item state: {e}")
        return api_response(False, error=f"Failed to retrieve item state: {str(e)}", status_code=500)


@agentic_bp.route('/threads/<item_thread_id>/state', methods=['PUT'])
@login_required
@handle_errors
def update_item_state(item_thread_id: str):
    """
    Update Item State
    ---
    tags:
      - Thread Management
    summary: Update specific fields in an item's state
    description: |
      Updates the persistent state for an item sub-thread.
      Only provided fields are updated; others are preserved.
    parameters:
      - in: path
        name: item_thread_id
        required: true
        type: string
        description: Item thread ID
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            zone:
              type: string
              description: Zone for storage
            status:
              type: string
              description: New status (identified, selected, searched, completed)
            search_results:
              type: array
              description: Product search results
            selected_product:
              type: object
              description: User-selected product
    responses:
      200:
        description: Item state updated successfully
      404:
        description: Item state not found
    """
    from common.utils.zone_detector import resolve_zone

    data = request.get_json() or {}

    # Get zone from request or detect
    zone = data.pop('zone', None)
    if not zone:
        zone = resolve_zone(request).value

    # Try to update state in Azure Blob Storage
    try:
        from ..state.checkpointing.azure import get_azure_blob_checkpointer
        checkpointer = get_azure_blob_checkpointer()

        success = checkpointer.update_item_state(item_thread_id, zone, data)

        if success:
            # Return updated state
            updated_state = checkpointer.get_item_state(item_thread_id, zone)
            return api_response(True, data=updated_state)
        else:
            return api_response(False, error="Item state not found", status_code=404)
    except Exception as e:
        logger.error(f"[THREAD_API] Failed to update item state: {e}")
        return api_response(False, error=f"Failed to update item state: {str(e)}", status_code=500)


@agentic_bp.route('/threads/user/<user_id>', methods=['GET'])
@login_required
@handle_errors
def get_user_threads(user_id: str):
    """
    List User's Thread Trees
    ---
    tags:
      - Thread Management
    summary: Get all thread trees for a user
    description: |
      Returns all main thread trees associated with a user.
      Can be filtered by zone.
    parameters:
      - in: path
        name: user_id
        required: true
        type: string
        description: User identifier
      - in: query
        name: zone
        type: string
        description: Optional zone filter
    responses:
      200:
        description: User threads retrieved successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                user_id:
                  type: string
                threads:
                  type: array
                  items:
                    type: object
                    properties:
                      thread_id:
                        type: string
                      zone:
                        type: string
                      workflow_type:
                        type: string
                      created_at:
                        type: string
    """
    zone = request.args.get('zone')

    try:
        from ..state.checkpointing.azure import get_azure_blob_checkpointer
        checkpointer = get_azure_blob_checkpointer()
        threads = checkpointer.get_user_threads(user_id, zone)

        return api_response(True, data={
            "user_id": user_id,
            "zone_filter": zone,
            "threads": threads,
            "total_count": len(threads)
        })
    except Exception as e:
        logger.warning(f"[THREAD_API] Azure checkpointer not available: {e}")
        return api_response(True, data={
            "user_id": user_id,
            "threads": [],
            "note": "Thread history not available (Azure Blob Storage not configured)"
        })


@agentic_bp.route('/threads/cleanup', methods=['POST'])
@login_required
@handle_errors
def cleanup_expired_threads():
    """
    Cleanup Expired Threads
    ---
    tags:
      - Thread Management
    summary: Clean up expired checkpoints based on TTL
    description: |
      Removes thread checkpoints older than the configured TTL.
      Can be run periodically or manually.
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            zone:
              type: string
              description: Optional zone to clean (cleans all if not provided)
    responses:
      200:
        description: Cleanup completed
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                removed_count:
                  type: integer
                cutoff_time:
                  type: string
    """
    data = request.get_json() or {}
    zone = data.get('zone')

    try:
        from ..state.checkpointing.azure import get_azure_blob_checkpointer
        checkpointer = get_azure_blob_checkpointer()
        result = checkpointer.cleanup_expired_checkpoints(zone)

        return api_response(result.get("success", True), data=result)
    except Exception as e:
        logger.error(f"[THREAD_API] Cleanup failed: {e}")
        return api_response(False, error=f"Cleanup failed: {str(e)}", status_code=500)


# ============================================================================
# WORKFLOW REGISTRY API (Level 4.5)
# ============================================================================

@agentic_bp.route('/workflows', methods=['GET'])
@login_required
@handle_errors
def list_workflows():
    """
    List All Registered Workflows
    ---
    tags:
      - Workflow Registry
    summary: Get all workflows registered in the WorkflowRegistry
    description: |
      Returns a list of all registered workflows with their metadata including:
      - name, display name, description
      - supported intents and keywords
      - capabilities and priority
      - enabled/disabled status
    responses:
      200:
        description: List of registered workflows
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                workflows:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      display_name:
                        type: string
                      description:
                        type: string
                      intents:
                        type: array
                        items:
                          type: string
                      priority:
                        type: integer
                      is_enabled:
                        type: boolean
                total:
                  type: integer
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        registry = get_workflow_registry()
        workflows = registry.list_all()
        
        return api_response(True, data={
            "workflows": [w.to_dict() for w in workflows],
            "total": len(workflows),
            "registry_stats": registry.get_stats()
        })
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Failed to list workflows: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/workflows/<workflow_name>', methods=['GET'])
@login_required
@handle_errors
def get_workflow_info(workflow_name: str):
    """
    Get Workflow Details
    ---
    tags:
      - Workflow Registry
    summary: Get detailed information about a specific workflow
    parameters:
      - name: workflow_name
        in: path
        type: string
        required: true
        description: Name of the workflow (e.g., 'solution', 'instrument_identifier')
    responses:
      200:
        description: Workflow details
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
      404:
        description: Workflow not found
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        registry = get_workflow_registry()
        workflow = registry.get(workflow_name)
        
        if not workflow:
            return api_response(False, error=f"Workflow '{workflow_name}' not found", status_code=404)
        
        return api_response(True, data={
            "workflow": workflow.to_dict()
        })
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Failed to get workflow {workflow_name}: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/workflows/<workflow_name>/invoke', methods=['POST'])
@login_required
@handle_errors
def invoke_workflow_by_name(workflow_name: str):
    """
    Invoke Workflow by Name
    ---
    tags:
      - Workflow Registry
    summary: Invoke a specific workflow directly by name
    description: |
      Invokes a workflow by name using the WorkflowRegistry.
      This bypasses intent classification and directly calls the workflow.
      
      The request body should contain the parameters expected by the workflow:
      - user_input: The user's query/input (required for most workflows)
      - session_id: Session identifier (optional, defaults to generated UUID)
      - Other workflow-specific parameters
      
      If use_guardrails is true (default), input validation is performed first.
    parameters:
      - name: workflow_name
        in: path
        type: string
        required: true
        description: Name of the workflow to invoke
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            user_input:
              type: string
              description: User query/input
            session_id:
              type: string
              description: Session ID
            use_guardrails:
              type: boolean
              description: Whether to validate input before invoking (default true)
    responses:
      200:
        description: Workflow execution result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
      400:
        description: Invalid request or guardrail blocked
      404:
        description: Workflow not found
      503:
        description: Workflow disabled or unavailable
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        data = request.get_json() or {}
        user_input = data.get('user_input', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        use_guardrails = data.get('use_guardrails', True)
        
        registry = get_workflow_registry()
        workflow = registry.get(workflow_name)
        
        if not workflow:
            return api_response(False, error=f"Workflow '{workflow_name}' not found", status_code=404)
        
        if not workflow.is_enabled:
            return api_response(False, error=f"Workflow '{workflow_name}' is disabled", status_code=503)
        
        logger.info(f"[REGISTRY_API] Invoking workflow '{workflow_name}' for session {session_id[:8]}...")
        
        # Build kwargs for the workflow
        kwargs = {
            'session_id': session_id
        }
        
        # Add user_input with the appropriate key (some workflows use different names)
        if user_input:
            # Try common parameter names
            if 'user_input' in str(workflow.entry_function.__code__.co_varnames):
                kwargs['user_input'] = user_input
            elif 'query' in str(workflow.entry_function.__code__.co_varnames):
                kwargs['query'] = user_input
            else:
                kwargs['user_input'] = user_input  # Default to user_input
        
        # Invoke with or without guardrails
        if use_guardrails and user_input:
            result = registry.invoke_safe(workflow_name, query=user_input, **kwargs)
            
            if not result.get('success'):
                # Guardrail blocked the request
                return api_response(False, 
                    error=result.get('error', 'Request blocked'),
                    data={
                        "guardrail_status": result.get('guardrail_status'),
                        "suggested_response": result.get('suggested_response')
                    },
                    status_code=400
                )
            
            workflow_result = result.get('data', {})
        else:
            workflow_result = registry.invoke(workflow_name, **kwargs)
        
        logger.info(f"[REGISTRY_API] Workflow '{workflow_name}' completed successfully")
        
        return api_response(True, data={
            "workflow": workflow_name,
            "session_id": session_id,
            "result": workflow_result
        })
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except ValueError as e:
        # Workflow not found or disabled
        return api_response(False, error=str(e), status_code=404)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Failed to invoke workflow {workflow_name}: {e}")
        import traceback
        traceback.print_exc()
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/workflows/<workflow_name>/match', methods=['POST'])
@login_required
@handle_errors
def match_workflow(workflow_name: str):
    """
    Match Intent to Workflow
    ---
    tags:
      - Workflow Registry
    summary: Check if a workflow matches given intent
    description: |
      Uses the WorkflowRegistry to check if a specific workflow matches
      the given intent and is_solution flag. Returns match details including
      confidence and alternatives.
    parameters:
      - name: workflow_name
        in: path
        type: string
        required: true
        description: Workflow to match against
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            intent:
              type: string
              description: Intent to match
            is_solution:
              type: boolean
              description: Whether this is a solution-type request
            confidence:
              type: number
              description: Classification confidence (0.0-1.0)
    responses:
      200:
        description: Match result
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        data = request.get_json() or {}
        intent = data.get('intent', '')
        is_solution = data.get('is_solution', False)
        confidence = data.get('confidence', 1.0)
        
        registry = get_workflow_registry()
        match_result = registry.match_intent(intent, is_solution, confidence)
        
        return api_response(True, data={
            "match": match_result.to_dict(),
            "requested_workflow": workflow_name,
            "is_match": match_result.workflow and match_result.workflow.name == workflow_name
        })
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Match failed: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/registry/stats', methods=['GET'])
@login_required
@handle_errors
def get_registry_stats():
    """
    Get Registry Statistics
    ---
    tags:
      - Workflow Registry
    summary: Get WorkflowRegistry statistics
    description: |
      Returns statistics about the WorkflowRegistry including:
      - Number of registered/enabled workflows
      - Intent mappings
      - Invocation and match counts
      - Guardrail block counts
    responses:
      200:
        description: Registry statistics
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        registry = get_workflow_registry()
        stats = registry.get_stats()
        
        return api_response(True, data=stats)
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Failed to get stats: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/registry/guardrails/validate', methods=['POST'])
@login_required
@handle_errors
def validate_input_guardrails():
    """
    Validate Input with Guardrails
    ---
    tags:
      - Workflow Registry
    summary: Validate user input against guardrails
    description: |
      Runs guardrail validation on user input without invoking a workflow.
      Useful for pre-flight validation before routing.
      
      Checks for:
      - Empty or too short input
      - Prompt injection attempts
      - Suspicious patterns
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: User input to validate
    responses:
      200:
        description: Validation result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                is_safe:
                  type: boolean
                status:
                  type: string
                  enum: [passed, blocked, needs_clarification, low_confidence]
                reason:
                  type: string
                suggested_response:
                  type: string
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        data = request.get_json() or {}
        query = data.get('query', '')
        
        registry = get_workflow_registry()
        result = registry.validate_input(query)
        
        return api_response(True, data={
            "is_safe": result.is_safe,
            "status": result.status.value,
            "reason": result.reason,
            "suggested_response": result.suggested_response,
            "confidence": result.confidence
        })
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Guardrail validation failed: {e}")
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# WORKFLOW REGISTRY API - LEVEL 5 (Semantic Matching, Retry, A/B Testing)
# ============================================================================

@agentic_bp.route('/workflows/semantic-match', methods=['POST'])
@login_required
@handle_errors
def semantic_match_workflow():
    """
    Semantic Workflow Match (Level 5)
    ---
    tags:
      - Workflow Registry L5
    summary: Find workflows using semantic similarity matching
    description: Uses Gemini embeddings to find the best matching workflows based on semantic similarity
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - query
            properties:
              query:
                type: string
                description: User query to match against workflow descriptions
              top_k:
                type: integer
                default: 3
                description: Number of top matches to return
    responses:
      200:
        description: Semantic match results
        content:
          application/json:
            schema:
              type: object
              properties:
                matches:
                  type: array
                  items:
                    type: object
                    properties:
                      workflow:
                        type: string
                      confidence:
                        type: number
                      reasoning:
                        type: string
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        data = request.get_json() or {}
        query = data.get('query', '')
        top_k = data.get('top_k', 3)
        
        if not query:
            return api_response(False, error="query is required", status_code=400)
        
        registry = get_workflow_registry()
        results = registry.match_semantic(query, top_k=top_k)
        
        return api_response(True, data={
            "query": query,
            "matches": [r.to_dict() for r in results],
            "method": "semantic_embedding"
        })
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Semantic match failed: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/workflows/<workflow_name>/invoke-with-retry', methods=['POST'])
@login_required
@handle_errors
def invoke_workflow_with_retry(workflow_name: str):
    """
    Invoke Workflow with Retry (Level 5)
    ---
    tags:
      - Workflow Registry L5
    summary: Invoke a workflow with automatic retry on transient failures
    description: Uses the workflow's RetryPolicy for exponential backoff on rate limits and other transient errors
    parameters:
      - name: workflow_name
        in: path
        required: true
        schema:
          type: string
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - user_input
            properties:
              user_input:
                type: string
              session_id:
                type: string
    responses:
      200:
        description: Workflow result
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        session_id = get_session_id()
        data = request.get_json() or {}
        user_input = data.get('user_input') or data.get('query') or data.get('message')
        
        if not user_input:
            return api_response(False, error="user_input is required", status_code=400)
        
        registry = get_workflow_registry()
        
        # Get workflow metadata for logging
        workflow = registry.get(workflow_name)
        if not workflow:
            return api_response(False, error=f"Workflow '{workflow_name}' not found", status_code=404)
        
        # Invoke with retry
        result = registry.invoke_with_retry(
            name=workflow_name,
            user_input=user_input,
            session_id=data.get('session_id', session_id)
        )
        
        return api_response(True, data={
            "workflow": workflow_name,
            "result": result,
            "retry_enabled": workflow.retry_policy is not None
        })
        
    except ValueError as e:
        return api_response(False, error=str(e), status_code=404)
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Invoke with retry failed: {e}")
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# A/B TESTING ENDPOINTS (Level 5)
# ============================================================================

@agentic_bp.route('/experiments', methods=['GET'])
@login_required
@handle_errors
def list_experiments():
    """
    List A/B Experiments (Level 5)
    ---
    tags:
      - A/B Testing
    summary: List all A/B testing experiments
    parameters:
      - name: active_only
        in: query
        schema:
          type: boolean
          default: false
    responses:
      200:
        description: List of experiments
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        
        registry = get_workflow_registry()
        experiments = registry.list_experiments(active_only=active_only)
        
        return api_response(True, data={"experiments": experiments})
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] List experiments failed: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/experiments', methods=['POST'])
@login_required
@handle_errors
def create_experiment():
    """
    Create A/B Experiment (Level 5)
    ---
    tags:
      - A/B Testing
    summary: Create a new A/B testing experiment
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - experiment_id
              - name
              - base_workflow
              - variant_workflow
            properties:
              experiment_id:
                type: string
              name:
                type: string
              base_workflow:
                type: string
              variant_workflow:
                type: string
              traffic_percentage:
                type: number
                default: 0.5
    responses:
      200:
        description: Created experiment
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        data = request.get_json() or {}
        
        required = ['experiment_id', 'name', 'base_workflow', 'variant_workflow']
        missing = [f for f in required if not data.get(f)]
        if missing:
            return api_response(False, error=f"Missing required fields: {missing}", status_code=400)
        
        registry = get_workflow_registry()
        experiment = registry.create_experiment(
            experiment_id=data['experiment_id'],
            name=data['name'],
            base_workflow=data['base_workflow'],
            variant_workflow=data['variant_workflow'],
            traffic_percentage=data.get('traffic_percentage', 0.5)
        )
        
        return api_response(True, data={
            "experiment": experiment.to_dict(),
            "message": f"Experiment '{data['experiment_id']}' created successfully"
        })
        
    except ValueError as e:
        return api_response(False, error=str(e), status_code=400)
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Create experiment failed: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/experiments/<experiment_id>', methods=['GET'])
@login_required
@handle_errors
def get_experiment_results(experiment_id: str):
    """
    Get Experiment Results (Level 5)
    ---
    tags:
      - A/B Testing
    summary: Get metrics and results for an experiment
    parameters:
      - name: experiment_id
        in: path
        required: true
        schema:
          type: string
    responses:
      200:
        description: Experiment results with metrics
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        registry = get_workflow_registry()
        results = registry.get_experiment_results(experiment_id)
        
        if not results:
            return api_response(False, error=f"Experiment '{experiment_id}' not found", status_code=404)
        
        return api_response(True, data=results)
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Get experiment results failed: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/experiments/<experiment_id>/stop', methods=['POST'])
@login_required
@handle_errors
def stop_experiment(experiment_id: str):
    """
    Stop Experiment (Level 5)
    ---
    tags:
      - A/B Testing
    summary: Stop an active experiment
    parameters:
      - name: experiment_id
        in: path
        required: true
        schema:
          type: string
    responses:
      200:
        description: Experiment stopped
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        registry = get_workflow_registry()
        success = registry.stop_experiment(experiment_id)
        
        if not success:
            return api_response(False, error=f"Experiment '{experiment_id}' not found", status_code=404)
        
        return api_response(True, data={
            "message": f"Experiment '{experiment_id}' stopped successfully"
        })
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Stop experiment failed: {e}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/workflows/refresh-embeddings', methods=['POST'])
@login_required
@handle_errors
def refresh_workflow_embeddings():
    """
    Refresh Workflow Embeddings (Level 5)
    ---
    tags:
      - Workflow Registry L5
    summary: Regenerate embeddings for all registered workflows
    description: Forces regeneration of all workflow embeddings using the current embedding model
    responses:
      200:
        description: Embeddings refreshed
    """
    try:
        from .workflow_registry import get_workflow_registry
        
        registry = get_workflow_registry()
        count = registry.refresh_embeddings()
        
        return api_response(True, data={
            "message": f"Successfully refreshed {count} workflow embeddings",
            "count": count
        })
        
    except ImportError:
        return api_response(False, error="WorkflowRegistry not available", status_code=503)
    except Exception as e:
        logger.error(f"[REGISTRY_API] Refresh embeddings failed: {e}")
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# HEALTH CHECK
# ============================================================================


@agentic_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return api_response(True, data={
        "status": "healthy",
        "service": "agentic-workflow"
    })
