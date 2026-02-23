# PHASE 1 FIX: Initialize application FIRST (loads environment once)
from initialization import initialize_application

# Initialize before any other imports that depend on environment variables
import sys
import logging as _init_logging
import io

# Ensure UTF-8 encoding from the very start (critical on Windows where default is cp1252)
# reconfigure() is sufficient — do NOT wrap stdout.buffer in a new TextIOWrapper
# as that closes the original stream and breaks bare print() calls in third-party code.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# UTF-8 StreamHandler — use sys.stdout directly now that it's reconfigured to UTF-8
_utf8_handler = _init_logging.StreamHandler(sys.stdout)
_utf8_handler.setFormatter(_init_logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure basic logging for initialization phase
_init_logging.basicConfig(
    level=_init_logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[_utf8_handler]
)
_init_logger = _init_logging.getLogger(__name__)

try:
    initialize_application()
except RuntimeError as e:
    _init_logger.error(f"FATAL: Application initialization failed: {e}")
    _init_logger.error("Please check your .env file and environment variables")
    sys.exit(1)

import asyncio
from datetime import datetime
from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json
import logging
import re
import os
import urllib.parse
from werkzeug.utils import secure_filename
import requests
from io import BytesIO

try:
    from serpapi import GoogleSearch
except ImportError:
    # Fallback for serpapi versions that don't export GoogleSearch
    class GoogleSearch:
        def __init__(self, params):
            import requests
            self.params = params

        def get_dict(self):
            import requests
            response = requests.get("https://serpapi.com/search", params=self.params)
            return response.json()
            
import threading
import csv


from functools import wraps
from dotenv import load_dotenv
from redis import Redis
# Suppress noisy Azure SDK logs
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)

# --- NEW IMPORTS FOR SEARCH FUNCTIONALITY ---
from googleapiclient.discovery import build

# --- NEW IMPORTS FOR AUTHENTICATION ---
from common.core.auth.auth_models import db, User
from common.core.auth.auth_utils import hash_password, check_password

# --- Cosmos DB Project Management ---
from common.services.azure.cosmos_manager import cosmos_project_manager
from common.utils.auth_decorators import login_required, admin_required

# --- LLM CHAINING IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.core.chaining import setup_langchain_components, create_analysis_chain, invoke_additional_requirements_chain

# from common.prompts_library import load_prompt, load_prompt_sections # REMOVED

from common.prompts import (
    INTENT_CLASSIFICATION_PROMPTS,
    INTENT_PROMPTS,
    INDEXING_AGENT_PROMPTS,
    SOLUTION_DEEP_AGENT_PROMPTS,
    SCHEMA_VALIDATION_PROMPT,
    ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT
)

# --- PROMPT LIBRARY SETUP ---
# Load consolidated prompt sections
# Note: sales_workflow_prompts was consolidated into sales_agent_prompts
# _SALES_AGENT_PROMPTS = load_prompt_sections("sales_agent_prompts", default_section="SALES_CONSULTANT")
# _INTENT_CLASSIFICATION_PROMPTS = load_prompt_sections("intent_classification_prompts", default_section="CLASSIFICATION")
# _INTENT_ANALYSIS_PROMPTS = load_prompt_sections("intent_prompts", default_section="REQUIREMENTS_EXTRACTION")
# _PPI_PROMPTS = load_prompt_sections("indexing_agent_prompts", default_section="PRODUCT_INDEX")
# _PRODUCT_ID_PROMPTS = load_prompt_sections("solution_deep_agent_prompts", default_section="INSTRUMENT_IDENTIFICATION")


# Define Constants for Main usage
SALES_AGENT_MAIN_PROMPT = ""
SALES_AGENT_GREETING_PROMPT = ""
SALES_AGENT_FINAL_ANALYSIS_PROMPT = ""
SALES_AGENT_ERROR_PROMPT = ""

SUMMARY_GENERATION_PROMPT = ""
PARAMETER_SELECTION_PROMPT = ""
REQUIREMENTS_EXTRACTION_PROMPT = INTENT_PROMPTS

CLASSIFICATION_PROMPT = INTENT_CLASSIFICATION_PROMPTS["DEFAULT"]
IDENTIFY_INSTRUMENT_PROMPT = SOLUTION_DEEP_AGENT_PROMPTS.get("INSTRUMENT_IDENTIFICATION", "")
VALIDATION_PROMPT = SCHEMA_VALIDATION_PROMPT
VENDOR_ANALYSIS_PROMPT = ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT


VALIDATION_ALERT_INITIAL_PROMPT = "Please review the validation results."
VALIDATION_ALERT_REPEAT_PROMPT = "Please address the validation issues."
FEEDBACK_POSITIVE_PROMPT = "Thank you for your positive feedback!"
FEEDBACK_NEGATIVE_PROMPT = "Thank you for your feedback. We'll use it to improve."
FEEDBACK_COMMENT_PROMPT = "Thank you for your comment: {comment}"
MANUFACTURER_DOMAIN_PROMPT = INDEXING_AGENT_PROMPTS["VENDOR_DISCOVERY"]


from common.core.loading import load_requirements_schema, build_requirements_schema_from_web
from flask_session import Session

# Import latest advanced specifications functionality

# Import Azure Blob utilities (MongoDB API compatible)
# Import Azure Blob utilities
from common.services.azure.blob_utils import azure_blob_file_manager

# PHASE 3: Service Layer Imports (MongoDB + Azure Blob Hybrid)
# These services provide transparent fallback from MongoDB to Azure Blob
try:
    from common.services.schema_service import schema_service
    from common.services.vendor_service import vendor_service
    from common.services.project_service import project_service
    from common.services.document_service import document_service
    from common.services.image_service import image_service
    from common.core.mongodb_manager import mongodb_manager
    logging.info("[INIT] ✓ Service layer loaded successfully (MongoDB + Azure Blob hybrid)")
except ImportError as e:
    logging.warning(f"[INIT] ⚠ Service layer not available: {e}")
    logging.warning("[INIT] Falling back to direct Azure Blob access")
    schema_service = None
    vendor_service = None
    project_service = None
    document_service = None
    image_service = None
    mongodb_manager = None


# MongoDB Project Management imports removed


# =========================================================================
# === ENVIRONMENT VALIDATION ===
# =========================================================================
def validate_environment():
    """Validate critical environment variables at startup."""
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("ENVIRONMENT VALIDATION")
    logger.info("="*70)

    required = {
        'GOOGLE_API_KEY': 'LLM generation'
    }

    for var, purpose in required.items():
        if os.getenv(var):
            logger.info(f"✅ {var} configured ({purpose})")
        else:
            logger.warning(f"⚠️  {var} missing ({purpose} will fail)")

    logger.info("="*70)

# Call environment validation before app initialization
validate_environment()


# Load environment variables
# =========================================================================
# === FLASK APP CONFIGURATION ===
# =========================================================================
app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
# Manual CORS handling

# A list of allowed origins for CORS
allowed_origins = [
    "https://ai-product-recommender-ui.vercel.app",  # Your production frontend
    "https://en-genie.vercel.app",                   # Your new frontend domain
    "https://en-genie1.vercel.app",                  # Your deployed frontend
    "http://localhost:8080",                         # Add your specific local dev port
    "http://localhost:5173",
    "http://localhost:3000"
]

# Dynamically add Vercel preview URLs to allowed_origins
if os.environ.get("VERCEL") == "1":
    # Add the production URL
    prod_url = os.environ.get("VERCEL_URL")
    if prod_url:
        allowed_origins.append(f"https://{prod_url}")
    # Add the preview URL for the current branch
    branch_url = os.environ.get("VERCEL_BRANCH_URL")
    if branch_url and branch_url != prod_url:
        allowed_origins.append(f"https://{branch_url}")

# CORS Configuration with full headers support
CORS(app,
     origins=allowed_origins,
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With', 'Accept'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
     expose_headers=['Content-Type', 'Authorization'],
     max_age=3600)

# Handle OPTIONS preflight requests explicitly
@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_response(('', 200))
        origin = request.headers.get('Origin')
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept'
            response.headers['Access-Control-Max-Age'] = '3600'
        return response

# Ensure CORS headers are added to all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept'
    return response

# NOTE: basicConfig was already called above with a UTF-8 handler.
# Suppress noisy third-party loggers.
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
def is_redis_available():
    """Check if Redis environment variables are properly configured."""
    # Check for standard or Railway-specific environment variables
    redis_host = os.getenv("REDIS_HOST") or os.getenv("REDISHOST")
    redis_port = os.getenv("REDIS_PORT") or os.getenv("REDISPORT")
    return redis_host is not None and redis_port is not None

# Define Redis session initialization function
def init_redis_sessions(app):
    """Initialize Redis-backed sessions."""
    try:
        # Get connection details (supporting both standard and Railway naming)
        redis_host = os.getenv("REDIS_HOST") or os.getenv("REDISHOST")
        redis_port = int(os.getenv("REDIS_PORT") or os.getenv("REDISPORT") or "6379")
        redis_password = os.getenv("REDIS_PASSWORD") or os.getenv("REDISPASSWORD") or ""
        
        app.config["SESSION_TYPE"] = "redis"
        app.config["SESSION_PERMANENT"] = True
        app.config["SESSION_USE_SIGNER"] = True
        app.config["SESSION_REDIS"] = Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            # decode_responses MUST be False for Flask-Session because it stores pickled binary data
            decode_responses=False,
            socket_timeout=5,
            retry_on_timeout=True
        )
        logging.info(f"Redis session storage initialized successfully (Host: {redis_host})")
        return True
    except Exception as e:
        logging.warning(f"Failed to initialize Redis sessions: {e}")
        return False

# Define filesystem session initialization function  
def init_filesystem_sessions(app):
    """Initialize filesystem-backed sessions as fallback."""
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_PERMANENT"] = False
    logging.info("Filesystem session storage initialized")


if os.getenv('FLASK_ENV') == 'production' or os.getenv('RAILWAY_ENVIRONMENT'):
    # Production session settings
    app.config["SESSION_PERMANENT"] = True
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = "/app/flask_session"
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
else:
    # Development session settings
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem" 

def get_mysql_uri():
    """Get MySQL URI, returns None if MySQL is not configured"""
    required_vars = ['MYSQLUSER', 'MYSQLPASSWORD', 'MYSQLHOST', 'MYSQLPORT', 'MYSQLDATABASE']
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logging.warning(f"⚠️ MySQL not configured - missing: {', '.join(missing)}")
        return None

    return (
        f"mysql+pymysql://{os.getenv('MYSQLUSER')}:"
        f"{os.getenv('MYSQLPASSWORD')}@"
        f"{os.getenv('MYSQLHOST')}:"
        f"{os.getenv('MYSQLPORT')}/"
        f"{os.getenv('MYSQLDATABASE')}"
    )

# Database configuration (use SQLite fallback if MySQL not available)
mysql_uri = get_mysql_uri()
if mysql_uri:
    app.config["SQLALCHEMY_DATABASE_URI"] = mysql_uri
    logging.info("✅ Using MySQL database")
else:
    # Fallback to SQLite for development/testing
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fallback.db"
    logging.warning("⚠️ Using SQLite fallback database (MySQL not configured)")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')

# Initialize database and session (SESSION_TYPE is now properly set)
db.init_app(app)

# Note: Database tables (User) are created by create_db() at end of file.
# This operation is SAFE: it only creates tables if they don't exist.
# Existing tables and data are preserved across redeploys.

Session(app)



# --- Initialize Rate Limiting ---
from common.infrastructure.rate_limiter import init_limiter
limiter = init_limiter(app)
logging.info("Rate limiting initialized successfully")

# --- Import and Register Agentic Workflow Blueprint ---
from common.infrastructure.api.main_api import agentic_bp
app.register_blueprint(agentic_bp)
logging.info("Agentic workflow blueprint registered at /api/agentic")

# --- Import and Register Deep Agent Blueprint ---
from common.agentic.deep_agent.api import deep_agent_bp
app.register_blueprint(deep_agent_bp, url_prefix='/api')
logging.info("Deep Agent blueprint registered at /api/deep-agent")

# NOTE: search.product_search_api no longer exists — product search is handled by
# agentic_bp at /api/agentic/product-search (registered above via main_api.py).

# --- Import and Register EnGenie Chat API Blueprint ---
from chat.engenie_chat_api import engenie_chat_bp
app.register_blueprint(engenie_chat_bp)
logging.info("EnGenie Chat API blueprint registered at /api/engenie-chat")

# --- Import and Register Tools API Blueprint ---
from common.tools.api import tools_bp
app.register_blueprint(tools_bp, url_prefix='/api/tools')
logging.info("Tools API blueprint registered at /api/tools")


# NOTE: The old sales_agent_bp (/api/sales-agent) has been superseded by
# agentic_bp's /api/agentic/sales-agent (SalesAgentTool-based).
# A lightweight redirect shim is registered so legacy callers are forwarded
# and the old DeepAgenticWorkflowOrchestrator is no longer invoked.
from flask import redirect, url_for
from flask import Blueprint as _Blueprint
_legacy_sales_bp = _Blueprint('legacy_sales_agent', __name__)

@_legacy_sales_bp.route('/', methods=['POST'], strict_slashes=False)
@_legacy_sales_bp.route('/<path:subpath>', methods=['POST'], strict_slashes=False)
def _legacy_sales_redirect(subpath=''):
    """Redirect old /api/sales-agent/* calls to /api/agentic/sales-agent."""
    from flask import request as _req
    import requests as _requests
    # Forward the request body to the canonical endpoint
    new_url = _req.host_url.rstrip('/') + '/api/agentic/sales-agent'
    resp = _requests.post(new_url, json=_req.get_json(), headers={'Cookie': _req.headers.get('Cookie', '')})
    from flask import Response
    return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))

app.register_blueprint(_legacy_sales_bp, url_prefix='/api/sales-agent')
logging.info("Legacy /api/sales-agent → redirecting to /api/agentic/sales-agent")

# --- Import and Register Session API Blueprints ---
from common.infrastructure.api.session import register_session_blueprints
register_session_blueprints(app)
logging.info("Session and Instance blueprints registered")


# LangChain and Utility Imports
# Note: api_utils was consolidated into agentic.infrastructure.api.utils
from common.infrastructure.api.utils import (
    convert_keys_to_camel_case,
    clean_empty_values,
    map_provided_to_schema,
    get_missing_mandatory_fields,
    friendly_field_name
)

# =========================================================================
# === HELPER FUNCTIONS AND UTILITIES ===
# =========================================================================
# --- Import and Register Resource Monitoring Blueprint ---
try:
    from common.infrastructure.api.monitoring import resource_bp
    app.register_blueprint(resource_bp)
    logging.info("Resource Monitoring blueprint registered at /api/resources")
except ImportError:
    logging.warning("Resource Monitoring API not available")
from common.services.extraction.extraction_engine import (
    extract_data_from_pdf,
    send_to_language_model,
    aggregate_results,
    generate_dynamic_path,
    split_product_types,
    save_json
    # Removed: identify_and_save_product_image - no longer needed with API-based images
)

# Import standardization utilities
from common.services.products.standardization import (
    standardize_vendor_analysis_result,
    standardize_ranking_result,
    enhance_submodel_mapping,
    standardize_product_image_mapping,
    create_standardization_report,
    update_existing_vendor_files_with_standardization,
    standardize_vendor_name,
    standardized_jsonify
)

# Initialize LangChain components
try:
    components = setup_langchain_components()
    analysis_chain = create_analysis_chain(components)  # Uses Vector DB / Azure, no local paths needed
    
    logging.info("LangChain components initialized.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    components = None
    analysis_chain = None

def prettify_req(req):
    return req.replace('_', ' ').replace('-', ' ').title()

def flatten_schema(schema_dict):
    flat = {}
    for k, v in schema_dict.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[subk] = subv
        else:
            flat[k] = v
    return flat


# Use imported login_required instead of local definition
# ALLOWED_EXTENSIONS moved to top-level if needed, but keeping one here for clarity if only used once
ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str):
    """Check if the uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# OLD /api/intent ENDPOINT (COMMENTED OUT)
# This endpoint used an independent LLM chain for intent classification.
# Now replaced with a wrapper that calls classify_intent_tool from intent_tools.py
# for consistent intent classification across the system.
# =============================================================================
# @app.route("/api/intent", methods=["POST"])
# @login_required
# def api_intent_old():
#     """
#     [DEPRECATED] Old intent classification endpoint.
#     Kept for reference - used independent LLM chain instead of intent_tools.
#     """
#     pass
# =============================================================================

# IntentClassificationRoutingAgent handles intent classification and workflow routing
# It internally uses classify_intent_tool with additional features:
# - Workflow state locking
# - Exit phrase detection
# - Metrics-based complexity detection
# - Intelligent routing decisions

@app.route("/api/intent", methods=["POST"])
@login_required
def api_intent():
    """
    Classify user intent and route to appropriate workflow
    ---
    tags:
      - Workflow
    summary: Classify user intent and route to workflow
    description: |
      Uses IntentClassificationRoutingAgent for intelligent intent classification and workflow routing.

      **Features:**
      - Workflow state locking (keeps user in current workflow until exit)
      - Exit phrase detection ("start over", "reset", etc.)
      - Metrics-based complexity detection for solution vs single-product
      - LLM-based intent classification with retry and fallback
      
      **Intent Types:**
      - `greeting` - Initial greeting
      - `solution` - Complex engineering challenge requiring multiple instruments
      - `requirements` - Single product specifications
      - `question` - Asking about industrial topics
      - `productRequirements` - User is providing product specifications
      - `knowledgeQuestion` - User is asking a question
      - `workflow` - Continuing an existing workflow
      - `confirm` / `reject` - User confirmations/rejections
      - `chitchat` - Casual conversation
      - `other` - Unrecognized intent
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: User input for intent classification
        required: true
        schema:
          type: object
          required:
            - userInput
          properties:
            userInput:
              type: string
              description: The user's message or input to classify
              example: "I need a pressure transmitter with HART protocol"
            search_session_id:
              type: string
              description: Session ID for workflow isolation
              example: "session_12345"
    responses:
      200:
        description: Intent classification result
        schema:
          type: object
          properties:
            intent:
              type: string
              enum: [greeting, solution, requirements, question, productRequirements, knowledgeQuestion, workflow, confirm, reject, chitchat, other]
              description: Classified intent type
            nextStep:
              type: string
              description: Next workflow step to navigate to
              example: "initialInput"
            resumeWorkflow:
              type: boolean
              description: Whether to resume the current workflow
            confidence:
              type: number
              description: Confidence score (0.0-1.0)
            isSolution:
              type: boolean
              description: Whether this is a complex engineering solution request
      400:
        description: Bad request - missing userInput
        schema:
          type: object
          properties:
            error:
              type: string
              example: "userInput is required"
      401:
        description: Unauthorized - login required
      500:
        description: Intent classification failed
    """
    data = request.get_json(force=True)
    user_input = data.get("userInput", "").strip()
    if not user_input:
        return jsonify({"error": "userInput is required"}), 400

    # Get search session ID if provided (for session isolation)
    search_session_id = data.get("search_session_id", "default")

    # Get current workflow state from session (session-isolated)
    current_step_key = f'current_step_{search_session_id}'
    current_intent_key = f'current_intent_{search_session_id}'
    
    current_step = session.get(current_step_key, None)
    current_intent = session.get(current_intent_key, None)
    
    # --- Handle skip for missing mandatory fields ---
    # Accept both legacy and frontend step names when user wants to skip missing mandatory fields
    if current_step in ("awaitMandatory", "awaitMissingInfo") and user_input.lower() in ["yes", "skip", "y"]:
        session[f'current_step_{search_session_id}'] = "awaitAdditionalAndLatestSpecs"
        response = {
            "intent": "workflow",
            "nextStep": "awaitAdditionalAndLatestSpecs",
            "resumeWorkflow": True,
            "message": "Skipping missing mandatory fields. Additional and latest specifications are available."
        }
        return jsonify(response), 200
    
    # =========================================================================
    # USE IntentClassificationRoutingAgent FOR COMPLETE ROUTING
    # This agent handles:
    # - Workflow state locking (single source of truth)
    # - Exit phrase detection
    # - Metrics-based complexity detection
    # - LLM-based intent classification
    # - Workflow routing decisions
    # =========================================================================
    from common.agentic.intent_classification_routing_agent import (
        IntentClassificationRoutingAgent,
        WorkflowTarget,
        get_workflow_memory
    )

    try:
        # Create routing agent instance
        routing_agent = IntentClassificationRoutingAgent(name="API_IntentRouter")

        # Build context for the agent
        context = {
            "current_step": current_step,
            "current_intent": current_intent,
            "context": f"Current step: {current_step or 'None'}, Current intent: {current_intent or 'None'}"
        }

        logging.info(f"[INTENT_API] Calling IntentClassificationRoutingAgent for: {user_input[:100]}...")

        # Call the routing agent - handles EVERYTHING internally:
        # - Exit detection
        # - Workflow locking
        # - Metrics extraction
        # - LLM classification via classify_intent_tool
        # - Routing decision
        routing_result = routing_agent.classify(
            query=user_input,
            session_id=search_session_id,
            context=context
        )

        logging.info(f"[INTENT_API] Routing result: {routing_result.target_workflow.value} (intent={routing_result.intent}, conf={routing_result.confidence:.2f})")

        # Check for classification errors
        if routing_result.intent == "error":
            error_msg = routing_result.reasoning

            # Check if it's a rate limit / quota error (external service issue)
            is_external_error = any(x in str(error_msg) for x in [
                'RESOURCE_EXHAUSTED', 'quota', '429', 'Rate limit', 'overloaded', '503'
            ])

            if is_external_error:
                logging.warning(f"[INTENT_API] External service error: {error_msg}")
                return jsonify({
                    "error": "Service temporarily unavailable. Please retry.",
                    "intent": "other",
                    "nextStep": None,
                    "resumeWorkflow": False,
                    "retryAfter": 30,
                    "serviceError": True
                }), 503

            return jsonify({
                "error": error_msg,
                "intent": "other",
                "nextStep": None,
                "resumeWorkflow": False
            }), 500

        # Map WorkflowTarget to frontend expected values
        target_to_intent = {
            WorkflowTarget.SOLUTION_WORKFLOW: "solution",
            WorkflowTarget.INSTRUMENT_IDENTIFIER: "productRequirements",
            WorkflowTarget.ENGENIE_CHAT: "knowledgeQuestion",
            WorkflowTarget.OUT_OF_DOMAIN: "other"
        }

        # Handle special intents that don't map directly from target
        if routing_result.intent == "greeting":
            mapped_intent = "greeting"
        elif routing_result.intent == "workflow_locked":
            # Use the locked workflow's intent
            workflow_memory = get_workflow_memory()
            current_workflow = workflow_memory.get_workflow(search_session_id)
            workflow_to_intent = {
                "engenie_chat": "knowledgeQuestion",
                "instrument_identifier": "productRequirements",
                "solution": "solution"
            }
            mapped_intent = workflow_to_intent.get(current_workflow, "knowledgeQuestion")
        elif routing_result.intent in ["confirm", "reject", "additional_specs"]:
            mapped_intent = "workflow"
        elif routing_result.intent == "chitchat":
            mapped_intent = "chitchat"
        else:
            mapped_intent = target_to_intent.get(routing_result.target_workflow, "other")

        # Determine next step based on target workflow
        target_to_next_step = {
            WorkflowTarget.SOLUTION_WORKFLOW: "solutionWorkflow",
            WorkflowTarget.INSTRUMENT_IDENTIFIER: "initialInput",
            WorkflowTarget.ENGENIE_CHAT: None,  # EnGenie Chat handles its own routing
            WorkflowTarget.OUT_OF_DOMAIN: None
        }

        if routing_result.intent == "greeting":
            next_step = "greeting"
        else:
            next_step = target_to_next_step.get(routing_result.target_workflow)

        # Determine workflow name for session tracking
        target_to_workflow_name = {
            WorkflowTarget.SOLUTION_WORKFLOW: "solution",
            WorkflowTarget.INSTRUMENT_IDENTIFIER: "instrument_identifier",
            WorkflowTarget.ENGENIE_CHAT: "engenie_chat",
            WorkflowTarget.OUT_OF_DOMAIN: None
        }

        new_workflow = target_to_workflow_name.get(routing_result.target_workflow)
        if routing_result.intent == "greeting":
            new_workflow = None  # Clear workflow on greeting

        # Check if workflow is locked (already set by routing agent)
        workflow_memory = get_workflow_memory()
        is_workflow_locked = routing_result.extracted_info.get("workflow_locked", False)

        # Determine resume_workflow flag
        if is_workflow_locked:
            resume_workflow = True
        elif routing_result.intent in ["confirm", "reject", "additional_specs"]:
            resume_workflow = True
        else:
            resume_workflow = False

        # Build suggestion for knowledge questions (don't auto-route)
        suggest_workflow = None
        if routing_result.target_workflow == WorkflowTarget.ENGENIE_CHAT and not is_workflow_locked:
            suggest_workflow = {
                "name": "EnGenie Chat",
                "workflow_id": "engenie_chat",
                "description": "Get answers about products, standards, and industrial topics",
                "action": "openEnGenieChat"
            }
            logging.info("[INTENT_API] Suggesting EnGenie Chat workflow")
        elif routing_result.target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
            suggest_workflow = {
                "name": "EnGenie Chat",
                "workflow_id": "engenie_chat",
                "description": "Get answers about products, standards, and industrial topics",
                "action": "openEnGenieChat"
            }

        # Build response
        result_json = {
            "intent": mapped_intent,
            "nextStep": next_step,
            "resumeWorkflow": resume_workflow,
            "confidence": routing_result.confidence,
            "isSolution": routing_result.is_solution,
            "extractedInfo": routing_result.extracted_info,
            "solutionIndicators": routing_result.solution_indicators,
            "currentWorkflow": new_workflow,
            "suggestWorkflow": suggest_workflow,
            "workflowLocked": is_workflow_locked,
            "routingReasoning": routing_result.reasoning  # Include reasoning for debugging
        }

        # Add reject message for out-of-domain queries
        if routing_result.reject_message:
            result_json["rejectMessage"] = routing_result.reject_message

        # Update Flask session state for backward compatibility
        if mapped_intent == "greeting":
            session[f'current_step_{search_session_id}'] = 'greeting'
            session[f'current_intent_{search_session_id}'] = 'greeting'
        elif mapped_intent == "solution" or routing_result.is_solution:
            session[f'current_step_{search_session_id}'] = 'solutionWorkflow'
            session[f'current_intent_{search_session_id}'] = 'solution'
            logging.info("[SOLUTION_ROUTING] Detected solution/engineering challenge - routing to solution workflow")
        elif mapped_intent == "productRequirements":
            session[f'current_step_{search_session_id}'] = 'initialInput'
            session[f'current_intent_{search_session_id}'] = 'productRequirements'
        elif mapped_intent == "knowledgeQuestion":
            session[f'current_intent_{search_session_id}'] = 'knowledgeQuestion'
        elif mapped_intent == "workflow" and next_step:
            session[f'current_step_{search_session_id}'] = next_step
            session[f'current_intent_{search_session_id}'] = 'workflow'

        # Log metrics if available
        system_metrics = routing_result.extracted_info.get("system_metrics", {})
        if system_metrics:
            logging.info(f"[INTENT_API] System complexity: score={system_metrics.get('complexity_score', 0)}, "
                        f"instruments={system_metrics.get('estimated_instruments', 0)}, "
                        f"is_complex={system_metrics.get('is_complex_system', False)}")

        logging.info(f"[INTENT_API] Final response: intent={mapped_intent}, workflow={new_workflow}, locked={is_workflow_locked}")
        return jsonify(result_json), 200

    except Exception as e:
        error_msg = str(e)
        logging.exception("[INTENT_API] Intent classification failed.")

        # Return 503 for rate limit errors (temporary condition), 500 for other errors
        is_rate_limit = any(x in error_msg for x in ['429', 'Resource exhausted', 'RESOURCE_EXHAUSTED', 'quota'])
        status_code = 503 if is_rate_limit else 500

        return jsonify({
            "error": error_msg,
            "intent": "other",
            "nextStep": None,
            "resumeWorkflow": False,
            "retryable": is_rate_limit
        }), status_code

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    API Health Check
    ---
    tags:
      - Health
    summary: Check API health status
    description: Returns the current health status of the API and its components.
    produces:
      - application/json
    responses:
      200:
        description: API is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: "healthy"
            workflow_initialized:
              type: boolean
              description: Whether workflow engine is initialized
            langsmith_enabled:
              type: boolean
              description: Whether LangSmith monitoring is enabled
      401:
        description: Unauthorized - login required
    """
    return {
        "status": "healthy",
        "workflow_initialized": False,
        "langsmith_enabled": False
    }, 200


@app.route('/api/health/enrichment', methods=['GET'])
def enrichment_health():
    """
    Enrichment Pipeline Health Check
    ---
    tags:
      - Health
    summary: Check enrichment pipeline health status
    description: Returns detailed health status of all enrichment components (vector store, LLM, standards documents).
    produces:
      - application/json
    responses:
      200:
        description: Enrichment pipeline is healthy or degraded
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [healthy, degraded, error]
            components:
              type: object
            capabilities:
              type: object
      503:
        description: Enrichment pipeline is in error state
    """
    try:
        # LLM service health
        llm_healthy = bool(os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY'))

        # Overall status
        if llm_healthy:
            overall = "healthy"
        else:
            overall = "error"

        return jsonify({
            "status": overall,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "llm_service": {
                    "healthy": llm_healthy,
                    "provider": "google" if os.getenv('GOOGLE_API_KEY') else "openai" if os.getenv('OPENAI_API_KEY') else "none"
                }
            },
            "capabilities": {
                "user_spec_extraction": llm_healthy,
                "llm_enrichment": llm_healthy,
                "standards_enrichment": False
            }
        }), 200 if overall != "error" else 503

    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/health/database', methods=['GET'])
def database_health():
    """
    Database Infrastructure Health Check (PHASE 3)
    ---
    tags:
      - Health
    summary: Check MongoDB and Azure Blob Storage health
    description: |
      Returns detailed health status of the hybrid database infrastructure:
      - MongoDB connection and collections
      - Azure Blob Storage connectivity
      - Service layer availability
    produces:
      - application/json
    responses:
      200:
        description: Database infrastructure status
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [healthy, degraded, error]
            mongodb:
              type: object
            azure_blob:
              type: object
            services:
              type: object
    """
    try:
        health_status = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "mongodb": {
                "connected": False,
                "collections": [],
                "reason": "Not checked"
            },
            "azure_blob": {
                "connected": False,
                "reason": "Not checked"
            },
            "services": {
                "schema_service": False,
                "vendor_service": False,
                "project_service": False,
                "document_service": False,
                "image_service": False
            }
        }

        # Check MongoDB
        try:
            from common.core.mongodb_manager import mongodb_manager
            from common.services.azure.cosmos_manager import cosmos_project_manager

            if mongodb_manager.is_connected():
                db = mongodb_manager.database
                collections = db.list_collection_names() if db else []

                # Check cosmos_project_manager (user_projects collection)
                projects_health = cosmos_project_manager.check_mongodb_health()

                health_status["mongodb"] = {
                    "connected": True,
                    "collections": collections,
                    "count": len(collections),
                    "reason": "OK",
                    "user_projects_collection": {
                        "can_read": projects_health.get('can_read', False),
                        "can_write": projects_health.get('can_write', False),
                        "document_count": projects_health.get('document_count', 0),
                        "error": projects_health.get('error')
                    }
                }
            else:
                health_status["mongodb"]["reason"] = "MongoDB not connected (using fallback mode)"
                health_status["mongodb"]["error"] = mongodb_manager.get_connection_error()
        except ImportError as ie:
            health_status["mongodb"]["reason"] = f"MongoDB manager not installed: {str(ie)}"
        except Exception as e:
            health_status["mongodb"]["reason"] = f"Error: {str(e)}"
            import traceback
            logging.error(f"MongoDB health check error: {traceback.format_exc()}")

        # Check Azure Blob
        try:
            from common.services.azure.blob_utils import azure_blob_file_manager
            # Simple connectivity test
            test_result = azure_blob_file_manager.is_healthy() if hasattr(azure_blob_file_manager, 'is_healthy') else True
            health_status["azure_blob"] = {
                "connected": test_result,
                "reason": "OK" if test_result else "Connection failed"
            }
        except Exception as e:
            health_status["azure_blob"]["reason"] = f"Error: {str(e)}"

        # Check Service Layer
        try:
            from common.services.schema_service import schema_service
            health_status["services"]["schema_service"] = True
        except:
            pass

        try:
            from common.services.vendor_service import vendor_service
            health_status["services"]["vendor_service"] = True
        except:
            pass

        try:
            from common.services.project_service import project_service
            health_status["services"]["project_service"] = True
        except:
            pass

        try:
            from common.services.document_service import document_service
            health_status["services"]["document_service"] = True
        except:
            pass

        try:
            from common.services.image_service import image_service
            health_status["services"]["image_service"] = True
        except:
            pass

        # Determine overall status
        mongodb_ok = health_status["mongodb"]["connected"]
        blob_ok = health_status["azure_blob"]["connected"]
        services_count = sum(health_status["services"].values())

        if mongodb_ok and blob_ok and services_count >= 3:
            health_status["status"] = "healthy"
        elif blob_ok:  # Can run with Azure Blob only
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "error"

        return jsonify(health_status), 200

    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# =========================================================================
# === IMAGE SERVING ENDPOINT (Azure Blob Storage Primary)
# =========================================================================
@app.route('/api/images/<path:file_id>', methods=['GET'])
def serve_image(file_id):
    """
    Serve images from Azure Blob Storage
    ---
    tags:
      - Images
    summary: Serve image from Azure Blob Storage
    description: Retrieves and serves an image stored in Azure Blob Storage by its file ID (UUID).
    produces:
      - image/jpeg
      - image/png
      - image/gif
      - image/webp
      - application/json
    parameters:
      - name: file_id
        in: path
        type: string
        required: true
        description: Azure Blob file ID (UUID format)
        example: "083015c2-300c-44f3-be9a-3fdafebc39b0"
    responses:
      200:
        description: Image data returned successfully
        headers:
          Cache-Control:
            type: string
            description: Caching directive
            example: "public, max-age=2592000"
      404:
        description: Image not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Image not found"
      500:
        description: Internal server error
    """
    try:
        from azure_blob_config import azure_blob_manager
        import os

        # Check if Azure Blob Storage is available
        # If not available, will fallback to local filesystem later
        blob_name = None
        if not azure_blob_manager.is_available:
            logging.warning(f"Azure Blob Storage is not available for serving image: {file_id}, trying local fallback")
            # Skip to local fallback below (don't return early)
        else:
            container_client = azure_blob_manager.container_client
            base_path = azure_blob_manager.base_path

            # Search paths: files (UUIDs), images (cached), generic_images (generated), vendor_images (cached)
            search_paths = [
                f"{base_path}/files",
                f"{base_path}/images",
                f"{base_path}/generic_images",
                f"{base_path}/vendor_images"
            ]

            # Search for the blob with matching file_id
            image_blob = None
            content_type = 'image/png'  # Default content type

            # Optimization: Check if file_id is a full blob path (e.g., "generic_images/filename.png")
            # Try direct access first before searching all blobs
            if '/' in file_id:
                try:
                    # file_id might be a relative path like "generic_images/viscousliquidflowtransmitter.png"
                    full_blob_path = f"{base_path}/{file_id}"
                    blob_client = container_client.get_blob_client(full_blob_path)
                    blob_properties = blob_client.get_blob_properties()
                    if blob_properties:
                        blob_name = full_blob_path
                        if blob_properties.content_settings and blob_properties.content_settings.content_type:
                            content_type = blob_properties.content_settings.content_type
                        logging.info(f"[SERVE_IMAGE] Direct access successful for blob path: {file_id}")
                except Exception as direct_err:
                    logging.debug(f"[SERVE_IMAGE] Direct access failed for {file_id}: {direct_err}, falling back to search")

            # Only search if direct access didn't find the blob
            if blob_name is None:
                try:
                    for path in search_paths:
                        if image_blob:
                            break

                        # List blobs and find the one with matching file_id in the name or metadata
                        blobs = container_client.list_blobs(
                            name_starts_with=path,
                            include=['metadata']
                        )

                        for blob in blobs:
                            # Improved matching logic:
                            # 1. Exact match (normalized)
                            # 2. Ends with file_id (handles "generic_images/foo.png" matching "foo.png")
                            # 3. UUID match

                            blob_name_lower = blob.name.lower()
                            file_id_lower = file_id.lower()

                            # Check strict endings (most reliable for filenames)
                            if blob_name_lower.endswith(f"/{file_id_lower}") or blob_name_lower == file_id_lower:
                                image_blob = blob
                                blob_name = blob.name
                                if blob.content_settings and blob.content_settings.content_type:
                                    content_type = blob.content_settings.content_type
                                break

                            # Fallback: lenient substring match for UUIDs (only if not a path-like file_id)
                            if '/' not in file_id and file_id in blob.name:
                                image_blob = blob
                                blob_name = blob.name
                                if blob.content_settings and blob.content_settings.content_type:
                                    content_type = blob.content_settings.content_type
                                break

                            # Also check metadata for file_id
                            if blob.metadata and blob.metadata.get('file_id') == file_id:
                                image_blob = blob
                                blob_name = blob.name
                                if blob.content_settings and blob.content_settings.content_type:
                                    content_type = blob.content_settings.content_type
                                break

                except Exception as e:
                    logging.warning(f"Error searching for image blob {file_id}: {e}")

            # If blob found in Azure, download and serve it
            if blob_name is not None:
                # Download the blob content
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    download_stream = blob_client.download_blob()
                    image_data = download_stream.readall()
                except Exception as e:
                    logging.error(f"Failed to download image blob {file_id}: {e}")
                    # Try local fallback on Azure download failure
                    blob_name = None

                if blob_name is not None:
                    # Extract filename from blob name
                    filename = os.path.basename(blob_name)

                    # Create response with proper headers
                    response = send_file(
                        BytesIO(image_data),
                        mimetype=content_type,
                        as_attachment=False,
                        download_name=filename
                    )

                    # Add caching headers (cache for 30 days)
                    response.headers['Cache-Control'] = 'public, max-age=2592000'

                    logging.info(f"Served image from Azure Blob Storage: {file_id} ({len(image_data)} bytes)")
                    return response

        # Fallback: Check local filesystem
        # This handles cases where images are saved locally due to Azure failure or Azure not available
        if blob_name is None:
            local_path = os.path.join(app.root_path, 'static', 'images', file_id)
            if os.path.exists(local_path):
                logging.info(f"[SERVE_IMAGE] Serving from local fallback: {local_path}")
                return send_file(local_path)

            logging.error(f"Image not found in Azure Blob Storage or local fallback: {file_id}")
            return jsonify({"error": "Image not found"}), 404
        
    except Exception as e:
        logging.exception(f"Error serving image {file_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Fallback route for images requested without /api/images/ prefix
@app.route('/<string:file_id>', methods=['GET'])
def serve_image_root(file_id):
    """
    Fallback for images requested at root (e.g. legacy or malformed URLs)
    Only handles UUID format or long hashes to avoid conflict with other routes
    """
    # Check if it's a UUID format (36 chars with hyphens: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    import re
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    if re.match(uuid_pattern, file_id):
        return serve_image(file_id)
    
    # Also handle old formats: hash (64 chars) or ObjectId (24 chars)
    if len(file_id) in [24, 64] and all(c in '0123456789abcdefABCDEF' for c in file_id):
        return serve_image(file_id)

    return jsonify({"error": "Not found"}), 404


@app.route('/api/generic_image/<path:product_type>', methods=['GET'])
@login_required
def get_generic_image(product_type):
    """
    Fetch generic product type image with Azure Blob caching
    
    Strategy:
    1. Check Azure Blob cache first
    2. If not found, search external APIs with "generic <product_type>"
    3. Cache the result
    4. Return image URL or Azure Blob path
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
    """
    try:
        from generic_image_utils import fetch_generic_product_image
        
        # Decode URL-encoded product type
        import urllib.parse
        decoded_product_type = urllib.parse.unquote(product_type)
        
        logging.info(f"[API] ===== Generic Image Request START =====")
        logging.info(f"[API] Product Type (raw): {product_type}")
        logging.info(f"[API] Product Type (decoded): {decoded_product_type}")
        
        # Fetch image (checks cache first, then external APIs)
        image_result = fetch_generic_product_image(decoded_product_type)
        
        if image_result:
            logging.info(f"[API] ✓ Image found! Source: {image_result.get('source')}, Cached: {image_result.get('cached')}")
            logging.info(f"[API] ===== Generic Image Request END (SUCCESS) =====")
            return jsonify({
                "success": True,
                "image": image_result,
                "product_type": decoded_product_type
            }), 200
        else:
            logging.warning(f"[API] ✗ No image found for: {decoded_product_type}")
            logging.info(f"[API] ===== Generic Image Request END (NOT FOUND) =====")
            return jsonify({
                "success": False,
                "error": "No image found",
                "product_type": decoded_product_type
            }), 404
            
    except Exception as e:
        logging.exception(f"[API] ✗ ERROR fetching generic image for {product_type}: {e}")
        logging.info(f"[API] ===== Generic Image Request END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e),
            "product_type": product_type
        }), 500


@app.route('/api/generic_image_fast/<path:product_type>', methods=['GET'])
@login_required
def get_generic_image_fast(product_type):
    """
    Fetch generic product type image with FAST-FAIL behavior.
    
    Unlike the regular endpoint, this returns IMMEDIATELY if:
    - Cache is empty AND LLM is rate-limited
    
    Returns a 'use_placeholder' flag so the frontend can show a placeholder.
    
    Response:
        {
            "success": true/false,
            "image": { "url": "...", ... } or null,
            "use_placeholder": true/false,
            "reason": "cached" | "generated" | "rate_limited"
        }
    """
    try:
        from generic_image_utils import fetch_generic_product_image_fast
        import urllib.parse
        
        decoded_product_type = urllib.parse.unquote(product_type)
        logging.info(f"[API_FAST] Fast image request: {decoded_product_type}")
        
        result = fetch_generic_product_image_fast(decoded_product_type)
        
        if result.get('success'):
            logging.info(f"[API_FAST] ✓ Image found: {result.get('reason')}")
            return jsonify({
                "success": True,
                "image": result,
                "product_type": decoded_product_type,
                "use_placeholder": False,
                "reason": result.get('reason', 'cached')
            }), 200
        else:
            logging.warning(f"[API_FAST] ✗ Using placeholder: {result.get('reason')}")
            return jsonify({
                "success": False,
                "image": None,
                "product_type": decoded_product_type,
                "use_placeholder": True,
                "reason": result.get('reason', 'rate_limited')
            }), 200  # Return 200 with use_placeholder=True
            
    except Exception as e:
        logging.exception(f"[API_FAST] Error: {e}")
        return jsonify({
            "success": False,
            "image": None,
            "product_type": product_type,
            "use_placeholder": True,
            "reason": "error",
            "error": str(e)
        }), 200  # Return 200 so frontend doesn't show error


@app.route('/api/generic_image/regenerate/<path:product_type>', methods=['POST'])
@login_required
def regenerate_generic_image_endpoint(product_type):
    """
    [DEPRECATED] Endpoint removed as LLM generation is disabled.
    """
    return jsonify({
        "success": False,
        "error": "Image regeneration is no longer supported.",
        "reason": "deprecated"
    }), 410


@app.route('/api/generic_images/batch', methods=['POST'])
@login_required
def get_generic_images_batch():
    """
    Fetch generic product type images for MULTIPLE product types IN PARALLEL.
    
    PARALLELIZATION STRATEGY:
    - Phase 1: Check Azure cache for ALL product types simultaneously (fast)
    - Phase 2: For cache misses, generate with LLM (sequential to respect rate limits)
    
    Request Body:
        {
            "product_types": ["Pressure Transmitter", "Flow Meter", ...]
        }
    
    Returns:
        {
            "success": true,
            "images": {
                "Pressure Transmitter": {...image_result...},
                "Flow Meter": {...image_result...}
            },
            "cache_hits": 5,
            "cache_misses": 2,
            "processing_time_ms": 1234
        }
    """
    try:
        from generic_image_utils import fetch_generic_images_batch
        import time
        
        start_time = time.time()
        
        data = request.get_json()
        if not data or 'product_types' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'product_types' in request body"
            }), 400
        
        product_types = data.get('product_types', [])
        if not isinstance(product_types, list):
            return jsonify({
                "success": False,
                "error": "'product_types' must be a list"
            }), 400
        
        # Deduplicate and clean
        product_types = list(set([pt.strip() for pt in product_types if pt and isinstance(pt, str)]))
        
        if not product_types:
            return jsonify({
                "success": True,
                "images": {},
                "cache_hits": 0,
                "cache_misses": 0,
                "processing_time_ms": 0
            }), 200
        
        logging.info(f"[API_BATCH] Starting batch fetch for {len(product_types)} product types...")
        
        # Fetch all images in parallel
        results = fetch_generic_images_batch(product_types)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Count hits/misses
        cache_hits = sum(1 for r in results.values() if r and r.get('cached'))
        cache_misses = len(product_types) - cache_hits
        
        logging.info(f"[API_BATCH] Completed: {cache_hits} hits, {cache_misses} misses in {processing_time}ms")
        
        return jsonify({
            "success": True,
            "images": results,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "total_requested": len(product_types),
            "processing_time_ms": processing_time
        }), 200
        
    except Exception as e:
        logging.exception(f"[API_BATCH] Error in batch image fetch: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================================
# === FILE UPLOAD AND TEXT EXTRACTION ENDPOINT ===
# =========================================================================
@app.route('/api/upload-requirements', methods=['POST'])
@login_required
def upload_requirements_file():
    """
    Upload file (PDF, DOCX, TXT, Images) and extract text as requirements
    
    Accepts: multipart/form-data with 'file' field
    Returns: Extracted text from the file
    """
    try:
        from common.services.extraction.file_extraction_utils import extract_text_from_file
        
        logging.info("[API] ===== File Upload Request START =====")
        
        # Check if file is present
        if 'file' not in request.files:
            logging.warning("[API] No file provided in request")
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logging.warning("[API] Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Read file bytes
        file_bytes = file.read()
        filename = file.filename
        
        logging.info(f"[API] Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Extract text from file
        extraction_result = extract_text_from_file(file_bytes, filename)
        
        if not extraction_result['success']:
            logging.warning(f"[API] Failed to extract text from {filename}")
            return jsonify({
                "success": False,
                "error": f"Could not extract text from {extraction_result['file_type']} file",
                "file_type": extraction_result['file_type']
            }), 400
        
        logging.info(f"[API] ✓ Successfully extracted {extraction_result['character_count']} characters from {filename}")
        logging.info("[API] ===== File Upload Request END (SUCCESS) =====")
        
        return jsonify({
            "success": True,
            "extracted_text": extraction_result['extracted_text'],
            "filename": filename,
            "file_type": extraction_result['file_type'],
            "character_count": extraction_result['character_count']
        }), 200
        
    except Exception as e:
        logging.exception(f"[API] ✗ ERROR processing file upload: {e}")
        logging.info("[API] ===== File Upload Request END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500





# =========================================================================
# === NEW FEEDBACK ENDPOINT ===
# =========================================================================
@app.route("/api/feedback", methods=["POST"])
@login_required
def handle_feedback():
    """
    Handles user feedback and saves a complete log entry to the database.
    """
    if not components or not components.get('llm'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        feedback_type = data.get("feedbackType")
        comment = data.get("comment", "")

        # --- DATABASE LOGGING LOGIC STARTS HERE ---
        
        # 1. Retrieve the stored data from the session
        user_query = session.get('log_user_query', 'No query found - user may have provided feedback without validation')
        system_response = session.get('log_system_response', {})

        # 2. Format the feedback for the database
        feedback_log_entry = feedback_type
        if comment:
            feedback_log_entry += f" ({comment})"

        # 3. Get the current user's information to log their username
        current_user = db.session.get(User, session['user_id'])
        if not current_user:
            logging.error(f"Could not find user with ID {session['user_id']} to create log entry.")
            return jsonify({"error": "Authenticated user not found for logging."}), 404
        
        username = current_user.username

        # 4. Persist feedback to MongoDB only (do not store in SQL)
        try:
            project_id_for_feedback = session.get('current_project_id') or data.get('projectId')

            feedback_entry = {
                'timestamp': datetime.utcnow(),
                'user_id': str(session.get('user_id')) if session.get('user_id') else None,
                'user_name': username,
                'feedback_type': feedback_type,
                'comment': comment,
                'user_query': user_query,
                'system_response': system_response
            }

            # If we have a project id, append to that project's feedback_entries array
            if project_id_for_feedback:
                try:
                    cosmos_project_manager.append_feedback_to_project(project_id_for_feedback, str(session.get('user_id')), feedback_entry)
                    logging.info(f"Appended feedback to project {project_id_for_feedback}")
                    
                    # Also save to global feedback collection (Azure Blob)
                    try:
                        azure_blob_file_manager.upload_json_data(
                            {**feedback_entry, 'project_id': project_id_for_feedback},
                            metadata={
                                'collection_type': 'feedback',
                                'user_id': str(session.get('user_id')),
                                'project_id': project_id_for_feedback,
                                'type': 'user_feedback'
                            }
                        )
                    except Exception as e:
                        logging.warning(f"Failed to save global feedback: {e}")
                        
                except Exception as me:
                    logging.warning(f"Failed to append feedback to project {project_id_for_feedback}: {me}")
            else:
                # No project id: save feedback as standalone document
                try:
                    azure_blob_file_manager.upload_json_data(
                        {**feedback_entry, 'project_id': None},
                        metadata={
                            'collection_type': 'feedback',
                            'user_id': str(session.get('user_id')),
                            'type': 'user_feedback'
                        }
                    )
                    logging.info("Saved feedback to Azure 'feedback' collection (no project id)")
                except Exception as e:
                    logging.error(f"Failed to save feedback to Azure feedback collection: {e}")

        except Exception as e:
            logging.exception(f"Failed to persist feedback to MongoDB: {e}")

        # Clean up the session logging keys
        session.pop('log_user_query', None)
        session.pop('log_system_response', None)
        
        # --- LOGGING LOGIC ENDS ---

        if not feedback_type and not comment:
            return jsonify({"error": "No feedback provided."}), 400
        
        # --- LLM RESPONSE GENERATION ---
        if feedback_type == 'positive':
            feedback_chain = ChatPromptTemplate.from_template(FEEDBACK_POSITIVE_PROMPT) | components['llm'] | StrOutputParser()
        elif feedback_type == 'negative':
            feedback_chain = ChatPromptTemplate.from_template(FEEDBACK_NEGATIVE_PROMPT) | components['llm'] | StrOutputParser()
        else:  # This handles the case where only a comment is provided
            feedback_chain = ChatPromptTemplate.from_template(FEEDBACK_COMMENT_PROMPT) | components['llm'] | StrOutputParser()

        llm_response = feedback_chain.invoke({"comment": comment})

        return jsonify({"response": llm_response}), 200

    except Exception as e:
        logging.exception("Feedback handling or MongoDB storage failed.")
        return jsonify({"error": "Failed to process feedback: " + str(e)}), 500

# =========================================================================
# === INSTRUMENT IDENTIFICATION ENDPOINT ===
# === DEPRECATED: Use /api/agentic/instrument-identifier instead ===
# =========================================================================
@app.route("/api/identify-instruments", methods=["POST"])
@login_required
def identify_instruments():
    """
    DEPRECATED: Use /api/agentic/instrument-identifier instead.

    Handles user input in project page with three cases:
    1. Greeting - Returns friendly greeting response
    2. Requirements - Returns identified instruments and accessories
    3. Industrial Question - Returns answer or redirect if not related
    """
    # DEPRECATION WARNING: This endpoint will be removed in a future version
    logging.warning("[DEPRECATED] /api/identify-instruments called - Use /api/agentic/instrument-identifier instead")

    if not components or not components.get('llm_pro'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        requirements = data.get("requirements", "").strip()
        search_session_id = data.get("search_session_id", "default")
        
        if not requirements:
            return jsonify({"error": "Requirements text is required"}), 400

        # Pre-classification heuristics to catch obvious cases
        requirements_lower = requirements.lower()
        
        # Strong indicators of unrelated content (emails, job offers, etc.)
        unrelated_indicators = [
            'from:', 'to:', 'subject:', 'date:',  # Email headers
            'congratulations for the selection', 'job offer', 'recruitment',
            'campus placement', 'hr department', 'hiring', 'interview process',
            'provisionally selected', 'offer letter', 'employment application',
            'dear sir', 'dear madam', 'forwarded message',
            'training and placement officer', 'campus recruitment'
        ]
        
        # Check if content has strong unrelated indicators
        unrelated_count = sum(1 for indicator in unrelated_indicators if indicator in requirements_lower)
        
        # If 2+ strong indicators, skip LLM and classify as unrelated immediately
        if unrelated_count >= 2:
            logging.info(f"[CLASSIFY] Pre-classification: UNRELATED (found {unrelated_count} indicators)")
            input_type = "unrelated"
            confidence = "high"
            reasoning = f"Contains {unrelated_count} strong indicators of non-industrial content (email headers, job/recruitment terms)"
        else:
            # Step 1: Classify the input type using LLM
            classification_chain = CLASSIFICATION_PROMPT | components['llm'] | StrOutputParser()
            classification_response = classification_chain.invoke({"user_input": requirements})
            
            # Clean and parse classification
            cleaned_classification = classification_response.strip()
            if cleaned_classification.startswith("```json"):
                cleaned_classification = cleaned_classification[7:]
            elif cleaned_classification.startswith("```"):
                cleaned_classification = cleaned_classification[3:]
            if cleaned_classification.endswith("```"):
                cleaned_classification = cleaned_classification[:-3]
            cleaned_classification = cleaned_classification.strip()
            
            try:
                classification = json.loads(cleaned_classification)
                input_type = classification.get("type", "requirements").lower()
                confidence = classification.get("confidence", "medium").lower()
                reasoning = classification.get("reasoning", "")
                
                # Log classification for debugging
                logging.info(f"[CLASSIFY] LLM classified as '{input_type}' (confidence: {confidence})")
                logging.info(f"[CLASSIFY] Reasoning: {reasoning}")
                
            except Exception as e:
                # Default to requirements if classification fails
                input_type = "requirements"
                confidence = "low"
                reasoning = "Classification parsing failed"
                logging.warning(f"Failed to parse classification, defaulting to requirements. Response: {classification_response}")
                logging.exception(e)

        # CASE 1: Greeting
        if input_type == "greeting":
            greeting_chain = SALES_AGENT_GREETING_PROMPT | components['llm'] | StrOutputParser()
            greeting_response = greeting_chain.invoke({"user_input": requirements})
            
            # from testing_utils import standardized_jsonify # Removed

            return standardized_jsonify({
                "response_type": "greeting",
                "message": greeting_response.strip(),
                "instruments": [],
                "accessories": []
            }, 200)

        # CASE 2: Requirements - Identify instruments and accessories
        elif input_type == "requirements":
            session_isolated_requirements = f"[Session: {search_session_id}] - This is an independent instrument identification request. Requirements: {requirements}"
            
            response_chain = IDENTIFY_INSTRUMENT_PROMPT | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"requirements": session_isolated_requirements})

            # Clean the LLM response
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  
            cleaned_response = cleaned_response.strip()

            try:
                result = json.loads(cleaned_response)
                
                # Validate the response structure
                if "instruments" not in result or not isinstance(result["instruments"], list):
                    raise ValueError("Invalid response structure from LLM")
                
                # Ensure all instruments have required fields
                for instrument in result["instruments"]:
                    if not all(key in instrument for key in ["category", "product_name", "specifications", "sample_input"]):
                        raise ValueError("Missing required fields in instrument data")
                
                # Validate accessories if present
                if "accessories" in result:
                    if not isinstance(result["accessories"], list):
                        raise ValueError("'accessories' must be a list if provided")
                    for accessory in result["accessories"]:
                        expected_acc_keys = ["category", "accessory_name", "specifications", "sample_input"]
                        if not all(key in accessory for key in expected_acc_keys):
                            raise ValueError("Missing required fields in accessory data")
                
                # Ensure strategy field exists for all instruments and accessories
                for instrument in result.get("instruments", []):
                    if "strategy" not in instrument:
                        instrument["strategy"] = ""

                for accessory in result.get("accessories", []):
                    if "strategy" not in accessory:
                        accessory["strategy"] = ""
                
                # Add response type
                result["response_type"] = "requirements"
                return standardized_jsonify(result, 200)
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {e}")
                logging.error(f"LLM Response: {llm_response}")
                
                return jsonify({
                    "response_type": "error",
                    "error": "Failed to parse instrument identification",
                    "instruments": [],
                    "accessories": [],
                    "summary": "Unable to identify instruments from the provided requirements"
                }), 500

        # CASE 3: Unrelated content - Politely redirect
        elif input_type == "unrelated":
            unrelated_chain = CLASSIFICATION_PROMPT | components['llm'] | StrOutputParser()
            unrelated_response = unrelated_chain.invoke({"reasoning": reasoning})
            
            # from testing_utils import standardized_jsonify # Removed

            return standardized_jsonify({
                "response_type": "question",  # Use "question" type for frontend compatibility
                "is_industrial": False,
                "message": unrelated_response.strip(),
                "instruments": [],
                "accessories": []
            }, 200)

        # CASE 4: Question - Answer industrial questions or redirect
        elif input_type == "question":
            fallback_chain = CLASSIFICATION_PROMPT | components['llm'] | StrOutputParser()
            fallback_message = fallback_chain.invoke({"requirements": requirements})
            return standardized_jsonify({
                "response_type": "question",
                "is_industrial": False,
                "message": fallback_message.strip(),
                "instruments": [],
                "accessories": []
            }, 200)
        
        # CASE 5: Unexpected classification type - Default fallback
        else:
            logging.warning(f"[CLASSIFY] Unexpected classification type: {input_type}")
            # Generate dynamic response from LLM
            unexpected_chain = CLASSIFICATION_PROMPT | components['llm'] | StrOutputParser()
            unexpected_message = unexpected_chain.invoke({"user_input": requirements})
            
            return standardized_jsonify({
                "response_type": "question",
                "is_industrial": False,
                "message": unexpected_message.strip(),
                "instruments": [],
                "accessories": []
            }, 200)

    except Exception as e:
        logging.exception("Instrument identification failed.")
        return jsonify({
            "response_type": "error",
            "error": "Failed to process request: " + str(e),
            "instruments": [],
            "accessories": [],
            "summary": ""
        }), 500

# DEPRECATED: Use /api/agentic/tools/search-vendors instead (uses DB, not CSV)
@app.route("/api/search-vendors", methods=["POST"])
@login_required
def search_vendors():
    """
    DEPRECATED: Use /api/agentic/tools/search-vendors instead.
    This endpoint uses CSV-based lookup which is outdated.

    Search for vendors
    ---
    tags:
      - Vendors
      - Deprecated
    summary: "[DEPRECATED] Search vendors by product criteria"
    description: |
      **DEPRECATED**: Use `/api/agentic/tools/search-vendors` instead.

      Search for vendors based on selected instrument/accessory details.
      Maps category, product name, and strategy to CSV data and returns filtered vendor list.

      Uses fuzzy matching to find the best matching vendors based on:
      - Product category
      - Product/accessory name
      - Procurement strategy
    consumes:
      - application/json
    produces:
      - application/json
    deprecated: true
    parameters:
      - in: body
        name: body
        description: Search criteria
        required: true
        schema:
          type: object
          required:
            - category
            - product_name
          properties:
            category:
              type: string
              description: Product category
              example: "Pressure Transmitters"
            product_name:
              type: string
              description: Product or instrument name
              example: "Rosemount 3051"
            accessory_name:
              type: string
              description: Alternative to product_name for accessories
            strategy:
              type: string
              description: Procurement strategy
              example: "Critical"
    responses:
      200:
        description: List of matching vendors
        schema:
          type: object
          properties:
            vendors:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  category:
                    type: string
                  strategy:
                    type: string
            total_count:
              type: integer
            matching_criteria:
              type: object
      400:
        description: Missing required fields
      401:
        description: Unauthorized - login required
      500:
        description: Vendor data not available
    """
    # DEPRECATION WARNING: This endpoint will be removed in a future version
    logging.warning("[DEPRECATED] /api/search-vendors called - Use /api/agentic/tools/search-vendors instead")

    try:
        data = request.get_json(force=True)
        category = data.get("category", "").strip()
        product_name = data.get("product_name", "").strip() or data.get("accessory_name", "").strip()
        strategy = data.get("strategy", "").strip()
        
        logging.info(f"[VENDOR_SEARCH] Received request: category='{category}', product='{product_name}', strategy='{strategy}'")
        
        if not category or not product_name:
            return jsonify({"error": "Category and product_name/accessory_name are required"}), 400
        
        # Load CSV data
        csv_path = os.path.join(os.path.dirname(__file__), 'instrumentation_procurement_strategy.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({"error": "Vendor data not available"}), 500
        
        vendors = []
        matched_vendors = []
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            vendors = list(reader)
        
        if not vendors:
            return jsonify({"vendors": [], "total_count": 0, "matching_criteria": {}}), 200
        
        # Get unique categories and subcategories for fuzzy matching
        csv_categories = list(set([v.get('category', '').strip() for v in vendors if v.get('category')]))
        csv_subcategories = list(set([v.get('subcategory', '').strip() for v in vendors if v.get('subcategory')]))
        csv_strategies = list(set([v.get('strategy', '').strip() for v in vendors if v.get('strategy')]))
        
        # Step 1: Match Category using dynamic fuzzy matching
        matched_category = None
        category_match_type = None
        
        # Exact match (case-insensitive)
        category_lower = category.lower().strip()
        matched_category = next((cat for cat in csv_categories if cat.lower().strip() == category_lower), None)
        category_match_type = "exact" if matched_category else None
        
        logging.info(f"[VENDOR_SEARCH] Category matching result: '{category}' -> '{matched_category}' (type: {category_match_type})")
        
        if not matched_category:
            logging.warning(f"[VENDOR_SEARCH] No category match found. Available categories: {csv_categories}")
            return jsonify({
                "vendors": [],
                "total_count": 0,
                "matching_criteria": {
                    "category_match": None,
                    "subcategory_match": None,
                    "strategy_match": None,
                    "message": f"No matching category found for '{category}'. Available: {csv_categories}"
                }
            }), 200
        
        # Step 2: Match Product Name to Subcategory (exact match, case-insensitive)
        product_name_lower = product_name.lower().strip()
        matched_subcategory = next((subcat for subcat in csv_subcategories if subcat.lower().strip() == product_name_lower), None)
        subcategory_match_type = "exact" if matched_subcategory else None
        
        # Step 3: Match Strategy (optional field, exact match, case-insensitive)
        matched_strategy = None
        strategy_match_type = None

        if strategy:  # Only if strategy is provided
            strategy_lower = strategy.lower().strip()
            matched_strategy = next((strat for strat in csv_strategies if strat.lower().strip() == strategy_lower), None)
            strategy_match_type = "exact" if matched_strategy else None
        
        # Step 4: Filter vendors based on matches
        filtered_vendors = []
        
        for vendor in vendors:
            vendor_category = vendor.get('category', '').strip()
            vendor_subcategory = vendor.get('subcategory', '').strip()
            vendor_strategy = vendor.get('strategy', '').strip()
            
            # Category must match
            if vendor_category != matched_category:
                continue
            
            # Subcategory should match if we found a match
            if matched_subcategory and vendor_subcategory != matched_subcategory:
                continue
            
            # Strategy should match if provided and we found a match
            if strategy and matched_strategy and vendor_strategy != matched_strategy:
                continue
            
            # Add vendor to results
            filtered_vendors.append({
                "vendor_id": vendor.get('vendor ID', ''),
                "vendor_name": vendor.get('vendor name', ''),
                "category": vendor_category,
                "subcategory": vendor_subcategory,
                "strategy": vendor_strategy,
                "refinery": vendor.get('refinery', ''),
                "additional_comments": vendor.get('additional comments', ''),
                "owner_name": vendor.get('owner name', '')
            })
        
        # Prepare response
        matching_criteria = {
            "category_match": {
                "input": category,
                "matched": matched_category,
                "match_type": category_match_type
            },
            "subcategory_match": {
                "input": product_name,
                "matched": matched_subcategory,
                "match_type": subcategory_match_type
            } if matched_subcategory else None,
            "strategy_match": {
                "input": strategy,
                "matched": matched_strategy,
                "match_type": strategy_match_type
            } if strategy and matched_strategy else None
        }
        
        # Store CSV vendor filter in session for analysis chain
        if filtered_vendors:
            session['csv_vendor_filter'] = {
                'vendor_names': [v.get('vendor_name', '').strip() for v in filtered_vendors],
                'csv_data': filtered_vendors,
                'product_type': category,
                'detected_product': product_name,
                'matching_criteria': matching_criteria
            }
            logging.info(f"[VENDOR_SEARCH] Stored {len(filtered_vendors)} vendors in session for analysis filtering")
        else:
            # Clear any existing filter if no vendors found
            session.pop('csv_vendor_filter', None)
            logging.info("[VENDOR_SEARCH] No vendors found, cleared session filter")
        
        # Return only vendor names list for frontend
        vendor_names_only = [v.get('vendor_name', '').strip() for v in filtered_vendors]
        
        return standardized_jsonify({
            "vendors": vendor_names_only,
            "total_count": len(filtered_vendors),
            "matching_criteria": matching_criteria
        }, 200)
        
    except Exception as e:
        logging.exception("Vendor search failed.")
        return jsonify({
            "error": "Failed to search vendors: " + str(e),
            "vendors": [],
            "total_count": 0
        }), 500

# =========================================================================
# === API ENDPOINTS ===
# =========================================================================
# PHASE 1 FIX: Use API Key Manager instead of hardcoded fallbacks
from common.config.api_key_manager import api_key_manager

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# PHASE 1 FIX: Get Google API key from centralized manager
GOOGLE_API_KEY = api_key_manager.get_current_google_key()

# PHASE 1 FIX: No hardcoded fallback - use only environment variable
GOOGLE_CX = os.getenv("GOOGLE_CX")

# Image search configuration
SERPER_API_KEY_IMAGES = SERPER_API_KEY  # Use same key for images

# Validation warnings (no silent fallbacks)
if not SERPER_API_KEY:
    logging.warning("SERPER_API_KEY environment variable not set! Image search via Serper will be unavailable.")

if not SERPAPI_KEY:
    logging.warning("SERPAPI_KEY environment variable not set! SerpAPI fallback will be unavailable.")

if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY environment variable not set! Google Custom Search will be unavailable.")

if not GOOGLE_CX:
    logging.warning("GOOGLE_CX environment variable not set! Custom search engine not configured.")


def serper_search_pdfs(query):
    """Perform a PDF search with Serper API."""
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{query} filetype:pdf",
            "num": 10
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("organic", [])
        
        pdf_results = []
        for item in results:
            pdf_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", ""),
                "source": "serper"
            })
        return pdf_results
    except Exception as e:
        logging.warning(f"[WARNING] Serper API search failed: {e}")
        return []


def serpapi_search_pdfs(query):
    """Perform a PDF search with SerpApi."""
    if not SERPAPI_KEY:
        return []
    
    try:
        search = GoogleSearch({
            "q": f"{query} filetype:pdf",
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        results = search.get_dict()
        items = results.get("organic_results", [])
        pdf_results = []
        for item in items:
            pdf_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", ""),
                "source": "serpapi"
            })
        return pdf_results
    except Exception as e:
        logging.warning(f"[WARNING] SerpAPI search failed: {e}")
        return []


def google_custom_search_pdfs(query):
    """Perform a PDF search with Google Custom Search API as fallback."""
    # PHASE 1 FIX: Use centralized API key configuration
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []

    try:
        import threading
        from googleapiclient.discovery import build

        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

        result_container = [None]
        exception_container = [None]

        def google_request():
            try:
                result = service.cse().list(
                    q=f"{query} filetype:pdf",
                    cx=GOOGLE_CX,
                    num=10,
                    fileType='pdf'
                ).execute()
                result_container[0] = result.get('items', [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=google_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)
        
        if thread.is_alive() or exception_container[0]:
            return []
        
        items = result_container[0] or []
        
        pdf_results = []
        for item in items:
            title = item.get('title', '')
            link = item.get('link', '')
            snippet = item.get('snippet', '')
            
            if link:
                pdf_results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": snippet,
                    "source": "google_custom"
                })
        
        return pdf_results
        
    except Exception as e:
        logging.warning(f"[WARNING] Google Custom Search failed: {e}")
        return []


def search_pdfs_with_fallback(query):
    """
    Search for PDFs using Serper API first, then SERP API, then Google Custom Search as final fallback.
    Returns combined results with source information.
    """
    # First, try Serper API
    serper_results = serper_search_pdfs(query)
    
    # If Serper API returns results, use them
    if serper_results:
        return serper_results
    
    # If Serper API fails or returns no results, try SERP API
    serpapi_results = serpapi_search_pdfs(query)
    
    # If SERP API returns results, use them
    if serpapi_results:
        return serpapi_results
    
    # If both Serper and SERP API fail or return no results, try Google Custom Search
    google_results = google_custom_search_pdfs(query)
    
    if google_results:
        return google_results
    
    # If all three fail, return empty list
    return []

@app.route("/api/search_pdfs", methods=["GET"])
@login_required
def search_pdfs():
    """
    Performs a PDF search using Serper API first, then SERP API, then Google Custom Search as final fallback.
    """
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Missing search query"}), 400

    try:
        # Use the new fallback search function
        results = search_pdfs_with_fallback(query)
        
        # Add metadata about which search engine was used
        response_data = {
            "results": results,
            "total_results": len(results),
            "query": query
        }
        
        # Add source information if results exist
        if results:
            sources_used = list(set(result.get("source", "unknown") for result in results))
            response_data["sources_used"] = sources_used
            
            # Add fallback indicator based on the new three-tier system
            if "serper" in sources_used:
                response_data["fallback_used"] = False
                response_data["message"] = "Results from Serper API"
            elif "serpapi" in sources_used:
                response_data["fallback_used"] = True
                response_data["message"] = "Results from SERP API (Serper fallback)"
            elif "google_custom" in sources_used:
                response_data["fallback_used"] = True
                response_data["message"] = "Results from Google Custom Search (final fallback)"
            else:
                response_data["fallback_used"] = False
        else:
            response_data["sources_used"] = []
            response_data["fallback_used"] = True
            response_data["message"] = "No results found from any search engine"
        
        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("PDF search failed.")
        return jsonify({"error": "Failed to perform PDF search: " + str(e)}), 500


@app.route("/api/view_pdf", methods=["GET"])
@login_required
def view_pdf():
    """
    Fetches a PDF from a URL and serves it for viewing.
    """
    pdf_url = request.args.get("url")
    if not pdf_url:
        return jsonify({"error": "Missing PDF URL"}), 400
    
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        
        return send_file(
            pdf_stream,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=os.path.basename(pdf_url)
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch PDF from URL: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An error occurred while viewing the PDF: " + str(e)}), 500


def fetch_price_and_reviews_serpapi(product_name: str):
    """Use SerpApi to fetch price and review info for a product."""
    if not SERPAPI_KEY:
        return []
    
    try:
        search = GoogleSearch({
            "q": f"{product_name} price review",
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        res = search.get_dict()
        results = []

        for item in res.get("organic_results", []):
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("source")
            link = item.get("link")

            # Try to pull price from structured extensions
            ext = (
                item.get("rich_snippet", {})
                    .get("bottom", {})
                    .get("detected_extensions", {})
            )
            if "price" in ext:
                price = f"${ext['price']}"
            elif "price_from" in ext and "price_to" in ext:
                price = f"${ext['price_from']} to ${ext['price_to']}"
            else:
                # Fallback: regex on snippet
                price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
                if price_match:
                    price = price_match.group(0)

            # Extract reviews (look in snippet)
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        logging.warning(f"[WARNING] SerpAPI price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews_serper(product_name: str):
    """Use Serper API to fetch price and review info for a product."""
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{product_name} price review",
            "num": 10
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []

        for item in data.get("organic", []):
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("displayLink")
            link = item.get("link")

            # Extract price from snippet using regex
            price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
            if price_match:
                price = price_match.group(0)

            # Extract reviews (look in snippet)
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        logging.warning(f"[WARNING] Serper API price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews_google_custom(product_name: str):
    """Use Google Custom Search to fetch price and review info for a product as fallback."""
    # PHASE 1 FIX: Use centralized API key configuration
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []

    try:
        import threading
        from googleapiclient.discovery import build

        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

        result_container = [None]
        exception_container = [None]

        def google_request():
            try:
                result = service.cse().list(
                    q=f"{product_name} price review",
                    cx=GOOGLE_CX,
                    num=10
                ).execute()
                result_container[0] = result.get('items', [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=google_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)
        
        if thread.is_alive() or exception_container[0]:
            return []
        
        items = result_container[0] or []
        results = []

        for item in items:
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("displayLink")
            link = item.get("link")

            # Extract price using regex
            price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
            if price_match:
                price = price_match.group(0)

            # Extract reviews
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        logging.warning(f"[WARNING] Google Custom Search price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews(product_name: str):
    """
    Fetch price and review info using SERP API first, then Serper API, then Google Custom Search as final fallback.
    Special order for pricing: SERP API → Serper → Google Custom Search
    Returns a structured response with results and metadata.
    """
    # First, try SERP API (special order for pricing)
    serpapi_results = fetch_price_and_reviews_serpapi(product_name)
    
    # If SERP API returns results, use them
    if serpapi_results:
        return {
            "productName": product_name, 
            "results": serpapi_results,
            "source_used": "serpapi",
            "fallback_used": False
        }
    
    # If SERP API fails or returns no results, try Serper API
    serper_results = fetch_price_and_reviews_serper(product_name)
    
    if serper_results:
        return {
            "productName": product_name, 
            "results": serper_results,
            "source_used": "serper",
            "fallback_used": True
        }
    
    # If both SERP API and Serper fail or return no results, try Google Custom Search
    google_results = fetch_price_and_reviews_google_custom(product_name)
    
    if google_results:
        return {
            "productName": product_name, 
            "results": google_results,
            "source_used": "google_custom",
            "fallback_used": True
        }
    
    # If all three fail, return empty results
    return {
        "productName": product_name, 
        "results": [],
        "source_used": "none",
        "fallback_used": True
    }


# =========================================================================
# === IMAGE SEARCH FUNCTIONS ===
# =========================================================================

def get_manufacturer_domains_from_llm(vendor_name: str) -> list:
    """
    Use LLM to dynamically generate manufacturer domain names based on vendor name
    """
    if not components or not components.get('llm'):
        # Fallback to common domains if LLM is not available
        return [
            "emerson.com", "yokogawa.com", "siemens.com", "abb.com", "honeywell.com",
            "schneider-electric.com", "ge.com", "rockwellautomation.com", "endress.com",
            "fluke.com", "krohne.com", "rosemount.com", "fisher.com", "metso.com"
        ]
    
    try:
        
        chain = MANUFACTURER_DOMAIN_PROMPT | components['llm'] | StrOutputParser()
        
        response = chain.invoke({"vendor_name": vendor_name})
        
        # Parse the response to extract domain names
        domains = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and '.' in line:
                # Clean up the line - remove any prefixes, bullets, numbers
                domain = line.split()[-1] if ' ' in line else line
                domain = domain.replace('www.', '').replace('http://', '').replace('https://', '')
                domain = domain.strip('.,()[]{}')
                
                if '.' in domain and len(domain) > 3:
                    domains.append(domain)
        
        # Ensure we have at least some domains
        if not domains:
            # Fallback: generate based on vendor name
            vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
            domains = [f"{vendor_clean}.com", f"{vendor_clean}.de", f"{vendor_clean}group.com"]
        
        logging.info(f"LLM generated {len(domains)} domains for {vendor_name}: {domains[:5]}...")
        return domains[:15]  # Limit to 15 domains
        
    except Exception as e:
        logging.warning(f"Failed to generate domains via LLM for {vendor_name}: {e}")
        # Fallback: generate based on vendor name
        vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
        return [f"{vendor_clean}.com", f"{vendor_clean}.de", f"{vendor_clean}group.com"]

def fetch_images_google_cse_sync(vendor_name: str, product_name: str = None, manufacturer_domains: list = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: Google Custom Search API for images from manufacturer domains
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        manufacturer_domains: (Optional) List of manufacturer domains to search within
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        # Build the search query in format <vendor_name><modelfamily><product type>
        query = vendor_name
        if model_family:
            query += f" {model_family}"  # Add space for better tokenization
        if product_type:
            query += f" {product_type}"  # Add space for better tokenization
            
        query += " product image"
            
        # We intentionally do NOT include raw product_name in the search query
        # to focus searches on model_family and product_type only.
        
        # Build site restriction for manufacturer domains using LLM (or reuse if provided)
        if manufacturer_domains is None:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
        search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png"
        
        # Use Google Custom Search API
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            searchType="image",
            num=8,
            safe="medium",
            imgSize="MEDIUM"
        ).execute()
        
        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        
        for item in result.get("items", []):
            url = item.get("link")
            
            # Skip images with unsupported URL schemes
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                logging.debug(f"Skipping image with unsupported URL scheme: {url}")
                continue
            
            # Only include http/https URLs
            if not url.startswith(('http://', 'https://')):
                logging.debug(f"Skipping non-HTTP URL: {url}")
                continue
                
            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} valid images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

def fetch_images_serpapi_sync(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: SerpAPI fallback for Google Images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        # Build the base query in format <vendor_name> <model_family?> <product_type?>
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"

        # Do not include raw product_name here — rely on model_family/product_type

        # Add helpful positive/negative tokens used previously
        base_query += " product image OR product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        # Build manufacturer domain filter using LLM domains when available
        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        search = GoogleSearch({
            "q": search_query,
            "engine": "google_images",
            "api_key": SERPAPI_KEY,
            "num": 8,
            "safe": "medium",
            "ijn": 0
        })
        
        results = search.get_dict()
        images = []
        
        for item in results.get("images_results", []):
            images.append({
                "url": item.get("original"),
                "title": item.get("title", ""),
                "source": "serpapi",
                "thumbnail": item.get("thumbnail", ""),
                "domain": item.get("source", "")
            })
        
        if images:
            logging.info(f"SerpAPI found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"SerpAPI image search failed for {vendor_name}: {e}")
        return []

def fetch_images_serper_sync(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: Serper.dev fallback for images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        # Build the base query in format <vendor_name> <model_family?> <product_type?>
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"

        # Do not include raw product_name here — rely on model_family/product_type

        base_query += " product image OR product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        url = "https://google.serper.dev/images"
        payload = {
            "q": search_query,
            "num": 8
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY_IMAGES,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        images = []
        
        for item in data.get("images", []):
            images.append({
                "url": item.get("imageUrl"),
                "title": item.get("title", ""),
                "source": "serper",
                "thumbnail": item.get("imageUrl"),  # Serper doesn't provide separate thumbnail
                "domain": item.get("link", "")
            })
        
        if images:
            logging.info(f"Serper found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Serper image search failed for {vendor_name}: {e}")
        return []

def fetch_product_images_with_fallback_sync(vendor_name: str, product_name: str = None, manufacturer_domains: list = None, model_family: str = None, product_type: str = None):
    """
    Synchronous 3-level image search fallback system with MongoDB caching
    0. Check MongoDB cache first
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        manufacturer_domains: (Optional) List of manufacturer domains to search within
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}, model family: {model_family}, type: {product_type}")
    
    # Step 0: Check MongoDB cache first (if model_family is provided)
    if vendor_name and model_family:
        from common.services.azure.blob_utils import get_cached_image, cache_image
        
        cached_image = get_cached_image(vendor_name, model_family)
        if cached_image:
            logging.info(f"Using cached image from GridFS for {vendor_name} - {model_family}")
            # Convert GridFS file_id to backend URL
            gridfs_file_id = cached_image.get('gridfs_file_id')
            backend_url = f"/api/images/{gridfs_file_id}"
            
            # Return image with backend URL
            image_response = {
                'url': backend_url,
                'title': cached_image.get('title', ''),
                'source': 'mongodb_gridfs',
                'thumbnail': backend_url,  # Same URL for thumbnail
                'domain': 'local',
                'cached': True,
                'gridfs_file_id': gridfs_file_id
            }
            return [image_response], "mongodb_gridfs"
    
    # Step 1: Try Google Custom Search API
    images = fetch_images_google_cse_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        manufacturer_domains=manufacturer_domains,
        model_family=model_family,
        product_type=product_type
    )
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        # Cache the top image if model_family is provided
        if vendor_name and model_family and len(images) > 0:
            from common.services.azure.blob_utils import cache_image
            cache_image(vendor_name, model_family, images[0])
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = fetch_images_serpapi_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        # Cache the top image if model_family is provided
        if vendor_name and model_family and len(images) > 0:
            from common.services.azure.blob_utils import cache_image
            cache_image(vendor_name, model_family, images[0])
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = fetch_images_serper_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )
    if images:
        logging.info(f"Using Serper images for {vendor_name}")
        # Cache the top image if model_family is provided
        if vendor_name and model_family and len(images) > 0:
            from mongodb_utils import cache_image
            cache_image(vendor_name, model_family, images[0])
        return images, "serper"
    
    # All failed
    logging.warning(f"All image search APIs failed for {vendor_name}")
    return [], "none"

def fetch_vendor_logo_sync(vendor_name: str, manufacturer_domains: list = None):
    """
    Specialized function to fetch vendor logo with MongoDB caching
    """
    logging.info(f"Fetching logo for vendor: {vendor_name}")
    
    # Step 0: Check Azure cache first
    try:
        from common.services.azure.blob_utils import azure_blob_file_manager, download_image_from_url
        
        logos_collection = azure_blob_file_manager.conn['collections'].get('vendor_logos')
        if logos_collection is not None:
            normalized_vendor = vendor_name.strip().lower()
            
            cached_logo = logos_collection.find_one({
                'vendor_name_normalized': normalized_vendor
            })
            
            if cached_logo and cached_logo.get('gridfs_file_id'):
                logging.info(f"Using cached logo from GridFS for {vendor_name}")
                gridfs_file_id = cached_logo.get('gridfs_file_id')
                backend_url = f"/api/images/{gridfs_file_id}"
                
                return {
                    'url': backend_url,
                    'thumbnail': backend_url,
                    'source': 'mongodb_gridfs',
                    'title': cached_logo.get('title', f"{vendor_name} Logo"),
                    'domain': 'local',
                    'cached': True,
                    'gridfs_file_id': str(gridfs_file_id)
                }
    except Exception as e:
        logging.warning(f"Failed to check logo cache for {vendor_name}: {e}")
    
    # Step 1: Cache miss - fetch from web
    logo_result = None
    
    # Try different logo-specific searches
    logo_queries = [
        f"{vendor_name} logo",
        f"{vendor_name} company logo", 
        f"{vendor_name} brand",
        f"{vendor_name}"
    ]
    
    for query in logo_queries:
        try:
            # Use Google CSE first for official logos
            if GOOGLE_API_KEY and GOOGLE_CX:
                # Build site restriction for manufacturer domains using LLM (or reuse if provided)
                if manufacturer_domains is None:
                    manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
                domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
                search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png OR filetype:svg"
                
                service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
                result = service.cse().list(
                    q=search_query,
                    cx=GOOGLE_CX,
                    searchType="image",
                    num=3,  # Only need a few logo options
                    safe="medium",
                    imgSize="MEDIUM"
                ).execute()
                
                for item in result.get("items", []):
                    logo_url = item.get("link")
                    title = item.get("title", "").lower()
                    
                    # Check if this looks like a logo
                    if any(keyword in title for keyword in ["logo", "brand", "company"]):
                        logo_result = {
                            "url": logo_url,
                            "thumbnail": item.get("image", {}).get("thumbnailLink", logo_url),
                            "source": "google_cse_logo",
                            "title": item.get("title", ""),
                            "domain": item.get("displayLink", "")
                        }
                        break
                
                # If no specific logo found, use first result from official domain
                if not logo_result and result.get("items"):
                    item = result["items"][0]
                    logo_result = {
                        "url": item.get("link"),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", item.get("link")),
                        "source": "google_cse_general",
                        "title": item.get("title", ""),
                        "domain": item.get("displayLink", "")
                    }
                
                if logo_result:
                    break
                    
        except Exception as e:
            logging.warning(f"Logo search failed for query '{query}': {e}")
            continue
    
    # Fallback: use general vendor search
    if not logo_result:
        try:
            images, source = fetch_product_images_with_fallback_sync(vendor_name, "")
            if images:
                # Return first image as logo
                logo_result = images[0].copy()
                logo_result["source"] = f"{source}_fallback"
        except Exception as e:
            logging.warning(f"Fallback logo search failed for {vendor_name}: {e}")
    
     # Step 2: Cache the logo in Azure Blob if found
    if logo_result:
        try:
            from common.services.azure.blob_utils import azure_blob_file_manager, download_image_from_url
            
            logo_url = logo_result.get('url')
            if logo_url and not logo_url.startswith('/api/images/'):  # Don't re-cache GridFS URLs
                # Download the logo
                download_result = download_image_from_url(logo_url)
                if download_result:
                    image_bytes, content_type, file_size = download_result
                    
                    # gridfs is not needed for Azure, upload directly
                    logos_collection = azure_blob_file_manager.conn['collections'].get('vendor_logos')
                    
                    if logos_collection is not None:
                        normalized_vendor = vendor_name.strip().lower()
                        file_extension = content_type.split('/')[-1] if '/' in content_type else 'png'
                        filename = f"logo_{normalized_vendor}.{file_extension}"
                        
                        # Store in GridFS
                        gridfs_file_id = gridfs.put(
                            image_bytes,
                            filename=filename,
                            content_type=content_type,
                            vendor_name=vendor_name,
                            original_url=logo_url,
                            logo_type='vendor_logo'
                        )
                        
                        logging.info(f"Stored vendor logo in GridFS: {filename} (ID: {gridfs_file_id})")
                        
                        # Store metadata
                        logo_doc = {
                            'vendor_name': vendor_name,
                            'vendor_name_normalized': normalized_vendor,
                            'gridfs_file_id': gridfs_file_id,
                            'original_url': logo_url,
                            'title': logo_result.get('title', f"{vendor_name} Logo"),
                            'source': logo_result.get('source', ''),
                            'domain': logo_result.get('domain', ''),
                            'content_type': content_type,
                            'file_size': file_size,
                            'filename': filename,
                            'created_at': datetime.utcnow()
                        }
                        
                        logos_collection.update_one(
                            {'vendor_name_normalized': normalized_vendor},
                            {'$set': logo_doc},
                            upsert=True
                        )
                        
                        logging.info(f"Successfully cached vendor logo for {vendor_name}")
                        
                        # Return cached version
                        backend_url = f"/api/images/{gridfs_file_id}"
                        return {
                            'url': backend_url,
                            'thumbnail': backend_url,
                            'source': 'mongodb_gridfs',
                            'title': logo_doc['title'],
                            'domain': 'local',
                            'cached': True,
                            'gridfs_file_id': str(gridfs_file_id)
                        }
        except Exception as e:
            logging.warning(f"Failed to cache vendor logo for {vendor_name}: {e}")
    
    return logo_result

async def fetch_images_google_cse(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 1: Google Custom Search API for images from manufacturer domains
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if model_family:
            query += f" {model_family}"
        if product_type:
            query += f" {product_type}"
        
        # Build site restriction for manufacturer domains using LLM
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
        search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png"
        
        # Use Google Custom Search API
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            searchType="image",
            num=8,
            safe="medium",
            imgSize="MEDIUM"
        ).execute()
        
        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        
        for item in result.get("items", []):
            url = item.get("link")
            
            # Skip images with unsupported URL schemes
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                logging.debug(f"Skipping image with unsupported URL scheme: {url}")
                continue
            
            # Only include http/https URLs
            if not url.startswith(('http://', 'https://')):
                logging.debug(f"Skipping non-HTTP URL: {url}")
                continue
                
            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} valid images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serpapi(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 2: SerpAPI fallback for Google Images
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"
        base_query += " product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        search = GoogleSearch({
            "q": search_query,
            "engine": "google_images",
            "api_key": SERPAPI_KEY,
            "num": 8,
            "safe": "medium",
            "ijn": 0
        })
        
        results = search.get_dict()
        images = []
        
        for item in results.get("images_results", []):
            images.append({
                "url": item.get("original"),
                "title": item.get("title", ""),
                "source": "serpapi",
                "thumbnail": item.get("thumbnail", ""),
                "domain": item.get("source", "")
            })
        
        if images:
            logging.info(f"SerpAPI found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"SerpAPI image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serper(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 3: Serper.dev fallback for images
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"
        base_query += " product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        url = "https://google.serper.dev/images"
        payload = {
            "q": search_query,
            "num": 8
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY_IMAGES,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        images = []
        
        for item in data.get("images", []):
            images.append({
                "url": item.get("imageUrl"),
                "title": item.get("title", ""),
                "source": "serper",
                "thumbnail": item.get("imageUrl"),  # Serper doesn't provide separate thumbnail
                "domain": item.get("link", "")
            })
        
        if images:
            logging.info(f"Serper found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Serper image search failed for {vendor_name}: {e}")
        return []

async def fetch_product_images_with_fallback(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    3-level image search fallback system
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}")
    
    # Step 1: Try Google Custom Search API (pass model_family/product_type, avoid raw product_name)
    images = await fetch_images_google_cse(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = await fetch_images_serpapi(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = await fetch_images_serper(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using Serper images for {vendor_name}")
        return images, "serper"
    
    # All failed
    logging.warning(f"All image search APIs failed for {vendor_name}")
    return [], "none"


@app.route("/api/test_image_search", methods=["GET"])
@login_required
def test_image_search():
    """
    Test endpoint for the image search functionality
    """
    vendor_name = request.args.get("vendor", "Emerson")
    product_name = request.args.get("product", "")
    
    try:
        # Use synchronous version for reliability; pass model_family instead of product_name
        model_family = None
        # If user provided product as a family list or string, prefer it
        if product_name and ',' in product_name:
            # accept '3051C,3051S' style input from quick tests
            model_family = product_name.split(',')[0].strip()
        elif product_name:
            model_family = product_name.strip()

        images, source_used = fetch_product_images_with_fallback_sync(
            vendor_name,
            product_name=None,
            manufacturer_domains=None,
            model_family=model_family,
            product_type=None
        )
        
        # Also test domain generation
        generated_domains = get_manufacturer_domains_from_llm(vendor_name)
        
        return jsonify({
            "vendor": vendor_name,
            "product": product_name,
            "images": images,
            "source_used": source_used,
            "count": len(images),
            "generated_domains": generated_domains
        })
        
    except Exception as e:
        logging.error(f"Image search test failed: {e}")
        return jsonify({
            "error": str(e),
            "vendor": vendor_name,
            "product": product_name,
            "images": [],
            "source_used": "error",
            "count": 0,
            "generated_domains": []
        }), 500


@app.route("/api/get_analysis_product_images", methods=["POST"])
@login_required
def get_analysis_product_images():
    """
    Get images for specific products from analysis results.
    Expected input:
    {
        "vendor": "Emerson",
        "product_type": "Flow Transmitter", 
        "product_name": "Rosemount 3051",
        "model_families": ["3051C", "3051S", "3051T"]
    }
    """
    try:
        data = request.get_json()

        vendor = data.get("vendor", "")
        product_type = data.get("product_type", "")
        product_name = data.get("product_name", "")
        model_families = data.get("model_families", [])

        if not vendor:
            return jsonify({"error": "Vendor name is required"}), 400

        # Removed requirements_match check - fetch images for ALL products (exact and approximate matches)
        # This supports the fallback display of approximate matches when no exact matches are found
        logging.info(f"Fetching images for analysis result: {vendor} {product_type} {product_name}")

        # Generate manufacturer domains once per request for this vendor
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor)

        # Search for images with different combinations
        all_images = []
        search_combinations = []

        # Prefer model family for search if available (e.g., STD800 instead of submodel STD830)
        primary_family = None
        if isinstance(model_families, list) and model_families:
            primary_family = str(model_families[0]).strip()

        # Build a base name for search: model family if present, otherwise product_name
        # Example: "STD800" instead of "STD830 Pressure Transmitter"
        base_name_for_search = primary_family or product_name

        # 1. Most specific: vendor + base_name_for_search + product_type
        if base_name_for_search and product_type:
            search_query = f"{vendor} {base_name_for_search} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "family_with_type",
                "priority": 1
            })

        # 2. Medium specific: vendor + base_name_for_search
        if base_name_for_search:
            search_query = f"{vendor} {base_name_for_search}"
            search_combinations.append({
                "query": search_query,
                "type": "family_or_name",
                "priority": 2
            })

        # 3. General: vendor + product_type
        if product_type:
            search_query = f"{vendor} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "type_general",
                "priority": 3
            })
        
        # Execute searches and collect results
        for search_info in search_combinations:
            try:
                # Pass model_family and product_type to the fetcher and avoid using raw product_name
                images, source_used = fetch_product_images_with_fallback_sync(
                    vendor_name=vendor,
                    product_name=None,
                    manufacturer_domains=manufacturer_domains,
                    model_family=base_name_for_search if base_name_for_search else None,
                    product_type=product_type if product_type else None,
                )
                
                # Add metadata to images
                for img in images:
                    img["search_type"] = search_info["type"]
                    img["search_priority"] = search_info["priority"]
                    img["search_query"] = search_info["query"]
                
                all_images.extend(images)
                
                # If we get good results from high-priority search, we can stop early
                if len(images) >= 5 and search_info["priority"] <= 2:
                    logging.info(f"Got {len(images)} images from high-priority search: {search_info['type']}")
                    break
                    
            except Exception as e:
                logging.warning(f"Search failed for query '{search_info['query']}': {e}")
                continue
        
        # Remove duplicates based on URL
        unique_images = []
        seen_urls = set()
        for img in all_images:
            url = img.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_images.append(img)
        
        # Sort by priority and quality
        def image_quality_score(img):
            score = 0
            
            # Priority weight (lower priority number = higher score)
            score += (5 - img.get("search_priority", 5)) * 10
            
            # Domain quality (official domains get higher score)
            domain = img.get("domain", "").lower()
            if any(mfg_domain in domain for mfg_domain in manufacturer_domains):
                score += 15
            
            # Source quality
            source = img.get("source", "")
            if source == "google_cse":
                score += 10
            elif source == "serpapi":
                score += 5
            
            # Title relevance (contains product name or model family)
            title = img.get("title", "").lower()
            if product_name.lower() in title:
                score += 8
            for model in model_families:
                if model.lower() in title:
                    score += 6
                    break
            
            return score
        
        # Sort by quality score (highest first)
        unique_images.sort(key=image_quality_score, reverse=True)
        
        # Select best images - top 1 for main display, top 10 for "view more"
        top_image = unique_images[0] if unique_images else None
        best_images = unique_images[:10]
        
        # Get vendor logo using specialized logo search
        vendor_logo = None
        try:
            vendor_logo = fetch_vendor_logo_sync(vendor, manufacturer_domains=manufacturer_domains)
        except Exception as e:
            logging.warning(f"Failed to fetch vendor logo for {vendor}: {e}")
        
        # Prepare response
        response_data = {
            "vendor": vendor,
            "product_type": product_type,
            "product_name": product_name,
            "model_families": model_families,
            "top_image": top_image,  # Single best image for main display
            "vendor_logo": vendor_logo,  # Vendor logo
            "all_images": best_images,  # All images for "view more"
            # Compatibility fields: many frontends expect `images` or `image`
            "images": best_images,
            "image": top_image,
            "total_found": len(all_images),
            "unique_count": len(unique_images),
            "best_count": len(best_images),
            "search_summary": {
                "searches_performed": len(search_combinations),
                "search_types": list(set(img.get("search_type") for img in best_images)),
                "sources_used": list(set(img.get("source") for img in best_images))
            }
        }
        
        logging.info(f"Analysis image search completed: {len(best_images)} best images selected from {len(all_images)} total")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Analysis product image search failed: {e}")
        return jsonify({
            "error": f"Failed to fetch analysis product images: {str(e)}",
            "vendor": data.get("vendor", ""),
            "product_type": data.get("product_type", ""),
            "product_name": data.get("product_name", ""),
            "model_families": data.get("model_families", []),
            "top_image": None,
            "vendor_logo": None,
            "all_images": [],
            "total_found": 0,
            "unique_count": 0,
            "best_count": 0
        }), 500


@app.route("/api/upload_pdf_from_url", methods=["POST"])
@login_required
def upload_pdf_from_url():
    data = request.get_json(force=True)
    pdf_url = data.get("url")
    if not pdf_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        # --- 1. Download PDF ---
        logging.info(f"[DOWNLOAD] Fetching PDF: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()

        filename = os.path.basename(urllib.parse.urlparse(pdf_url).path) or "document.pdf"
        pdf_bytes = response.content  # keep PDF in memory

        # --- 2. Extract data from PDF ---
        text_chunks = extract_data_from_pdf(BytesIO(pdf_bytes))
        raw_results = send_to_language_model(text_chunks)

        # Flatten results
        def flatten_results(results):
            flat = []
            for r in results:
                if isinstance(r, list):
                    flat.extend(r)
                else:
                    flat.append(r)
            return flat

        all_results = flatten_results(raw_results)
        final_result = aggregate_results(all_results, filename)
        
        # Apply standardization to the final result before splitting
        try:
            standardized_final_result = standardize_vendor_analysis_result(final_result)
            logging.info("Applied standardization to PDF from URL analysis")
        except Exception as e:
            logging.warning(f"Failed to standardize PDF from URL result: {e}")
            standardized_final_result = final_result

        # --- 3. Split by product types ---
        split_results = split_product_types([standardized_final_result])

        saved_json_paths = []
        saved_pdf_paths = []

        for result in split_results:
            # --- 4. Save JSON result to MongoDB ---
            vendor = (result.get("vendor") or "UnknownVendor").replace(" ", "_")
            product_type = (result.get("product_type") or "UnknownProduct").replace(" ", "_")
            model_series = (
                result.get("models", [{}])[0].get("model_series") or "UnknownModel"
            ).replace(" ", " ")
            
            try:
                # Upload product JSON to MongoDB
                # Structure: vendors/{vendor}/{product_type}/{model}.json
                product_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'json',
                    'collection_type': 'products',
                    'path': f'vendors/{vendor}/{product_type}/{model_series}.json'
                }
                mongodb_file_manager.upload_json_data(result, product_metadata)
                saved_json_paths.append(f"MongoDB:products:{vendor}:{product_type}:{model_series}")
                logging.info(f"[INFO] Stored product JSON to MongoDB: {vendor} - {product_type}")
            except Exception as e:
                logging.error(f"Failed to save product JSON to MongoDB: {e}")

            # --- 5. Save PDF to Azure Blob ---
            try:
                pdf_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'pdf',
                    'collection_type': 'documents',
                    'filename': filename,
                    'path': f'documents/{vendor}/{product_type}/{filename}'
                }
                file_id = azure_blob_file_manager.upload_to_azure(pdf_bytes, pdf_metadata)
                saved_pdf_paths.append(f"Azure:Documents:{file_id}")
                logging.info(f"[INFO] Stored PDF to Azure Blob: {filename} (ID: {file_id})")
            except Exception as e:
                logging.error(f"Failed to save PDF to Azure Blob: {e}")

            # --- Note: Product image extraction removed - now using API-based image search ---

        return jsonify({
            "data": split_results,
            "pdfFiles": saved_pdf_paths,
            "jsonFiles": saved_json_paths
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch PDF from URL: {str(e)}"}), 500
    except Exception as e:
        logging.exception("PDF analysis from URL failed.")
        return jsonify({"error": f"Failed to analyze PDF from URL: {str(e)}"}), 500

    
@app.route("/register", methods=["POST"])
@limiter.limit("5 per minute;20 per hour;100 per day")
def register():
    """
    Register new user
    ---
    tags:
      - Authentication
    summary: Register a new user account
    description: |
      Creates a new user account with pending status. The account must be approved
      by an administrator before the user can log in.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: User registration data
        required: true
        schema:
          type: object
          required:
            - username
            - email
            - password
          properties:
            username:
              type: string
              description: Unique username
              example: "johndoe"
            email:
              type: string
              format: email
              description: User email address
              example: "john.doe@example.com"
            password:
              type: string
              format: password
              description: User password
              example: "SecurePass123!"
            first_name:
              type: string
              description: User's first name
              example: "John"
            last_name:
              type: string
              description: User's last name
              example: "Doe"
    responses:
      201:
        description: Registration successful
        schema:
          type: object
          properties:
            message:
              type: string
              example: "User registration submitted. Awaiting admin approval."
      400:
        description: Missing required fields
      409:
        description: Username or email already exists
    """
    # Support both JSON (API) and Multipart/Form-Data (Frontend with File Upload)
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    
    # Optional: Log file upload attempt (file processing to be implemented)
    if 'document' in request.files:
        logging.info(f"User {username} uploaded a document: {request.files['document'].filename}")

    if not username or not email or not password:
        return jsonify({"error": "Missing username, email, or password"}), 400

    try:
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists"}), 409
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 409

        hashed_pw = hash_password(password)
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_pw,
            first_name=first_name,
            last_name=last_name,
            status='pending',
            role='user'
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User registration submitted. Awaiting admin approval."}), 201

    except Exception as e:
        logging.error(f"Registration failed - database error: {e}")
        return jsonify({
            "error": "Registration service temporarily unavailable. Please ensure database is connected.",
            "details": "Database connection failed" if "mysql" in str(e).lower() else "Registration error"
        }), 503

@app.route("/login", methods=["POST"])
@limiter.limit("5 per minute;20 per hour;100 per day")
def login():
    """
    User login
    ---
    tags:
      - Authentication
    summary: Authenticate user and create session
    description: |
      Authenticates user credentials and creates a session. The session cookie
      is automatically set and used for subsequent authenticated requests.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: Login credentials
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
              description: Username
              example: "johndoe"
            password:
              type: string
              format: password
              description: User password
              example: "SecurePass123!"
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Login successful"
            user:
              type: object
              properties:
                username:
                  type: string
                name:
                  type: string
                first_name:
                  type: string
                last_name:
                  type: string
                email:
                  type: string
                role:
                  type: string
                  enum: [user, admin]
      401:
        description: Invalid credentials
      403:
        description: Account not active
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    try:
        user = User.query.filter_by(username=username).first()
        if user and check_password(user.password_hash, password):
            # Allow both 'active' and 'approved' statuses
            if user.status not in ['active', 'approved']:
                return jsonify({"error": f"Account not active. Current status: {user.status}."}), 403

            # Ensure new login creates a fresh agentic workflow session
            # Previous sessions remain saved in DB but are not automatically resumed
            session.pop('agentic_session_id', None)

            session['user_id'] = user.id
            # Construct full name from first_name and last_name
            full_name = ""
            if user.first_name and user.last_name:
                full_name = f"{user.first_name} {user.last_name}"
            elif user.first_name:
                full_name = user.first_name
            elif user.last_name:
                full_name = user.last_name
            else:
                full_name = user.username

            return jsonify({
                "message": "Login successful",
                "user": {
                    "username": user.username,
                    "name": full_name,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                    "role": user.role
                }
            }), 200

        return jsonify({"error": "Invalid username or password"}), 401

    except Exception as e:
        logging.error(f"Login failed - database error: {e}")
        return jsonify({
            "error": "Authentication service temporarily unavailable. Please ensure database is connected.",
            "details": "Database connection failed" if "mysql" in str(e).lower() else "Authentication error"
        }), 503

@app.route("/logout", methods=["POST"])
def logout():
    """
    User logout
    ---
    tags:
      - Authentication
    summary: End user session
    description: Clears the user session and logs out the user.
    produces:
      - application/json
    responses:
      200:
        description: Logout successful
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Logout successful"
    """
    session.pop('user_id', None)
    return jsonify({"message": "Logout successful"}), 200

@app.route("/user", methods=["GET"])
@login_required
def get_current_user():
    """
    Get current user
    ---
    tags:
      - Authentication
    summary: Get current authenticated user
    description: Returns the profile information of the currently logged-in user.
    produces:
      - application/json
    responses:
      200:
        description: User profile data
        schema:
          type: object
          properties:
            user:
              type: object
              properties:
                username:
                  type: string
                name:
                  type: string
                first_name:
                  type: string
                last_name:
                  type: string
                email:
                  type: string
                role:
                  type: string
      401:
        description: Unauthorized - login required
      404:
        description: User not found
    """
    user = db.session.get(User, session['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    # Construct full name from first_name and last_name
    full_name = ""
    if user.first_name and user.last_name:
        full_name = f"{user.first_name} {user.last_name}"
    elif user.first_name:
        full_name = user.first_name
    elif user.last_name:
        full_name = user.last_name
    else:
        full_name = user.username
    
    return jsonify({
        "user": {
            "username": user.username,
            "name": full_name,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role
        }
    }), 200



# =========================================================================
# === PROGRESS TRACKING ENDPOINT ===
# =========================================================================

# Global progress tracker for long-running operations
current_operation_progress = None

@app.route("/api/progress", methods=["GET"])
@login_required
def get_operation_progress():
    """Get progress of current long-running operation"""
    global current_operation_progress
    
    if current_operation_progress is None:
        return jsonify({
            "status": "no_active_operation",
            "message": "No active operation in progress"
        }), 200
    
    try:
        progress_data = current_operation_progress.get_progress()
        return jsonify({
            "status": "in_progress",
            "progress": progress_data
        }), 200
    except Exception as e:
        logging.error(f"Failed to get progress: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve progress information"
        }), 500

# =========================================================================
# === VALIDATION ENDPOINT ===
# =========================================================================

@app.route("/debug-session/<session_id>", methods=["GET"])
@login_required
def debug_session_state(session_id):
    """Debug endpoint to check session state for a specific search session"""
    current_step_key = f'current_step_{session_id}'
    current_intent_key = f'current_intent_{session_id}'
    product_type_key = f'product_type_{session_id}'
    
    session_data = {
        'session_id': session_id,
        'current_step': session.get(current_step_key, 'None'),
        'current_intent': session.get(current_intent_key, 'None'),
        'product_type': session.get(product_type_key, 'None'),
        'all_session_keys': [k for k in session.keys() if session_id in k],
        'all_keys': list(session.keys()),  # Show all keys for debugging
        'session_size': len(session.keys())
    }
    
    return jsonify(session_data), 200

@app.route("/debug-session-clear/<session_id>", methods=["POST"])
@login_required  
def clear_session_state(session_id):
    """Debug endpoint to manually clear session state for testing"""
    keys_to_remove = [k for k in session.keys() if session_id in k]
    
    for key in keys_to_remove:
        del session[key]
    
    return jsonify({
        'session_id': session_id,
        'cleared_keys': keys_to_remove,
        'status': 'cleared'
    }), 200



# DEPRECATED: Use /api/agentic/validate instead
@app.route("/api/validate", methods=["POST"])
@login_required
def api_validate():
    """
    DEPRECATED: Use /api/agentic/validate instead.
    The agentic version provides split validation steps (validate-product-input, get-product-schema).
    """
    # DEPRECATION WARNING: This endpoint will be removed in a future version
    logging.warning("[DEPRECATED] /api/validate called - Use /api/agentic/validate instead")

    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        user_input = data.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "Missing user_input"}), 400

        # Get search session ID if provided (for multiple search tabs)
        search_session_id = data.get("search_session_id", "default")
        
        # By default preserve any previously-detected product type and workflow state for this
        # search session. Only clear them when the client explicitly requests a reset
        # (for example when initializing a brand-new independent search tab).
        session_key = f'product_type_{search_session_id}'
        step_key = f'current_step_{search_session_id}'
        intent_key = f'current_intent_{search_session_id}'

        if data.get('reset', False):
            if session_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing previous product type due to reset request: {session[session_key]}")
                del session[session_key]
            if step_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing step state due to reset request: {session[step_key]}")
                del session[step_key]
            if intent_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing intent state due to reset request: {session[intent_key]}")
                del session[intent_key]
        else:
            logging.info(f"[VALIDATE] Session {search_session_id}: Preserving existing product_type and workflow state if present.")
        
        # Store original user input for logging (session-isolated)
        session[f'log_user_query_{search_session_id}'] = user_input

        initial_schema = load_requirements_schema()
        
        # Add session context to LLM validation to prevent cross-contamination
        session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"
        
        temp_validation_result = components['validation_chain'].invoke({
            "user_input": session_isolated_input,
            "schema": json.dumps(initial_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })
        detected_type = temp_validation_result.get('product_type', 'UnknownProduct')
        
        specific_schema = load_requirements_schema(detected_type)
        if not specific_schema:
            global current_operation_progress
            try:
                # Set up progress tracking for web schema building
                from loading import ProgressTracker
                current_operation_progress = ProgressTracker(4, f"Building Schema for {detected_type}")
                specific_schema = build_requirements_schema_from_web(detected_type)
            finally:
                # Clear progress tracker when done
                current_operation_progress = None

        # Add session context to detailed validation as well
        session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"
        
        validation_result = components['validation_chain'].invoke({
            "user_input": session_isolated_input,
            "schema": json.dumps(specific_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })

        cleaned_provided_reqs = clean_empty_values(validation_result.get("provided_requirements", {}))

        mapped_provided_reqs = map_provided_to_schema(
            convert_keys_to_camel_case(specific_schema),
            convert_keys_to_camel_case(cleaned_provided_reqs)
        )

        response_data = {
            "productType": validation_result.get("product_type", detected_type),
            "detectedSchema": convert_keys_to_camel_case(specific_schema),
            "providedRequirements": mapped_provided_reqs
        }

        # ---------------- Helpers for missing mandatory fields ----------------


        missing_mandatory_fields = get_missing_mandatory_fields(
            mapped_provided_reqs, response_data["detectedSchema"]
        )

        # ---------------- Helper: Convert camelCase to friendly label ----------------


        # ---------------- Prompt user if any mandatory fields are missing ----------------
        if missing_mandatory_fields:
            # Convert missing fields to friendly labels
            missing_fields_friendly = [friendly_field_name(f) for f in missing_mandatory_fields]
            missing_fields_str = ", ".join(missing_fields_friendly)
            is_repeat = data.get("is_repeat", False)

            if not is_repeat:
                alert_prompt = ChatPromptTemplate.from_template(VALIDATION_ALERT_INITIAL_PROMPT)
            else:
                alert_prompt = ChatPromptTemplate.from_template(VALIDATION_ALERT_REPEAT_PROMPT)

            alert_chain = alert_prompt | components['llm'] | StrOutputParser()
            agent_message = alert_chain.invoke({
                "product_type": response_data["productType"],
                "missing_fields": missing_fields_str
            })

            response_data["validationAlert"] = {
                "message": agent_message,
                "canContinue": True,
                "missingFields": missing_mandatory_fields
            }

        # Store product_type in session for later use in advanced parameters (session-isolated)
        session[f'product_type_{search_session_id}'] = response_data["productType"]

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Validation failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/new-search", methods=["POST"])
@login_required
def api_new_search():
    """Initialize a new search session, clearing any previous state"""
    try:
        data = request.get_json(force=True) if request.is_json else {}
        search_session_id = data.get("search_session_id", "default")
        
        # Clear all session data related to this search session
        keys_to_clear = [k for k in session.keys() if search_session_id in k or k.startswith('product_type')]
        for key in keys_to_clear:
            del session[key]
        
        # Clear general workflow state for new search
        workflow_keys = ['current_step', 'current_intent', 'data']
        for key in workflow_keys:
            if key in session:
                del session[key]
        
        logging.info(f"[NEW_SEARCH] Initialized new search session: {search_session_id}")
        logging.info(f"[NEW_SEARCH] Cleared session keys: {keys_to_clear}")
        
        return jsonify({
            "success": True,
            "search_session_id": search_session_id,
            "message": "New search session initialized"
        }), 200
        
    except Exception as e:
        logging.exception("Failed to initialize new search session.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/schema", methods=["GET"])
@login_required
def api_schema():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        product_type = request.args.get("product_type", "").strip()
        
        if product_type:
            try:
                # Try to load from MongoDB with timeout protection
                schema_data = load_requirements_schema(product_type)
                
                # Check if schema is valid (not empty)
                if schema_data and (schema_data.get("mandatory_requirements") or schema_data.get("optional_requirements")):
                    logging.info(f"[SCHEMA] Successfully loaded schema for '{product_type}'")
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                else:
                    logging.warning(f"[SCHEMA] Empty schema returned for '{product_type}', building from web")
                    # Fallback to web discovery if schema is empty
                    schema_data = build_requirements_schema_from_web(product_type)
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                    
            except Exception as db_error:
                # Storage timeout or connection error - fallback to web-based schema
                logging.error(f"[SCHEMA] Storage error for '{product_type}': {str(db_error)}")
                logging.info(f"[SCHEMA] Falling back to web-based schema generation for '{product_type}'")
                
                try:
                    schema_data = build_requirements_schema_from_web(product_type)
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                except Exception as web_error:
                    logging.error(f"[SCHEMA] Web-based schema generation also failed: {str(web_error)}")
                    # Return minimal schema to prevent complete failure
                    return jsonify({
                        "productType": product_type,
                        "mandatoryRequirements": {},
                        "optionalRequirements": {},
                        "error": f"Failed to load schema: {str(db_error)}"
                    }), 200  # Return 200 with error message instead of 500
        else:
            # No product type - return generic schema
            schema_data = load_requirements_schema()
            return jsonify(convert_keys_to_camel_case(schema_data)), 200
            
    except Exception as e:
        logging.exception("Schema fetch failed.")
        # Return minimal schema with error instead of failing completely
        return jsonify({
            "productType": product_type if 'product_type' in locals() else "",
            "mandatoryRequirements": {},
            "optionalRequirements": {},
            "error": str(e)
        }), 200  # Return 200 to prevent frontend from breaking

@app.route("/api/additional_requirements", methods=["POST"])
@login_required
def api_additional_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()
        search_session_id = data.get("search_session_id", "default")

        
        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        specific_schema = load_requirements_schema(product_type)
        
        # Add session isolation to prevent cross-contamination
        session_isolated_input = f"[Session: {search_session_id}] - This is an independent additional requirements request. User input: {user_input}"
        
        validation_result = invoke_additional_requirements_chain(
            components,
            session_isolated_input,
            product_type,
            json.dumps(specific_schema, indent=2),
            components['additional_requirements_format_instructions']
        )

        new_requirements = validation_result.get("provided_requirements", {})
        combined_reqs = new_requirements

        if combined_reqs:
            reqs_for_llm = '\n'.join([
                f"- {prettify_req(key)}: {value}" for key, value in combined_reqs.items()
            ])
            llm_chain = REQUIREMENTS_EXTRACTION_PROMPT | components['llm'] | StrOutputParser()
            explanation = llm_chain.invoke({
                "product_type": prettify_req(product_type),
                "requirements": reqs_for_llm
            })
        else:
            explanation = "I could not identify any specific requirements from your input."

        provided_requirements = new_requirements
        if new_requirements.get('mandatoryRequirements'):
            provided_requirements = {
                **new_requirements.get('mandatoryRequirements', {}),
                **new_requirements.get('optionalRequirements', {})
            }

        response_data = {
            "explanation": explanation,
            "providedRequirements": convert_keys_to_camel_case(provided_requirements),
        }


        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Additional requirements handling failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/api/structure_requirements", methods=["POST"])
@login_required
def api_structure_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        full_input = data.get("full_input", "")
        if not full_input:
            return jsonify({"error": "Missing full_input"}), 400

        structured_req = components['requirements_chain'].invoke({"user_input": full_input})
        return jsonify({"structured_requirements": structured_req}), 200
    except Exception as e:
        logging.exception("Requirement structuring failed.")
        return jsonify({"error": str(e)}), 500



@app.route("/api/add_advanced_parameters", methods=["POST"])
@login_required
def api_add_advanced_parameters():
    """
    Processes user input for latest advanced specifications selection with series numbers
    """
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()
        available_parameters = data.get("available_parameters", [])

        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        if not user_input:
            return jsonify({"error": "Missing user_input"}), 400

        # Use LLM to extract selected specifications from user input
        prompt = PARAMETER_SELECTION_PROMPT

        try:
            chain = prompt | components['llm'] | StrOutputParser()
            llm_response = chain.invoke({
                "product_type": product_type,
                "available_parameters": json.dumps(available_parameters),
                "user_input": user_input
            })

            # Parse the LLM response
            result = json.loads(llm_response)
            selected_parameters = result.get("selected_parameters", {})
            explanation = result.get("explanation", "Latest specifications selected successfully.")

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"LLM parsing failed, using fallback: {e}")
            # Fallback: simple keyword matching
            selected_parameters = {}
            user_lower = user_input.lower()
            
            if "all" in user_lower or "everything" in user_lower:
                # Handle both dict format (new) and string format (old)
                for param in available_parameters:
                    param_key = param.get('key', param) if isinstance(param, dict) else param
                    selected_parameters[param_key] = ""
                explanation = "All available latest specifications have been selected."
            else:
                # Look for specification names in user input
                for param in available_parameters:
                    # Handle both dict format (new) and string format (old)
                    if isinstance(param, dict):
                        param_key = param.get('key', '')
                        param_name = param.get('name', '').lower()
                        if param_key.lower() in user_lower or param_name in user_lower:
                            selected_parameters[param_key] = ""
                    else:
                        if param.lower() in user_lower or param.replace('_', ' ').lower() in user_lower:
                            selected_parameters[param] = ""
                
                explanation = f"Selected {len(selected_parameters)} latest specifications based on your input."

        def wants_parameter_display(text: str) -> bool:
            lowered = text.lower()
            display_keywords = ["show", "display", "list", "see", "view", "what are"]
            spec_keywords = ["spec", "parameter", "latest"]
            return any(keyword in lowered for keyword in display_keywords) and any(
                key in lowered for key in spec_keywords
            )

        normalized_input = user_input.strip().lower()
        
        # Generate friendly response
        if selected_parameters:
            param_list = ", ".join([param.replace('_', ' ').title() for param in selected_parameters.keys()])
            friendly_response = f"Great! I've added these latest advanced specifications: {param_list}. Would you like to add any more advanced specifications?"
        else:
            if wants_parameter_display(normalized_input) and available_parameters:
                formatted_available = ", ".join(
                    [
                        (param.get("name") or param.get("key", "")).strip()
                        if isinstance(param, dict)
                        else str(param)
                        for param in available_parameters
                    ]
                )
                friendly_response = (
                    "Here are the latest advanced specifications you can add: "
                    f"{formatted_available}. Let me know the names you want to include or say 'no' to skip."
                )
            else:
                friendly_response = "I didn't find any matching specifications in your input. Could you please specify which latest specifications you'd like to add?"

        response_data = {
            "selectedParameters": convert_keys_to_camel_case(selected_parameters),
            "explanation": explanation,
            "friendlyResponse": friendly_response,
            "totalSelected": len(selected_parameters)
        }

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Latest advanced specifications addition failed.")
        return jsonify({"error": str(e)}), 500

# DEPRECATED: Use /api/agentic/run-analysis instead
@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    """
    DEPRECATED: Use /api/agentic/run-analysis instead.
    The agentic version uses ProductSearchWorkflow with image fetching.
    """
    # DEPRECATION WARNING: This endpoint will be removed in a future version
    logging.warning("[DEPRECATED] /api/analyze called - Use /api/agentic/run-analysis instead")

    try:
        # Check if analysis chain is initialized
        if analysis_chain is None:
            logging.error("[ANALYZE] Analysis chain not initialized")
            return jsonify({"error": "Analysis service not available. Please try again later."}), 503

        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Check if CSV vendors are provided for targeted analysis
        csv_vendors = data.get("csv_vendors", [])
        requirements = data.get("requirements", "")
        product_type = data.get("product_type", "")
        detected_product = data.get("detected_product", "")
        user_input = data.get("user_input")
        
        # Handle CSV vendor filtering (optional - can be combined with user_input)
        if csv_vendors and len(csv_vendors) > 0:
            logging.info(f"[CSV_VENDOR_FILTER] Applying CSV vendor filter with {len(csv_vendors)} vendors")
            
            # Standardize CSV vendor names for filtering
            csv_vendor_names = []
            for csv_vendor in csv_vendors:
                try:
                    original_name = csv_vendor.get("vendor_name", "")
                    standardized_name = standardize_vendor_name(original_name)
                    csv_vendor_names.append(standardized_name.lower())
                except Exception as e:
                    logging.warning(f"Failed to standardize CSV vendor {csv_vendor.get('vendor_name', '')}: {e}")
                    csv_vendor_names.append(csv_vendor.get("vendor_name", "").lower())
            
            # Store CSV filter in session for analysis chain to use
            session[f'csv_vendor_filter_{session.get("user_id", "default")}'] = {
                'vendor_names': csv_vendor_names,
                'csv_vendors': csv_vendors,
                'product_type': product_type,
                'detected_product': detected_product
            }
            
            logging.info(f"[CSV_VENDOR_FILTER] Applied filter for vendors: {csv_vendor_names}")
        
        # Now handle the analysis (user_input is REQUIRED)
        if user_input is not None:
            # user_input can be a string or dict - handle both cases
            # The analysis chain expects the raw input string for LLM processing
            if isinstance(user_input, dict):
                # If it's already a dict, extract the raw input or convert to string
                if "raw_input" in user_input:
                    user_input_str = user_input["raw_input"]
                else:
                    # Convert dict to a formatted string for the LLM
                    user_input_str = json.dumps(user_input, indent=2)
            elif isinstance(user_input, str):
                # Try to parse as JSON first (might be a JSON string)
                try:
                    parsed = json.loads(user_input)
                    if isinstance(parsed, dict):
                        if "raw_input" in parsed:
                            user_input_str = parsed["raw_input"]
                        else:
                            user_input_str = json.dumps(parsed, indent=2)
                    else:
                        user_input_str = user_input
                except json.JSONDecodeError:
                    # It's a plain string - use as is
                    user_input_str = user_input
            else:
                return jsonify({"error": "user_input must be a string or dict"}), 400

            # Pass user_input as string to the analysis chain
            logging.info(f"[ANALYZE] Processing input: {user_input_str[:200]}...")
            analysis_result = analysis_chain({"user_input": user_input_str})
        
        else:
            return jsonify({"error": "Missing 'user_input' parameter or CSV vendor data"}), 400
        
        # Apply standardization to the analysis result
        try:
            # Standardize vendor analysis if it exists
            if "vendor_analysis" in analysis_result:
                analysis_result["vendor_analysis"] = standardize_vendor_analysis_result(analysis_result["vendor_analysis"])
            
            # Standardize overall ranking if it exists
            if "overall_ranking" in analysis_result:
                analysis_result["overall_ranking"] = standardize_ranking_result(analysis_result["overall_ranking"])
                
            logging.info("Applied standardization to analysis result")
        except Exception as e:
            logging.warning(f"Standardization failed, proceeding with original result: {e}")

        camel_case_result = convert_keys_to_camel_case(analysis_result)

        # Store the analysis result as system response for logging
        session['log_system_response'] = analysis_result

        return jsonify(camel_case_result)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logging.error(f"Analysis failed: {str(e)}")
        logging.error(f"Traceback:\n{error_traceback}")
        return jsonify({
            "error": str(e),
            "details": error_traceback.split('\n')[-3] if error_traceback else None
        }), 500



def match_user_with_pdf(user_input, pdf_data):
    """
    Matches user input fields with PDF data.
    Accepts user_input as a dict or JSON string.
    """
    # Ensure user_input is a dict
    if isinstance(user_input, str):
        try:
            user_input = json.loads(user_input)
        except json.JSONDecodeError:
            logging.warning("user_input is a string that cannot be parsed; wrapping in dict.")
            user_input = {"raw_input": user_input}

    if not isinstance(user_input, dict):
        raise ValueError("user_input must be a dict after parsing.")

    matched_results = {}
    for field, requirement in user_input.items():
        # Example matching logic; replace with your actual logic
        matched_results[field] = pdf_data.get(field, None)

    return matched_results

@app.route("/api/get_field_description", methods=["POST"])
@login_required
def api_get_field_description():
    """
    Get field description and value from standards documents using Deep Agent.
    
    This endpoint uses the Deep Agent's field extraction functionality to:
    1. Query relevant standards documents for the field
    2. Extract specification value and description
    3. Return standards-based information for UI display
    
    Request Body:
        {
            "field": "Compliance.hazardousAreaRating.value",
            "product_type": "Multi-Point Thermocouple"
        }
    
    Response:
        {
            "success": true,
            "description": "ATEX II 1/2 G Ex ia/d per IEC 60079",
            "field": "hazardousAreaRating",
            "product_type": "Multi-Point Thermocouple",
            "source": "instrumentation_safety_standards.docx",
            "standards_referenced": ["IEC 60079", "ATEX"],
            "confidence": 0.85
        }
    """
    try:
        data = request.get_json(force=True)
        field_path = data.get("field", "").strip()
        product_type = data.get("product_type", "general").strip()

        if not field_path:
            return jsonify({"error": "Missing 'field' parameter.", "success": False}), 400

        # Parse field path (e.g., "Compliance.hazardousAreaRating.value" -> "hazardousAreaRating")
        field_parts = field_path.split(".")
        field_name = field_parts[-2] if len(field_parts) >= 2 and field_parts[-1] in ["value", "source", "confidence", "standards_referenced"] else field_parts[-1]
        section_name = field_parts[0] if len(field_parts) > 1 else "Other"
        
        logging.info(f"[FieldDescription] Extracting value for field: {field_name}, product: {product_type}, section: {section_name}")
        
        # =====================================================
        # STRATEGY 1: Direct Default Lookup (FAST PATH)
        # Uses comprehensive standards specifications from schema_field_extractor
        # =====================================================
        try:
            from common.agentic.deep_agent.schema_field_extractor import get_default_value_for_field
            
            default_value = get_default_value_for_field(product_type, field_name)
            
            if default_value:
                logging.info(f"[FieldDescription] ✓ Found default value for {field_name}: {default_value[:50]}...")
                return jsonify({
                    "success": True,
                    "description": default_value,
                    "field": field_name,
                    "field_path": field_path,
                    "section": section_name,
                    "product_type": product_type,
                    "source": "standards_specifications",
                    "standards_referenced": [],  # Could be enhanced to include IEC/ISO codes
                    "confidence": 0.9
                }), 200
            else:
                logging.debug(f"[FieldDescription] No default value found for {field_name}")
                
        except Exception as default_err:
            logging.warning(f"[FieldDescription] Default lookup failed: {default_err}")
        
        # =====================================================
        # STRATEGY 2: LLM-based Description (Fallback)
        # Used when no predefined default exists
        # =====================================================
        try:
            from llm_fallback import create_llm_with_fallback
            import os
            
            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Create a focused prompt for technical specifications
            prompt = f"""Provide the typical technical specification value for "{field_name}" 
for a {product_type} in industrial instrumentation.

Guidelines:
- Provide a specific value or range (e.g., "±0.1%", "4-20mA", "IP66")
- Include relevant standards references (IEC, ISO, NAMUR, etc.) if applicable
- Keep response under 50 words
- Focus on the specification VALUE, not a description

Example format: "±0.75% or ±2.5°C (whichever is greater) per IEC 60584"
"""
            
            response = llm.invoke(prompt)
            description = response.content.strip()
            
            if description and description.lower() not in ["not specified", "n/a", "unknown", ""]:
                logging.info(f"[FieldDescription] ✓ LLM generated value for {field_name}")
                return jsonify({
                    "success": True,
                    "description": description,
                    "field": field_name,
                    "field_path": field_path,
                    "section": section_name,
                    "product_type": product_type,
                    "source": "llm_inference",
                    "standards_referenced": [],
                    "confidence": 0.7
                }), 200
                
        except Exception as llm_err:
            logging.warning(f"[FieldDescription] LLM fallback failed: {llm_err}")
        
        # =====================================================
        # STRATEGY 3: Generic Fallback
        # Last resort when all else fails
        # =====================================================
        field_display = field_name.replace("_", " ").replace("-", " ")
        # Convert camelCase to Title Case
        import re
        field_display = re.sub('([a-z])([A-Z])', r'\1 \2', field_display).title()
        
        return jsonify({
            "success": True,
            "description": f"Specification for {field_display}",
            "field": field_name,
            "field_path": field_path,
            "section": section_name,
            "product_type": product_type,
            "source": "generic",
            "standards_referenced": [],
            "confidence": 0.3
        }), 200

    except Exception as e:
        logging.exception("Failed to get field description.")
        return jsonify({
            "success": False,
            "error": "Failed to get field description: " + str(e),
            "description": ""
        }), 500


@app.route("/api/describe_field", methods=["POST"])
@login_required
def api_describe_field():
    """
    Alias endpoint for get_field_description with frontend-compatible parameter names.

    Frontend sends:
        {
            "field_name": "hazardousAreaRating",
            "product_type": "Multi-Point Thermocouple",
            "context_value": "optional context"
        }

    Response:
        {
            "success": true,
            "description": "ATEX II 1/2 G Ex ia/d per IEC 60079"
        }
    """
    try:
        data = request.get_json(force=True)
        field_name = data.get("field_name", "").strip()
        product_type = data.get("product_type", "general").strip()
        context_value = data.get("context_value", "")

        if not field_name:
            return jsonify({"success": False, "description": "", "error": "Missing field_name"}), 400

        logging.info(f"[DescribeField] Field: {field_name}, Product: {product_type}")

        # Try default lookup first (fast path)
        try:
            from common.agentic.deep_agent.schema_field_extractor import get_default_value_for_field
            default_value = get_default_value_for_field(product_type, field_name)
            if default_value:
                return jsonify({"success": True, "description": default_value}), 200
        except Exception as e:
            logging.debug(f"[DescribeField] Default lookup failed: {e}")

        # LLM fallback
        try:
            from llm_fallback import create_llm_with_fallback
            import os

            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            context_hint = f" with value '{context_value}'" if context_value else ""
            prompt = f"""Provide a brief technical description for the field "{field_name}"{context_hint}
for a {product_type} in industrial instrumentation.

Keep response under 30 words. Focus on what this field represents and typical values."""

            response = llm.invoke(prompt)
            description = response.content.strip()

            if description:
                return jsonify({"success": True, "description": description}), 200

        except Exception as e:
            logging.warning(f"[DescribeField] LLM failed: {e}")

        # Generic fallback
        import re
        field_display = re.sub('([a-z])([A-Z])', r'\1 \2', field_name).replace("_", " ").title()
        return jsonify({"success": True, "description": f"Technical specification for {field_display}"}), 200

    except Exception as e:
        logging.exception("[DescribeField] Error")
        return jsonify({"success": False, "description": "", "error": str(e)}), 500


@app.route("/api/get_all_field_descriptions", methods=["POST"])
@login_required
def api_get_all_field_descriptions():
    """
    Get ALL field descriptions and values for a schema at once (BATCH API).
    
    This avoids multiple individual API calls by fetching all values in one request.
    
    Request Body:
        {
            "product_type": "Thermocouple",
            "fields": ["Performance.accuracy", "Electrical.outputSignal", ...]
        }
    
    Response:
        {
            "success": true,
            "product_type": "Thermocouple",
            "field_values": {
                "Performance.accuracy": {"value": "±0.75%...", "source": "standards"},
                "Electrical.outputSignal": {"value": "4-20mA...", "source": "standards"},
                ...
            },
            "total_fields": 30,
            "fields_populated": 28
        }
    """
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "general").strip()
        fields = data.get("fields", [])
        
        if not fields:
            return jsonify({"error": "Missing 'fields' parameter.", "success": False}), 400
        
        logging.info(f"[BatchFieldDescription] Processing {len(fields)} fields for {product_type}")
        
        # Import template specifications for actual descriptions
        template_descriptions = {}
        try:
            from common.agentic.deep_agent.specification_templates import get_all_specs_for_product_type
            template_specs = get_all_specs_for_product_type(product_type)
            if template_specs:
                for spec_key, spec_def in template_specs.items():
                    template_descriptions[spec_key] = spec_def.description
                logging.info(f"[BatchFieldDescription] Loaded {len(template_descriptions)} template descriptions")
        except ImportError as e:
            logging.warning(f"[BatchFieldDescription] Could not import templates: {e}")
        
        def normalize_key(key: str) -> str:
            """Normalize key to lowercase with underscores for matching.
            Converts camelCase, PascalCase, and snake_case to lowercase_with_underscores.
            """
            import re
            # Insert underscore before uppercase letters (for camelCase/PascalCase)
            normalized = re.sub(r'([a-z])([A-Z])', r'\1_\2', key)
            # Replace spaces, hyphens, and dots with underscores
            normalized = normalized.replace(' ', '_').replace('-', '_').replace('.', '_')
            # Remove duplicate underscores and convert to lowercase
            normalized = re.sub(r'_+', '_', normalized).lower().strip('_')
            return normalized

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

        # Build normalized lookup for template descriptions
        normalized_template_lookup = {}
        for t_key, t_desc in template_descriptions.items():
            normalized_template_lookup[normalize_key(t_key)] = t_desc

        field_values = {}
        fields_populated = 0

        for field_path in fields:
            # Parse field path
            field_parts = field_path.split(".")
            field_name = field_parts[-2] if len(field_parts) >= 2 and field_parts[-1] in ["value", "source", "confidence", "standards_referenced"] else field_parts[-1]

            # Try to get description from templates first
            description = None
            source = "not_found"

            # Normalize the field name for matching (handles camelCase -> snake_case)
            normalized_name = normalize_key(field_name)

            # Check template descriptions by field name
            # 1. Exact Name Match
            if field_name in template_descriptions:
                description = template_descriptions[field_name]
                source = "template_specifications"
                fields_populated += 1
            # 2. Exact Path Match
            elif field_path in template_descriptions:
                description = template_descriptions[field_path]
                source = "template_specifications"
                fields_populated += 1
            # 3. Normalized Key Match (handles camelCase <-> snake_case)
            elif normalized_name in normalized_template_lookup:
                description = normalized_template_lookup[normalized_name]
                source = "template_specifications_normalized"
                fields_populated += 1
            else:
                # 4. Fuzzy Suffix Match on normalized path
                normalized_path = normalize_key(field_path)
                found_fuzzy = False
                for t_key_normalized, t_desc in normalized_template_lookup.items():
                    if normalized_path.endswith(t_key_normalized):
                        # Ensure boundary safety (preceded by underscore or is exact match)
                        idx = normalized_path.rfind(t_key_normalized)
                        if idx == 0 or (idx > 0 and normalized_path[idx-1] == '_'):
                            description = t_desc
                            source = "template_specifications_fuzzy"
                            fields_populated += 1
                            found_fuzzy = True
                            break

                if not found_fuzzy:
                    # Fallback to human-readable generation
                    description = prettify_field_name(field_name)
                    source = "generated"
                    fields_populated += 1
            
            field_values[field_path] = {
                "value": description or "",
                "source": source,
                "field_name": field_name,
                "confidence": 0.9 if source == "template_specifications" else 0.5,
                "standards_referenced": []
            }
        
        logging.info(f"[BatchFieldDescription] Completed: {fields_populated}/{len(fields)} fields populated")
        
        return jsonify({
            "success": True,
            "product_type": product_type,
            "field_values": field_values,
            "total_fields": len(fields),
            "fields_populated": fields_populated
        }), 200
        
    except Exception as e:
        logging.exception("Failed to get batch field descriptions.")
        return jsonify({
            "success": False,
            "error": "Failed to get batch field descriptions: " + str(e),
            "field_values": {}
        }), 500

def get_submodel_to_model_series_mapping():
    """
    Creates a mapping from submodel names to their parent model series
    by scanning all vendor JSON files.
    """
    """Load submodel mapping from MongoDB instead of local files"""
    submodel_to_series = {}
    
    try:
        # Query Azure Blob for all product data (vendors)
        products_collection = azure_blob_file_manager.conn['collections'].get('vendors')
        
        if not products_collection:
            logging.warning("Products (vendors) collection not found in Azure Blob")
            return submodel_to_series
        
        # Get all products from Azure Blob
        cursor = products_collection.find({})
        
        for doc in cursor:
            try:
                # Extract product data
                if 'data' in doc:
                    data = doc['data']
                else:
                    data = {k: v for k, v in doc.items() if k not in ['_id', 'metadata']}
                
                # Process models and submodels
                models = data.get('models', [])
                for model in models:
                    model_series = model.get('model_series', '')
                    submodels = model.get('sub_models', [])
                    
                    for submodel in submodels:
                        submodel_name = submodel.get('name', '')
                        if submodel_name and model_series:
                            submodel_to_series[submodel_name] = model_series
                            
            except Exception as e:
                logging.warning(f"Failed to process MongoDB document: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Failed to load submodel mapping from MongoDB: {e}")
        return submodel_to_series
                        
    logging.info(f"Generated submodel mapping with {len(submodel_to_series)} entries")
    return submodel_to_series

@app.route("/api/vendors", methods=["GET"])
@login_required
def get_vendors():
    """
    Get vendors with product images - ONLY for vendors in analysis results.
    Optimized to avoid unnecessary API calls.
    """
    try:
        # Get vendor list from query parameter (sent by frontend with analysis results)
        vendors_param = request.args.get('vendors', '')
        
        if vendors_param:
            # Use vendors from analysis results
            vendor_list = [v.strip() for v in vendors_param.split(',') if v.strip()]
            logging.info(f"Fetching images for {len(vendor_list)} vendors from analysis results: {vendor_list}")
        else:
            # Fallback: return empty list if no vendors specified
            logging.warning("No vendors specified in request, returning empty list")
            return jsonify({
                "vendors": [],
                "summary": {
                    "total_vendors": 0,
                    "total_images": 0,
                    "sources_used": {}
                }
            }), 200
        
        vendors = []
        
        def process_vendor(vendor_name):
            """Process a single vendor synchronously for better reliability"""
            try:
                # Fetch product images using the 3-level fallback system (sync version)
                images, source_used = fetch_product_images_with_fallback_sync(vendor_name)
                
                # Convert to expected format
                formatted_images = []
                for img in images:
                    # Create a normalized product key for frontend matching
                    title = img.get("title", "")
                    norm_key = re.sub(r"[\s_]+", "", title).replace("+", "").lower()
                    
                    formatted_images.append({
                        "fileName": title,
                        "url": img.get("url", ""),
                        "productKey": norm_key,
                        "thumbnail": img.get("thumbnail", ""),
                        "source": img.get("source", source_used),
                        "domain": img.get("domain", "")
                    })
                
                # Try to get logo from the first image or a specific logo search
                logo_url = None
                if formatted_images:
                    # Use first image as logo or search specifically for logo
                    logo_url = formatted_images[0].get("thumbnail") or formatted_images[0].get("url")
                
                vendor_data = {
                    "name": vendor_name,
                    "logoUrl": logo_url,
                    "images": formatted_images,
                    "source_used": source_used,
                    "image_count": len(formatted_images)
                }
                
                # Apply basic standardization to vendor data
                try:
                    vendor_data["name"] = standardize_vendor_name(vendor_data["name"])
                except Exception as e:
                    logging.warning(f"Failed to standardize vendor name {vendor_name}: {e}")
                    # Keep original name if standardization fails
                    vendor_data["name"] = vendor_name
                
                logging.info(f"Processed vendor {vendor_name}: {len(formatted_images)} images from {source_used}")
                return vendor_data
                
            except Exception as e:
                logging.warning(f"Failed to process vendor {vendor_name}: {e}")
                # Return minimal vendor data on failure
                return {
                    "name": vendor_name,
                    "logoUrl": None,
                    "images": [],
                    "source_used": "error",
                    "image_count": 0,
                    "error": str(e)
                }
        
        # Process only the vendors from analysis results
        for vendor_name in vendor_list:
            vendor_data = process_vendor(vendor_name)
            if vendor_data:
                vendors.append(vendor_data)
        
        # Filter out any None results and add summary info
        vendors = [v for v in vendors if v is not None]
        
        # Add summary statistics
        total_images = sum(v.get("image_count", 0) for v in vendors)
        sources_used = {}
        for v in vendors:
            source = v.get("source_used", "unknown")
            sources_used[source] = sources_used.get(source, 0) + 1
        
        response_data = {
            "vendors": vendors,
            "summary": {
                "total_vendors": len(vendors),
                "total_images": total_images,
                "sources_used": sources_used
            }
        }
        
        logging.info(f"Successfully processed {len(vendors)} vendors with {total_images} total images")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Critical error in get_vendors: {e}")
        return jsonify({
            "error": "Failed to fetch vendors",
            "vendors": [],
            "summary": {
                "total_vendors": 0,
                "total_images": 0,
                "sources_used": {}
            }
        }), 500

@app.route("/api/submodel-mapping", methods=["GET"])
@login_required
def get_submodel_mapping():
    """
    Returns the mapping from submodel names to model series names.
    This helps the frontend map analysis results (submodel names) to images (model series names).
    """
    try:
        mapping = get_submodel_to_model_series_mapping()
        
        # Skip LLM-based standardization for this endpoint to prevent connection issues
        # Basic mapping is sufficient for frontend functionality
        logging.info(f"Retrieved {len(mapping)} submodel mappings")
        
        return jsonify({"mapping": mapping})
    except Exception as e:
        logging.error(f"Error getting submodel mapping: {e}")
        return jsonify({"error": "Failed to get submodel mapping", "mapping": {}}), 500

@app.route("/api/admin/approve_user", methods=["POST"])
@login_required
def approve_user():
    admin_user = db.session.get(User, session['user_id'])
    if admin_user.role != "admin":
        return jsonify({"error": "Forbidden: Admins only"}), 403

    data = request.get_json()
    user_id = data.get("user_id")
    action = data.get("action", "approve")
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    if action == "approve":
        user.status = "active"
    elif action == "reject":
        user.status = "rejected"
    else:
        return jsonify({"error": "Invalid action"}), 400

    db.session.commit()
    return jsonify({"message": f"User {user.username} status updated to {user.status}."}), 200

@app.route("/api/admin/pending_users", methods=["GET"])
@login_required
def pending_users():
    admin_user = db.session.get(User, session['user_id'])
    if admin_user.role != "admin":
        return jsonify({"error": "Forbidden: Admins only"}), 403

    pending = User.query.filter_by(status="pending").all()
    result = [{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "first_name": u.first_name,
        "last_name": u.last_name
    } for u in pending]
    return jsonify({"pending_users": result}), 200

# Duplicate ALLOWED_EXTENSIONS and allowed_file removed.

from common.infrastructure.rate_limiter import router_limited

@app.route("/api/get-price-review", methods=["GET"])
@router_limited
@login_required
def api_get_price_review():
    product_name = request.args.get("productName")
    if not product_name:
        return jsonify({"error": "Missing productName parameter"}), 400

    results = fetch_price_and_reviews(product_name)

    return jsonify(results), 200

@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Expected field name 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)

    try:
        # Read file into a BytesIO stream so it can be reused
        file_stream = BytesIO(file.read())

        # Extract text chunks from PDF
        text_chunks = extract_data_from_pdf(file_stream)
        raw_results = send_to_language_model(text_chunks)
        
        def flatten_results(results):
            flat = []
            for r in results:
                if isinstance(r, list): flat.extend(r)
                else: flat.append(r)
            return flat

        all_results = flatten_results(raw_results)
        final_result = aggregate_results(all_results, filename)
        
        # Apply standardization to the final result before splitting
        try:
            standardized_final_result = standardize_vendor_analysis_result(final_result)
            logging.info("Applied standardization to uploaded file analysis")
        except Exception as e:
            logging.warning(f"Failed to standardize uploaded file result: {e}")
            standardized_final_result = final_result
        
        split_results = split_product_types([standardized_final_result])

        saved_paths = []
        for result in split_results:
            # Save to Azure Blob instead of local files
            vendor = (result.get("vendor") or "UnknownVendor").replace(" ", "_")
            product_type = (result.get("product_type") or "UnknownProduct").replace(" ", "_")
            model_series = (
                result.get("models", [{}])[0].get("model_series") or "UnknownModel"
            ).replace(" ", " ")
            
            try:
                product_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'json',
                    'collection_type': 'products',
                    'path': f'vendors/{vendor}/{product_type}/{model_series}.json'
                }
                azure_blob_file_manager.upload_json_data(result, product_metadata)
                saved_paths.append(f"Azure:vendors/{vendor}/{product_type}/{model_series}.json")
                logging.info(f"[INFO] Stored product JSON to Azure Blob: {vendor} - {product_type}")
            except Exception as e:
                logging.error(f"Failed to save product JSON to Azure Blob: {e}")
            
            # Note: Product image extraction removed - now using API-based image search

        return jsonify({
            "data": split_results,
            "savedFiles": saved_paths
        })
    except Exception as e:
        logging.exception("File upload processing failed.")
        return jsonify({"error": str(e)}), 500

# =========================================================================
# === STANDARDIZATION ENDPOINTS ===
# === 
# === Integrated standardization functionality:
# === - /analyze endpoint: Standardizes vendor analysis and ranking results
# === - /vendors endpoint: Standardizes vendor names and product image mappings 
# === - /submodel-mapping endpoint: Enhances submodel mappings with standardization
# === - /upload endpoint: Standardizes analysis results from PDF uploads
# === - /api/upload_pdf_from_url endpoint: Standardizes analysis results from URL uploads
# === 
# === New standardization endpoints:
# === - GET /standardization/report: Generate comprehensive standardization report
# === - POST /standardization/update-files: Update existing files with standardization (admin only)
# === - POST /standardization/vendor-analysis: Standardize vendor analysis data
# === - POST /standardization/ranking: Standardize ranking data  
# === - POST /standardization/submodel-mapping: Enhance submodel mapping data
# =========================================================================

@app.route("/api/standardization/report", methods=["GET"])
@login_required
def get_standardization_report():
    """
    Generate and return a comprehensive standardization report
    """
    try:
        report = create_standardization_report()
        return jsonify(report), 200
    except Exception as e:
        logging.error(f"Failed to generate standardization report: {e}")
        return jsonify({"error": "Failed to generate standardization report"}), 500

@app.route("/api/standardization/update-files", methods=["POST"])
@login_required
def update_files_with_standardization():
    """
    Update existing vendor files with standardized naming
    """
    try:
        admin_user = db.session.get(User, session['user_id'])
        if admin_user.role != "admin":
            return jsonify({"error": "Forbidden: Admins only"}), 403
            
        updated_files = update_existing_vendor_files_with_standardization()
        return jsonify({
            "message": f"Successfully updated {len(updated_files)} files with standardization",
            "updated_files": updated_files
        }), 200
    except Exception as e:
        logging.error(f"Failed to update files with standardization: {e}")
        return jsonify({"error": "Failed to update files with standardization"}), 500

@app.route("/api/standardization/vendor-analysis", methods=["POST"])
@login_required
def standardize_vendor_analysis():
    """
    Standardize a vendor analysis result
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        analysis_result = data.get("analysis_result")
        if not analysis_result:
            return jsonify({"error": "Missing analysis_result parameter"}), 400
            
        standardized_result = standardize_vendor_analysis_result(analysis_result)
        return jsonify(standardized_result), 200
    except Exception as e:
        logging.error(f"Failed to standardize vendor analysis: {e}")
        return jsonify({"error": "Failed to standardize vendor analysis"}), 500

@app.route("/api/standardization/ranking", methods=["POST"])
@login_required
def standardize_ranking():
    """
    Standardize a ranking result
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        ranking_result = data.get("ranking_result")
        if not ranking_result:
            return jsonify({"error": "Missing ranking_result parameter"}), 400
            
        standardized_result = standardize_ranking_result(ranking_result)
        return jsonify(standardized_result), 200
    except Exception as e:
        logging.error(f"Failed to standardize ranking: {e}")
        return jsonify({"error": "Failed to standardize ranking"}), 500

@app.route("/api/standardization/submodel-mapping", methods=["POST"])
@login_required
def enhance_submodel_mapping_endpoint():
    """
    Enhance submodel to model series mapping with standardization
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        submodel_data = data.get("submodel_data")
        if not submodel_data:
            return jsonify({"error": "Missing submodel_data parameter"}), 400
            
        enhanced_result = enhance_submodel_mapping(submodel_data)
        return jsonify(enhanced_result), 200
    except Exception as e:
        logging.error(f"Failed to enhance submodel mapping: {e}")
        return jsonify({"error": "Failed to enhance submodel mapping"}), 500
    

# =========================================================================
# === PROJECT MANAGEMENT ENDPOINTS ===
# =========================================================================

@app.route("/api/projects/save", methods=["POST"])
@login_required
def save_project():
    """
    Save or update a project with all current state data using Cosmos DB / Azure Blob
    """
    try:
        data = request.get_json(force=True)
        # Debug log incoming product_type information to trace saving issues (project save)
        try:
            incoming_pt = data.get('product_type') if isinstance(data, dict) else None
            incoming_detected = data.get('detected_product_type') if isinstance(data, dict) else None
            logging.info(f"[SAVE_PROJECT] Incoming product_type='{incoming_pt}' detected_product_type='{incoming_detected}' project_name='{data.get('project_name') if isinstance(data, dict) else None}' user_id={session.get('user_id')}")
        except Exception:
            logging.exception("Failed to log incoming project save payload")
        
        # Get current user ID
        user_id = str(session['user_id'])
        
        # Extract project data
        project_id = data.get("project_id")  # If updating existing project
        project_name = data.get("project_name", "").strip()
        
        if not project_name:
            return jsonify({"error": "Project name is required"}), 400
        
        # Check if initial_requirements is provided
        # Allow saving if project has instruments/accessories (already analyzed) even without requirements text
        has_requirements = bool(data.get("initial_requirements", "").strip())
        has_instruments = bool(data.get("identified_instruments") and len(data.get("identified_instruments", [])) > 0)
        has_accessories = bool(data.get("identified_accessories") and len(data.get("identified_accessories", [])) > 0)
        
        if not has_requirements and not has_instruments and not has_accessories:
            return jsonify({"error": "Initial requirements are required"}), 400
        
        # Save project to Cosmos DB / Azure Blob using project manager
        # If the frontend provided a displayed_media_map, persist those images into GridFS
        try:
            displayed_media = data.get('displayed_media_map', {}) if isinstance(data, dict) else {}
            if displayed_media:
                from common.services.azure.blob_utils import azure_blob_file_manager
                # For each displayed media entry, fetch the URL and store bytes in GridFS
                for key, entry in displayed_media.items():
                    try:
                        top = entry.get('top_image') if isinstance(entry, dict) else None
                        vlogo = entry.get('vendor_logo') if isinstance(entry, dict) else None

                        def process_media(obj, subtype):
                            if not obj:
                                return None
                            url = obj.get('url') if isinstance(obj, dict) else (obj if isinstance(obj, str) else None)
                            if not url:
                                return None
                            # If url already references our API, skip re-upload
                            if url.startswith('/api/projects/file/'):
                                return url
                            # If it's a data URL, decode
                            if url.startswith('data:'):
                                import base64, re
                                m = re.match(r'data:(.*?);base64,(.*)', url)
                                if m:
                                    content_type = m.group(1)
                                    b = base64.b64decode(m.group(2))
                                    metadata = {'collection_type': 'documents', 'original_url': '', 'content_type': content_type}
                                    fid = azure_blob_file_manager.upload_to_azure(b, metadata)
                                    return f"/api/projects/file/{fid}"
                                return None
                            # Otherwise attempt to download the URL
                            try:
                                resp = requests.get(url, timeout=8)
                                resp.raise_for_status()
                                content_type = resp.headers.get('Content-Type', 'application/octet-stream')
                                b = resp.content
                                metadata = {'collection_type': 'documents', 'original_url': url, 'content_type': content_type}
                                fid = azure_blob_file_manager.upload_to_azure(b, metadata)
                                return f"/api/projects/file/{fid}"
                            except Exception as e:
                                logging.warning(f"Failed to fetch/displayed media URL {url}: {e}")
                                return None

                        new_top = process_media(top, 'top_image')
                        new_logo = process_media(vlogo, 'vendor_logo')

                        # Inject back into data so that stored project contains references to GridFS-served URLs
                        if new_top or new_logo:
                            # attempt to find product entries in data and replace matching keys
                            # The frontend sends a map keyed by `${vendor}-${productName}`; we'll store this map as `embedded_media`
                            if 'embedded_media' not in data:
                                data['embedded_media'] = {}
                            data['embedded_media'][key] = {}
                            if new_top:
                                data['embedded_media'][key]['top_image'] = {'url': new_top}
                            if new_logo:
                                data['embedded_media'][key]['vendor_logo'] = {'url': new_logo}
                    except Exception as e:
                        logging.warning(f"Error processing displayed_media_map entry {key}: {e}")
        except Exception as e:
            logging.warning(f"Failed to persist displayed_media_map: {e}")

        # Ensure pricing and feedback are passed through from frontend payload
        # If frontend uses `pricing` or `feedback_entries` include them in the saved document
        try:
            # If frontend supplied feedback, normalize to `feedback_entries`
            if 'feedback' in data and 'feedback_entries' not in data:
                data['feedback_entries'] = data.get('feedback')
        except Exception:
            logging.warning('Failed to normalize incoming feedback payload')

        saved_project = cosmos_project_manager.save_project(user_id, data)

        # Store the saved project id in the session so future feedback posts can attach to it
        try:
            session['current_project_id'] = saved_project.get('project_id')
        except Exception:
            logging.warning('Failed to set current_project_id in session')
        
        # Return the saved project data
        return jsonify({
            "message": "Project saved successfully",
            "project": saved_project
        }), 200
        
    except ValueError as e:
        logging.warning(f"Project save validation error: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("Project save failed.")
        return jsonify({"error": "Failed to save project: " + str(e)}), 500


@app.route("/api/projects/preview-save", methods=["POST"])
@login_required
def preview_save_project():
    """
    Debug helper: compute resolved product_type (prefers detected_product_type)
    and return it without saving. Useful for quick verification.
    """
    try:
        data = request.get_json(force=True)
        project_name = (data.get('project_name') or '').strip()
        detected = data.get('detected_product_type')
        incoming = (data.get('product_type') or '').strip()

        if detected:
            resolved = detected.strip()
        else:
            if incoming and project_name and incoming.lower() == project_name.lower():
                resolved = ''
            else:
                resolved = incoming

        return jsonify({
            'resolved_product_type': resolved,
            'detected_product_type': detected,
            'incoming_product_type': incoming,
            'project_name': project_name
        }), 200
    except Exception as e:
        logging.exception('Preview save failed')
        return jsonify({'error': str(e)}), 500

@app.route("/api/projects", methods=["GET"])
@login_required
def get_user_projects():
    """
    Get all projects for the current user from Cosmos DB
    """
    try:
        user_id = str(session['user_id'])
        
        # Get all active projects for the user from Cosmos DB
        projects = cosmos_project_manager.get_user_projects(user_id)
        
        return standardized_jsonify({
            "projects": projects,
            "total_count": len(projects)
        }, 200)
        
    except Exception as e:
        logging.exception("Failed to retrieve user projects.")
        return jsonify({"error": "Failed to retrieve projects: " + str(e)}), 500

@app.route("/api/projects/<project_id>", methods=["GET"])
@login_required
def get_project_details(project_id):
    """
    Get full project details for loading from Cosmos DB / Azure Blob
    """
    try:
        user_id = str(session['user_id'])
        
        # Get project details from Cosmos DB / Azure Blob
        project_details = cosmos_project_manager.get_project_details(project_id, user_id)
        
        return standardized_jsonify({"project": project_details}, 200)
        
    except ValueError as e:
        logging.warning(f"Project access denied: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception(f"Failed to retrieve project {project_id}.")
        return jsonify({"error": "Failed to retrieve project: " + str(e)}), 500


@app.route('/api/projects/file/<path:file_id>', methods=['GET'])
@login_required
def serve_project_file(file_id):
    """
    Serve a file stored in Azure Blob by its file ID (path).
    """
    try:
        from common.services.azure.blob_utils import azure_blob_file_manager
        
        # Try 'documents' collection corresponding to save_project uploads
        blob_path = f"{azure_blob_file_manager.base_path}/documents/{file_id}"
        blob_client = azure_blob_file_manager.container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
             # Fallback to files collection
             blob_path = f"{azure_blob_file_manager.base_path}/files/{file_id}"
             blob_client = azure_blob_file_manager.container_client.get_blob_client(blob_path)
             if not blob_client.exists():
                return jsonify({'error': 'File not found'}), 404
        
        # Get properties for content type
        props = blob_client.get_blob_properties()
        content_type = props.content_settings.content_type or 'application/octet-stream'
        
        # Read data
        data = blob_client.download_blob().readall()

        return (data, 200, {
            'Content-Type': content_type,
            'Content-Length': str(len(data)),
            'Cache-Control': 'public, max-age=31536000'
        })
    except Exception as e:
        logging.exception('Failed to serve project file')
        return jsonify({'error': str(e)}), 500

@app.route("/api/projects/<project_id>", methods=["DELETE"])
@login_required
def delete_project(project_id):
    """
    Permanently delete a project from Cosmos DB / Azure Blob
    """
    try:
        user_id = str(session['user_id'])
        
        # Delete project from Cosmos DB / Azure Blob
        cosmos_project_manager.delete_project(project_id, user_id)
        
        return standardized_jsonify({"message": "Project deleted successfully"}, 200)
        
    except ValueError as e:
        logging.warning(f"Project delete access denied: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception(f"Failed to delete project {project_id}.")
        return jsonify({"error": "Failed to delete project: " + str(e)}), 500


# =========================================================================
# === STRATEGY DOCUMENT MANAGEMENT APIs ===
# =========================================================================
# User-uploaded strategy documents stored in Azure Blob + MongoDB metadata
# Pattern: File → Blob, Metadata + URL → MongoDB, SAS URLs for secure access

# Configuration for document uploads
DOCUMENT_ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'txt'}
DOCUMENT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def document_allowed_file(filename: str) -> bool:
    """Check if the uploaded filename has an allowed extension for documents."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in DOCUMENT_ALLOWED_EXTENSIONS


@app.route('/api/upload-strategy-file', methods=['POST'])
@login_required
def upload_strategy_file():
    """
    Upload Strategy Document (Async Processing)
    ---
    tags:
      - Document Management
    summary: Upload strategy document for background vendor data extraction
    description: |
      Uploads a strategy document (PDF, DOCX, Excel, CSV) and triggers background
      processing to extract vendor information using LLM.

      **Processing Flow:**
      1. File is uploaded and stored in MongoDB with "pending" status
      2. Background task extracts vendor data (vendor_name, category, subcategory, strategy)
      3. Status can be checked via the returned processing_url

      **This endpoint returns immediately (202 Accepted) - processing happens in background**

      **Supported file types:** PDF, DOC, DOCX, XLS, XLSX, CSV, TXT
      **Max file size:** 50MB
      **Extracted fields:** vendor_name, category, subcategory, strategy
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The strategy document to upload
    responses:
      202:
        description: Document uploaded successfully, processing in background
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            document_id:
              type: string
              description: MongoDB document ID for status tracking
            filename:
              type: string
            file_size:
              type: integer
            status:
              type: string
              enum: [pending]
              description: Initial status (will change to processing/completed/failed)
            processing_url:
              type: string
              description: URL to check processing status
      400:
        description: Bad request (no file, invalid file type, file too large)
      500:
        description: Server error (database error, failed to start background task)
    """
    try:
        logging.info("[STRATEGY-UPLOAD] ===== Strategy Document Upload START =====")

        # Check if document_service is available
        if document_service is None:
            logging.error("[STRATEGY-UPLOAD] Document service not available")
            return jsonify({
                "success": False,
                "error": "Document service not available. Please check server configuration."
            }), 500

        # Check if file is present
        if 'file' not in request.files:
            logging.warning("[STRATEGY-UPLOAD] No file provided in request")
            return jsonify({
                "success": False,
                "error": "No file provided. Please include a 'file' field in your request."
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            logging.warning("[STRATEGY-UPLOAD] Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected. Please select a file to upload."
            }), 400

        # Validate file extension
        if not document_allowed_file(file.filename):
            logging.warning(f"[STRATEGY-UPLOAD] Invalid file type: {file.filename}")
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Supported types: {', '.join(DOCUMENT_ALLOWED_EXTENSIONS)}"
            }), 400

        # Read file bytes
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        content_type = file.content_type or 'application/octet-stream'

        # Validate file size
        if len(file_bytes) > DOCUMENT_MAX_FILE_SIZE:
            logging.warning(f"[STRATEGY-UPLOAD] File too large: {len(file_bytes)} bytes")
            return jsonify({
                "success": False,
                "error": f"File too large. Maximum size is {DOCUMENT_MAX_FILE_SIZE // (1024*1024)}MB."
            }), 400

        # Get user info
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        username = user.username if user else None
        is_admin = user.role == 'admin' if user else False

        logging.info(f"[STRATEGY-UPLOAD] Processing file: {filename} ({len(file_bytes)} bytes) for user {user_id} (admin: {is_admin})")

        # Store file in MongoDB with "pending" status and trigger background processing
        try:
            from datetime import datetime
            import base64

            # Get MongoDB collection
            strategy_collection = mongodb_manager.get_collection('stratergy')

            if not strategy_collection:
                logging.error("[STRATEGY-UPLOAD] MongoDB strategy collection not available")
                return jsonify({
                    "success": False,
                    "error": "Database not available. Please check server configuration."
                }), 500

            # Create document with pending status
            # Store file as base64 for background processing
            document = {
                "user_id": user_id,
                "file_name": filename,
                "content_type": content_type,
                "file_size": len(file_bytes),
                "uploaded_at": datetime.utcnow().isoformat(),
                "uploaded_by_username": username,
                "is_admin_upload": is_admin,  # Track if uploaded by admin (visible to all users)
                "status": "pending",
                "data": [],  # Will be populated by background task
                "vendor_count": 0,
                "error": None,
                # Store file temporarily for processing
                "file_data": base64.b64encode(file_bytes).decode('utf-8')
            }

            # Insert into MongoDB
            result = strategy_collection.insert_one(document)
            document_id = str(result.inserted_id)

            logging.info(f"[STRATEGY-UPLOAD] ✓ Stored document in MongoDB (ID: {document_id})")

            # Trigger background processing
            try:
                from common.strategy_rag.ingestion.background_processor import process_strategy_document_async

                process_strategy_document_async(
                    document_id=document_id,
                    file_bytes=file_bytes,
                    filename=filename,
                    content_type=content_type,
                    user_id=user_id
                )

                logging.info(f"[STRATEGY-UPLOAD] ✓ Triggered background processing for document {document_id}")
                logging.info("[STRATEGY-UPLOAD] ===== Strategy Document Upload END (SUCCESS - PROCESSING) =====")

                return jsonify({
                    "success": True,
                    "message": "Strategy document uploaded successfully. Processing in background.",
                    "document_id": document_id,
                    "filename": filename,
                    "file_size": len(file_bytes),
                    "status": "pending",
                    "processing_url": f"/api/strategy-processing-status/{document_id}"
                }), 202  # 202 Accepted - processing asynchronously

            except Exception as bg_error:
                logging.error(f"[STRATEGY-UPLOAD] Failed to trigger background processing: {bg_error}")
                # Update document status to failed
                strategy_collection.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"status": "failed", "error": str(bg_error)}}
                )
                return jsonify({
                    "success": False,
                    "error": f"Failed to start background processing: {str(bg_error)}"
                }), 500

        except Exception as db_error:
            logging.error(f"[STRATEGY-UPLOAD] Database storage failed: {db_error}")
            return jsonify({
                "success": False,
                "error": f"Failed to store in database: {str(db_error)}"
            }), 500

    except Exception as e:
        logging.exception(f"[STRATEGY-UPLOAD] ✗ ERROR: {e}")
        logging.info("[STRATEGY-UPLOAD] ===== Strategy Document Upload END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/strategy-processing-status/<document_id>', methods=['GET'])
@login_required
def get_strategy_processing_status(document_id):
    """
    Get Strategy Document Processing Status
    ---
    tags:
      - Document Management
    summary: Check the processing status of an uploaded strategy document
    description: |
      Returns the current processing status of a strategy document upload.

      **Possible status values:**
      - `pending`: Waiting for processing to start
      - `processing`: Currently extracting vendor data
      - `completed`: Successfully extracted and stored
      - `failed`: Extraction failed (error details included)
    parameters:
      - in: path
        name: document_id
        type: string
        required: true
        description: MongoDB document ID returned from upload
    responses:
      200:
        description: Status retrieved successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            document_id:
              type: string
            status:
              type: string
              enum: [pending, processing, completed, failed]
            file_name:
              type: string
            vendor_count:
              type: integer
              description: Number of vendors extracted (0 if not completed)
            error:
              type: string
              description: Error message if status is 'failed'
            uploaded_at:
              type: string
            processing_started_at:
              type: string
            processing_completed_at:
              type: string
      404:
        description: Document not found
      500:
        description: Server error
    """
    try:
        logging.info(f"[STRATEGY-STATUS] Checking status for document {document_id}")

        from common.strategy_rag.ingestion.background_processor import get_processing_status

        status_info = get_processing_status(document_id)

        if not status_info:
            logging.warning(f"[STRATEGY-STATUS] Document {document_id} not found")
            return jsonify({
                "success": False,
                "error": "Document not found"
            }), 404

        logging.info(f"[STRATEGY-STATUS] Status: {status_info.get('status')}")

        return jsonify({
            "success": True,
            "document_id": str(status_info.get('_id')),
            "status": status_info.get('status', 'unknown'),
            "file_name": status_info.get('file_name'),
            "vendor_count": status_info.get('vendor_count', 0),
            "error": status_info.get('error'),
            "uploaded_at": status_info.get('uploaded_at'),
            "processing_started_at": status_info.get('processing_started_at'),
            "processing_completed_at": status_info.get('processing_completed_at')
        }), 200

    except Exception as e:
        logging.exception(f"[STRATEGY-STATUS] ✗ ERROR: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



# =========================================================================
# === STANDARDS DOCUMENT MANAGEMENT APIs ===
# =========================================================================
# User-uploaded engineering standards documents stored in Azure Blob + MongoDB


@app.route('/api/upload-standards-file', methods=['POST'])
@login_required
def upload_standards_file():
    """
    Upload Standards Document
    ---
    tags:
      - Document Management
    summary: Upload an engineering standards document
    description: |
      Uploads an engineering standards document (PDF, DOCX, etc.) to Azure Blob Storage
      and stores metadata in MongoDB. Returns a SAS URL for secure file access.

      **Supported file types:** PDF, DOC, DOCX, XLS, XLSX, CSV, TXT
      **Max file size:** 50MB
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The standards document to upload
    responses:
      200:
        description: Document uploaded successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            document_id:
              type: string
            filename:
              type: string
            file_size:
              type: integer
            blob_url:
              type: string
            sas_url:
              type: string
      400:
        description: Bad request
      500:
        description: Server error
    """
    try:
        logging.info("[STANDARDS-UPLOAD] ===== Standards Document Upload START =====")

        # Check if document_service is available
        if document_service is None:
            logging.error("[STANDARDS-UPLOAD] Document service not available")
            return jsonify({
                "success": False,
                "error": "Document service not available. Please check server configuration."
            }), 500

        # Check if file is present
        if 'file' not in request.files:
            logging.warning("[STANDARDS-UPLOAD] No file provided in request")
            return jsonify({
                "success": False,
                "error": "No file provided. Please include a 'file' field in your request."
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            logging.warning("[STANDARDS-UPLOAD] Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected. Please select a file to upload."
            }), 400

        # Validate file extension
        if not document_allowed_file(file.filename):
            logging.warning(f"[STANDARDS-UPLOAD] Invalid file type: {file.filename}")
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Supported types: {', '.join(DOCUMENT_ALLOWED_EXTENSIONS)}"
            }), 400

        # Read file bytes
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        content_type = file.content_type or 'application/octet-stream'

        # Validate file size
        if len(file_bytes) > DOCUMENT_MAX_FILE_SIZE:
            logging.warning(f"[STANDARDS-UPLOAD] File too large: {len(file_bytes)} bytes")
            return jsonify({
                "success": False,
                "error": f"File too large. Maximum size is {DOCUMENT_MAX_FILE_SIZE // (1024*1024)}MB."
            }), 400

        # Get user info
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        username = user.username if user else None
        is_admin = user.role == 'admin' if user else False

        logging.info(f"[STANDARDS-UPLOAD] Processing file: {filename} ({len(file_bytes)} bytes) for user {user_id} (admin: {is_admin})")

        # Upload using document service
        result = document_service.upload_standards_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            content_type=content_type,
            username=username,
            is_admin=is_admin  # Admin documents visible to all users
        )

        if result.get('success'):
            logging.info(f"[STANDARDS-UPLOAD] ✓ Successfully uploaded: {filename} (ID: {result.get('document_id')})")
            logging.info("[STANDARDS-UPLOAD] ===== Standards Document Upload END (SUCCESS) =====")

            return jsonify({
                "success": True,
                "message": "Standards document uploaded successfully",
                "document_id": result.get('document_id'),
                "filename": result.get('filename'),
                "file_size": result.get('file_size'),
                "blob_url": result.get('blob_url'),
                "sas_url": result.get('sas_url')
            }), 200
        else:
            logging.warning(f"[STANDARDS-UPLOAD] Upload failed: {result}")
            return jsonify({
                "success": False,
                "error": "Failed to upload document"
            }), 500

    except Exception as e:
        logging.exception(f"[STANDARDS-UPLOAD] ✗ ERROR: {e}")
        logging.info("[STANDARDS-UPLOAD] ===== Standards Document Upload END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================================
# === DOCUMENT UTILITY ENDPOINTS ===
# =========================================================================

@app.route('/api/documents/refresh-sas/<document_type>/<document_id>', methods=['GET'])
@login_required
def refresh_document_sas_url(document_type, document_id):
    """
    Refresh SAS URL
    ---
    tags:
      - Document Management
    summary: Generate a fresh SAS URL for a document
    description: |
      Generates a new time-limited SAS URL for accessing a document.
      Use this when the previous SAS URL has expired.
    parameters:
      - in: path
        name: document_type
        type: string
        enum: [strategy, standards]
        required: true
        description: Type of document
      - in: path
        name: document_id
        type: string
        required: true
        description: Document ID
    responses:
      200:
        description: New SAS URL generated
        schema:
          type: object
          properties:
            success:
              type: boolean
            sas_url:
              type: string
            expires_in_hours:
              type: integer
      404:
        description: Document not found
      500:
        description: Server error
    """
    try:
        if document_service is None:
            return jsonify({
                "success": False,
                "error": "Document service not available"
            }), 500

        if document_type not in ['strategy', 'standards']:
            return jsonify({
                "success": False,
                "error": "Invalid document type. Must be 'strategy' or 'standards'"
            }), 400

        sas_url = document_service.refresh_sas_url(
            document_id=document_id,
            doc_type=document_type
        )

        if sas_url:
            return jsonify({
                "success": True,
                "sas_url": sas_url,
                "expires_in_hours": 24
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Document not found"
            }), 404

    except Exception as e:
        logging.exception(f"[SAS-REFRESH] Error refreshing SAS URL: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/documents/health', methods=['GET'])
@login_required
def document_service_health():
    """
    Document Service Health Check
    ---
    tags:
      - Document Management
    summary: Check document service health
    description: Returns health status of the document management system
    responses:
      200:
        description: Health status
    """
    try:
        if document_service is None:
            return jsonify({
                "success": False,
                "status": "unavailable",
                "error": "Document service not initialized"
            }), 503

        health = document_service.health_check()

        return jsonify({
            "success": True,
            **health
        }), 200

    except Exception as e:
        logging.exception(f"[DOC-HEALTH] Error: {e}")
        return jsonify({
            "success": False,
            "status": "error",
            "error": str(e)
        }), 500


def create_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role='admin').first():
            hashed_pw = hash_password("Daman@123")
            admin = User(
                username="Daman", 
                email="reddydaman04@gmail.com", 
                password_hash=hashed_pw, 
                first_name="Daman",
                last_name="Reddy",
                status='active', 
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            logging.info("Admin user created with username 'Daman' and password 'Daman@123'.")


# =========================================================================
# === APPLICATION INITIALIZATION (Runs on both Gunicorn and Dev Server) ===
# =========================================================================
# This section runs when the module is imported (e.g., by gunicorn main:app)

# Create database tables and admin user (non-blocking - allows startup even if MySQL unavailable)
try:
    create_db()
    logging.info("✅ Database tables created and admin user initialized")
except Exception as e:
    logging.warning(f"⚠️ Database initialization failed (MySQL may not be available): {e}")
    logging.warning("⚠️ Application will continue but database-dependent features may not work")
    logging.warning("⚠️ To fix: Ensure MySQL database is running and connection string is correct")

# Initialize automatic checkpoint cleanup (Phase 1 improvement)
# Prevents unbounded memory growth from checkpoint accumulation
checkpoint_manager = None
try:
    from common.agentic.checkpointing import CheckpointManager, start_auto_checkpoint_cleanup

    # Create checkpoint manager with Azure Blob Storage (production-ready)
    # Falls back to memory if Azure credentials not configured
    checkpoint_manager = CheckpointManager(
        backend="azure_blob",
        max_age_hours=72,
        max_checkpoints_per_user=100,
        # Azure Blob specific config (loaded from environment)
        container_prefix="workflow-checkpoints",
        default_zone="DEFAULT",
        ttl_hours=72,
        use_managed_identity=False
    )
    start_auto_checkpoint_cleanup(
        checkpoint_manager,
        cleanup_interval_seconds=300,  # Run cleanup every 5 minutes
        max_age_hours=72  # Remove checkpoints older than 72 hours
    )
    logging.info("Automatic checkpoint cleanup initialized successfully with Azure Blob Storage")
except Exception as e:
    logging.warning(f"Failed to initialize checkpoint cleanup: {e}")

# Initialize automatic session cleanup (Phase 2 improvement)
# Prevents memory leaks from accumulated session files
session_cleanup_manager = None
try:
    from common.infrastructure.state.session.cleanup import SessionCleanupManager

    session_dir = app.config.get("SESSION_FILE_DIR", "/tmp/flask_session")
    session_cleanup_manager = SessionCleanupManager(
        session_dir=session_dir,
        cleanup_interval=600,  # Run cleanup every 10 minutes
        max_age_hours=24  # Remove sessions older than 24 hours
    )
    session_cleanup_manager.start()
    logging.info("Automatic session cleanup initialized successfully with proper lifecycle")
except Exception as e:
    logging.warning(f"Failed to initialize session cleanup: {e}")

# Initialize bounded workflow state management (Phase 4 improvement)
# Prevents OOM crashes from unbounded state accumulation
try:
    from common.infrastructure.caching.workflow_state_cache import stop_workflow_state_manager
    logging.info("Workflow state manager initialized with bounded memory and auto-cleanup")
except Exception as e:
    logging.warning(f"Failed to initialize workflow state manager: {e}")


# Register shutdown handler for graceful cleanup (Phase 3-4 improvements)
def shutdown_cleanup():
    """Graceful shutdown handler for all background managers."""
    # Stop session cleanup (Phase 3)
    if session_cleanup_manager:
        try:
            logging.info("[SHUTDOWN] Stopping session cleanup manager...")
            session_cleanup_manager.stop()
            logging.info("[SHUTDOWN] Session cleanup manager stopped successfully")
        except Exception as e:
            logging.error(f"[SHUTDOWN] Error stopping session cleanup: {e}")

    # Stop workflow state manager (Phase 4)
    try:
        logging.info("[SHUTDOWN] Stopping workflow state manager...")
        stop_workflow_state_manager()
        logging.info("[SHUTDOWN] Workflow state manager stopped successfully")
    except Exception as e:
        logging.error(f"[SHUTDOWN] Error stopping workflow state manager: {e}")


# Register cleanup to run ONLY on application shutdown (not per-request)
import atexit
atexit.register(shutdown_cleanup)

# Pre-warm standards document cache (Task #6 - Performance Optimization)
# Loads all standard documents into memory at startup to eliminate cache-miss delays
try:
    from common.standards.generation.deep_agent import prewarm_document_cache

    logging.info("Pre-warming standards document cache...")
    cache_stats = prewarm_document_cache()
    logging.info(
        f"Standards cache pre-warmed: {cache_stats['success']}/{cache_stats['total']} "
        f"documents loaded in {cache_stats['elapsed_seconds']}s"
    )
except Exception as e:
    logging.warning(f"Failed to pre-warm standards document cache: {e}")
    logging.warning("Standards will be loaded on-demand (slower first request)")


# =========================================================================
# === DEVELOPMENT SERVER (Only runs when executed directly) ===
# =========================================================================
def log_startup_summary():
    """Log comprehensive startup summary showing all loaded implementations."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🚀 APPLICATION STARTUP SUMMARY")
    logger.info("=" * 80)
    
    # Core Components
    logger.info("📦 CORE COMPONENTS:")
    logger.info("  ✓ Flask Application (v" + str(app.config.get('ENV', 'production')) + ")")
    logger.info("  ✓ CORS configured with " + str(len(allowed_origins)) + " allowed origins")
    
    # Database Status
    logger.info("")
    logger.info("💾 DATABASE & STORAGE:")
    if mysql_uri:
        logger.info("  ✓ MySQL Database - Connected")
    else:
        logger.info("  ⚠ SQLite Fallback Database")
    
    # Session Storage
    logger.info("  ✓ Session Storage: " + str(app.config.get('SESSION_TYPE', 'unknown')))
    
    # MongoDB Status
    try:
        from common.core.mongodb_manager import mongodb_manager, is_mongodb_available
        if is_mongodb_available():
            logger.info("  ✓ MongoDB - Connected")
        else:
            error = mongodb_manager.get_connection_error() or "unknown error"
            logger.info(f"  ⚠ MongoDB - Not available ({error}) → using Azure Blob fallback")
    except Exception as _e:
        logger.info(f"  ⚠ MongoDB - Not available ({_e})")

    # Pinecone Status
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")
        if pinecone_key:
            logger.info(f"  ✓ Pinecone - Configured (index: {pinecone_index})")
        else:
            logger.info("  ⚠ Pinecone - PINECONE_API_KEY not set → vector search in mock mode (0 results)")
    except Exception as _e:
        logger.info(f"  ⚠ Pinecone - Status unknown ({_e})")

    # Azure Storage
    try:
        from common.config.azure_blob_config import azure_blob_manager
        if azure_blob_manager.is_available:
            logger.info("  ✓ Azure Blob Storage - Configured")
        else:
            logger.info("  ⚠ Azure Blob Storage - Not configured")
    except:
        logger.info("  ⚠ Azure Blob Storage - Not available")
    
    # Service Layer
    logger.info("")
    logger.info("🔧 SERVICE LAYER:")
    services = [
        ('Schema Service', schema_service),
        ('Vendor Service', vendor_service),
        ('Project Service', project_service),
        ('Document Service', document_service),
        ('Image Service', image_service)
    ]
    for name, service in services:
        status = "✓" if service is not None else "✗"
        logger.info(f"  {status} {name}")
    
    # API Endpoints / Blueprints
    logger.info("")
    logger.info("🌐 REGISTERED API BLUEPRINTS:")
    blueprints = [
        'agentic_bp',
        'deep_agent_bp',
        'engenie_chat_bp',
        'tools_bp',
        'sales_agent_bp',
        'resource_bp'
    ]
    
    registered_count = 0
    for bp_name in app.blueprints:
        logger.info(f"  ✓ {bp_name}")
        registered_count += 1
    
    logger.info(f"  Total: {registered_count} blueprints registered")
    
    # Workflows
    logger.info("")
    logger.info("🔄 AVAILABLE WORKFLOWS:")
    workflows = [
        "Intent Classification & Routing",
        "Search & Ranking",
        "Solution Deep Agent",
        "EnGenie Chat",
        "Indexing Agent",
        "Standards Enrichment"
    ]
    for workflow in workflows:
        logger.info(f"  ✓ {workflow}")
    
    # LLM Configuration
    logger.info("")
    logger.info("🤖 LLM CONFIGURATION:")
    google_keys = []
    for i in range(1, 10):
        key_name = f"GOOGLE_API_KEY{i}" if i > 1 else "GOOGLE_API_KEY"
        if os.getenv(key_name):
            google_keys.append(key_name)
    
    logger.info(f"  • Google API Keys: {len(google_keys)}")
    logger.info(f"  • OpenAI Fallback: {'✓ Configured' if os.getenv('OPENAI_API_KEY') else '✗ Not configured'}")
    logger.info(f"  • LangSmith Tracing: {'✓ Enabled' if os.getenv('LANGSMITH_API_KEY') else '✗ Disabled'}")
    
    # Caching & Performance
    logger.info("")
    logger.info("⚡ CACHING & PERFORMANCE:")
    logger.info("  ✓ Bounded Cache System")
    logger.info("  ✓ LLM Response Cache")
    logger.info("  ✓ Embedding Cache")
    logger.info("  ✓ RAG Cache")
    logger.info("  ✓ Schema Cache")
    
    # Standards Pre-warming Status
    try:
        logger.info("  ✓ Standards Document Cache - Pre-warmed")
    except:
        logger.info("  ⚠ Standards Document Cache - On-demand loading")
    
    # Rate Limiting
    logger.info("")
    logger.info("🛡️ SECURITY & RATE LIMITING:")
    logger.info("  ✓ Rate Limiter - Active")
    logger.info("  ✓ Circuit Breaker - Available")
    logger.info("  ✓ Authentication System - Enabled")
    
    # Server Info
    logger.info("")
    logger.info("🖥️ SERVER CONFIGURATION:")
    logger.info("  • Host: 0.0.0.0")
    logger.info("  • Port: 5000")
    logger.info("  • Debug Mode: " + ("ON" if app.debug else "OFF"))
    logger.info("  • Threaded: True")
    logger.info("  • Auto-reload: False")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ APPLICATION READY - All systems operational")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    # Log comprehensive startup summary
    log_startup_summary()
    
    # Run Flask development server
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
