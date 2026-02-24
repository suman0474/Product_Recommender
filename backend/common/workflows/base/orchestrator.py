# agentic/workflows/base/orchestrator.py
# LangGraph Workflow Orchestrator - Prompt Integration System
#
# This module provides the integration layer between prompts_library
# and LangGraph workflows. It creates a unified system for:
# 1. Managing prompt-based agents
# 2. Tool definitions for each prompt type
# 3. Workflow orchestration with LangGraph
#
# NOTE: This excludes Strategy → Judge → Ranking → Comparison chains (per user request)

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from enum import Enum
from functools import wraps

from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# Local imports
from common.services.llm.fallback import create_llm_with_fallback
from common.utils.llm_manager import get_cached_llm
from common.prompts import (
    INTENT_CLASSIFICATION_PROMPTS,
    INTENT_PROMPTS,
    SOLUTION_DEEP_AGENT_PROMPTS,
    SCHEMA_VALIDATION_PROMPT,
    INDEX_RAG_PROMPTS
)
logger = logging.getLogger(__name__)

# Using consolidated prompts directly:
_INTENT_CLASSIFICATION_PROMPTS = {
    "CLASSIFICATION": INTENT_CLASSIFICATION_PROMPTS.get("DEFAULT", ""),
    "QUICK_CLASSIFICATION": INTENT_CLASSIFICATION_PROMPTS.get("QUICK_CLASSIFICATION", "")
}

_INTENT_ANALYSIS_PROMPTS = {
    "REQUIREMENTS_EXTRACTION": INTENT_PROMPTS
}

# Load unified product identification prompts
_PRODUCT_ID_PROMPTS = SOLUTION_DEEP_AGENT_PROMPTS


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

T = TypeVar('T')


class PromptCategory(str, Enum):
    """Categories of prompts for organization"""
    VALIDATION = "validation"
    INTENT = "intent"
    IDENTIFICATION = "identification"
    SALES_AGENT = "sales_agent"
    FEEDBACK = "feedback"
    RAG = "rag"
    SEARCH = "search"
    KNOWLEDGE = "knowledge"


# ============================================================================
# PROMPT REGISTRY - Central registry of all available prompts
# ============================================================================

class PromptRegistry:
    """
    Central registry for all prompts from prompts.py
    
    Provides:
    - Organized access to prompts by category
    - Validation that prompts exist
    - Documentation of prompt usage
    """
    
    def __init__(self):
        self._prompts: Dict[str, ChatPromptTemplate] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Load all prompts from prompts_library"""

        # Load individual prompts
        validation_prompt = SCHEMA_VALIDATION_PROMPT
        identify_instrument_prompt = _PRODUCT_ID_PROMPTS.get("INSTRUMENT_IDENTIFICATION", "")
        advanced_parameter_selection_prompt = ""
        
        # Validation prompts
        self.register("validation", validation_prompt, {
            "category": PromptCategory.VALIDATION,
            "description": "Validates user requirements against schema",
            "inputs": ["user_input", "schema", "format_instructions"],
            "output_type": "json"
        })
        
        self.register("requirements", _INTENT_ANALYSIS_PROMPTS["REQUIREMENTS_EXTRACTION"], {
            "category": PromptCategory.VALIDATION,
            "description": "Extracts structured requirements from user input",
            "inputs": ["user_input"],
            "output_type": "text"
        })

        self.register("additional_requirements", _INTENT_ANALYSIS_PROMPTS["REQUIREMENTS_EXTRACTION"], {
            "category": PromptCategory.VALIDATION,
            "description": "Extracts new requirements from user input",
            "inputs": ["product_type", "user_input", "schema", "format_instructions"],
            "output_type": "json"
        })

        # Intent classification prompts
        self.register("intent_classifier", _INTENT_CLASSIFICATION_PROMPTS["CLASSIFICATION"], {
            "category": PromptCategory.INTENT,
            "description": "Classifies user intent for workflow routing",
            "inputs": ["current_step", "current_intent", "user_input"],
            "output_type": "json"
        })

        self.register("identify_classification", _INTENT_CLASSIFICATION_PROMPTS["CLASSIFICATION"], {
            "category": PromptCategory.INTENT,
            "description": "Classifies input as greeting/requirements/question/unrelated",
            "inputs": ["user_input"],
            "output_type": "json"
        })
        
        # Identification prompts
        self.register("instrument_identifier", identify_instrument_prompt, {
            "category": PromptCategory.IDENTIFICATION,
            "description": "Identifies instruments from requirements",
            "inputs": ["requirements", "format_instructions"],
            "output_type": "json"
        })
        
        self.register("identify_instrument", identify_instrument_prompt, {
            "category": PromptCategory.IDENTIFICATION,
            "description": "Full BOM identification from requirements",
            "inputs": ["requirements"],
            "output_type": "json"
        })
        
        # Sales Agent prompts
        # Sales Agent prompts removed

        
        # Feedback prompts
        self.register("feedback_positive", "Thank you for your positive feedback!", {
            "category": PromptCategory.FEEDBACK,
            "description": "Response to positive feedback",
            "inputs": ["comment"],
            "output_type": "text"
        })
        
        self.register("feedback_negative", "Thank you for your feedback. We'll use it to improve.", {
            "category": PromptCategory.FEEDBACK,
            "description": "Response to negative feedback",
            "inputs": ["comment"],
            "output_type": "text"
        })
        
        self.register("feedback_comment", "Thank you for your comment: {comment}", {
            "category": PromptCategory.FEEDBACK,
            "description": "Response to general comment",
            "inputs": ["comment"],
            "output_type": "text"
        })
        
        # RAG/Knowledge prompts
        # Load grounded_chat from RAG prompts if available, otherwise use basic template
        try:
            grounded_chat_prompt = INDEX_RAG_PROMPTS.get("CHAT_AGENT", "Answer the question: {user_question}")
        except:
            grounded_chat_prompt = "Answer the question: {user_question}"
        
        self.register("grounded_chat", grounded_chat_prompt, {
            "category": PromptCategory.RAG,
            "description": "RAG-grounded response with 3 sources",
            "inputs": ["product_type", "specifications", "user_question", 
                      "strategy_context", "standards_context", "inventory_context",
                      "format_instructions"],
            "output_type": "json"
        })
        
        # Search prompts
        self.register("advanced_parameter_selection", advanced_parameter_selection_prompt, {
            "category": PromptCategory.SEARCH,
            "description": "Extract advanced parameters from user input",
            "inputs": ["product_type", "available_parameters", "user_input"],
            "output_type": "json"
        })
        
        # Validation alert prompts
        self.register("validation_alert_initial", "Please review the validation results.", {
            "category": PromptCategory.VALIDATION,
            "description": "First alert about missing fields",
            "inputs": ["product_type", "missing_fields"],
            "output_type": "text"
        })
        
        self.register("validation_alert_repeat", "Please address the validation issues.", {
            "category": PromptCategory.VALIDATION,
            "description": "Repeat alert about missing fields",
            "inputs": ["missing_fields"],
            "output_type": "text"
        })
        
        # Identification response prompts
        self.register("identify_greeting", "", {
            "category": PromptCategory.IDENTIFICATION,
            "description": "Response to greeting in identification flow",
            "inputs": ["user_input"],
            "output_type": "text"
        })
        
        self.register("identify_unrelated", _INTENT_CLASSIFICATION_PROMPTS["CLASSIFICATION"], {
            "category": PromptCategory.IDENTIFICATION,
            "description": "Response to non-industrial content",
            "inputs": ["reasoning"],
            "output_type": "text"
        })

        self.register("identify_fallback", _INTENT_CLASSIFICATION_PROMPTS["CLASSIFICATION"], {
            "category": PromptCategory.IDENTIFICATION,
            "description": "Fallback response for identification",
            "inputs": ["requirements"],
            "output_type": "text"
        })
        
        logger.info(f"[ORCHESTRATOR] Registered {len(self._prompts)} prompts")
    
    def register(
        self, 
        name: str, 
        prompt: ChatPromptTemplate, 
        metadata: Dict[str, Any]
    ):
        """Register a prompt with metadata"""
        self._prompts[name] = prompt
        self._metadata[name] = metadata
    
    def get(self, name: str) -> ChatPromptTemplate:
        """Get a prompt by name"""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found in registry")
        return self._prompts[name]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a prompt"""
        return self._metadata.get(name, {})
    
    def list_by_category(self, category: PromptCategory) -> List[str]:
        """List all prompts in a category"""
        return [
            name for name, meta in self._metadata.items()
            if meta.get("category") == category
        ]
    
    def list_all(self) -> List[str]:
        """List all registered prompt names"""
        return list(self._prompts.keys())


# Global prompt registry instance
prompt_registry = PromptRegistry()


# ============================================================================
# TOOL FACTORY - Creates LangChain tools from prompts
# ============================================================================

class ToolFactory:
    """
    Factory for creating LangChain tools from prompts.
    
    Each tool wraps a prompt with:
    - Input validation
    - LLM invocation
    - Output parsing
    - Error handling
    - Logging
    """
    
    def __init__(self, default_model: str = "gemini-2.5-flash"):
        self.default_model = default_model
        self.tools: Dict[str, BaseTool] = {}
    
    def get_llm(self, temperature: float = 0.1):
        """Get cached LLM instance for better performance"""
        return get_cached_llm(model=self.default_model, temperature=temperature)
    
    def create_tool_from_prompt(
        self,
        name: str,
        prompt: ChatPromptTemplate,
        description: str,
        output_type: str = "text",
        temperature: float = 0.1
    ) -> BaseTool:
        """
        Create a LangChain tool from a prompt template.
        
        Args:
            name: Tool name
            prompt: ChatPromptTemplate to use
            description: Tool description
            output_type: "text" or "json"
            temperature: LLM temperature
            
        Returns:
            StructuredTool instance
        """
        llm = self.get_llm(temperature)
        
        if output_type == "json":
            parser = JsonOutputParser()
        else:
            parser = StrOutputParser()
        
        chain = prompt | llm | parser
        
        def tool_func(**kwargs) -> Union[str, Dict[str, Any]]:
            """Execute the prompt chain"""
            try:
                logger.debug(f"[TOOL:{name}] Invoking with: {list(kwargs.keys())}")
                result = chain.invoke(kwargs)
                logger.debug(f"[TOOL:{name}] Result type: {type(result)}")
                return result
            except Exception as e:
                logger.error(f"[TOOL:{name}] Error: {e}")
                if output_type == "json":
                    return {"error": str(e), "success": False}
                return f"Error: {str(e)}"
        
        # Get input schema from prompt
        input_variables = prompt.input_variables
        
        tool_instance = StructuredTool.from_function(
            func=tool_func,
            name=name,
            description=description
        )
        
        self.tools[name] = tool_instance
        return tool_instance
    
    def create_all_tools(self) -> Dict[str, BaseTool]:
        """Create tools for all registered prompts"""
        
        for prompt_name in prompt_registry.list_all():
            prompt = prompt_registry.get(prompt_name)
            metadata = prompt_registry.get_metadata(prompt_name)
            
            # Skip if tool already exists
            if prompt_name in self.tools:
                continue
            
            self.create_tool_from_prompt(
                name=f"{prompt_name}_tool",
                prompt=prompt,
                description=metadata.get("description", f"Tool for {prompt_name}"),
                output_type=metadata.get("output_type", "text")
            )
        
        logger.info(f"[ORCHESTRATOR] Created {len(self.tools)} tools")
        return self.tools
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: PromptCategory) -> List[BaseTool]:
        """Get all tools in a category"""
        prompt_names = prompt_registry.list_by_category(category)
        return [
            self.tools.get(f"{name}_tool")
            for name in prompt_names
            if f"{name}_tool" in self.tools
        ]


# Global tool factory instance
tool_factory = ToolFactory()


# ============================================================================
# INDIVIDUAL TOOL DEFINITIONS
# ============================================================================

@tool("classify_user_intent")
def classify_user_intent_tool(
    user_input: str,
    current_step: str = "start",
    current_intent: str = "none"
) -> Dict[str, Any]:
    """
    Classify user intent for workflow routing.
    
    Returns:
        - intent: greeting, productRequirements, knowledgeQuestion, workflow, chitchat, other
        - nextStep: suggested next workflow step
        - resumeWorkflow: whether to resume current workflow
    """
    import time
    
    max_retries = 3
    base_retry_delay = 15
    last_error = None
    
    for attempt in range(max_retries):
        try:
            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            prompt = _INTENT_CLASSIFICATION_PROMPTS["CLASSIFICATION"]
            chain = prompt | llm | JsonOutputParser()
            
            result = chain.invoke({
                "user_input": user_input,
                "current_step": current_step,
                "current_intent": current_intent
            })
            
            return {
                "success": True,
                "intent": result.get("intent", "other"),
                "nextStep": result.get("nextStep"),
                "resumeWorkflow": result.get("resumeWorkflow", False)
            }
            
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = any(x in error_msg for x in ['429', 'Resource exhausted', 'RESOURCE_EXHAUSTED', 'quota'])
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = base_retry_delay * (2 ** attempt)
                logger.warning(f"[TOOL] Intent classification rate limit, retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)
                last_error = e
                continue
            
            logger.error(f"[TOOL] Intent classification error: {e}")
            return {
                "success": False,
                "intent": "other",
                "error": str(e)
            }


@tool("identify_instruments")
def identify_instruments_tool(requirements: str) -> Dict[str, Any]:
    """
    Identify instruments and accessories from requirements.
    
    Returns:
        - project_name: Identified project name
        - instruments: List of identified instruments
        - accessories: List of identified accessories
        - summary: Brief summary
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = _PRODUCT_ID_PROMPTS.get("INSTRUMENT_IDENTIFICATION", "")
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({"requirements": requirements})
        
        return {
            "success": True,
            "project_name": result.get("project_name", "Unknown Project"),
            "instruments": result.get("instruments", []),
            "accessories": result.get("accessories", []),
            "summary": result.get("summary", "")
        }
        
    except Exception as e:
        logger.error(f"[TOOL] Instrument identification error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool("validate_requirements")
def validate_requirements_tool(
    user_input: str,
    product_schema: str,
    format_instructions: str = ""
) -> Dict[str, Any]:
    """
    Validate user requirements against a product schema.
    
    Returns:
        - is_valid: Whether requirements are valid
        - product_type: Detected product type
        - provided_requirements: Extracted requirements
        - missing_fields: List of missing required fields
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = load_prompt("schema_validation_prompt")
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_input": user_input,
            "schema": product_schema,
            "format_instructions": format_instructions
        })
        
        return {
            "success": True,
            "is_valid": result.get("is_valid", False),
            "product_type": result.get("product_type"),
            "provided_requirements": result.get("provided_requirements", {}),
            "missing_fields": result.get("missing_fields", [])
        }
        
    except Exception as e:
        logger.error(f"[TOOL] Validation error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool("extract_additional_requirements")
def extract_additional_requirements_tool(
    user_input: str,
    product_type: str,
    product_schema: str
) -> Dict[str, Any]:
    """
    Extract additional requirements from user input.
    
    Returns:
        - provided_requirements: New requirements extracted
        - explanation: Brief explanation of extraction
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = _INTENT_ANALYSIS_PROMPTS["REQUIREMENTS_EXTRACTION"]
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "schema": product_schema,
            "format_instructions": ""
        })
        
        return {
            "success": True,
            "provided_requirements": result.get("provided_requirements", {}),
            "explanation": result.get("explanation", "")
        }
        
    except Exception as e:
        logger.error(f"[TOOL] Additional requirements error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Sales agent respond tool removed



@tool("handle_feedback")
def handle_feedback_tool(
    feedback_type: str,
    comment: str
) -> str:
    """
    Generate response to user feedback.
    
    feedback_type: positive, negative, or comment
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.5,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        if feedback_type == "positive":
            prompt_text = "Thank you for your positive feedback!"
        elif feedback_type == "negative":
            prompt_text = "Thank you for your feedback. We'll use it to improve."
        else:
            prompt_text = "Thank you for your comment: {comment}"
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({"comment": comment})
        
    except Exception as e:
        logger.error(f"[TOOL] Feedback handling error: {e}")
        return "Thank you for your feedback!"


@tool("grounded_knowledge_chat")
def grounded_knowledge_chat_tool(
    user_question: str,
    product_type: str,
    specifications: str,
    strategy_context: str = "",
    standards_context: str = "",
    inventory_context: str = ""
) -> Dict[str, Any]:
    """
    Answer questions using grounded knowledge from RAG sources.
    
    Combines Strategy, Standards, and Inventory RAG context.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Try to load grounded chat prompt from RAG prompts
        try:
            prompt_text = INDEX_RAG_PROMPTS.get("CHAT_AGENT", "Answer: {user_question}")
        except:
            prompt_text = "Answer: {user_question}"
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_question": user_question,
            "product_type": product_type,
            "specifications": specifications,
            "strategy_context": strategy_context or "No strategy data available",
            "standards_context": standards_context or "No standards data available",
            "inventory_context": inventory_context or "No inventory data available",
            "format_instructions": ""
        })
        
        return {
            "success": True,
            "answer": result.get("answer", ""),
            "key_facts": result.get("key_facts", []),
            "recommendation": result.get("recommendation", ""),
            "sources": result.get("sources", [])
        }
        
    except Exception as e:
        logger.error(f"[TOOL] Grounded chat error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool("classify_identification_input")
def classify_identification_input_tool(user_input: str) -> Dict[str, Any]:
    """
    Classify input for instrument identification workflow.
    
    Returns:
        - type: greeting, requirements, question, unrelated
        - confidence: high, medium, low
        - reasoning: Brief explanation
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = prompts.identify_classification_prompt
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({"user_input": user_input})
        
        return {
            "success": True,
            "type": result.get("type", "unrelated"),
            "confidence": result.get("confidence", "low"),
            "reasoning": result.get("reasoning", "")
        }
        
    except Exception as e:
        logger.error(f"[TOOL] Classification error: {e}")
        return {
            "success": False,
            "type": "unrelated",
            "error": str(e)
        }


@tool("extract_advanced_parameters")
def extract_advanced_parameters_tool(
    user_input: str,
    product_type: str,
    available_parameters: str
) -> Dict[str, Any]:
    """
    Extract advanced parameter selections from user input.
    
    Returns:
        - selected_parameters: Dict of parameter name to value
        - explanation: Brief explanation
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = prompts.advanced_parameter_selection_prompt
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "available_parameters": available_parameters
        })
        
        return {
            "success": True,
            "selected_parameters": result.get("selected_parameters", {}),
            "explanation": result.get("explanation", "")
        }
        
    except Exception as e:
        logger.error(f"[TOOL] Advanced params error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# TOOL COLLECTIONS
# ============================================================================

def get_sales_agent_tools() -> List[BaseTool]:
    """Get tools for sales agent workflow"""
    return [
        classify_user_intent_tool,
        sales_agent_respond_tool,
        extract_additional_requirements_tool,
        extract_advanced_parameters_tool,
        handle_feedback_tool
    ]


def get_identification_tools() -> List[BaseTool]:
    """Get tools for instrument identification workflow"""
    return [
        classify_identification_input_tool,
        identify_instruments_tool,
        validate_requirements_tool
    ]


def get_knowledge_tools() -> List[BaseTool]:
    """Get tools for knowledge/RAG workflows"""
    return [
        grounded_knowledge_chat_tool,
        classify_user_intent_tool
    ]


def get_all_tools() -> List[BaseTool]:
    """Get all available tools"""
    return [
        classify_user_intent_tool,
        identify_instruments_tool,
        validate_requirements_tool,
        extract_additional_requirements_tool,
        sales_agent_respond_tool,
        handle_feedback_tool,
        grounded_knowledge_chat_tool,
        classify_identification_input_tool,
        extract_advanced_parameters_tool
    ]


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class WorkflowOrchestrator:
    """
    Main orchestrator for LangGraph workflows.
    
    Provides:
    - Access to prompt registry
    - Tool factory
    - Workflow execution
    """
    
    def __init__(self):
        self.prompt_registry = prompt_registry
        self.tool_factory = tool_factory
        self.tools = get_all_tools()
        logger.info(f"[ORCHESTRATOR] Initialized with {len(self.tools)} tools")
    
    def get_prompt(self, name: str) -> ChatPromptTemplate:
        """Get a prompt by name"""
        return self.prompt_registry.get(name)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def list_prompts(self) -> List[str]:
        """List all available prompts"""
        return self.prompt_registry.list_all()
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return [tool.name for tool in self.tools]
    
    def get_tools_for_workflow(self, workflow_name: str) -> List[BaseTool]:
        """Get appropriate tools for a workflow type"""
        workflow_tools = {
            "sales_agent": get_sales_agent_tools(),
            "identification": get_identification_tools(),
            "knowledge": get_knowledge_tools(),
            "grounded_chat": get_knowledge_tools(),
        }
        return workflow_tools.get(workflow_name, self.tools)


# Global orchestrator instance
orchestrator = WorkflowOrchestrator()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Registry
    "prompt_registry",
    "PromptRegistry",
    "PromptCategory",
    
    # Factory
    "tool_factory",
    "ToolFactory",
    
    # Orchestrator
    "orchestrator",
    "WorkflowOrchestrator",
    
    # Individual tools
    "classify_user_intent_tool",
    "identify_instruments_tool",
    "validate_requirements_tool",
    "extract_additional_requirements_tool",
    "sales_agent_respond_tool",
    "handle_feedback_tool",
    "grounded_knowledge_chat_tool",
    "classify_identification_input_tool",
    "extract_advanced_parameters_tool",
    
    # Tool collections
    "get_sales_agent_tools",
    "get_identification_tools",
    "get_knowledge_tools",
    "get_all_tools"
]
