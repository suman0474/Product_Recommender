# solution/prompts.py
# =============================================================================
# SOLUTION DEEP AGENT - INLINE PROMPT CONSTANTS
# =============================================================================
#
# These prompt templates are used by solution/workflow.py for:
#   - SOLUTION_INTENT_PROMPT        : LLM-based intent refinement
#   - MODIFICATION_PROCESSING_PROMPT: Applying BOM/requirement modifications
#   - CLARIFICATION_PROMPT          : Asking clarifying questions
#   - RESET_CONFIRMATION_PROMPT     : Confirming session reset
#
# =============================================================================

SOLUTION_INTENT_PROMPT = """You are an intent classifier for an industrial instrumentation solution assistant.

Current user input:
{user_input}

Conversation context:
{conversation_context}

Classify the user's intent into EXACTLY ONE of the following types:
- "requirements"   : New solution request or listing out instrument/equipment needs
- "modification"   : Updating, changing, or editing items already identified (e.g. "change the transmitter to HART", "update quantity to 2")
- "clarification"  : The input is ambiguous and needs clarification before proceeding
- "concise_bom"    : User explicitly wants a short/concise bill of materials
- "reset"          : User wants to clear the session and start fresh
- "router_needed"  : The request is not related to solutions (e.g. product search, general chat)
- "invalid_input"  : The input is gibberish, irrelevant, or lacks any useful data to identify instruments and accessories

Return a JSON object with the following fields:
{{
  "type": "<one of the types above>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}

Return ONLY valid JSON. No markdown, no code blocks."""


MODIFICATION_PROCESSING_PROMPT = """You are an assistant that processes modification requests to an existing instrument/solution bill of materials (BOM).

Current BOM and context:
{current_state}

User's modification request:
{modification_request}

Apply the requested modifications and return the updated BOM data as a JSON object with structure:
{{
  "updated_items": [
    {{
      "number": <item number>,
      "name": "<instrument or accessory name>",
      "type": "<instrument|accessory>",
      "category": "<category>",
      "quantity": <quantity>,
      "specifications": {{}},
      "sample_input": "<updated sample input string>",
      "purpose": "<purpose>"
    }}
  ],
  "changes_made": ["<description of each change>"],
  "summary": "<brief summary of modifications applied>"
}}

Rules:
- Only update items explicitly mentioned in the modification request.
- Keep all other items unchanged.
- Preserve existing specifications unless overridden.
- Return ONLY valid JSON. No markdown, no code blocks."""


CLARIFICATION_PROMPT = """You are a helpful assistant for an industrial instrumentation solution platform.

The user has provided input that requires clarification before you can proceed.

User input:
{user_input}

Conversation context:
{conversation_context}

Generate 1-3 targeted clarifying questions that would help you understand exactly what the user needs.

Return a JSON object:
{{
  "missing_information": "<what is unclear or missing>",
  "reasoning": "<why clarification is needed>",
  "clarification_questions": [
    "<question 1>",
    "<question 2>"
  ],
  "message": "<a friendly message to the user asking for clarification>"
}}

Return ONLY valid JSON. No markdown, no code blocks."""


RESET_CONFIRMATION_PROMPT = """You are an assistant for an industrial instrumentation solution platform.

The user has requested to reset or clear their session.

User input:
{user_input}

Generate a friendly confirmation message asking the user if they are sure they want to reset (clear all identified instruments and start fresh).

Return a JSON object:
{{
  "message": "<friendly confirmation message asking if they want to reset>",
  "confirmed": false
}}

Return ONLY valid JSON. No markdown, no code blocks."""
