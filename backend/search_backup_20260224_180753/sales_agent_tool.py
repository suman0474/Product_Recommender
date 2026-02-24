"""
Sales Agent Tool for Product Search Workflow
=============================================

Step 3 of Product Search Workflow:
- Step-by-step requirements collection workflow
- LLM-powered conversational interface
- Session-based state management
- Advanced parameters integration
- Knowledge question handling

This tool replicates the /api/sales-agent functionality as a modular,
reusable component that can be integrated into workflows.

Workflow Steps:
1. initialInput - Initial product requirements
2. awaitMissingInfo - Collect missing mandatory fields
3. awaitAdditionalAndLatestSpecs - Additional specifications
4. awaitAdvancedSpecs - Advanced parameter specifications
5. showSummary - Display requirements summary
6. finalAnalysis - Complete analysis

"""

import re
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Enumeration of workflow steps"""
    GREETING = "greeting"
    INITIAL_INPUT = "initialInput"
    AWAIT_MISSING_INFO = "awaitMissingInfo"
    AWAIT_ADDITIONAL_SPECS = "awaitAdditionalAndLatestSpecs"
    AWAIT_ADVANCED_SPECS = "awaitAdvancedSpecs"
    AWAIT_ADVANCED_SELECTION = "await_advanced_selection"
    SHOW_SUMMARY = "showSummary"
    FINAL_ANALYSIS = "finalAnalysis"
    ANALYSIS_ERROR = "analysisError"
    DEFAULT = "default"


class SalesAgentTool:
    """
    Sales Agent Tool - Step 3 of Product Search Workflow

    Responsibilities:
    1. Manage step-by-step requirements collection
    2. Handle conversational interactions with LLM
    3. Maintain session state and context
    4. Integrate with advanced parameters discovery
    5. Generate summaries and proceed to analysis
    """

    def __init__(self, llm=None):
        """
        Initialize the sales agent tool.

        Args:
            llm: LLM instance for generating responses (optional)
        """
        self.llm = llm
        self.session_states = {}  # Session-isolated state storage
        logger.info("[SalesAgentTool] Initialized")

    def process_step(
        self,
        step: str,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str,
        intent: str = "workflow",
        save_immediately: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single workflow step.

        Args:
            step: Current workflow step
            user_message: User's message or input
            data_context: Context data for the current step
            session_id: Session identifier for state isolation
            intent: Classified intent ("workflow" or "knowledgeQuestion")
            save_immediately: Skip greeting and save immediately

        Returns:
            Response with:
            {
                "content": str,  # AI-generated response message
                "nextStep": str,  # Next workflow step
                "maintainWorkflow": bool,  # Whether to maintain current workflow
                "dataContext": dict,  # Updated context data
                "discoveredParameters": list  # Optional: discovered parameters
            }
        """
        logger.info(f"[SalesAgentTool] Session {session_id}: Step={step}, Intent={intent}")

        # Initialize session state if not exists
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "current_step": None,
                "current_intent": None,
                "product_type": None,
                "data_context": {},
                "additional_specs_input": None,
                "selected_parameters": {},
                "awaiting_additional_specs_yesno": True
            }

        session_state = self.session_states[session_id]

        try:
            # Handle knowledge questions
            if intent == "knowledgeQuestion":
                return self._handle_knowledge_question(
                    user_message, step, session_id
                )

            # Route to appropriate step handler
            if step == WorkflowStep.INITIAL_INPUT.value:
                return self._handle_initial_input(
                    user_message, data_context, session_id, save_immediately
                )

            elif step == WorkflowStep.AWAIT_MISSING_INFO.value:
                return self._handle_missing_info(
                    user_message, data_context, session_id
                )

            elif step == WorkflowStep.AWAIT_ADDITIONAL_SPECS.value:
                return self._handle_additional_specs(
                    user_message, data_context, session_id
                )

            elif step == WorkflowStep.AWAIT_ADVANCED_SPECS.value:
                return self._handle_advanced_specs(
                    user_message, data_context, session_id
                )

            elif step == WorkflowStep.AWAIT_ADVANCED_SELECTION.value:
                return self._handle_advanced_selection(
                    user_message, data_context, session_id
                )

            elif step == WorkflowStep.SHOW_SUMMARY.value:
                return self._handle_show_summary(
                    user_message, data_context, session_id
                )

            elif step == WorkflowStep.FINAL_ANALYSIS.value:
                return self._handle_final_analysis(
                    data_context, session_id
                )

            elif step == WorkflowStep.GREETING.value:
                return self._handle_greeting(session_id)

            else:
                # Default fallback
                return self._handle_default(user_message, session_id)

        except Exception as e:
            logger.error(f"[SalesAgentTool] Error processing step: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error. Please try again.",
                "nextStep": step,
                "maintainWorkflow": True,
                "error": str(e)
            }

    def _handle_knowledge_question(
        self,
        user_message: str,
        current_step: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle knowledge questions and resume workflow"""
        logger.info(f"[SalesAgentTool] Handling knowledge question in step: {current_step}")

        # Generate context hint based on current step
        context_hints = {
            WorkflowStep.AWAIT_MISSING_INFO.value: (
                "Once you have the information you need, please provide the missing details "
                "so we can continue with your product selection."
            ),
            WorkflowStep.AWAIT_ADDITIONAL_SPECS.value: (
                "Now, let's continue - would you like to add additional specifications?"
            ),
            WorkflowStep.AWAIT_ADVANCED_SPECS.value: (
                "Now, let's continue with advanced specifications."
            ),
            WorkflowStep.SHOW_SUMMARY.value: (
                "Now, let's proceed with your product analysis."
            )
        }

        context_hint = context_hints.get(
            current_step,
            "Now, let's continue with your product selection."
        )

        # Generate LLM response for knowledge question
        if self.llm:
            response = self._generate_llm_response(
                "knowledge_question",
                user_message=user_message,
                context_hint=context_hint
            )
        else:
            response = (
                f"Thank you for your question. {context_hint}"
            )

        return {
            "content": response,
            "nextStep": current_step,  # Resume at same step
            "maintainWorkflow": True
        }

    def _handle_initial_input(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str,
        save_immediately: bool = False
    ) -> Dict[str, Any]:
        """Handle initial product input step"""
        product_type = data_context.get('productType') or data_context.get('product_type', 'a product')
        missing_fields = data_context.get('missingFields') or data_context.get('missing_fields', [])
        is_valid = data_context.get('isValid') if data_context.get('isValid') is not None else data_context.get('is_valid', True)
        schema = data_context.get('schema', {})

        # Save product type to session
        session_state = self.session_states[session_id]
        session_state['product_type'] = product_type
        session_state['current_step'] = WorkflowStep.INITIAL_INPUT.value
        session_state['schema'] = schema
        session_state['missing_fields'] = missing_fields

        logger.info(f"[SalesAgentTool] Initial input - product_type: {product_type}")
        logger.info(f"[SalesAgentTool] Initial input - is_valid: {is_valid}, missing_fields: {len(missing_fields) if missing_fields else 0}")

        if save_immediately:
            # Skip greeting, save and proceed
            logger.info(f"[SalesAgentTool] Saved product type: {product_type}")
            return {
                "content": f"Saved product type: {product_type}",
                "nextStep": WorkflowStep.AWAIT_ADDITIONAL_SPECS.value
            }

        # Check if there are missing mandatory fields
        has_missing_fields = missing_fields and len(missing_fields) > 0

        if has_missing_fields:
            # Format missing fields for display
            formatted_fields = [self._format_field_name(f) for f in missing_fields]
            fields_list = ", ".join(formatted_fields[:5])  # Show first 5
            if len(formatted_fields) > 5:
                fields_list += f" and {len(formatted_fields) - 5} more"

            response = (
                f"I've identified that you're looking for a **{product_type}**. "
                f"I've loaded the product schema and captured your requirements.\n\n"
                f"However, I notice some key specifications are missing:\n"
                f"**Missing fields:** {fields_list}\n\n"
                f"Would you like to provide these details, or should I proceed with the information I have?"
            )
            next_step = WorkflowStep.AWAIT_MISSING_INFO.value
            logger.info(f"[SalesAgentTool] Missing fields detected, routing to awaitMissingInfo")
        else:
            # All fields provided - ask about advanced parameters
            response = (
                f"Great! I've identified that you need a **{product_type}** and "
                f"captured all your requirements.\n\n"
                f"Would you like to add any additional or advanced specifications? "
                f"This can help narrow down the best product matches."
            )
            next_step = WorkflowStep.AWAIT_ADDITIONAL_SPECS.value
            logger.info(f"[SalesAgentTool] All fields valid, routing to awaitAdditionalAndLatestSpecs")

        return {
            "content": response,
            "nextStep": next_step,
            "hasMissingFields": has_missing_fields,
            "missingFields": missing_fields
        }

    def _format_field_name(self, field: str) -> str:
        """Convert camelCase or snake_case to Title Case."""
        # Replace underscores and split on capital letters
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field)
        words = words.replace('_', ' ')
        return words.title()

    def _handle_missing_info(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle missing information collection step.

        This step is triggered when validation detects missing mandatory fields.
        The user can either:
        1. Provide the missing information
        2. Say "proceed" or "skip" to continue without providing all fields

        Args:
            user_message: User's response (could be additional requirements or skip command)
            data_context: Context containing schema, missing_fields, etc.
            session_id: Session identifier

        Returns:
            Response with next step decision
        """
        session_state = self.session_states[session_id]
        user_lower = user_message.lower().strip()

        # Check for skip/proceed commands
        skip_keywords = ['skip', 'proceed', 'continue', 'no', 'n', 'move on', 'next']
        is_skip = any(keyword in user_lower for keyword in skip_keywords)

        # Get current missing fields from context or session
        missing_fields = data_context.get('missingFields') or data_context.get('missing_fields', [])
        if not missing_fields:
            missing_fields = session_state.get('missing_fields', [])

        product_type = data_context.get('productType') or data_context.get('product_type') or session_state.get('product_type', 'product')
        schema = data_context.get('schema', session_state.get('schema', {}))

        logger.info(f"[SalesAgentTool] Handling missing info - skip: {is_skip}, missing_fields: {len(missing_fields)}")

        if is_skip:
            # User wants to skip providing missing fields
            response = (
                f"Understood. I'll proceed with the information we have. "
                f"Some specifications may not be as precise without the missing details.\n\n"
                f"Would you like to add any additional or advanced specifications?"
            )
            next_step = WorkflowStep.AWAIT_ADDITIONAL_SPECS.value
            logger.info(f"[SalesAgentTool] User skipped missing fields, moving to additional specs")

        elif user_message.strip():
            # User provided some information - this should be re-validated
            # Return a flag to trigger re-validation on the frontend/backend
            response = (
                f"Thank you for the additional information. "
                f"Let me process these requirements..."
            )

            return {
                "content": response,
                "nextStep": WorkflowStep.AWAIT_MISSING_INFO.value,
                "requiresRevalidation": True,
                "userInput": user_message,
                "productType": product_type
            }

        else:
            # Empty message - remind user about missing fields
            formatted_fields = [self._format_field_name(f) for f in missing_fields[:5]]
            fields_display = ", ".join(formatted_fields)
            if len(missing_fields) > 5:
                fields_display += f" and {len(missing_fields) - 5} more"

            response = (
                f"I still need some information to find the best match. "
                f"Missing specifications: **{fields_display}**\n\n"
                f"Please provide these details, or type 'proceed' to continue without them."
            )
            next_step = WorkflowStep.AWAIT_MISSING_INFO.value

        return {
            "content": response,
            "nextStep": next_step,
            "missingFields": missing_fields
        }

    def _handle_additional_specs(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Handle additional and latest specifications step"""
        session_state = self.session_states[session_id]
        user_lower = user_message.lower().strip()

        # Keywords for yes/no detection
        affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
        negative_keywords = ['no', 'n', 'nope', 'skip']

        is_awaiting_yesno = session_state.get('awaiting_additional_specs_yesno', True)
        is_yes = any(keyword in user_lower for keyword in affirmative_keywords)
        is_no = any(keyword in user_lower for keyword in negative_keywords)

        if is_awaiting_yesno:
            # First interaction - asking yes/no
            if is_no:
                # User says NO -> skip to summary
                session_state['awaiting_additional_specs_yesno'] = False
                response = "Understood. Let's proceed to the summary."
                next_step = WorkflowStep.SHOW_SUMMARY.value

            elif is_yes:
                # User says YES -> collect additional specs
                session_state['awaiting_additional_specs_yesno'] = False

                # Show available parameters if any
                available_parameters = data_context.get('availableParameters', [])
                if available_parameters:
                    params_display = self._format_parameters(available_parameters)
                    response = (
                        f"Great! Here are the latest specifications:\n\n{params_display}\n\n"
                        "Please enter your additional specifications."
                    )
                else:
                    response = "Great! Please enter your additional specifications."

                next_step = WorkflowStep.AWAIT_ADDITIONAL_SPECS.value

            else:
                # Invalid response
                response = "Please respond with yes or no. Would you like to add additional specifications?"
                next_step = WorkflowStep.AWAIT_ADDITIONAL_SPECS.value

        else:
            # Collecting actual specifications
            session_state['additional_specs_input'] = user_message
            session_state['awaiting_additional_specs_yesno'] = True  # Reset

            response = "Thank you for the additional specifications. Proceeding to advanced parameters."
            next_step = WorkflowStep.AWAIT_ADVANCED_SPECS.value

        return {
            "content": response,
            "nextStep": next_step
        }

    def _handle_advanced_specs(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Handle advanced specifications step with discovery integration"""
        session_state = self.session_states[session_id]
        user_lower = user_message.lower().strip()

        # Get context data
        product_type = data_context.get('productType') or session_state.get('product_type')
        available_parameters = data_context.get('availableParameters', [])
        selected_parameters = data_context.get('selectedParameters', {})
        total_selected = data_context.get('totalSelected', 0)

        # Keywords
        affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay', 'proceed']
        negative_keywords = ['no', 'n', 'nope', 'skip', 'none']
        display_keywords = ['show', 'display', 'list', 'see', 'view', 'what are']

        # Check if parameters need to be discovered
        if not available_parameters:
            logger.info(f"[SalesAgentTool] No parameters available, attempting discovery")

            # Check if user wants to skip or discover
            if user_lower in negative_keywords:
                response = "Understood. Proceeding to summary without advanced parameters."
                next_step = WorkflowStep.SHOW_SUMMARY.value
            elif user_lower in affirmative_keywords or not user_message.strip():
                # Trigger discovery (in actual implementation, call discovery function)
                response = (
                    "Let me discover the available advanced parameters for you. "
                    "This may take a moment..."
                )
                next_step = WorkflowStep.AWAIT_ADVANCED_SPECS.value

                # Return with flag to trigger discovery in workflow
                return {
                    "content": response,
                    "nextStep": next_step,
                    "triggerDiscovery": True,
                    "productType": product_type
                }
            else:
                response = "Would you like me to discover advanced parameters, or shall we proceed to summary?"
                next_step = WorkflowStep.AWAIT_ADVANCED_SPECS.value

        else:
            # Parameters already available
            wants_display = any(keyword in user_lower for keyword in display_keywords)
            user_affirmed = any(keyword in user_lower for keyword in affirmative_keywords)
            user_denied = any(keyword in user_lower for keyword in negative_keywords)

            if user_affirmed:
                # User wants to add parameters
                params_display = self._format_parameters(available_parameters)
                response = (
                    f"Here are the available advanced parameters:\n\n{params_display}\n\n"
                    "Please specify which parameters you'd like to add."
                )
                next_step = WorkflowStep.AWAIT_ADVANCED_SPECS.value

            elif user_denied:
                # User doesn't want advanced parameters
                response = "Understood. Proceeding to summary without advanced parameters."
                next_step = WorkflowStep.SHOW_SUMMARY.value

            elif wants_display or not user_message.strip():
                # Display parameters
                params_display = self._format_parameters(available_parameters)
                response = (
                    f"Available advanced parameters:\n\n{params_display}\n\n"
                    "Would you like to add any of these?"
                )
                next_step = WorkflowStep.AWAIT_ADVANCED_SPECS.value

            elif total_selected > 0 or user_message.strip():
                # User provided selections
                selected_names = [
                    param.replace('_', ' ').title()
                    for param in selected_parameters.keys()
                ] if selected_parameters else []

                if selected_names:
                    selected_display = ", ".join(selected_names)
                    response = f"**Added Advanced Parameters:** {selected_display}\n\nProceeding to summary."
                else:
                    response = "Thank you for the specifications. Proceeding to summary."

                next_step = WorkflowStep.SHOW_SUMMARY.value

            else:
                # Default fallback
                response = "Please respond with yes or no. Would you like to add advanced parameters?"
                next_step = WorkflowStep.AWAIT_ADVANCED_SPECS.value

        return {
            "content": response,
            "nextStep": next_step,
            "dataContext": data_context
        }

    def _handle_advanced_selection(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Handle selection of discovered advanced specifications"""
        user_lower = user_message.lower().strip()
        
        # Get discovered specs from context
        # Note: frontend might pass it as 'available_advanced_params' or 'discoveredParameters'
        discovered_specs = (
            data_context.get('available_advanced_params') or 
            data_context.get('availableAdvancedParams') or 
            data_context.get('discoveredParameters') or 
            []
        )
        
        selected_specs = []
        
        # Check for keywords
        is_all = any(k in user_lower for k in ['all', 'everything', 'yes', 'sure'])
        is_none = any(k in user_lower for k in ['no', 'skip', 'none', 'proceed'])
        
        if is_none:
            response = "No problem! Proceeding without advanced specifications."
            next_step = WorkflowStep.SHOW_SUMMARY.value
            
        elif is_all:
            selected_specs = discovered_specs
            response = f"Great! I've added all {len(selected_specs)} advanced specifications."
            next_step = WorkflowStep.SHOW_SUMMARY.value
            
        else:
            # Try to match names
            valid_names = []
            for spec in discovered_specs:
                spec_name = spec.get('name', spec.get('key', '')).lower()
                spec_key = spec.get('key', '').lower()
                
                if spec_name in user_lower or spec_key in user_lower:
                    selected_specs.append(spec)
                    valid_names.append(spec.get('name', spec.get('key')))
            
            if selected_specs:
                names_str = ", ".join(valid_names)
                response = f"Great! I've added: {names_str}."
                
                # Check if we should ask for values or proceed
                # For now, proceed to summary where values can be refined
                next_step = WorkflowStep.SHOW_SUMMARY.value
            else:
                # Ambiguous input fallback
                names_list = self._format_parameters(discovered_specs)
                response = (
                    f"I'm not sure which specifications you want to add.\n"
                    f"Available specs:\n{names_list}\n\n"
                    f"Say 'all', 'none', or list the names you want."
                )
                next_step = WorkflowStep.AWAIT_ADVANCED_SELECTION.value
                return {
                    "content": response,
                    "nextStep": next_step
                }

        # Update context with selected specs
        # We need to format it as key-value pairs for the summary/analysis
        selected_params_dict = {}
        for spec in selected_specs:
            key = spec.get('key')
            if key:
                selected_params_dict[key] = ""  # Value handles by future steps or left blank
                
        # Merge with existing
        existing_selected = data_context.get('selectedParameters', {})
        existing_selected.update(selected_params_dict)
        data_context['selectedParameters'] = existing_selected

        return {
            "content": response,
            "nextStep": next_step,
            "dataContext": data_context,
            "selectedParameters": selected_params_dict
        }

    def _handle_show_summary(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Handle summary display and confirmation"""
        user_lower = user_message.lower().strip()
        proceed_keywords = ['yes', 'y', 'proceed', 'continue', 'run', 'analyze', 'ok', 'okay', 'start']

        # Check if user wants to proceed
        wants_to_proceed = any(keyword in user_lower for keyword in proceed_keywords)

        if wants_to_proceed:
            # User confirmed to proceed
            response = "Perfect! Starting the product analysis now. This may take a moment..."
            next_step = WorkflowStep.FINAL_ANALYSIS.value
            return {
                "content": response,
                "nextStep": next_step,
                "triggerAnalysis": True  # Flag to tell frontend to run analysis
            }

        # Generate summary from collected data
        product_type = data_context.get('productType') or data_context.get('product_type', 'Not specified')
        collected_data = data_context.get('collectedData', data_context)

        # Build summary sections
        summary_parts = [f"**Product Type:** {product_type}"]

        # Extract mandatory requirements
        mandatory = collected_data.get('mandatoryRequirements', collected_data.get('mandatory', {}))
        if mandatory and isinstance(mandatory, dict):
            filled_mandatory = {k: v for k, v in mandatory.items() if v}
            if filled_mandatory:
                summary_parts.append("\n**Mandatory Requirements:**")
                for key, value in filled_mandatory.items():
                    field_name = self._format_field_name(key)
                    summary_parts.append(f"- {field_name}: {value}")

        # Extract optional requirements
        optional = collected_data.get('optionalRequirements', collected_data.get('optional', {}))
        if optional and isinstance(optional, dict):
            filled_optional = {k: v for k, v in optional.items() if v}
            if filled_optional:
                summary_parts.append("\n**Optional Requirements:**")
                for key, value in filled_optional.items():
                    field_name = self._format_field_name(key)
                    summary_parts.append(f"- {field_name}: {value}")

        # Extract advanced parameters
        advanced = collected_data.get('selectedAdvancedParams', collected_data.get('advancedSpecs', {}))
        if advanced and isinstance(advanced, dict):
            filled_advanced = {k: v for k, v in advanced.items() if v}
            if filled_advanced:
                summary_parts.append("\n**Advanced Specifications:**")
                for key, value in filled_advanced.items():
                    field_name = self._format_field_name(key)
                    summary_parts.append(f"- {field_name}: {value}")

        # If nothing was collected except product type
        if len(summary_parts) == 1:
            summary_parts.append("\nNo specific requirements have been provided yet.")

        summary_text = "\n".join(summary_parts)

        response = (
            f"Here's a summary of your requirements:\n\n{summary_text}\n\n"
            "Would you like to proceed with the product analysis? "
            "Type **'yes'** or **'proceed'** to continue."
        )
        next_step = WorkflowStep.SHOW_SUMMARY.value

        return {
            "content": response,
            "nextStep": next_step
        }

    def _handle_final_analysis(
        self,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Handle final analysis step"""
        # Try multiple paths to find ranked products (handle different API response structures)
        ranked_products = []
        
        # Path 1: analysisResult.overallRanking.rankedProducts (camelCase)
        if data_context.get('analysisResult'):
            analysis = data_context['analysisResult']
            if isinstance(analysis, dict):
                overall = analysis.get('overallRanking', {})
                if isinstance(overall, dict):
                    ranked_products = overall.get('rankedProducts', overall.get('ranked_products', []))
        
        # Path 2: analysis_result.overall_ranking.ranked_products (snake_case)
        if not ranked_products and data_context.get('analysis_result'):
            analysis = data_context['analysis_result']
            if isinstance(analysis, dict):
                overall = analysis.get('overall_ranking', analysis.get('overallRanking', {}))
                if isinstance(overall, dict):
                    ranked_products = overall.get('ranked_products', overall.get('rankedProducts', []))
        
        # Path 3: Direct overallRanking at top level
        if not ranked_products and data_context.get('overallRanking'):
            overall = data_context['overallRanking']
            if isinstance(overall, dict):
                ranked_products = overall.get('rankedProducts', overall.get('ranked_products', []))
        
        # Path 4: Direct overall_ranking at top level
        if not ranked_products and data_context.get('overall_ranking'):
            overall = data_context['overall_ranking']
            if isinstance(overall, list):
                ranked_products = overall
            elif isinstance(overall, dict):
                ranked_products = overall.get('ranked_products', overall.get('rankedProducts', []))
        
        # Path 5: rankedProducts directly in context
        if not ranked_products:
            ranked_products = data_context.get('rankedProducts', data_context.get('ranked_products', []))
        
        # Count matching products (handle both camelCase and snake_case)
        count = 0
        for p in ranked_products:
            if isinstance(p, dict):
                # Check both field name variants
                match = p.get('requirementsMatch') or p.get('requirements_match') or False
                if match is True:
                    count += 1
        
        # If no requirementsMatch field found, use total count as approximate matches
        total_count = len(ranked_products) if ranked_products else 0
        
        logger.info(f"[SalesAgentTool] Final analysis: {total_count} products, {count} exact matches")

        if total_count > 0:
            if count > 0:
                response = (
                    f"Analysis complete! I found {count} products that exactly match your requirements"
                    f"{f' (plus {total_count - count} alternatives)' if total_count > count else ''}. "
                    "You can now view the detailed results."
                )
            else:
                response = (
                    f"Analysis complete! I found {total_count} products that may meet your needs. "
                    "While none are exact matches, these are the closest alternatives available. "
                    "You can now view the detailed results."
                )
        else:
            response = (
                "Analysis complete! While I didn't find exact matches, "
                "I have some similar products that may meet your needs."
            )

        return {
            "content": response,
            "nextStep": None,  # End of workflow
            "analysisComplete": True,
            "totalProducts": total_count,
            "exactMatches": count
        }

    def _handle_greeting(self, session_id: str) -> Dict[str, Any]:
        """Handle greeting step"""
        response = (
            "Hello! I'm your AI Product Advisor. "
            "I'll help you find the perfect product for your needs. "
            "What product are you looking for?"
        )

        return {
            "content": response,
            "nextStep": WorkflowStep.INITIAL_INPUT.value
        }

    def _handle_default(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """Handle default/unrecognized steps"""
        response = "I'm not sure how to proceed. Could you please rephrase your request?"

        return {
            "content": response,
            "nextStep": WorkflowStep.INITIAL_INPUT.value
        }

    def _format_parameters(self, parameters: List) -> str:
        """
        Format parameters for display.

        Args:
            parameters: List of parameter dicts or strings

        Returns:
            Formatted string with bullet points
        """
        formatted = []
        for param in parameters:
            if isinstance(param, dict):
                name = param.get('name') or param.get('key') or str(param)
            else:
                name = str(param)

            # Format name: replace underscores, title case
            name = name.replace('_', ' ')
            name = re.split(r'[\(\[\{]', name, 1)[0].strip()
            name = " ".join(name.split())
            name = name.title()

            formatted.append(f"- {name}")

        return "\n".join(formatted)

    def _generate_llm_response(
        self,
        prompt_type: str,
        **kwargs
    ) -> str:
        """
        Generate LLM response for a given prompt type.

        Args:
            prompt_type: Type of prompt to use
            **kwargs: Variables for the prompt

        Returns:
            Generated response string
        """
        if not self.llm:
            return "LLM not available. Please configure LLM instance."

        # This would integrate with the LLM instance
        # For now, return a placeholder
        return f"Generated response for {prompt_type}"

    def get_session_state(self, session_id: str, key: str = None) -> Any:
        """
        Get session state or specific key.

        Args:
            session_id: Session identifier
            key: Optional specific key to retrieve

        Returns:
            Session state dict or specific value
        """
        if session_id not in self.session_states:
            return None

        session_state = self.session_states[session_id]

        if key:
            return session_state.get(key)

        return session_state

    def clear_session(self, session_id: str):
        """
        Clear session state.

        Args:
            session_id: Session identifier
        """
        if session_id in self.session_states:
            del self.session_states[session_id]
            logger.info(f"[SalesAgentTool] Cleared session: {session_id}")


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example usage of SalesAgentTool"""
    print("\n" + "="*70)
    print("SALES AGENT TOOL - STANDALONE EXAMPLE")
    print("="*70)

    # Initialize tool
    tool = SalesAgentTool()
    session_id = "test_session_001"

    # Example 1: Initial input
    print("\n[Example 1] Initial input:")
    result = tool.process_step(
        step="initialInput",
        user_message="I need a pressure transmitter",
        data_context={"productType": "Pressure Transmitter"},
        session_id=session_id
    )

    print(f"Response: {result['content']}")
    print(f"Next Step: {result['nextStep']}")

    # Example 2: Additional specs - say no
    print("\n[Example 2] Skip additional specs:")
    result = tool.process_step(
        step="awaitAdditionalAndLatestSpecs",
        user_message="no",
        data_context={"productType": "Pressure Transmitter"},
        session_id=session_id
    )

    print(f"Response: {result['content']}")
    print(f"Next Step: {result['nextStep']}")

    # Example 3: Summary
    print("\n[Example 3] Show summary:")
    result = tool.process_step(
        step="showSummary",
        user_message="",
        data_context={"productType": "Pressure Transmitter"},
        session_id=session_id
    )

    print(f"Response: {result['content']}")
    print(f"Next Step: {result['nextStep']}")


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    example_usage()
