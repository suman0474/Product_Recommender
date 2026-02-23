# agentic/workflows/specification_utils.py
# =============================================================================
# SHARED SPECIFICATION UTILITIES
# =============================================================================
#
# Shared functions for building sample_input from enriched specifications.
# Used by: optimized_agent.py (during enrichment), identifier.py, solution/workflow.py
#

import ast
import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE EXTRACTION & CLEANING
# =============================================================================

MAX_SPEC_VALUE_WORDS = 7   # Threshold: values longer than this are considered verbose
TRIM_TARGET_WORDS = 4      # Trim verbose values down to this many words

# Patterns that indicate a value is a sentence/instruction rather than a spec value.
# Values starting with these (case-insensitive) are discarded entirely.
_SENTENCE_START_PATTERNS = [
    "for ", "mount ", "use ", "ensure ", "install ", "apply ", "check ",
    "the ", "a ", "an ", "this ", "these ", "those ", "it ", "they ",
    "should ", "must ", "shall ", "will ", "can ", "may ", "would ",
    "refer ", "see ", "note ", "per ", "as per ", "according ",
    "vendor ", "consult ", "contact ", "select ", "verify ",
    "recommend", "typical", "based on", "depending on", "in accordance",
    "to be ", "to ensure ", "to prevent ", "to maintain ",
    # Error message patterns (ROOT CAUSE FIX: detect AI/API error messages)
    "i found relevant", "i found ", "error:", "exception:",
    "failed to", "unable to", "could not", "cannot ",
    "an error occurred", "temporarily unavailable", "service is ",
]

# Patterns that indicate the value contains error messages (anywhere in the string)
_ERROR_CONTENT_PATTERNS = [
    "api quota", "rate limit", "temporarily unavailable", "service unavailable",
    "please try again", "check your api", "ai service", "llm service",
    "timeout", "connection error", "network error", "authentication failed",
    "standards documents", "relevant documents",  # Common in error fallback messages
    ".docx)", ".doc)", ".pdf)",  # File references in error messages
]

# Maximum length for a valid spec value (error messages are typically much longer)
MAX_VALID_SPEC_VALUE_LENGTH = 150


def is_error_message(value: str) -> bool:
    """
    Detect if a value looks like an error message rather than a spec value.

    ROOT CAUSE FIX: Filters out error messages that were incorrectly stored as values
    when AI service calls failed but returned fallback messages.

    Args:
        value: The string value to check

    Returns:
        True if the value appears to be an error message, False otherwise
    """
    if not value or not isinstance(value, str):
        return False

    val_lower = value.lower().strip()

    # Check 1: Length check - spec values rarely exceed 150 chars
    if len(val_lower) > MAX_VALID_SPEC_VALUE_LENGTH:
        # Long values are likely error messages or descriptions
        return True

    # Check 2: Error content patterns (anywhere in the string)
    for pattern in _ERROR_CONTENT_PATTERNS:
        if pattern in val_lower:
            logger.debug(f"[is_error_message] Detected error pattern '{pattern}' in value: {value[:50]}...")
            return True

    # Check 3: Starts with error-like patterns
    for pattern in _SENTENCE_START_PATTERNS:
        if val_lower.startswith(pattern):
            # Only consider it an error if it's also suspiciously long (>50 chars)
            if len(val_lower) > 50:
                return True

    return False


def trim_spec_value(value: str, max_words: int = MAX_SPEC_VALUE_WORDS) -> str:
    """
    Trim a specification value to keep it concise for UI display.
    
    Rules:
    - Values with <= 7 words are kept as-is (they are concise enough).
    - Values with > 7 words are trimmed to 4 words.
    - If the resulting value looks like a sentence/instruction (starts with
      patterns like "For", "Mount", "Use", "Ensure", etc.), the value is
      discarded entirely (returns empty string "").
    - Callers should check for empty return and skip the key-value pair.
    
    Examples:
        "316L SS"                              -> "316L SS" (unchanged, <= 7 words)
        "4-20mA HART"                          -> "4-20mA HART" (unchanged)
        "0-100 bar gauge pressure range"       -> "0-100 bar gauge pressure range" (7 words, kept)
        "Carbon steel body with 316SS trim"    -> "Carbon steel body with 316SS trim" (7 words, kept)
        "Mount sensor at 1/4 of vessel dia..." -> "" (discarded - sentence starting with "Mount")
        "vendor safety IS barriers signal..."  -> "vendor safety IS barriers" (trimmed to 4)
    
    Args:
        value: The specification value string
        max_words: Threshold word count above which trimming occurs (default: 7)
        
    Returns:
        Trimmed value string, or empty string "" if the value should be discarded.
    """
    if not value or not isinstance(value, str):
        return value or ""
    
    value = value.strip()
    words = value.split()
    
    # Short enough — keep as-is
    if len(words) <= max_words:
        return value
    
    # Value is verbose (> 7 words) — check if it's a sentence/instruction
    val_lower = value.lower()
    for pattern in _SENTENCE_START_PATTERNS:
        if val_lower.startswith(pattern):
            # This is a sentence/instruction, discard entirely
            return ""
    
    # Trim to 4 words
    trimmed = " ".join(words[:TRIM_TARGET_WORDS])
    return trimmed

def clean_spec_string_value(val_str: str) -> str:
    """
    Detect and clean string values that contain Python dict/list syntax.

    Converts stringified dicts back to clean values:
    - "{'value': 'Carbon steel', 'confidence': 1.0}" -> "Carbon steel"
    - "{'key1': 'val1', 'key2': 'val2'}" -> "key1: val1; key2: val2"
    - "[{'value': 'a'}, {'value': 'b'}]" -> "a, b"
    - "plain string" -> "plain string" (unchanged)

    ROOT CAUSE FIX: Now filters out error messages that were stored as values.
    """
    if not val_str or not isinstance(val_str, str):
        return val_str or ""

    stripped = val_str.strip()

    # ROOT CAUSE FIX: Check for error messages before any other processing
    if is_error_message(stripped):
        logger.debug(f"[clean_spec_string_value] Filtered error message: {stripped[:50]}...")
        return ""
    
    # Quick check: does it look like a Python dict/list?
    if not (stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("({")):
        return val_str
    
    # Try to parse it
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(stripped.replace("'", '"'))
        except json.JSONDecodeError:
            value_match = re.search(r"['\"]value['\"]\s*:\s*['\"]([^'\"]+)['\"]", stripped)
            if value_match:
                return value_match.group(1)
            return val_str
        
    if isinstance(parsed, dict):
        if 'value' in parsed:
            val = parsed['value']
            if isinstance(val, dict) and 'value' in val:
                val = val['value']
            if val and str(val).lower() not in ['null', 'none', 'n/a', '']:
                return str(val)
            return ""
        
        parts = []
        for k, v in parsed.items():
            if isinstance(v, dict) and 'value' in v:
                clean_v = str(v['value'])
            else:
                clean_v = str(v)
            if clean_v and clean_v.lower() not in ['null', 'none', 'n/a', '']:
                clean_k = k.replace('_', ' ').title()
                parts.append(f"{clean_k}: {clean_v}")
        return "; ".join(parts) if parts else ""
    
    elif isinstance(parsed, list):
        values = []
        for item in parsed:
            if isinstance(item, dict) and 'value' in item:
                v = str(item['value'])
                if v and v.lower() not in ['null', 'none', '']:
                    values.append(v)
            elif item:
                values.append(str(item))
        return ", ".join(values) if values else ""
    
    return val_str


def extract_spec_value(spec_value: Any) -> str:
    """
    Extract ONLY the value from any specification format.

    Handles:
    - dict with 'value' key: {'value': 'x', 'confidence': 0.9} -> 'x'
    - category wrapper dict: {'key1': 'val1'} -> 'key1: val1'
    - plain string: 'plain value' -> 'plain value'
    - list: [{'value': 'a'}, {'value': 'b'}] -> 'a, b'
    - stringified dict: "{'value': 'x'}" -> 'x'

    ROOT CAUSE FIX: Filters out error messages and empty values.
    """
    if spec_value is None:
        return ""

    # ROOT CAUSE FIX: Check for error dict structure (e.g., {"error": "..."})
    if isinstance(spec_value, dict) and "error" in spec_value and "value" not in spec_value:
        logger.debug(f"[extract_spec_value] Filtered error dict: {spec_value}")
        return ""

    if isinstance(spec_value, dict):
        if 'value' in spec_value:
            val = spec_value['value']
            if isinstance(val, dict) and 'value' in val:
                val = val['value']
            return str(val) if val and str(val).lower() not in ['null', 'none', 'n/a'] else ""
        
        parts = []
        for k, v in spec_value.items():
            if isinstance(v, dict) and 'value' in v:
                clean_v = str(v['value'])
            elif isinstance(v, dict):
                clean_v = "; ".join(f"{ik}: {iv}" for ik, iv in v.items() 
                                     if iv and str(iv).lower() not in ['null', 'none', ''])
            else:
                clean_v = str(v) if v else ""
            if clean_v and clean_v.lower() not in ['null', 'none', 'n/a', '']:
                clean_k = k.replace('_', ' ').title()
                parts.append(f"{clean_k}: {clean_v}")
        return "; ".join(parts) if parts else ""
    
    if isinstance(spec_value, list):
        values = []
        for v in spec_value:
            extracted = extract_spec_value(v)
            if extracted:
                values.append(extracted)
        return ", ".join(values) if values else ""
    
    result = str(spec_value) if spec_value else ""
    cleaned = clean_spec_string_value(result)

    # ROOT CAUSE FIX: Final validation to catch any error messages that slipped through
    if is_error_message(cleaned):
        logger.debug(f"[extract_spec_value] Final filter caught error message: {cleaned[:50]}...")
        return ""

    return cleaned


def flatten_nested_specs(spec_dict: dict, preserve_metadata: bool = False) -> dict:
    """
    Flatten nested specification dictionaries into simple key-value pairs.

    Input:  {'environmental_specs': {'temperature_range': {'value': '200C', 'confidence': 0.9}}}
    Output: {'temperature_range': '200C'}  (if preserve_metadata=False)
    Output: {'temperature_range': {'value': '200C', 'confidence': 0.9}}  (if preserve_metadata=True)

    ROOT CAUSE FIX: Now filters out error messages and empty values during flattening.

    Args:
        spec_dict: Nested specification dictionary
        preserve_metadata: If True, preserve the original dict structure (including source, confidence)

    Returns:
        Flattened specification dictionary (with error values filtered out)
    """
    flattened = {}

    # Keys that typically contain array values with verbose content (not spec values)
    SKIP_ARRAY_KEYS = {
        'requirements', 'recommendations', 'constraints', 'notes', 'warnings',
        'considerations', 'guidelines', 'best_practices', 'additional_notes'
    }

    # ROOT CAUSE FIX: Skip keys that indicate error status
    SKIP_ERROR_KEYS = {'error', 'error_message', 'exception', 'traceback'}

    for key, value in spec_dict.items():
        key_lower = key.lower()

        # Skip keys that typically contain verbose arrays
        if key_lower in SKIP_ARRAY_KEYS:
            continue

        # ROOT CAUSE FIX: Skip error-related keys
        if key_lower in SKIP_ERROR_KEYS:
            logger.debug(f"[flatten_nested_specs] Skipping error key: {key}")
            continue

        # ROOT CAUSE FIX: Skip error dict structures
        if isinstance(value, dict) and 'error' in value and 'value' not in value:
            logger.debug(f"[flatten_nested_specs] Skipping error dict for key: {key}")
            continue

        # Skip list/array values entirely - they don't represent proper spec values
        if isinstance(value, list):
            # Only join if it's a short list of simple values (e.g., ["HART", "Modbus"])
            if len(value) <= 3 and all(isinstance(v, str) and len(v) < 30 for v in value):
                joined = ", ".join(str(v) for v in value if v)
                if joined:
                    flattened[key] = joined
            # Otherwise skip - likely verbose requirements/recommendations array
            continue

        if isinstance(value, dict):
            if 'value' in value:
                if preserve_metadata:
                    # Keep the entire dict (for source extraction)
                    flattened[key] = value
                else:
                    # Extract just the value (for display)
                    extracted = extract_spec_value(value)
                    if extracted:
                        flattened[key] = extracted
            else:
                for inner_key, inner_value in value.items():
                    inner_key_lower = inner_key.lower()
                    # Skip nested verbose keys
                    if inner_key_lower in SKIP_ARRAY_KEYS:
                        continue
                    # Skip nested array values
                    if isinstance(inner_value, list):
                        if len(inner_value) <= 3 and all(isinstance(v, str) and len(v) < 30 for v in inner_value):
                            joined = ", ".join(str(v) for v in inner_value if v)
                            if joined:
                                flattened[inner_key] = joined
                        continue
                    if isinstance(inner_value, dict) and 'value' in inner_value:
                        if preserve_metadata:
                            flattened[inner_key] = inner_value
                        else:
                            extracted = extract_spec_value(inner_value)
                            if extracted:
                                flattened[inner_key] = extracted
                    elif inner_value and str(inner_value).lower() not in ['null', 'none', '']:
                        flattened[inner_key] = str(inner_value)
        elif value and str(value).lower() not in ['null', 'none', '']:
            cleaned = clean_spec_string_value(str(value))
            if cleaned and cleaned.lower() not in ['null', 'none', '']:
                flattened[key] = cleaned

    # ROOT CAUSE FIX: Final pass to filter any error messages that slipped through
    filtered_flattened = {}
    for key, value in flattened.items():
        if isinstance(value, str) and is_error_message(value):
            logger.debug(f"[flatten_nested_specs] Final filter removed error value for '{key}'")
            continue
        if value:  # Skip empty values
            filtered_flattened[key] = value

    return filtered_flattened


# =============================================================================
# SAMPLE INPUT BUILDER
# =============================================================================

# Semantic groups for categorizing spec keys
_MATERIAL_KEYWORDS = {'material', 'wetted', 'construction', 'housing', 'probe', 'column_material'}
_RANGE_KEYWORDS = {'range', 'pressure', 'temperature', 'accuracy', 'repeatability',
                   'resolution', 'viscosity', 'density', 'flow', 'level', 'turndown'}
_SERVICE_KEYWORDS = {'service', 'application', 'fluid', 'process', 'medium', 'suitability'}
_COMM_KEYWORDS = {'protocol', 'communication', 'output', 'fieldbus', 'hart', 'ethernet',
                  'signal', 'dcs', 'integration'}
_CERT_KEYWORDS = {'certification', 'approval', 'hazardous', 'explosion', 'atex',
                  'enclosure_rating', 'ip', 'nema', 'safety', 'sil'}

# Low-value spec keys to skip (only verbose procedural/installation details)
SKIP_KEYS = {
    # Installation procedures (verbose, not searchable)
    "manufacturers_cable_requirements", "cable_requirements", "cable_specifications",
    "installation_guidelines", "installation_route", "installation_requirements",
    "installation_recommendations",
    # Documentation (meta, not spec values)
    "general_description", "general_information", "description",
    "data_sheet_requirements", "documentation_requirements",
    # Cable routing details (implementation-specific)
    "cable_entry", "cable_gland_certification", "cable_routing",
    "cable_segregation_standard", "cable_specification", "cable_testing",
    "cable_insulation_resistance_test_frequency", "cable_replacement_criteria",
    "instrument_cable_type",
    # Fieldbus segment infrastructure (not product specs)
    "fieldbus_segment_limits", "segment_limits", "spur_length",
    "fieldbus_cable_requirements", "fieldbus_spur_length_limit",
    "fieldbus_segment_components",
    # Verification/commissioning procedures
    "verify_zero_span", "verify_linearity", "verify_communication",
    "verify_device_registration", "check_barrier_status",
    "test_devices_for_proper_signal_handling", "check_network_drop",
    "fieldbus_commissioning_verification", "communication_parameter_verification",
    "barrier_voltage_drop_testing",
    # Mounting/orientation details (too specific)
    "enclosure_orientation",
    "barrier_isolator_mounting_location",
    "mount_barriers_and_isolators",
    # Maintenance/security (operational, not spec)
    "firmware_configuration_maintenance",
    "cyber_security_measures", "maintain_security",
    # Tagging/addressing (configuration, not spec)
    "node_address", "device_tag",
    # Site-specific (not product specs)
    "perform_site_survey", "locate_gateways", "provide_clear_labelling",
    "condensation_prone_areas",
    # Grounding (installation detail)
    "grounding_practice", "ground_shields",
    # Impulse line details (installation)
    "impulse_line_slope", "impulse_tubing_material", "impulse_tubing_od",
    # Verbose aggregate keys from batch processing (contain arrays/lists, not spec values)
    "recommendations", "recommendation", "requirements", "requirement",
    "constraints", "constraint", "notes", "note", "warnings", "warning",
    "considerations", "consideration", "guidelines", "guideline",
    "best_practices", "best_practice", "additional_notes", "additional_requirements",
    # Generic verbose keys that contain descriptions instead of values
    "general_notes", "special_considerations", "application_notes",
    "selection_criteria", "design_considerations", "compliance_notes",
}


def _categorize_key(key: str) -> str:
    """Categorize a specification key into a semantic group."""
    key_lower = key.lower()
    for kw in _MATERIAL_KEYWORDS:
        if kw in key_lower:
            return "materials"
    for kw in _RANGE_KEYWORDS:
        if kw in key_lower:
            return "ranges"
    for kw in _SERVICE_KEYWORDS:
        if kw in key_lower:
            return "service"
    for kw in _COMM_KEYWORDS:
        if kw in key_lower:
            return "communication"
    for kw in _CERT_KEYWORDS:
        if kw in key_lower:
            return "certifications"
    return "features"


def build_sample_input(item: dict, project_name: str = "Project", max_specs: int = 50) -> str:
    """
    Build a natural language sample_input string from an item's specifications.
    
    Uses combined_specifications (enriched) if available,
    falls back to specifications (initial).
    
    This function is idempotent — it always builds from scratch using the item's
    current spec data, avoiding duplication issues.
    
    STRATEGY:
    1. Skip generic/low-value specs
    2. Prioritize by source: user_specified > standards > llm_generated
    3. Categorize specs semantically (materials, ranges, service, etc.)
    4. Assemble into natural language prose (no semicolons, no key-value pairs)
    5. Cap at max_specs to prevent validation errors
    
    Args:
        item: Item dict with name, category, specifications/combined_specifications
        project_name: Project name for context
        max_specs: Maximum number of spec values to include (default: 50)
        
    Returns:
        Natural language description string for product search
    """
    name = item.get("name") or item.get("accessory_name") or item.get("product_name") or "Component"
    category = item.get("category", "Industrial Item")

    # Get ALL specifications from BOTH sources (combined_specifications has enriched specs)
    combined_specs = item.get("combined_specifications", {})
    original_specs = item.get("specifications", {})

    # Flatten and use best available (preserve metadata for source extraction)
    specs = {}
    if combined_specs:
        specs = flatten_nested_specs(combined_specs, preserve_metadata=True)
    elif original_specs:
        specs = flatten_nested_specs(original_specs, preserve_metadata=True) if isinstance(original_specs, dict) else {}

    if not specs:
        return f"{name} ({category}) for {project_name}."

    # ── Step 1: Extract & prioritize values ────────────────────────────
    # Each entry: (key, clean_value, source_priority)
    # source_priority: 0=user, 1=standards, 2=llm, 3=unknown
    SOURCE_PRIORITY = {"user_specified": 0, "standards": 1, "llm_generated": 2, "database": 2}
    
    raw_entries = []
    skipped_by_filter = []
    for k, v in specs.items():
        if k.lower() in SKIP_KEYS:
            skipped_by_filter.append(k)
            continue

        # Extract clean value
        if isinstance(v, (dict, list)):
            val = extract_spec_value(v)
            source = v.get("source", "") if isinstance(v, dict) else ""
        else:
            val = clean_spec_string_value(str(v))
            source = ""

        if not val or val.lower() in ["null", "none", "n/a", ""]:
            continue

        val_clean = val.replace("(Standards)", "").replace("(Inferred)", "").replace("(LLM)", "").replace("(USER)", "").strip()
        if not val_clean:
            continue

        val_clean = trim_spec_value(val_clean)
        priority = SOURCE_PRIORITY.get(source, 3)
        raw_entries.append((k, val_clean, priority))

    # Sort by source priority (user first, then standards, then llm)
    raw_entries.sort(key=lambda x: x[2])
    
    # Log skipped specs for debugging
    if skipped_by_filter:
        import logging
        logging.getLogger(__name__).warning(f"[SAMPLE_INPUT] Skipped {len(skipped_by_filter)} specs by SKIP_KEYS: {skipped_by_filter}")

    # Cap at max_specs
    capped_entries = raw_entries[:max_specs]

    if not capped_entries:
        return f"{name} ({category}) for {project_name}."

    # ── Step 2: Categorize semantically ────────────────────────────────
    groups = {
        "materials": [],
        "ranges": [],
        "service": [],
        "communication": [],
        "certifications": [],
        "features": [],
    }

    for k, val, _ in capped_entries:
        group = _categorize_key(k)
        groups[group].append(val)

    # Limit each group to keep the output concise
    for g in groups:
        groups[g] = groups[g][:8]

    # ── Step 3: Assemble natural language description programmatically ─────────
    parts = [f"{name}"]
    
    def deduplicate(lst: list) -> list:
        seen = set()
        res = []
        for v in lst:
            # removing standard tags brackets if any crept through
            cleaned_v = v.replace("[STANDARDS]", "").replace("[INFERRED]", "").replace("[LLM]", "").replace("[USER]", "")
            cleaned_v = cleaned_v.strip()
            if cleaned_v and cleaned_v.lower() not in seen:
                seen.add(cleaned_v.lower())
                res.append(cleaned_v)
        return res

    # Service / application / Purpose
    purpose = item.get("purpose") or item.get("solution_purpose") or item.get("related_instrument")
    if groups["service"] or purpose:
        service_parts = groups["service"]
        if purpose and purpose not in service_parts:
            service_parts.insert(0, purpose)
        service_parts = deduplicate(service_parts)
        if service_parts:
            parts.append(f"for {', '.join(service_parts)}")

    # Materials
    if groups["materials"]:
        materials_parts = deduplicate(groups['materials'])
        if materials_parts:
            parts.append(f"with {', '.join(materials_parts)}")

    # Ranges / ratings
    if groups["ranges"]:
        ranges_parts = deduplicate(groups['ranges'])
        if ranges_parts:
            parts.append(f"rated at {', '.join(ranges_parts)}")

    # Communication
    if groups["communication"]:
        comm_parts = deduplicate(groups['communication'])
        if comm_parts:
            parts.append(f"using {', '.join(comm_parts)}")

    # Certifications
    if groups["certifications"]:
        cert_parts = deduplicate(groups['certifications'])
        if cert_parts:
            parts.append(f"certified {', '.join(cert_parts)}")

    # Features (remaining)
    if groups["features"]:
        feat_parts = deduplicate(groups['features'][:6])
        if feat_parts:
            parts.append(f"with {', '.join(feat_parts)}")

    description_string = " ".join(parts) + "."
    return description_string
