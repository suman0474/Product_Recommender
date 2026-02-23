"""
Generic Product Type Image Utilities
Handles fetching cached generic product type images using Azure Blob Storage and MongoDB.

ARCHITECTURE:
1. Check Azure Blob Storage (L1 Cache - Primary)
2. Check MongoDB (L2 Cache - Secondary)
3. Return URL/metadata or None if not found (NO Generation)

STORAGE: Azure Blob Storage (primary) and MongoDB (metadata).
"""

import logging
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Issue-specific debug logging for terminal log analysis
try:
    from debug_flags import issue_debug
except ImportError:
    issue_debug = None  # Fallback if debug_flags not available


# --- Main Utilities ---

def get_generic_image_from_azure(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached generic product type image from Azure Blob Storage (L1 Cache)

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")

    Returns:
        Dict containing image metadata or None if not found
    """
    try:
        from common.config.azure_blob_config import Collections, azure_blob_manager
        from azure.core.exceptions import ResourceNotFoundError
        import json

        # Product type aliases for better matching
        PRODUCT_TYPE_ALIASES = {
            'temperaturetransmitter': ['temptransmitter', 'temperaturesensor', 'rtdtransmitter'],
            'pressuretransmitter': ['pressuretransducer', 'pressuresensor'],
            'flowmeter': ['flowtransmitter', 'flowsensor'],
            'leveltransmitter': ['levelindicator', 'levelmeter', 'levelsensor'],
            'controlvalve': ['controlvalves', 'valve'],
            'variablefrequencydrive': ['vfd', 'frequencyinverter', 'acdrive'],
            'thermocouple': ['thermocouplesensor', 'tcassembly'],
            'junctionbox': ['junctionboxes', 'jboxes', 'enclosure'],
            'mountingbracket': ['bracket', 'mountinghardware'],
        }

        # Normalize product type for Azure path
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        
        # Check if this normalized type has a preferred alias
        for canonical, aliases in PRODUCT_TYPE_ALIASES.items():
            if normalized_type in aliases:
                # logger.info(f"[AZURE_CHECK] Alias mapping: '{normalized_type}' -> '{canonical}'")
                normalized_type = canonical
                break

        # Check if azure_blob_manager is available
        if not azure_blob_manager.is_available:
             raise Exception("Azure Blob Manager not initialized or available")

        # Get the underlying file manager
        file_manager = azure_blob_manager.get_client()
        
        # Try finding the image blob path exactly as uploaded by azure_blob_file_manager
        product_normalized = product_type.lower().replace(' ', '_')
        image_blob_path_exact = f"generic_{product_normalized}.png"
        
        found_blob_path = None
        if file_manager.file_exists(image_blob_path_exact, container_name=Collections.GENERIC_IMAGES):
            found_blob_path = image_blob_path_exact
        else:
            # Fallback to the alias normalized_type if different
            alias_blob_path = f"generic_{normalized_type}.png"
            if file_manager.file_exists(alias_blob_path, container_name=Collections.GENERIC_IMAGES):
                found_blob_path = alias_blob_path

        if not found_blob_path:
            raise ResourceNotFoundError(f"Image file not found: {image_blob_path_exact}")

        logger.info(f"[AZURE_CHECK] ✓ Found cached generic image in Azure Blob for: {product_type}")
        if issue_debug:
            issue_debug.cache_hit("azure_blob", normalized_type)

        return {
            'azure_blob_path': found_blob_path,
            'product_type': product_type,
            'source': 'gemini_imagen',
            'content_type': 'image/png',
            'file_size': 0,
            'generation_method': 'llm',
            'cached': True,
            'storage_location': 'azure_blob'
        }

    except ResourceNotFoundError:
        # logger.info(f"[AZURE_CHECK] No cached generic image in Azure Blob for: {product_type} (normalized: {normalized_type})")
        # Try local fallback here if not found in Azure
        return _get_local_generic_image(product_type, normalized_type)
        
    except Exception as e:
        # logger.warning(f"[AZURE_CHECK] Failed to retrieve generic image from Azure Blob for '{product_type}': {e}")
        # Try local fallback on error
        return _get_local_generic_image(product_type, normalized_type)


def _get_local_generic_image(product_type: str, normalized_type: str = None) -> Optional[Dict[str, Any]]:
    """Helper to check local disk for generic images."""
    try:
        import json
        if not normalized_type:
             normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
             
        local_dir = os.path.join(os.getcwd(), 'static', 'images', 'generic_images')
        metadata_path = os.path.join(local_dir, f"{normalized_type}.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"[LOCAL_CHECK] ✓ Found cached generic image locally for: {product_type}")
            return {
                'azure_blob_path': f"{normalized_type}.png", # Simplified path for local
                'product_type': metadata.get('product_type'),
                'source': metadata.get('source'),
                'content_type': metadata.get('content_type', 'image/png'),
                'file_size': metadata.get('file_size', 0),
                'generation_method': metadata.get('generation_method', 'llm'),
                'cached': True,
                'storage_location': 'local'
            }
    except Exception as e:
        logger.warning(f"[LOCAL_CHECK] Failed local lookup for {product_type}: {e}")
    
    return None


# PARALLEL BATCH IMAGE FETCHING
# =============================================================================

def fetch_generic_images_batch(product_types: list, max_parallel_cache_checks: int = 10) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch generic images for multiple product types IN PARALLEL.
    
    STRATEGY:
    - Check Azure Blob (L1) and MongoDB (L2) in parallel.
    - NO LLM generation.

    Args:
        product_types: List of product type strings to fetch images for
        max_parallel_cache_checks: Number of parallel cache checks

    Returns:
        Dict mapping product_type -> image result (or None if failed)
    """
    if not product_types:
        return {}

    results = {}
    
    # Define check function
    def check_caches(product_type):
        return (product_type, fetch_generic_product_image(product_type))

    logger.info(f"[BATCH_IMAGE] Starting parallel image fetch for {len(product_types)} product types...")
    
    with ThreadPoolExecutor(max_workers=max_parallel_cache_checks) as executor:
        future_to_type = {executor.submit(check_caches, pt): pt for pt in product_types}
        
        for future in as_completed(future_to_type):
            product_type = future_to_type[future]
            try:
                pt, result = future.result()
                results[pt] = result
            except Exception as exc:
                logger.error(f"[BATCH_IMAGE] Exception for '{product_type}': {exc}")
                results[product_type] = None
                
    return results


def _try_fallback_cache_lookup(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Try fallback cache lookups for a product type.
    
    This only checks cache (Azure/Local), no LLM generation.
    """
    import re
    
    # Common prefixes to strip
    COMMON_PREFIXES = [
        # General modifiers (safe to strip - don't change visual appearance)
        "Process ", "Industrial ", "Digital ", "Smart ", "Advanced ", 
        "Precision ", "High Performance ", "Multi ", "Dual ", "Single ",
        "Compact ", "Integrated ", "Electronic ", "Intelligent ", "Programmable ",
        "Wireless ", "Remote ", "Field ", "Panel ", "Portable ", "Handheld ",
        "Ex-rated ", "ATEX ", "Hazardous Area ", "Explosion Proof ",
        # Communication protocol prefixes (don't change physical appearance)
        "HART ", "Foundation Fieldbus ", "Profibus ", "Modbus ", "4-20mA ",
        # Material/application prefixes (minor visual differences)
        "Stainless Steel ", "SS ", "316L ", "Hastelloy ", "Titanium ",
        "Sanitary ", "Hygienic ", "Food Grade ", "Pharmaceutical ",
        # Size/range prefixes
        "High Pressure ", "Low Pressure ", "High Temperature ", "Low Temperature ",
        "Wide Range ", "Narrow Range ", "High Accuracy ",
    ]
    
    fallback_types = []
    product_upper = product_type.strip()
    
    for prefix in COMMON_PREFIXES:
        if product_upper.lower().startswith(prefix.lower()):
            fallback_type = product_upper[len(prefix):].strip()
            if fallback_type and fallback_type != product_type:
                fallback_types.append(fallback_type)
    
    # Remove parenthetical content
    base_type = re.sub(r'\s*\([^)]*\)\s*', '', product_type).strip()
    if base_type and base_type != product_type and base_type not in fallback_types:
        fallback_types.append(base_type)
    
    # Extract base instrument type
    words = product_type.strip().split()
    if len(words) >= 2:
        last_word = words[-1]
        if last_word not in fallback_types and last_word != product_type:
            fallback_types.append(last_word)
        last_two = " ".join(words[-2:])
        if last_two not in fallback_types and last_two != product_type:
            fallback_types.append(last_two)
    
    # Try each fallback
    for fallback in fallback_types:
        # Try Azure Cache (L1)
        azure_image = get_generic_image_from_azure(fallback)
        if azure_image:
            logger.info(f"[BATCH_IMAGE] ✓ Fallback cache hit (Azure): '{fallback}' for '{product_type}'")
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
            return {
                'url': backend_url,
                'product_type': product_type,
                'source': azure_image.get('source', 'gemini_imagen'),
                'cached': True,
                'generation_method': azure_image.get('generation_method', 'llm'),
                'fallback_used': fallback
            }
            
        # Try MongoDB Cache (L2) for fallback
        try:
            from common.services.image_service import image_service
            if image_service:
                mongo_image = image_service.get_generic_image(fallback)
                if mongo_image:
                    logger.info(f"[BATCH_IMAGE] ✓ Fallback cache hit (MongoDB): '{fallback}' for '{product_type}'")
                    return {
                        'url': mongo_image.get('image_url', ''),
                        'product_type': product_type,
                        'source': mongo_image.get('source', 'gemini_imagen'),
                        'cached': True,
                        'cache_source': 'mongodb',
                        'generation_method': mongo_image.get('generation_method', 'llm'),
                        'fallback_used': fallback
                    }
        except:
            pass

    
    # Category-based fallback (last resort)
    CATEGORY_MAPPINGS = {
        # Instruments
        'transmitter': ['Transmitter', 'Pressure Transmitter', 'Temperature Transmitter'],
        'sensor': ['Sensor', 'Temperature Sensor', 'Pressure Sensor'],
        'valve': ['Valve', 'Control Valve', 'Globe Valve', 'Ball Valve'],
        'actuator': ['Actuator', 'Pneumatic Actuator'],
        'meter': ['Meter', 'Flow Meter'],
        'gauge': ['Gauge', 'Pressure Gauge'],
        'cable': ['Cable', 'Instrument Cable', 'Cable Gland'],
        'fitting': ['Fitting', 'Pipe Fitting'],
        'bracket': ['Bracket', 'Mounting Bracket'],
        'junction': ['Junction Box'],
        'switch': ['Switch', 'Pressure Switch'],
        'motor': ['Motor', 'Electric Motor'],
        'drive': ['Drive', 'Variable Frequency Drive'],
        'pump': ['Pump'],
        'controller': ['Controller', 'Digital Valve Controller'],
        'analyzer': ['Analyzer'],
        # Accessories - Temperature
        'thermowell': ['Thermowell', 'Protective Sleeve', 'Temperature Well'],
        'terminal': ['Terminal Head', 'Connection Head', 'Terminal Box'],
        # Accessories - Pressure
        'manifold': ['Manifold', '3-Valve Manifold', '5-Valve Manifold', 'Valve Manifold'],
        'impulse': ['Impulse Line', 'Impulse Tubing'],
        'snubber': ['Snubber', 'Pulsation Dampener'],
        # Accessories - Flow
        'gasket': ['Gasket', 'Flange Gasket', 'Gasket Kit', 'Sealing Gasket'],
        'bolt': ['Bolt', 'Bolts and Nuts', 'Stud Bolt', 'Bolt Kit', 'Fastener'],
        'flange': ['Flange', 'Mounting Flange', 'Process Flange', 'Flange Adapter'],
        'straightener': ['Flow Straightener', 'Flow Conditioner'],
        # Accessories - General
        'gland': ['Cable Gland', 'Gland', 'Connector Gland'],
        'regulator': ['Air Filter Regulator', 'Filter Regulator', 'Pressure Regulator'],
        'sunshield': ['Sunshield', 'Sun Shield', 'Weather Shield'],
        'enclosure': ['Enclosure', 'Junction Box', 'Terminal Enclosure'],
        'positioner': ['Positioner', 'Valve Positioner', 'Digital Positioner'],
    }
    
    product_lower = product_type.lower()
    for category, generic_types in CATEGORY_MAPPINGS.items():
        if category in product_lower:
            for generic_type in generic_types:
                if generic_type.lower() != product_lower and generic_type not in fallback_types:
                    # Check Azure (L1)
                    azure_image = get_generic_image_from_azure(generic_type)
                    if azure_image:
                        logger.info(f"[BATCH_IMAGE] ✓ Category fallback (Azure): '{generic_type}' for '{product_type}'")
                        backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
                        return {
                            'url': backend_url,
                            'product_type': product_type,
                            'source': azure_image.get('source', 'gemini_imagen'),
                            'cached': True,
                            'generation_method': azure_image.get('generation_method', 'llm'),
                            'fallback_used': generic_type,
                            'fallback_type': 'category'
                        }
                    
                    # Check MongoDB (L2)
                    try:
                        from common.services.image_service import image_service
                        if image_service:
                            mongo_image = image_service.get_generic_image(generic_type)
                            if mongo_image:
                                logger.info(f"[BATCH_IMAGE] ✓ Category fallback (MongoDB): '{generic_type}' for '{product_type}'")
                                return {
                                    'url': mongo_image.get('image_url', ''),
                                    'product_type': product_type,
                                    'source': mongo_image.get('source', 'gemini_imagen'),
                                    'cached': True,
                                    'cache_source': 'mongodb',
                                    'generation_method': mongo_image.get('generation_method', 'llm'),
                                    'fallback_used': generic_type,
                                    'fallback_type': 'category'
                                }
                    except:
                        pass
    
    return None


def fetch_generic_product_image(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch generic product type image using Azure Blob as L1 and MongoDB as L2.
    NO LLM Generation.

    FLOW:
    1. Check Azure Blob Storage (Primary L1)
    2. Check MongoDB (Secondary L2)
    3. Try fallback cache lookups
    4. Return None if not found

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")

    Returns:
        Dict containing image URL and metadata, or None if not found
    """
    logger.info(f"[FETCH] Fetching generic image for product type: {product_type}")

    # Step 1: Check Azure Blob Storage (Primary L1 Cache)
    azure_image = get_generic_image_from_azure(product_type)
    if azure_image:
        logger.info(f"[FETCH] ✓ Using cached generic image from {azure_image.get('storage_location', 'Azure')} for '{product_type}' (L1 Cache)")
        
        # Determine URL based on storage location
        if azure_image.get('storage_location') == 'local':
            # Local URL
            filename = os.path.basename(azure_image.get('azure_blob_path', ''))
            backend_url = f"/api/images/generic_images/{filename}"
        else:
            # Azure URL
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

        return {
            'url': backend_url,
            'product_type': product_type,
            'source': azure_image.get('source', 'gemini_imagen'),
            'cached': True,
            'generation_method': azure_image.get('generation_method', 'llm')
        }

    # Step 2: Check MongoDB Cache (Secondary L2 Cache)
    try:
        from common.services.image_service import image_service
        if image_service:
            mongo_image = image_service.get_generic_image(product_type)
            if mongo_image:
                logger.info(f"[FETCH] ✓ MongoDB cache HIT for '{product_type}' (L2 Cache)")
                return {
                    'url': mongo_image.get('image_url', ''),
                    'product_type': product_type,
                    'source': mongo_image.get('source', 'gemini_imagen'),
                    'cached': True,
                    'cache_source': 'mongodb',
                    'generation_method': mongo_image.get('generation_method', 'llm'),
                    'cache_id': mongo_image.get('cache_id')
                }
            else:
                 # logger.info(f"[FETCH] MongoDB cache MISS for '{product_type}'")
                 pass
    except ImportError:
        logger.debug(f"[FETCH] image_service not available, skipping MongoDB cache")
    except Exception as e:
        logger.warning(f"[FETCH] MongoDB cache error: {e}")

    # Step 3: Try fallback strategies (using Azure/Local lookup)
    
    # Pattern 1: "X for Y" (accessories)
    extracted_base_type = None
    extraction_pattern = None

    if ' for ' in product_type:
        extracted_base_type = product_type.split(' for ')[0].strip()
        extraction_pattern = 'X for Y'
    elif ' with ' in product_type:
        extracted_base_type = product_type.split(' with ')[0].strip()
        extraction_pattern = 'X with Y'
    elif ' including ' in product_type:
        extracted_base_type = product_type.split(' including ')[0].strip()
        extraction_pattern = 'X including Y'
    elif ' featuring ' in product_type:
        extracted_base_type = product_type.split(' featuring ')[0].strip()
        extraction_pattern = 'X featuring Y'

    if extracted_base_type and extracted_base_type.lower() != product_type.lower():
        logger.info(f"[FETCH] Extracted base type from '{extraction_pattern}' pattern: '{extracted_base_type}'")

        # Try exact match for extracted type (Azure first)
        azure_image = get_generic_image_from_azure(extracted_base_type)
        if azure_image:
            logger.info(f"[FETCH] ✓ Found cached image for extracted base type: '{extracted_base_type}' (Azure)")
            backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"
            return {
                'url': backend_url,
                'product_type': product_type,
                'source': azure_image.get('source', 'gemini_imagen'),
                'cached': True,
                'generation_method': azure_image.get('generation_method', 'llm'),
                'extracted_from': product_type,
                'extracted_type': extracted_base_type,
                'extraction_pattern': extraction_pattern
            }
        
        # Try MongoDB
        try:
            from common.services.image_service import image_service
            if image_service:
                mongo_image = image_service.get_generic_image(extracted_base_type)
                if mongo_image:
                    logger.info(f"[FETCH] ✓ Found cached image for extracted base type: '{extracted_base_type}' (MongoDB)")
                    return {
                        'url': mongo_image.get('image_url', ''),
                        'product_type': product_type,
                        'source': mongo_image.get('source', 'gemini_imagen'),
                        'cached': True,
                        'cache_source': 'mongodb',
                        'generation_method': mongo_image.get('generation_method', 'llm'),
                        'extracted_from': product_type,
                        'extracted_type': extracted_base_type
                    }
        except:
            pass

    # Step 4: Try fallbacks (prefixes, categories)
    # If we extracted a base type (e.g. "Mounting Bracket" from "Mounting Bracket for X"),
    # we perform fallbacks on the base type to avoid returning the main instrument's image.
    target_for_fallback = extracted_base_type if extracted_base_type else product_type
    fallback_result = _try_fallback_cache_lookup(target_for_fallback)
    if fallback_result:
        fallback_result['product_type'] = product_type
        return fallback_result

    # Complete failure - no image available
    logger.info(f"[FETCH] No cached image found for '{product_type}'")
    return None


def fetch_generic_product_image_fast(product_type: str) -> Dict[str, Any]:
    """
    Fetch generic product image (fast mode).
    Same as main fetch but specifically for UI responsiveness.
    """
    result = fetch_generic_product_image(product_type)
    
    if result:
        return {
            'success': True,
            'url': result['url'],
            'product_type': product_type,
            'source': 'cache',
            'use_placeholder': False,
            'reason': 'cached'
        }
    
    return {
        'success': False,
        'url': None,
        'product_type': product_type,
        'source': None,
        'use_placeholder': True,
        'reason': 'not_found'
    }
