import json
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Standalone Module Imports
from .standalone_vector_store import PineconeStandaloneStore
from .standalone_llm import create_standalone_llm

logger = logging.getLogger(__name__)

class TaxonomyRAG:
    """
    RAG Service for Taxonomy.
    Handles indexing and retrieval of taxonomy terms.
    """

    def __init__(self):
        # Use our isolated vector store
        self.vector_store = PineconeStandaloneStore()
        self.collection_type = "taxonomy"

    def index_taxonomy(self, taxonomy_data: Dict[str, Any]) -> None:
        """
        Index the taxonomy into the vector store.
        Skips if already populated (checked via collection stats).
        """
        try:
            stats = self.vector_store.get_collection_stats()
            if stats.get("success"):
                cols = stats.get("collections", {})
                if self.collection_type in cols and cols[self.collection_type].get("document_count", 0) > 0:
                    logger.info("[TaxonomyRAG] Index already populated. Skipping ingestion.")
                    return

            logger.info("[TaxonomyRAG] Indexing taxonomy...")

            for item in taxonomy_data.get("instruments", []):
                doc_content = f"Instrument: {item['name']}\n"
                doc_content += f"Category: {item.get('category', 'General')}\n"
                doc_content += f"Definition: {item.get('definition', '')}\n"
                if item.get("aliases"):
                    doc_content += f"Aliases: {', '.join(item['aliases'])}\n"

                metadata = {
                    "type": "instrument",
                    "name": item['name'],
                    "aliases": item.get('aliases', []),
                    "category": item.get('category')
                }

                self.vector_store.add_document(
                    collection_type=self.collection_type,
                    content=doc_content,
                    metadata=metadata,
                    doc_id=f"tax_inst_{item['name'].replace(' ', '_')}"
                )

            for item in taxonomy_data.get("accessories", []):
                doc_content = f"Accessory: {item['name']}\n"
                doc_content += f"Category: {item.get('category', 'General')}\n"
                doc_content += f"Definition: {item.get('definition', '')}\n"
                if item.get("related_instruments"):
                    doc_content += f"Related Instruments: {', '.join(item['related_instruments'])}\n"
                if item.get("aliases"):
                    doc_content += f"Aliases: {', '.join(item['aliases'])}\n"

                metadata = {
                    "type": "accessory",
                    "name": item['name'],
                    "aliases": item.get('aliases', []),
                    "related_instruments": item.get('related_instruments', [])
                }

                self.vector_store.add_document(
                    collection_type=self.collection_type,
                    content=doc_content,
                    metadata=metadata,
                    doc_id=f"tax_acc_{item['name'].replace(' ', '_')}"
                )

            logger.info(
                f"[TaxonomyRAG] Successfully indexed {len(taxonomy_data.get('instruments', []))} instruments and {len(taxonomy_data.get('accessories', []))} accessories."
            )

        except Exception as e:
            logger.error(f"[TaxonomyRAG] Failed to index taxonomy: {e}")

    def retrieve(self, query: str, top_k: int = 5, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant taxonomy terms for a query."""
        try:
            filters = {}
            if item_type:
                filters = {"type": item_type}

            search_result = self.vector_store.search(
                collection_type=self.collection_type,
                query=query,
                top_k=top_k,
                filter_metadata=filters if filters else None
            )

            if not search_result.get("success"):
                logger.warning(f"[TaxonomyRAG] Search failed: {search_result.get('error')}")
                return []

            results = []
            for item in search_result.get("results", []):
                meta = item.get("metadata", {})
                results.append({
                    "name": meta.get("name"),
                    "aliases": meta.get("aliases", []),
                    "type": meta.get("type"),
                    "score": item.get("relevance_score"),
                    "content": item.get("content")
                })

            return results

        except Exception as e:
            logger.error(f"[TaxonomyRAG] Retrieval failed: {e}")
            return []

    def get_top_files_by_similarity(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top files by cosine similarity for specification extraction.
        Returns a list of individual file content strings (one per matched document).
        """
        try:
            search_result = self.vector_store.search(
                collection_type="taxonomy",  # Target the taxonomy namespace
                query=query,
                top_k=top_k
            )

            if not search_result.get("success"):
                logger.warning(f"[TaxonomyRAG] File search failed: {search_result.get('error')}")
                return []

            files = []
            for idx, item in enumerate(search_result.get("results", [])):
                text = item.get("content", "").strip()
                score = item.get("relevance_score", 0.0)
                if text:
                    logger.info(
                        f"[TaxonomyRAG] File {idx+1}/{top_k}: score={score:.4f} | length={len(text)} chars"
                    )
                    files.append(text)

            logger.info(f"[TaxonomyRAG] Retrieved {len(files)}/{top_k} files for query: '{query[:60]}'")
            return files
        except Exception as e:
            logger.error(f"[TaxonomyRAG] File retrieval failed: {e}")
            return []

    def extract_specifications_from_files(
        self, product_name: str, files: List[str]
    ) -> dict:
        """
        Iterate over each of the top retrieved files individually,
        extract specifications from each, and deep-merge all results.
        Falls back to a mechanical category placeholder if no specs are found.
        """
        if not files:
            logger.warning(f"[TaxonomyRAG] No files provided for {product_name}. Returning fallback.")
            return self._mechanical_fallback(product_name)

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import JsonOutputParser

            llm = create_standalone_llm(temperature=0.0)

            prompt_template = """
You are an expert technical data extractor. 
Analyze the following document or definition for: "{product_name}".

Your task is to extract ANY technical features, functional characteristics, 
measured variables, primary purposes, materials, input/output types, or operational concepts you can find.

Even if the text is just a short high-level definition without hard numbers (e.g. no temperature ranges), 
you MUST extract its conceptual properties into a structured format. For example, if it says "Measures fluid flow", 
extract {{"measured_variable": "Fluid Flow", "primary_function": "Measurement"}}.

Document Contents:
{file_content}

Return ONLY a valid JSON dictionary mapping these extracted characteristics to their values.
Do not hallucinate, but be thorough in extracting the conceptual engineering features present.
Output JSON Format:
"""
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | JsonOutputParser()

            merged_specs: dict = {}

            for idx, file_content in enumerate(files):
                if not file_content.strip():
                    continue

                # Trim individual file content to safe token limit
                if len(file_content) > 10000:
                    file_content = file_content[:10000] + "... (truncated)"

                logger.info(
                    f"[TaxonomyRAG] Extracting specs from file {idx+1}/{len(files)} for '{product_name}'..."
                )
                try:
                    result = chain.invoke({
                        "product_name": product_name,
                        "file_content": file_content
                    })

                    if isinstance(result, dict) and result:
                        new_keys = 0
                        for k, v in result.items():
                            # Deep merge: only add if key not already present
                            if k not in merged_specs:
                                merged_specs[k] = v
                                new_keys += 1
                        logger.info(
                            f"[TaxonomyRAG] File {idx+1}: added {new_keys} new spec keys "
                            f"(total so far: {len(merged_specs)})"
                        )
                    else:
                        logger.info(
                            f"[TaxonomyRAG] File {idx+1}: no extractable specs found."
                        )

                except Exception as file_err:
                    logger.warning(
                        f"[TaxonomyRAG] Spec extraction failed for file {idx+1} of '{product_name}': {file_err}"
                    )
                    continue

            if not merged_specs:
                logger.warning(
                    f"[TaxonomyRAG] No specs extracted from any file for '{product_name}'. "
                    f"Applying fallback."
                )
                return self._mechanical_fallback(product_name)

            logger.info(
                f"[TaxonomyRAG] Extracted {len(merged_specs)} total specs for '{product_name}' "
                f"from {len(files)} files (iterative merge)."
            )
            return merged_specs

        except Exception as e:
            logger.error(f"[TaxonomyRAG] Specification extraction failed: {e}")
            return self._mechanical_fallback(product_name)

    def _mechanical_fallback(self, product_name: str) -> dict:
        """
        Fallback for items (e.g. mechanical accessories) where the taxonomy
        database doesn't contain instrument specifications.
        """
        logger.info(f"[TaxonomyRAG] Applying mechanical fallback for '{product_name}'.")
        return {
            "product_name": product_name,
            "category": "Mechanical Accessory",
            "primary_function": "Mechanical mounting, support or protection",
            "typical_materials": ["stainless steel", "aluminum", "carbon steel"],
            "note": "No functional engineering specifications found in taxonomy. "
                    "This item is likely a mechanical/structural accessory."
        }





_taxonomy_rag_instance = None


def get_taxonomy_rag() -> TaxonomyRAG:
    global _taxonomy_rag_instance
    if _taxonomy_rag_instance is None:
        _taxonomy_rag_instance = TaxonomyRAG()
    return _taxonomy_rag_instance

class SpecificationRetriever:
    """
    Retrieves specifications for normalized product types from MongoDB or JSON files.
    
    Supports two modes:
    1. MongoDB mode: Queries a MongoDB collection for product specifications
    2. JSON file mode: Loads specifications from JSON catalog files
    """
    
    def __init__(
        self, 
        mongodb_uri: Optional[str] = None,
        mongodb_database: str = None,
        mongodb_collection: str = None,
        json_catalog_path: Optional[str] = None
    ):
        """
        Initialize the specification retriever.
        
        Args:
            mongodb_uri: MongoDB connection string (optional, uses MONGODB_URI env var)
            mongodb_database: Database name (optional, uses MONGODB_DATABASE env var or "engenie")
            mongodb_collection: Collection name (optional, uses MONGODB_COLLECTION env var or "product_specifications")
            json_catalog_path: Path to JSON catalog file or directory (optional)
        """
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        self.mongodb_database = mongodb_database or os.getenv("MONGODB_DATABASE", "engenie")
        self.mongodb_collection = mongodb_collection or os.getenv("MONGODB_COLLECTION", "product_specifications")
        self.json_catalog_path = json_catalog_path
        
        self._mongo_client = None
        self._json_catalog = None
        
        # Determine which mode to use
        self.mode = self._determine_mode()
        
        logger.info(f"[SpecRetriever] Initialized in {self.mode} mode")
    
    def _determine_mode(self) -> str:
        """Determine whether to use MongoDB or JSON file mode."""
        if self.mongodb_uri:
            return "mongodb"
        elif self.json_catalog_path:
            return "json"
        else:
            # Default to JSON mode with default path
            default_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "data", 
                "product_catalog.json"
            )
            self.json_catalog_path = default_path
            return "json"
    
    def _get_mongo_client(self):
        """Lazy initialization of MongoDB client."""
        if self._mongo_client is None and self.mongodb_uri:
            try:
                from pymongo import MongoClient
                self._mongo_client = MongoClient(self.mongodb_uri)
                # Test connection
                self._mongo_client.admin.command('ping')
                logger.info("[SpecRetriever] MongoDB connection established")
            except Exception as e:
                logger.error(f"[SpecRetriever] MongoDB connection failed: {e}")
                self._mongo_client = None
        
        return self._mongo_client
    
    def _load_json_catalog(self) -> Dict[str, Any]:
        """Load JSON catalog from file."""
        if self._json_catalog is not None:
            return self._json_catalog
        
        if not self.json_catalog_path:
            logger.warning("[SpecRetriever] No JSON catalog path specified")
            return {}
        
        try:
            catalog_path = Path(self.json_catalog_path)
            
            if not catalog_path.exists():
                logger.warning(f"[SpecRetriever] JSON catalog not found: {catalog_path}")
                return {}
            
            if catalog_path.is_file():
                # Single JSON file
                with open(catalog_path, 'r', encoding='utf-8-sig') as f:
                    self._json_catalog = json.load(f)
                logger.info(f"[SpecRetriever] Loaded catalog from {catalog_path}")
            
            elif catalog_path.is_dir():
                # Directory of JSON files - merge them
                self._json_catalog = {}
                for json_file in catalog_path.glob("*.json"):
                    with open(json_file, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                        self._json_catalog.update(data)
                logger.info(f"[SpecRetriever] Loaded catalog from directory {catalog_path}")
            
            return self._json_catalog or {}
            
        except Exception as e:
            logger.error(f"[SpecRetriever] Failed to load JSON catalog: {e}")
            return {}
    
    def get_specification(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """
        Get specification for a single product by canonical name.
        
        Args:
            canonical_name: Normalized product name (e.g., "Temperature Transmitter")
            
        Returns:
            Dictionary with product specifications or None if not found
        """
        if self.mode == "mongodb":
            return self._get_spec_from_mongodb(canonical_name)
        else:
            return self._get_spec_from_json(canonical_name)
    
    def _get_spec_from_mongodb(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve specification from MongoDB."""
        try:
            client = self._get_mongo_client()
            if not client:
                return None
            
            db = client[self.mongodb_database]
            collection = db[self.mongodb_collection]
            
            # Query by canonical name (case-insensitive)
            spec = collection.find_one(
                {"canonical_name": {"$regex": f"^{canonical_name}$", "$options": "i"}}
            )
            
            if spec:
                # Remove MongoDB _id field
                spec.pop('_id', None)
                logger.debug(f"[SpecRetriever] Found spec for '{canonical_name}' in MongoDB")
                return spec
            else:
                logger.debug(f"[SpecRetriever] No spec found for '{canonical_name}' in MongoDB")
                return None
                
        except Exception as e:
            logger.error(f"[SpecRetriever] MongoDB query failed for '{canonical_name}': {e}")
            return None
    
    def _get_spec_from_json(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve specification from JSON catalog."""
        catalog = self._load_json_catalog()
        
        if not catalog:
            return None
        
        # Try exact match first
        if canonical_name in catalog:
            logger.debug(f"[SpecRetriever] Found spec for '{canonical_name}' in JSON (exact)")
            return catalog[canonical_name]
        
        # Try case-insensitive match
        canonical_lower = canonical_name.lower()
        for key, value in catalog.items():
            if key.lower() == canonical_lower:
                logger.debug(f"[SpecRetriever] Found spec for '{canonical_name}' in JSON (case-insensitive)")
                return value
        
        logger.debug(f"[SpecRetriever] No spec found for '{canonical_name}' in JSON")
        return None
    
    def get_specifications_batch(
        self, 
        normalized_items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve specifications for all normalized items in batch.
        
        Args:
            normalized_items: List of items with 'canonical_name' field
            
        Returns:
            Dictionary mapping item index/ID to specification data
        """
        logger.info(f"[SpecRetriever] Batch retrieval for {len(normalized_items)} items")
        
        results = {}
        found_count = 0
        
        for idx, item in enumerate(normalized_items):
            item_key = f"item_{idx}"
            canonical_name = item.get("canonical_name", "")
            original_name = item.get("product_type") or item.get("product_name") or item.get("name") or canonical_name
            
            if not canonical_name:
                logger.warning(f"[SpecRetriever] Item {idx} has no canonical_name, skipping")
                results[item_key] = {
                    "canonical_name": "",
                    "original_name": original_name,
                    "specifications": {},
                    "spec_found": False,
                    "error": "No canonical name"
                }
                continue
            
            # Retrieve specification
            spec = self.get_specification(canonical_name)
            
            if spec:
                found_count += 1
                results[item_key] = {
                    "canonical_name": canonical_name,
                    "original_name": original_name,
                    "specifications": spec,
                    "spec_found": True,
                    "category": item.get("category", "unknown"),
                    "quantity": item.get("quantity", 1)
                }
            else:
                results[item_key] = {
                    "canonical_name": canonical_name,
                    "original_name": original_name,
                    "specifications": {},
                    "spec_found": False,
                    "category": item.get("category", "unknown"),
                    "quantity": item.get("quantity", 1)
                }
        
        logger.info(
            f"[SpecRetriever] Batch complete: {found_count}/{len(normalized_items)} "
            f"specifications found ({found_count/len(normalized_items)*100:.1f}%)"
        )
        
        return results
    
    def close(self):
        """Close MongoDB connection if open."""
        if self._mongo_client:
            self._mongo_client.close()
            logger.info("[SpecRetriever] MongoDB connection closed")
