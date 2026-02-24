import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PineconeStandaloneStore:
    """
    Standalone Pinecone Vector Database Document Store for Taxonomy RAG.
    It removes dependencies on EnGenie's 'common' folder (no caches, no complex batching)
    making it perfectly portable to other applications.
    """
    
    def __init__(self, api_key: str = None, index_name: str = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")
        
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError as e:
            logger.error(f"[TaxonomyStandaloneStore] Missing dependency: {e}")
            raise

        if not self.api_key:
            logger.warning("[TaxonomyStandaloneStore] PINECONE_API_KEY not set.")
            self._use_mock = True
        else:
            try:
                from pinecone import Pinecone
                self.pc = Pinecone(api_key=self.api_key)
                self.index = self.pc.Index(self.index_name)
                self._use_mock = False
                logger.info(f"[TaxonomyStandaloneStore] Connected to Pinecone index: {self.index_name}")
            except Exception as e:
                logger.error(f"[TaxonomyStandaloneStore] Failed to initialize Pinecone: {e}")
                self._use_mock = True

    def _get_namespace(self, collection_type: str) -> str:
        # Default taxonomy namespace
        return "taxonomy_documents" if collection_type == "taxonomy" else f"{collection_type}_documents"

    def get_collection_stats(self) -> Dict:
        if getattr(self, '_use_mock', False):
            return {"success": False, "error": "Mock mode"}
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            collections = {}
            for ns, ns_data in namespaces.items():
                col_name = ns.replace("_documents", "")
                collections[col_name] = {"document_count": ns_data.get("vector_count", 0)}
                
            return {
                "success": True,
                "total_documents": stats.get("total_vector_count", 0),
                "collections": collections
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_document(self, collection_type: str, content: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Dict:
        if getattr(self, '_use_mock', False):
            return {"success": False, "error": "Mock mode active"}

        try:
            doc_id = doc_id or str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "doc_id": doc_id,
                "collection_type": collection_type,
                "ingested_at": datetime.now().isoformat()
            })
            
            chunks = self.text_splitter.split_text(content)
            namespace = self._get_namespace(collection_type)

            vectors = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata["text"] = chunk
                
                # Direct embedding sync call instead of batch processor
                embedding = self.embeddings.embed_query(chunk)
                vectors.append({"id": chunk_id, "values": embedding, "metadata": chunk_metadata})
            
            # Upsert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                self.index.upsert(vectors=vectors[i:i+batch_size], namespace=namespace)
            
            return {"success": True, "doc_id": doc_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, collection_type: str, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict:
        if getattr(self, '_use_mock', False):
            return {"success": False, "error": "Mock mode active"}

        try:
            namespace = self._get_namespace(collection_type)
            query_embedding = self.embeddings.embed_query(query)

            # Build query parameters
            query_params = {
                "namespace": namespace,
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }

            # Add metadata filter if provided
            if filter_metadata:
                query_params["filter"] = filter_metadata

            results = self.index.query(**query_params)
            
            formatted_results = []
            for match in results.get("matches", []):
                formatted_results.append({
                    "id": match["id"],
                    "content": match.get("metadata", {}).get("text", ""),
                    "metadata": match.get("metadata", {}),
                    "relevance_score": match.get("score", 0)
                })
            
            return {"success": True, "results": formatted_results}
        except Exception as e:
            return {"success": False, "results": [], "error": str(e)}
