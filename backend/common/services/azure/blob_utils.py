"""
Azure Blob Storage Helper Utilities for File Operations
Provides high-level functions for storing and retrieving files using Azure Blob Storage
Drop-in replacement for mongodb_utils.py
"""

import os
import json
import io
import logging
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Union, BinaryIO
from datetime import datetime

from common.config.azure_blob_config import get_azure_blob_connection, Collections, azure_blob_manager

# Azure SDK imports
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class AzureBlobFileManager:
    """High-level file management for Azure Blob Storage (replaces MongoDBFileManager)"""

    def __init__(self):
        # Lazy initialization - do NOT connect here
        self._conn = None
        
        # Index caches for fast lookups
        self._vendor_index_cache = None
        self._specs_index_cache = None

    @property
    def conn(self):
        if self._conn is None:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            container_name = os.getenv("AZURE_CONTAINER_NAME", "engenie-validation")
            
            if not connection_string:
                logger.error("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
                
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            
            # Ensure container exists
            try:
                if not container_client.exists():
                    container_client.create_container()
                    logger.info(f"Auto-created Azure Blob container: {container_name}")
            except Exception as e:
                logger.warning(f"Could not verify/create container {container_name}: {e}")
            
            # Cache the connection as a dictionary for compatibility with property accessors
            self._conn = {
                'service_client': blob_service_client,
                'container_client': container_client,
                'base_path': f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}"
            }
        return self._conn

    @property
    def container_client(self):
        return self.conn['container_client']

    @property
    def base_path(self):
        return self.conn['base_path']

    # ==================== UPLOAD OPERATIONS ====================

    def upload_to_azure(self, file_path_or_data: Union[str, bytes, BinaryIO],
                        metadata: Dict[str, Any]) -> str:
        """
        Upload file to Azure Blob Storage with metadata

        Args:
            file_path_or_data: File path, bytes, or file-like object
            metadata: File metadata including collection_type, product_type, vendor_name, etc.

        Returns:
            str: Blob file ID (path)
        """
        try:
            # Determine file data
            if isinstance(file_path_or_data, str):
                # File path
                with open(file_path_or_data, 'rb') as f:
                    file_data = f.read()
                filename = os.path.basename(file_path_or_data)
            elif isinstance(file_path_or_data, bytes):
                # Bytes data
                file_data = file_path_or_data
                filename = metadata.get('filename', f'file_{uuid.uuid4().hex[:8]}')
            else:
                # File-like object
                file_data = file_path_or_data.read()
                filename = metadata.get('filename', f'file_{uuid.uuid4().hex[:8]}')

            # Determine collection and path
            collection_type = metadata.get('collection_type', 'files')

            # Generate unique file ID
            file_id = f"{uuid.uuid4().hex}_{filename}"
            blob_path = f"{self.base_path}/{collection_type}/{file_id}"

            # Prepare metadata for blob
            blob_metadata = {
                'filename': filename,
                'upload_date': datetime.utcnow().isoformat(),
                'file_size': str(len(file_data)),
                'collection_type': collection_type,
                'product_type': metadata.get('product_type', ''),
                'vendor_name': metadata.get('vendor_name', ''),
            }

            # Determine content type
            content_type = metadata.get('content_type', 'application/octet-stream')
            if filename.endswith('.json'):
                content_type = 'application/json'
            elif filename.endswith('.pdf'):
                content_type = 'application/pdf'
            elif filename.endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif filename.endswith('.png'):
                content_type = 'image/png'

            # Upload to Azure Blob
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(
                file_data,
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type=content_type)
            )

            logger.info(f"Successfully uploaded file to Azure Blob: {filename} (Path: {blob_path})")
            return file_id

        except Exception as e:
            logger.error(f"Failed to upload file to Azure Blob: {e}")
            raise

    def upload_json_data(self, json_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Upload JSON data to Azure Blob Storage

        Args:
            json_data: JSON data to store
            metadata: Metadata including collection_type, product_type, vendor_name, etc.

        Returns:
            str: Document ID (blob path)
        """
        try:
            collection_type = metadata.get('collection_type', 'documents')
            product_type = metadata.get('product_type', '')
            vendor_name = metadata.get('vendor_name', '')

            # Generate document ID based on product_type and vendor for uniqueness
            if product_type and vendor_name:
                doc_id = self._normalize_name(f"{product_type}_{vendor_name}")
            elif product_type:
                doc_id = self._normalize_name(product_type)
            else:
                doc_id = uuid.uuid4().hex

            filename = f"{doc_id}.json"
            blob_path = f"{self.base_path}/{collection_type}/{filename}"

            # Create document structure
            document = {
                'data': json_data,
                'metadata': {
                    'upload_date': datetime.utcnow().isoformat(),
                    **metadata
                },
                'product_type': product_type,
                'vendor_name': vendor_name,
                'file_type': 'json'
            }

            # Prepare blob metadata
            blob_metadata = {
                'product_type': product_type,
                'vendor_name': vendor_name,
                'collection_type': collection_type,
                'upload_date': datetime.utcnow().isoformat()
            }

            # Upload to Azure Blob
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(
                json.dumps(document, indent=2, default=str),
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type='application/json')
            )

            # Update index
            self._update_collection_index(collection_type, doc_id, metadata)

            logger.info(f"Successfully uploaded JSON data to Azure Blob: {blob_path}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to upload JSON data to Azure Blob: {e}")
            raise

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for use as blob path"""
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

    def _update_collection_index(self, collection_type: str, doc_id: str, metadata: Dict[str, Any]):
        """Update the collection index with new document reference"""
        try:
            index_path = f"{self.base_path}/{Collections.INDEXES}/{collection_type}_index.json"
            blob_client = self.container_client.get_blob_client(index_path)

            # Try to load existing index
            try:
                index_data = json.loads(blob_client.download_blob().readall().decode('utf-8'))
            except ResourceNotFoundError:
                index_data = {"documents": {}, "updated_at": None}

            # Add document reference
            index_data["documents"][doc_id] = {
                "product_type": metadata.get('product_type', ''),
                "vendor_name": metadata.get('vendor_name', ''),
                "updated_at": datetime.utcnow().isoformat()
            }
            index_data["updated_at"] = datetime.utcnow().isoformat()

            # Save updated index
            blob_client.upload_blob(
                json.dumps(index_data, indent=2),
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json')
            )

        except Exception as e:
            logger.warning(f"Failed to update collection index: {e}")

    # ==================== RETRIEVAL OPERATIONS ====================

    def get_file_from_mongodb(self, collection_name: str, query: Dict[str, Any]) -> Optional[bytes]:
        """
        Alias for get_file_from_azure for backward compatibility with code expecting MongoDB interface.
        Required by loading.py for PDF extraction.
        """
        return self.get_file_from_azure(collection_name, query)

    def get_file_from_azure(self, collection_name: str, query: Dict[str, Any]) -> Optional[bytes]:
        """
        Retrieve file from Azure Blob Storage

        Args:
            collection_name: Collection type (documents, vendors, static, specs)
            query: Query to find the file (product_type, vendor_name, filename, etc.)

        Returns:
            bytes: File content or None if not found
        """
        try:
            # Strategy 1: Direct blob path if provided
            if 'blob_path' in query:
                blob_path = query['blob_path']
                if not blob_path.startswith(self.base_path):
                    blob_path = f"{self.base_path}/{collection_name}/{blob_path}"
                blob_client = self.container_client.get_blob_client(blob_path)
                return blob_client.download_blob().readall()

            # Strategy 2: Search by product_type and/or vendor_name
            product_type = query.get('product_type', '')
            vendor_name = query.get('vendor_name', '')
            filename = query.get('filename', '')

            # List blobs in collection and search
            prefix = f"{self.base_path}/{collection_name}/"
            blobs = self.container_client.list_blobs(name_starts_with=prefix, include=['metadata'])

            for blob in blobs:
                blob_metadata = blob.metadata or {}

                # Match criteria
                matches = True
                if product_type:
                    blob_pt = blob_metadata.get('product_type', '').lower()
                    if product_type.lower() not in blob_pt and blob_pt not in product_type.lower():
                        matches = False
                if vendor_name and matches:
                    blob_vn = blob_metadata.get('vendor_name', '').lower()
                    if vendor_name.lower() not in blob_vn and blob_vn not in vendor_name.lower():
                        matches = False
                if filename and matches:
                    if filename.lower() not in blob.name.lower():
                        matches = False

                if matches:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    return blob_client.download_blob().readall()

            return None

        except ResourceNotFoundError:
            logger.warning(f"File not found in Azure Blob: {query}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve file from Azure Blob: {e}")
            raise

    def get_json_data_from_azure(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve JSON data from Azure Blob Storage

        Args:
            collection_name: Collection name
            query: Query to find the document

        Returns:
            Dict: JSON data or None if not found
        """
        try:
            file_data = self.get_file_from_azure(collection_name, query)
            if file_data:
                document = json.loads(file_data.decode('utf-8'))
                # Handle both new format (with 'data' field) and legacy format
                if 'data' in document:
                    return document['data']
                else:
                    return {k: v for k, v in document.items() if k not in ['_id', 'metadata']}
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve JSON data from Azure Blob: {e}")
            raise

    def get_schema_from_azure(self, product_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve schema from Azure Blob Storage (product-documents container).

        Args:
            product_type: Product type to get schema for

        Returns:
            Dict: Schema data or None if not found
        """
        try:
            from common.core.azure_blob_file_manager import azure_blob_file_manager as core_fm
            normalized_type = self._normalize_name(product_type)

            # Try direct filename match first via core file manager
            try:
                file_data = core_fm.download_file(
                    f"{normalized_type}.json",
                    container_name=Collections.PRODUCT_DOCUMENTS
                )
                if file_data:
                    document = json.loads(file_data.decode('utf-8'))
                    if 'data' in document:
                        return document['data']
                    return {k: v for k, v in document.items() if k not in ['_id', 'metadata']}
            except ResourceNotFoundError:
                pass
            except Exception:
                pass

            # Search through all blobs in product-documents container
            container_client = core_fm._get_container_client(Collections.PRODUCT_DOCUMENTS)
            if container_client is None:
                return None

            blobs = container_client.list_blobs(include=['metadata'])

            for blob in blobs:
                if blob.name.endswith('.metadata'):
                    continue

                blob_metadata = blob.metadata or {}
                blob_pt = str(blob_metadata.get('product_type', '') or '').lower()

                # Check for match
                if (normalized_type in blob_pt or
                    blob_pt in normalized_type or
                    product_type.lower() in blob_pt or
                    blob_pt in product_type.lower()):

                    blob_client = container_client.get_blob_client(blob.name)
                    file_data = blob_client.download_blob().readall()
                    document = json.loads(file_data.decode('utf-8'))

                    if 'data' in document:
                        return document['data']
                    return {k: v for k, v in document.items() if k not in ['_id', 'metadata']}

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve schema from Azure Blob: {e}")
            raise

    # ==================== QUERY OPERATIONS ====================

    def list_files(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List files in a collection with optional filters

        Args:
            collection_name: Collection name
            filters: Optional filters to apply

        Returns:
            List of file metadata
        """
        try:
            prefix = f"{self.base_path}/{collection_name}/"
            blobs = self.container_client.list_blobs(name_starts_with=prefix, include=['metadata'])

            results = []
            filters = filters or {}

            for blob in blobs:
                if blob.name.endswith('.metadata'):
                    continue

                blob_metadata = blob.metadata or {}

                # Apply filters
                matches = True
                for key, value in filters.items():
                    if key in blob_metadata:
                        if str(value).lower() not in str(blob_metadata[key]).lower():
                            matches = False
                            break

                if matches:
                    results.append({
                        'blob_name': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified.isoformat() if blob.last_modified else None,
                        **blob_metadata
                    })

            return results

        except Exception as e:
            logger.error(f"Failed to list files from Azure Blob: {e}")
            return []

    def file_exists(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Check if file exists in Azure Blob Storage

        Args:
            collection_name: Collection name
            query: Query to check

        Returns:
            bool: True if file exists
        """
        try:
            result = self.get_file_from_azure(collection_name, query)
            return result is not None
        except Exception:
            return False

    # ==================== DELETE OPERATIONS ====================

    def delete_file(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Delete file from Azure Blob Storage

        Args:
            collection_name: Collection name
            query: Query to find file to delete

        Returns:
            bool: True if deleted successfully
        """
        try:
            # Strategy 1: Direct blob path if provided
            if 'blob_path' in query:
                blob_path = query['blob_path']
                if not blob_path.startswith(self.base_path):
                    blob_path = f"{self.base_path}/{collection_name}/{blob_path}"
                blob_client = self.container_client.get_blob_client(blob_path)
                blob_client.delete_blob()
                logger.info(f"Successfully deleted file from Azure Blob: {blob_path}")
                return True

            # Find the blob first
            prefix = f"{self.base_path}/{collection_name}/"
            blobs = self.container_client.list_blobs(name_starts_with=prefix, include=['metadata'])

            for blob in blobs:
                blob_metadata = blob.metadata or {}

                # Match criteria
                matches = True
                for key, value in query.items():
                    if key in blob_metadata:
                        if str(value).lower() not in str(blob_metadata[key]).lower():
                            matches = False
                            break

                if matches:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    blob_client.delete_blob()
                    logger.info(f"Successfully deleted file from Azure Blob: {blob.name}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete file from Azure Blob: {e}")
            return False





# ==================== IMAGE CACHING FUNCTIONS ====================

def download_image_from_url(url: str, timeout: int = 30) -> Optional[tuple]:
    """
    Download image from URL and return binary data with metadata

    Args:
        url: Image URL to download
        timeout: Request timeout in seconds

    Returns:
        Tuple of (image_data: bytes, content_type: str, file_size: int) or None if failed
    """
    try:
        import requests

        # Validate URL scheme
        if not url or not isinstance(url, str):
            logger.warning(f"Invalid URL provided: {url}")
            return None

        # Check for unsupported URL schemes
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        if any(url.startswith(scheme) for scheme in unsupported_schemes):
            logger.warning(f"Unsupported URL scheme, skipping: {url}")
            return None

        # Ensure URL starts with http:// or https://
        if not url.startswith(('http://', 'https://')):
            logger.warning(f"URL must start with http:// or https://, got: {url}")
            return None

        # Add headers to appear like a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }

        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()

        # Get content type
        content_type = response.headers.get('content-type', 'image/jpeg').lower()

        # Validate it's an image
        if 'image' not in content_type:
            logger.warning(f"URL does not return an image: {url} (content-type: {content_type})")
            return None

        # Download image data
        image_data = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                image_data += chunk

        file_size = len(image_data)

        # Validate minimum size
        if file_size < 100:
            logger.warning(f"Downloaded image is too small ({file_size} bytes): {url}")
            return None

        logger.info(f"Successfully downloaded image: {file_size} bytes, type: {content_type}")
        return (image_data, content_type, file_size)

    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def get_cached_image(vendor_name: str, model_family: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached product image from Azure Blob Storage

    Args:
        vendor_name: Vendor/manufacturer name
        model_family: Model family name

    Returns:
        Dict containing image metadata or None if not found
    """
    try:
        normalized_vendor = vendor_name.strip().lower()
        normalized_model = model_family.strip().lower()
        cache_key = f"{normalized_vendor}_{normalized_model}"

        # Try to find cached image
        img_container_client = _get_core_container_client(Collections.PRODUCT_IMAGES)
        if img_container_client is None:
            return None

        # List blobs with this prefix
        blobs = list(img_container_client.list_blobs(
            name_starts_with=cache_key,
            include=['metadata']
        ))

        for blob in blobs:
            if not blob.name.endswith('.json'):
                continue

            blob_client = img_container_client.get_blob_client(blob.name)
            metadata_data = json.loads(blob_client.download_blob().readall().decode('utf-8'))

            # Extract filename for gridfs_file_id (for compatibility)
            # blob name is path/to/images/file.jpg. We want "file.jpg" or the UUID part
            from pathlib import Path
            filename = Path(metadata_data.get('image_blob_path', '')).name

            return {
                'blob_path': metadata_data.get('image_blob_path'),
                # Compatibility field for main.py
                'gridfs_file_id': filename,
                'title': metadata_data.get('title'),
                'source': metadata_data.get('source'),
                'domain': metadata_data.get('domain'),
                'content_type': metadata_data.get('content_type', 'image/jpeg'),
                'file_size': metadata_data.get('file_size', 0),
                'original_url': metadata_data.get('original_url'),
                'cached': True
            }

        return None

    except Exception as e:
        logger.error(f"Failed to retrieve cached image: {e}")
        return None


def cache_image(vendor_name: str, model_family: str, image_data: Dict[str, Any]) -> bool:
    """
    Download and store product image in Azure Blob Storage

    Args:
        vendor_name: Vendor/manufacturer name
        model_family: Model family name
        image_data: Image data dict containing url, title, source, etc.

    Returns:
        bool: True if successfully cached
    """
    try:
        # Get image URL
        image_url = image_data.get('url')
        if not image_url:
            logger.warning(f"No URL provided for image caching: {vendor_name} - {model_family}")
            return False

        # Download the actual image
        download_result = download_image_from_url(image_url)
        if not download_result:
            logger.warning(f"Failed to download image from {image_url}, skipping cache")
            return False

        image_bytes, content_type, file_size = download_result

        # Normalize keys
        normalized_vendor = vendor_name.strip().lower()
        normalized_model = model_family.strip().lower()
        cache_key = f"{normalized_vendor}_{normalized_model}"

        # Determine file extension
        file_extension = content_type.split('/')[-1] if '/' in content_type else 'jpg'

        img_container_client = _get_core_container_client(Collections.PRODUCT_IMAGES)
        if img_container_client is None:
            logger.error(f"[CACHE_IMAGE] Core container client unavailable for product-images")
            return False

        # Upload image blob
        image_blob_path = f"{cache_key}.{file_extension.split('/')[-1] if '/' in file_extension else file_extension}"
        image_blob_client = img_container_client.get_blob_client(image_blob_path)
        image_blob_client.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
            metadata={
                'vendor_name': vendor_name,
                'model_family': model_family,
                'original_url': image_url
            }
        )

        # Upload metadata JSON
        metadata_doc = {
            'vendor_name': vendor_name,
            'vendor_name_normalized': normalized_vendor,
            'model_family': model_family,
            'model_family_normalized': normalized_model,
            'image_blob_path': image_blob_path,
            'original_url': image_url,
            'title': image_data.get('title', ''),
            'source': image_data.get('source', ''),
            'domain': image_data.get('domain', ''),
            'content_type': content_type,
            'file_size': file_size,
            'created_at': datetime.utcnow().isoformat()
        }

        metadata_blob_path = f"{cache_key}.json"
        metadata_blob_client = img_container_client.get_blob_client(metadata_blob_path)
        metadata_blob_client.upload_blob(
            json.dumps(metadata_doc, indent=2),
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json')
        )

        logger.info(f"Successfully cached image in Azure Blob for {vendor_name} - {model_family}")
        return True

    except Exception as e:
        logger.error(f"Failed to cache image in Azure Blob: {e}")
        return False


def get_available_vendors() -> List[str]:
    """
    Get list of all available vendor names from Azure Blob Storage.
    """
    try:
        container_client = _get_core_container_client(Collections.PRODUCT_DOCUMENTS)
        if container_client is None:
            logger.error(f"Failed to get vendors from Azure Blob: core container client unavailable")
            return []

        blobs = container_client.list_blobs(include=['metadata'])

        vendor_names = set()
        for blob in blobs:
            if blob.name.endswith('.metadata'):
                continue
            metadata = blob.metadata or {}
            vendor_name = metadata.get('vendor_name')
            if vendor_name and vendor_name.strip():
                vendor_names.add(vendor_name.strip())

        result = sorted(list(vendor_names))
        logger.info(f"Retrieved {len(result)} vendors from Azure Blob")
        return result

    except Exception as e:
        logger.error(f"Failed to get vendors from Azure Blob: {e}")
        return []


def _get_core_container_client(container_name: str):
    """
    Get a container client from the core AzureBlobFileManager.
    Returns None if unavailable.
    """
    try:
        from common.core.azure_blob_file_manager import azure_blob_file_manager as core_fm
        return core_fm._get_container_client(container_name)
    except Exception as e:
        logger.warning(f"[BLOB_UTILS] Failed to get core container client for '{container_name}': {e}")
        return None


def get_vendors_for_product_type(product_type: str) -> List[str]:
    """
    Get list of vendor names that have products for the specified product type.
    Uses the core AzureBlobFileManager to list blobs in the product-documents container.
    """
    try:
        from common.services.products.standardization import get_analysis_search_categories
        search_categories = get_analysis_search_categories(product_type)

        logger.info(f"[VENDOR_LOADING] Searching for vendors with product categories: {search_categories}")

        container_client = _get_core_container_client(Collections.PRODUCT_DOCUMENTS)
        if container_client is None:
            logger.error(f"Failed to get vendors from Azure Blob: core container client unavailable")
            return []

        blobs = container_client.list_blobs(include=['metadata'])

        vendor_names = set()
        for blob in blobs:
            if blob.name.endswith('.metadata'):
                continue

            metadata = blob.metadata or {}
            # Normalize product_type: replace underscores with spaces for matching
            blob_product_type = metadata.get('product_type', '').lower().replace('_', ' ')
            vendor_name = metadata.get('vendor_name')

            # Check if product type matches any search category
            matches = False
            for category in search_categories:
                category_normalized = category.lower().replace('_', ' ')
                if category_normalized in blob_product_type or blob_product_type in category_normalized:
                    matches = True
                    break

            if matches and vendor_name and vendor_name.strip():
                vendor_names.add(vendor_name.strip())

        result = sorted(list(vendor_names))
        logger.info(f"Retrieved {len(result)} vendors for product type '{product_type}': {result[:10]}...")
        return result

    except Exception as e:
        logger.error(f"Failed to get vendors for product type '{product_type}': {e}")
        return get_available_vendors()


def get_pdf_content_for_vendors(vendors: List[str], product_type: str = None) -> Dict[str, str]:
    """
    Get PDF text content for a list of vendors.
    
    Args:
        vendors: List of vendor names
        product_type: Optional product type to filter PDFs
        
    Returns:
        Dict mapping vendor name to PDF text content
    """
    try:
        from common.config.azure_blob_config import Collections
        
        pdf_content = {}
        
        pdf_container_client = _get_core_container_client(Collections.PRODUCT_DOCUMENTS)

        for vendor in vendors:
            try:
                if pdf_container_client is None:
                    logger.warning(f"[PDF_CONTENT] Core container client unavailable, skipping {vendor}")
                    continue

                blobs = list(pdf_container_client.list_blobs(include=['metadata']))

                for blob in blobs:
                    if not blob.name.endswith('.pdf') and not blob.name.endswith('.txt'):
                        continue

                    metadata = blob.metadata or {}
                    blob_vendor = metadata.get('vendor_name', '').lower()

                    # Check if vendor matches
                    if vendor.lower() in blob_vendor or blob_vendor in vendor.lower():
                        # If product_type specified, also check that
                        if product_type:
                            blob_pt = metadata.get('product_type', '').lower()
                            if product_type.lower() not in blob_pt and blob_pt not in product_type.lower():
                                continue

                        # Download and decode content
                        blob_client = pdf_container_client.get_blob_client(blob.name)
                        content = blob_client.download_blob().readall()
                        
                        # If it's a PDF, we need to extract text (simplified - assume already extracted)
                        if blob.name.endswith('.txt'):
                            pdf_content[vendor] = content.decode('utf-8', errors='ignore')
                        else:
                            # For actual PDFs, try to extract text
                            try:
                                import fitz  # PyMuPDF
                                pdf_doc = fitz.open(stream=content, filetype="pdf")
                                text = ""
                                for page in pdf_doc:
                                    text += page.get_text()
                                pdf_content[vendor] = text
                            except:
                                # Fallback - store raw content info
                                pdf_content[vendor] = f"[PDF content for {vendor} - {len(content)} bytes]"
                        
                        logger.info(f"[PDF_CONTENT] Loaded content for vendor: {vendor}")
                        break
                        
            except Exception as e:
                logger.warning(f"[PDF_CONTENT] Failed to load PDF for vendor {vendor}: {e}")
                
        logger.info(f"[PDF_CONTENT] Loaded PDF content for {len(pdf_content)}/{len(vendors)} vendors")
        return pdf_content
        
    except Exception as e:
        logger.error(f"[PDF_CONTENT] Failed to get PDF content: {e}")
        return {}


def get_products_for_vendors(vendors: List[str], product_type: str = None) -> Dict[str, List[Dict]]:
    """
    Get product JSON data for a list of vendors from Azure Blob Storage.
    
    Args:
        vendors: List of vendor names
        product_type: Optional product type to filter products
        
    Returns:
        Dict mapping vendor name to list of product dictionaries
    """
    try:
        from common.config.azure_blob_config import Collections

        products_data = {}

        # First, list all JSON blobs in the product-documents container
        container_client = _get_core_container_client(Collections.PRODUCT_DOCUMENTS)
        if container_client is None:
            logger.error("[PRODUCTS] Core container client unavailable for product-documents")
            return {}
        blobs = list(container_client.list_blobs(include=['metadata']))

        logger.info(f"[PRODUCTS] Searching {len(blobs)} blobs for vendors: {vendors}")

        for vendor in vendors:
            vendor_lower = vendor.lower().strip()

            for blob in blobs:
                if not blob.name.endswith('.json'):
                    continue

                metadata = blob.metadata or {}
                blob_vendor = metadata.get('vendor_name', '').lower().strip()

                # Matching strategies (exact and substring only)
                matched = False

                # Strategy 1: Exact match (case-insensitive)
                if vendor_lower == blob_vendor:
                    matched = True
                    logger.info(f"[PRODUCTS] Exact matched '{vendor}' to '{metadata.get('vendor_name')}'")
                # Strategy 2: Substring match
                elif vendor_lower in blob_vendor or blob_vendor in vendor_lower:
                    matched = True
                    logger.info(f"[PRODUCTS] Substring matched '{vendor}' to '{metadata.get('vendor_name')}'")

                if not matched:
                    continue
                    
                # If product_type specified, check using category expansion (same logic as vendor discovery)
                if product_type:
                    from common.services.products.standardization import get_analysis_search_categories
                    search_categories = get_analysis_search_categories(product_type)
                    
                    # Normalize blob product type
                    blob_pt = metadata.get('product_type', '').lower().replace('_', ' ')
                    
                    # Check if blob's product_type matches ANY of the search categories
                    category_matched = False
                    for category in search_categories:
                        category_normalized = category.lower().replace('_', ' ')
                        if category_normalized in blob_pt or blob_pt in category_normalized:
                            category_matched = True
                            break
                    
                    if not category_matched:
                        continue
                
                # Download and parse JSON
                try:
                    blob_client = container_client.get_blob_client(blob.name)
                    content = blob_client.download_blob().readall()
                    data = json.loads(content.decode('utf-8'))
                    
                    logger.info(f"[PRODUCTS] Loaded JSON for vendor {vendor} from {blob.name}")
                    
                    # Handle different JSON structures
                    # The JSON may have been wrapped during MongoDB→Azure migration
                    # Common format: {"data": {"vendor": "...", "product_type": "...", "models": [...]}, "metadata": {...}}
                    if isinstance(data, list):
                        products_data[vendor] = data
                    elif isinstance(data, dict):
                        # Check for common structures
                        if 'products' in data:
                            products_data[vendor] = data['products']
                        elif 'models' in data:
                            # This is the vendor JSON format with models array
                            products_data[vendor] = [data]  # Wrap as list
                        elif 'data' in data:
                            # Handle the migration wrapper format: {"data": {...}, "metadata": {...}}
                            inner_data = data['data']
                            if isinstance(inner_data, list):
                                products_data[vendor] = inner_data
                            elif isinstance(inner_data, dict):
                                # Check if inner data has models - this is the actual product catalog
                                if 'models' in inner_data:
                                    # The inner data IS the product catalog, wrap it as a list
                                    products_data[vendor] = [inner_data]
                                    logger.info(f"[PRODUCTS] Extracted catalog with {len(inner_data.get('models', []))} models for {vendor}")
                                elif 'products' in inner_data:
                                    products_data[vendor] = inner_data['products']
                                else:
                                    # Treat inner data as a single product
                                    products_data[vendor] = [inner_data]
                            else:
                                products_data[vendor] = [inner_data]
                        else:
                            # Treat the whole object as a product
                            products_data[vendor] = [data]
                    
                    product_count = len(products_data.get(vendor, []))
                    
                    # Enhanced logging - show structure details
                    product_list = products_data.get(vendor, [])
                    if product_list and isinstance(product_list[0], dict) and 'models' in product_list[0]:
                        models_count = len(product_list[0].get('models', []))
                        submodels_count = sum(len(m.get('sub_models', [])) for m in product_list[0].get('models', []))
                        logger.info(f"[PRODUCTS] Vendor '{vendor}': {product_count} catalog(s), {models_count} models, {submodels_count} submodels")
                    else:
                        logger.info(f"[PRODUCTS] Vendor '{vendor}': {product_count} product entries loaded")
                    
                    break  # Found data for this vendor
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"[PRODUCTS] Invalid JSON for vendor {vendor}: {e}")
                except Exception as e:
                    logger.warning(f"[PRODUCTS] Error loading blob {blob.name}: {e}")
        
        logger.info(f"[PRODUCTS] Loaded product data for {len(products_data)}/{len(vendors)} vendors")
        return products_data
        
    except Exception as e:
        logger.error(f"[PRODUCTS] Failed to get products: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


# Global instance
azure_blob_file_manager = AzureBlobFileManager()
