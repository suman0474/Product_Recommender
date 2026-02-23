
import os
import logging
from common.core.azure_blob_file_manager import azure_blob_file_manager

logger = logging.getLogger(__name__)

class Collections:
    PRODUCT_IMAGES = 'product-images'
    GENERIC_IMAGES = 'generic-images'
    VENDOR_LOGOS = 'vendor-logos'
    STRATEGY_DOCUMENTS = 'strategy-documents'
    STANDARDS_DOCUMENTS = 'standards-documents'
    USER_PROJECTS = 'user-projects'
    PRODUCT_DOCUMENTS = 'product-documents'
    FILES = 'files'
    VENDORS = 'vendors'

class AzureBlobManager:
    """
    Wrapper around AzureBlobFileManager for configuration purposes.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._manager = azure_blob_file_manager
        logger.info("AzureBlobManager initialized")

    @property
    def is_available(self) -> bool:
        """Check if Azure Blob is configured and available"""
        # Check env vars
        if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            return False
        return True

    def get_client(self):
        return self._manager

def get_azure_blob_connection():
    """Compatibility function"""
    return azure_blob_manager.get_client()

azure_blob_manager = AzureBlobManager()
