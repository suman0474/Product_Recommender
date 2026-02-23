
# models.py
# Contains Pydantic models for structured output
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union

class ProductMatch(BaseModel):
    product_name: str = Field(default="", description="Specific submodel name (e.g., 'STD850', '3051CD') - the exact model being matched")
    model_family: str = Field(default="", description="Model family/series name (e.g., 'STD800', '3051C') - the broader product family this submodel belongs to")
    product_type: str = Field(default="", description="type of the matching product")

    vendor: str = Field(default="", description="Vendor/manufacturer name")
    match_score: int = Field(default=0, description="Match score from 0-100")
    requirements_match: bool = Field(default=False, description="Whether the product meets ALL critical/mandatory requirements (True) or has fundamental gaps (False)")
    
    # Standards compliance tracking
    standards_compliance: Dict[str, Any] = Field(default_factory=dict, description="Standards compliance information including compliant_standards, non_compliant_standards, and compliance_notes")
    
    # Structured requirements matching
    matched_requirements: Dict[str, Any] = Field(default_factory=dict, description="Detailed parameter-by-parameter analysis with user_requirement, product_specification, source_reference, status, and match_explanation for each parameter")
    unmatched_requirements: List[str] = Field(default_factory=list, description="List of requirements that were not matched")
    
    # Analysis fields
    reasoning: str = Field(default="", description="Overall reasoning for the match including standards compliance summary")
    limitations: str = Field(default="", description="Any limitations or concerns")
    key_strengths: List[str] = Field(default_factory=list, description="List of key strengths")
    product_description: str = Field(default="", description="Rich formatted description with parameter-by-parameter breakdown including user requirements, product specifications with datasheet references, and standards compliance")
    
    # Pricing information
    pricing_url: str = Field(default="", description="URL to a pricing page or 'where to buy' page for this product")
    pricing_source: str = Field(default="", description="Source of the pricing information (e.g., 'google_search', 'vendor_site')")
    

class VendorAnalysis(BaseModel):
    vendor_matches: List[ProductMatch] = Field(description="Best products match from each vendor")

class RankedProduct(BaseModel):
    rank: Union[int, str] = Field(default=0, description="Overall ranking position")
    product_name: str = Field(default="", description="Only the specific submodel/series name (e.g., 'STD800', 'SMV800') - just the short model identifier, not full descriptions")
    vendor: str = Field(default="", description="Vendor name")
    model_family: str = Field(default="", description="Model family/series name (e.g., '3051C', 'STD800') for more specific image searches")
    
    overall_score: int = Field(default=0, description="Overall score 0-100")
    requirements_match: bool = Field(default=False, description="Whether the product meets ALL critical/mandatory requirements (True) or has fundamental gaps (False)")
    key_strengths: Union[str, List[Any]] = Field(default="", description="Detailed parameter-by-parameter analysis of key strengths showing user requirement vs product specification and basis for match")
    concerns: Union[str, List[Any]] = Field(default="", description="Detailed parameter-by-parameter analysis of potential concerns, limitations, or verification needs")
    product_description: str = Field(default="", description="Rich formatted description with parameter-by-parameter breakdown including user requirements, product specifications with datasheet references, and standards compliance")
    pricing_url: str = Field(default="", description="URL to a pricing page or 'where to buy' page for this product")
    pricing_source: str = Field(default="", description="Source of the pricing information (e.g., 'google_search', 'vendor_site')")


class OverallRanking(BaseModel):
    ranked_products: List[RankedProduct] = Field(description="All products ranked by suitability")

class RequirementValidation(BaseModel):
    
    provided_requirements: Dict[str, Any] = Field(description="Requirements that were provided")
    product_type: str = Field(description="Detected product type from user input")