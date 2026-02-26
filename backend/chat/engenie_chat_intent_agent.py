"""
EnGenie Chat Intent Classification Agent

Classifies user queries into intent categories for metadata and logging.
Classification result is used for:
- Structured logging/analytics
- Frontend routing decisions (is_engenie_chat_intent)
- Keyword fallback scoring when LLM classifier is unavailable

The orchestrator no longer routes to RAG sources — all queries go through
web search → LLM (grounded). Classification is informational only.
"""

import logging
import re
import json
from typing import Dict, Tuple, List, Optional
from enum import Enum

# Import for LLM-based semantic classification
try:
    from common.services.llm.fallback import create_llm_with_fallback
    from langchain_core.prompts import ChatPromptTemplate
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources for EnGenie Chat."""
    INDEX_RAG = "index_rag"         # Product database + web search
    STANDARDS_RAG = "standards_rag" # IEC, ISO, API standards
    STRATEGY_RAG = "strategy_rag"   # Vendor procurement strategies
    DEEP_AGENT = "deep_agent"       # Detailed spec extraction
    SOLUTION = "solution"           # System/Solution design
    WEB_SEARCH = "web_search"       # Web search with verification (parallel source)
    HYBRID = "hybrid"               # Multiple sources
    LLM = "llm"                     # General LLM fallback


# =============================================================================
# ENHANCED KEYWORD LISTS (Based on Debugs.md Test Cases)
# =============================================================================

# 1. Index RAG - Product Specifications
INDEX_RAG_KEYWORDS = [
    # Product types (instruments) - Core list
    "transmitter", "sensor", "valve", "actuator", "controller",
    "flowmeter", "flow meter", "coriolis", "magnetic flowmeter",
    "level transmitter", "level sensor", "level gauge",
    "pressure transmitter", "pressure sensor", "pressure gauge",
    "temperature sensor", "temperature transmitter", "thermocouple", "rtd",
    "analyzer", "ph analyzer", "conductivity", "oxygen analyzer",
    "positioner", "valve positioner", "indicator", "recorder",
    "gauge", "switch", "limit switch", "proximity switch",
    "control valve", "safety valve", "relief valve", "shutoff valve",
    "actuator", "pneumatic actuator", "electric actuator",

    # Product types (from Azure Blob extraction - 71 types)
    "coriolis flow meter", "flow switch", "flow transmitter", "pressure switch",
    "thermowell", "temperature transmitter", "digital temperature indicator",
    "isolation valve", "instrument isolation ball valve", "globe flow control valve",
    "safety relief valve", "pressure relief valve", "pressure safety valve",
    "junction box", "hazardous area junction box", "field instrument junction box",
    "vfd junction box", "variable frequency drive", "motor vfd",
    "cable gland", "cables and connectors", "instrument cable", "electrical cable",
    "mounting bracket", "mounting hardware", "mounting support",
    "thermocouple terminal head", "multipoint thermocouple", "k-type thermocouple",
    "resistance temperature detector", "surface thermocouple",
    "calibration kit", "data logger", "pid controller", "power supply",
    "pulsation dampener", "rotary lobe pump", "rotary pd pump",
    "instrumentation tubing", "stainless steel tubing", "impulse line",
    "flange gasket", "flange hardware", "process connection fittings",
    "manifold", "2-valve manifold", "3-valve manifold",

    # Brand names (major vendors) - Core list
    "rosemount", "yokogawa", "emerson", "honeywell", "siemens",
    "endress", "endress+hauser", "e+h", "abb", "foxboro",
    "fisher", "masoneilan", "metso", "neles", "samson",
    "krohne", "vega", "ifm", "pepperl", "turck",
    "danfoss", "burkert", "asco", "parker", "swagelok",

    # Brand names (from Azure Blob extraction - 224 vendors)
    "micro motion", "magnetrol", "mettler toledo", "wika", "fluke",
    "omega", "phoenix contact", "pepperl+fuchs", "rockwell", "schneider",
    "schneider electric", "ametek", "ashcroft", "baumer", "brooks instrument",
    "flowserve", "hach", "hima", "honeywell", "keyence",
    "msa", "national instruments", "panametrics", "pilz", "puls",
    "rittal", "rotork", "servomex", "sick", "skf",
    "smc", "thermo fisher", "vaisala", "watlow", "xylem",
    "anderson greenwood", "alpha laval", "anton paar", "beamex", "belden",
    "eaton", "gems sensors", "hubbell", "moore industries", "moxa",
    "panasonic", "pentair", "pyromation", "r stahl", "wago", "weidmuller",

    # Product identifiers and series
    "3051", "3051s", "ejx", "ejx series", "ejx110", "ejx310", "ejx510",
    "644", "3144", "5400", "5300", "2088", "3100",
    "dvc6200", "fieldvue", "fisher ez", "ez valve",

    # General product queries
    "model", "product", "specification", "specifications", "specs",
    "datasheet", "data sheet", "catalog", "catalogue",
    "features", "capabilities", "performance", "accuracy", "range",
    "accuracy range", "measurement range", "flow characteristics",

    # Query patterns
    "what is", "what are", "tell me about", "describe", "find",
    "show me", "get me", "information about", "details of",
    "how does", "how do"
]

# 2. Standards RAG - Compliance & Safety
STANDARDS_RAG_KEYWORDS = [
    # Standards organizations
    "iec", "iso", "api", "asme", "astm", "nfpa", "ansi", "din", "en",

    # Specific standard numbers
    "61508", "61511", "62443", "61131", "60079",
    "526", "api 526", "api 520", "api 521",
    "isa", "isa-84", "isa-88", "isa-95",

    # Safety Integrity Levels
    "sil", "sil 1", "sil 2", "sil 3", "sil 4",
    "sil-1", "sil-2", "sil-3", "sil-4",
    "sil1", "sil2", "sil3", "sil4",
    "functional safety", "safety integrity", "safety integrity level",
    "safety instrumented", "sis", "sif", "safety function",

    # Hazardous Area Classifications
    "atex", "iecex", "explosion", "explosion-proof", "explosionproof",
    "hazardous area", "hazardous areas", "hazardous zone",
    "zone 0", "zone 1", "zone 2", "zone 20", "zone 21", "zone 22",
    "class i", "class ii", "class iii",
    "division 1", "division 2", "div 1", "div 2",
    "intrinsic safety", "intrinsically safe", "flameproof",
    "increased safety", "encapsulation", "pressurization",
    "ex d", "ex e", "ex i", "ex ia", "ex ib", "ex ic",

    # Compliance & Certification
    "certification", "certified", "compliance", "compliant",
    "standard", "standards", "regulation", "regulations",
    "requirement", "requirements", "conformance", "approval",
    "installation requirement", "installation requirements",

    # Safety-related queries
    "difference between", "compare", "explain",
    "what is the requirement", "according to"
]

# 3. Strategy RAG - Vendor & Procurement
STRATEGY_RAG_KEYWORDS = [
    # Vendor-related
    "vendor", "vendors", "supplier", "suppliers",
    "manufacturer", "manufacturers", "make", "makes",
    "who manufactures", "who makes", "who supplies",

    # Preference & Approval
    "preferred", "preferred vendor", "preferred supplier",
    "approved", "approved vendor", "approved supplier", "approved suppliers",
    "strategic", "strategic partner", "partner",
    "recommended", "recommended vendor",
    "best vendor", "top vendor",

    # Procurement Strategy
    "procurement", "procurement strategy", "sourcing", "sourcing strategy",
    "selection", "vendor selection", "supplier selection",
    "priority", "priorities", "prioritize",
    "strategy", "strategies",

    # Commercial Terms
    "cost", "price", "pricing", "budget",
    "lead time", "delivery", "delivery time",
    "support", "service", "warranty",
    "relationship", "long-term", "contract",
    "compare suppliers", "compare vendors",

    # Specific procurement queries
    "who is our", "who are our", "which vendor", "which supplier",
    "do they also", "do they make"
]

# 4. Deep Agent - Detailed Extraction
DEEP_AGENT_KEYWORDS = [
    # Extraction commands
    "extract", "extraction", "pull out", "get from",
    "from the standard", "from the table", "from the document",
    "according to standard", "according to the standard",
    "as per standard", "as per the standard",

    # Specific document references
    "table", "tables", "figure", "figures",
    "section", "sections", "clause", "clauses",
    "annex", "annexes", "appendix", "appendices",
    "paragraph", "page",

    # Detailed requirements
    "specific requirement", "specific requirements",
    "detailed specs", "detailed specifications",
    "technical specification", "technical specifications",
    "parameter from", "value from", "limit from",
    "what does the standard say", "what is the specific",

    # Data extraction patterns
    "pressure limits", "temperature limits", "response time requirement",
    "material requirements", "dimensional requirements",
    "tolerance", "tolerances", "allowable", "permissible"
]

# 5. Web Search Keywords
WEB_SEARCH_KEYWORDS = [
    # Recency indicators
    "latest", "recent", "current", "newest", "updated",
    "new", "news", "update", "updates", "announcement",

    # Market/Industry
    "market", "market trend", "industry trend", "industry news",
    "competitor", "competition", "alternative", "alternatives",
    "comparison", "versus", "vs",

    # External information
    "external", "outside", "beyond", "internet",
    "online", "website", "web", "search",

    # General queries not in database
    "general information", "overview", "introduction",
    "what is happening", "what's new", "developments",

    # Price/availability (often needs current web data)
    "price", "pricing", "cost", "availability",
    "where to buy", "purchase", "order",

    # Reviews and opinions
    "review", "reviews", "opinion", "opinions",
    "feedback", "experience", "experiences"
]

# 6. Solution Design Keywords
SOLUTION_KEYWORDS = [
    # Design intent
    "design", "designing", "designed",
    "solution", "solutions",
    "system", "systems",
    "package", "packages", "packaged",
    "skid", "skids",
   "unit", "units",
    
    # Context indicators
    "metering skid", "control system", "safety system",
    "instrumentation system", "scada system",
    "distributed control", "dcs", "plc system",
    "custody transfer", "fiscal metering",
    "boiler control", "burner management",
    "compressor control", "pump control",
    
    # Design-specific phrases
    "i need to design", "we're designing", "design a",
    "solution for", "complete solution",
    "integrated solution", "turnkey solution",
    "system integration", "full system",
    "build a system", "create a system",
    "engineering solution", "custom solution"
]

# =============================================================================
# CLASSIFICATION PATTERNS
# =============================================================================

# Patterns that strongly indicate Index RAG (product queries)
INDEX_RAG_PATTERNS = [
    r"what (is|are) the (specifications?|specs|features|capabilities) (for|of)",
    r"tell me about (the )?[a-z0-9\-\s]+ (transmitter|sensor|valve|meter|analyzer)",
    r"find (the |a )?(datasheet|catalog|specs?) for",
    r"(rosemount|yokogawa|emerson|honeywell|siemens|fisher|abb)\s+[a-z0-9\-]+",
    r"[a-z]+ (series|model|type)\s+[a-z0-9\-]+",
    r"accuracy (range|of|for)",
    r"flow characteristics",
    r"(ejx|3051|644|dvc)\s*[a-z0-9]*"
]

# Patterns that strongly indicate Standards RAG (compliance queries)
STANDARDS_RAG_PATTERNS = [
    r"(installation|safety|certification) requirements? (for|according|per)",
    r"(what|explain).*(difference|comparison) between sil",
    r"(iec|iso|api|atex|iecex)\s*[0-9]+",
    r"sil[- ]?[1-4]",
    r"zone [0-2]",
    r"(hazardous|explosive) area",
    r"(according to|per|as per) (iec|iso|api|atex)",
    r"(certified|compliant|approval) for",
    r"standard requirements?"
]

# Patterns that strongly indicate Strategy RAG (vendor queries)
STRATEGY_RAG_PATTERNS = [
    r"(who is|who are) (our )?(preferred|approved|strategic)",
    r"(preferred|approved|recommended) (vendor|supplier|manufacturer)",
    r"(preferred|approved) vendor for",  # "preferred vendor for control valves"
    r"(our )?(preferred|approved) (vendor|supplier) for [a-z\s]+",
    r"procurement strategy",
    r"(compare|comparison of) (the )?(approved )?(suppliers|vendors)",
    r"long lead time",
    r"who (manufactures|makes|supplies)",
    r"do they (also )?(make|manufacture|supply)",
    r"which (vendor|supplier) (should we|do we|to) use"
]

# Patterns that strongly indicate Deep Agent (extraction queries)
DEEP_AGENT_PATTERNS = [
    r"extract (the )?[a-z\s]+ from (the )?(standard|table|document)",
    r"(pressure|temperature|flow) limits? (for|from|in)",
    r"(specific|detailed) (requirement|specification) (for|in)",
    r"(clause|section|annex|table) [0-9]+",
    r"response time requirement",
    r"what does the standard say about"
]

# Patterns that strongly indicate Solution Design (system/package design)
SOLUTION_PATTERNS = [
    r"design (a|the|an) .*(system|skid|package|unit|solution)",
    r"(solution|package) for .*(process|application|plant|facility)",
    r"(metering|control|safety|instrumentation) (skid|system|package)",
    r"(custody transfer|fiscal metering|burner management)",
    r"(i need|we need|looking for|require) (a |an )?(complete |integrated )?(solution|system|package)",
    r"build (a|an) .*(control|safety|metering) system",
    r"(design|create|build|engineer) .*(scada|dcs|plc) system",
    r"turnkey (solution|package|system)",
    r"system integration for",
    r"full system (for|to)"
]

# Patterns that strongly indicate Web Search (external/current info)
WEB_SEARCH_PATTERNS = [
    r"(latest|recent|current|newest) (news|update|development|trend)",
    r"(market|industry) (trend|news|update)",
    r"(competitor|alternative|comparison).*(product|vendor|supplier)",
    r"(price|pricing|cost|availability) (for|of)",
    r"where (to|can i) (buy|purchase|order)",
    r"(review|feedback|experience) (of|for|with)",
    r"what('s| is) (new|happening|trending)",
    r"(search|find).*(online|web|internet)"
]

# Patterns indicating hybrid queries (multiple classification domains needed)
HYBRID_PATTERNS = [
    # Product + Standards (e.g., "Is Rosemount 3051S certified for SIL 3?")
    # Asking about a specific product's certification status
    r"(rosemount|yokogawa|emerson|fisher|abb)\s+[a-z0-9\-]+.*(sil|atex|zone|certified|hazardous)",
    r"(sil|atex|zone|certified|hazardous).*(rosemount|yokogawa|emerson|fisher|abb)",
    r"is (the |a )?(rosemount|yokogawa|emerson|fisher|abb).*certified",

    # Vendor + Standards (e.g., "Which preferred vendor supplies for Zone 0?")
    # Asking about vendors that meet specific safety/certification requirements
    r"(preferred|approved) (vendor|supplier).*(suitable for|certified for|rated for).*(sil|atex|zone|hazardous)",
    r"(vendor|supplier).*(suitable for|certified for).*(zone|sil|atex|hazardous)",
    r"(sil|atex|zone|hazardous).*(preferred|approved) (vendor|supplier)",
    r"which.*(vendor|supplier).*supplies.*(zone|sil|atex|hazardous)",

    # Specific combined queries
    r"(coriolis|magnetic|vortex) meter.*(zone|hazardous|sil|atex)",
    r"(zone|hazardous|sil|atex).*(coriolis|magnetic|vortex) meter"
]


# =============================================================================
# SEMANTIC LLM CLASSIFICATION (For Uncertain Queries)
# =============================================================================

SEMANTIC_CLASSIFICATION_PROMPT = """You are an industrial instrumentation query classifier.

Classify this query into ONE category:

Query: {query}

Categories:
- INDEX_RAG: Searching for SPECIFIC PRODUCTS - datasheets, specs, individual instruments (e.g., "Rosemount 3051 specs", "pressure transmitter datasheet")
- STANDARDS_RAG: Safety standards (SIL, ATEX, IECEx), certifications, compliance requirements (IEC, ISO, API standards), hazardous area classifications
- STRATEGY_RAG: Vendor procurement decisions - approved/preferred supplier lists, vendor selection criteria, commercial terms, procurement rules
- SOLUTION: SYSTEM/PACKAGE DESIGN - designing complete solutions, skids, integrated systems requiring multiple instruments (e.g., "design a metering skid", "control system for boiler")
- DEEP_AGENT: Extract specific values from standards documents - tables, clauses, annexes (detailed requirement extraction from PDFs)
- HYBRID: Query needs BOTH product info AND standards/strategy (e.g., "Is Rosemount 3051 certified for SIL 3?")
- LLM: General non-instrumentation question, conversational/greeting

IMPORTANT DISTINCTIONS:
- INDEX_RAG = Finding a single product ("get me a transmitter")
- SOLUTION = Designing a complete system ("design a custody transfer skid")
- STRATEGY_RAG = Choosing vendors ("who is our preferred vendor")

Respond ONLY with valid JSON (no markdown):
{{"category": "CATEGORY_NAME", "confidence": 0.7, "reasoning": "brief reason"}}
"""


def _classify_with_llm(query: str) -> Tuple[DataSource, float, str]:
    """
    Use LLM for semantic classification when keyword matching is uncertain.
    
    This provides more accurate classification for ambiguous queries like:
    - "How do I maintain this transmitter?" (intent unclear from keywords)
    - "What should I consider for hazardous areas?" (could be multiple sources)
    
    Returns:
        Tuple of (DataSource, confidence, reasoning)
    """
    if not LLM_AVAILABLE:
        logger.debug("[INTENT] LLM semantic classification unavailable")
        return DataSource.LLM, 0.3, "LLM classification unavailable"
    
    try:
        llm = create_llm_with_fallback(model="gemini-2.5-flash", temperature=0.1)
        prompt = ChatPromptTemplate.from_template(SEMANTIC_CLASSIFICATION_PROMPT)
        
        response = (prompt | llm).invoke({"query": query})
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up response (remove markdown if present)
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        category_map = {
            "INDEX_RAG": DataSource.INDEX_RAG,
            "STANDARDS_RAG": DataSource.STANDARDS_RAG,
            "STRATEGY_RAG": DataSource.STRATEGY_RAG,
            "DEEP_AGENT": DataSource.DEEP_AGENT,
            "SOLUTION": DataSource.SOLUTION,
            "WEB_SEARCH": DataSource.WEB_SEARCH,
            "HYBRID": DataSource.HYBRID,
            "LLM": DataSource.LLM,
        }
        
        category = result.get("category", "LLM").upper()
        source = category_map.get(category, DataSource.LLM)
        confidence = float(result.get("confidence", 0.7))
        reasoning = result.get("reasoning", "LLM semantic classification")
        
        logger.info(f"[INTENT] LLM semantic classification: {source.value} (confidence: {confidence:.2f})")
        return source, confidence, f"[LLM] {reasoning}"
        
    except json.JSONDecodeError as e:
        logger.warning(f"[INTENT] LLM response not valid JSON: {e}")
        return DataSource.LLM, 0.3, "LLM classification parse error"
    except Exception as e:
        logger.warning(f"[INTENT] LLM classification failed: {e}")
        return DataSource.LLM, 0.3, f"LLM classification failed: {str(e)[:50]}"


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_query(query: str, use_semantic_llm: bool = True, conversation_history: list = None) -> Tuple[DataSource, float, str]:
    """
    Classify a query to determine the best data source.

    Uses an LLM-first classification approach:
    1. Fast-path for unambiguous identifiers (brand+model, standard codes)
    2. Hybrid detection for multi-domain queries
    3. LLM semantic classification (PRIMARY classifier)
    4. Keyword scoring fallback (only if LLM unavailable)

    Args:
        query: User query string
        use_semantic_llm: Whether to use LLM for classification (default: True)
        conversation_history: Optional conversation history for context-aware routing

    Returns:
        Tuple of (DataSource, confidence, reasoning)
    """
    query_lower = query.lower().strip()

    # Stage 1: Fast-path for unambiguous identifiers only
    # These are patterns so specific they don't need LLM (brand+model, standard codes)
    fast_path_result = _fast_path_classify(query_lower)
    if fast_path_result:
        source, confidence, reasoning = fast_path_result
        logger.info(f"[INTENT] Fast-path match: {source.value} (confidence: {confidence:.2f})")
        return source, confidence, reasoning

    # Stage 2: Check for hybrid queries (product + standards combination)
    hybrid_detected = _detect_hybrid_pattern(query_lower)
    if hybrid_detected:
        logger.info(f"[INTENT] Hybrid pattern detected")
        return DataSource.HYBRID, 0.9, "Hybrid query detected (multiple data sources needed)"

    # Stage 3: LLM Semantic Classification (PRIMARY classifier)
    if use_semantic_llm:
        logger.info(f"[INTENT] Using LLM semantic classification (primary)")
        llm_result = _classify_with_llm(query)
        source, confidence, reasoning = llm_result
        # Trust LLM result if confidence is reasonable
        if confidence >= 0.4:
            return source, confidence, reasoning
        logger.info(f"[INTENT] LLM confidence too low ({confidence:.2f}), falling back to keywords")

    # Stage 4: Keyword-based scoring (fallback when LLM unavailable or low confidence)
    scores, matches = _calculate_keyword_scores(query_lower)

    primary_source = DataSource.LLM
    max_score = 0.0
    for source, score in scores.items():
        if score > max_score:
            max_score = score
            primary_source = source

    # Calculate confidence from keyword scores
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = min(max_score / total_score + 0.3, 1.0)
    else:
        confidence = 0.3

    # Build reasoning
    if primary_source != DataSource.LLM and matches.get(primary_source):
        matched_keywords = matches[primary_source][:5]
        reasoning = f"Keyword fallback - matched: {', '.join(matched_keywords)}"
    else:
        reasoning = "No specific keywords matched, defaulting to LLM"

    logger.info(f"[INTENT] Query classified as {primary_source.value} (confidence: {confidence:.2f}) [keyword fallback]")
    return primary_source, confidence, reasoning


def _fast_path_classify(query_lower: str) -> Optional[Tuple[DataSource, float, str]]:
    """
    Fast-path classification for unambiguous queries only.
    
    Only matches patterns that are so specific they don't need LLM:
    - Brand name + model number (e.g., "Rosemount 3051", "EJX110")
    - Standard code references (e.g., "IEC 61508", "API 526")
    - Explicit hybrid patterns (product + certification)
    - Explicit solution/design patterns
    """
    # Check hybrid patterns first (highest specificity)
    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.HYBRID, 0.95, f"Fast-path: hybrid pattern matched"

    # Brand + model number (very specific product queries)
    FAST_PATH_PRODUCT = [
        r"(rosemount|yokogawa|emerson|honeywell|siemens|fisher|abb|endress)\s+[a-z0-9]{2,}[a-z0-9\-]*",
        r"(ejx|3051|644|dvc|5400|5300|2088|3144)\s*[a-z0-9]*",
    ]
    for pattern in FAST_PATH_PRODUCT:
        if re.search(pattern, query_lower):
            return DataSource.INDEX_RAG, 0.95, f"Fast-path: specific product identifier"

    # Explicit document/standard extraction requests — checked BEFORE STANDARDS_RAG
    # so "extract clauses for SIL-2" routes to DEEP_AGENT, not STANDARDS_RAG.
    FAST_PATH_DEEP_AGENT = [
        r"extract (the )?[a-z\s]+ from (the )?(standard|table|document)",
        r"(clause|section|annex|table) [0-9]+",
        # Extraction intent without "from the standard" suffix
        r"extract (all |the )?(specification|requirement|spec)s? (clause|section|paragraph|table)s?",
        r"(specification|requirement) clauses? (for|from|in|related to)",
        r"clauses? (for|related to|from) (sil|iec|iso|atex|api)",
        r"(all |the )?(specification|requirement)s? (clauses?|sections?|paragraphs?)",
    ]
    for pattern in FAST_PATH_DEEP_AGENT:
        if re.search(pattern, query_lower):
            return DataSource.DEEP_AGENT, 0.95, f"Fast-path: document extraction request"

    # Standard code references (e.g., "IEC 61508", "API 526", "SIL 3")
    FAST_PATH_STANDARDS = [
        r"(iec|iso|api|atex|iecex)\s*[0-9]{3,5}",
        r"sil[- ]?[1-4]",
        r"(according to|per|as per) (iec|iso|api|atex|isa)",
    ]
    for pattern in FAST_PATH_STANDARDS:
        if re.search(pattern, query_lower):
            return DataSource.STANDARDS_RAG, 0.95, f"Fast-path: standard code reference"

    # Explicit solution/design intent
    FAST_PATH_SOLUTION = [
        r"design (a|the|an) .*(system|skid|package|unit|solution)",
        r"(i need|we need|looking for) (a |an )?(complete |integrated )?(solution|system|package)",
        r"(custody transfer|fiscal metering|burner management) (skid|system|package)",
        r"build (a|an) .*(control|safety|metering) system",
        r"(create|develop|implement|engineer) (a |an |the )?(full |complete |new )?.*(dcs|plc|scada|control system|safety system)",
    ]
    for pattern in FAST_PATH_SOLUTION:
        if re.search(pattern, query_lower):
            return DataSource.SOLUTION, 0.95, f"Fast-path: solution/design request"

    # Explicit vendor/procurement queries
    FAST_PATH_STRATEGY = [
        r"(who is|who are) (our )?(preferred|approved|strategic) (vendor|supplier)",
        r"(preferred|approved) (vendor|supplier) for",
        r"procurement strategy",
    ]
    for pattern in FAST_PATH_STRATEGY:
        if re.search(pattern, query_lower):
            return DataSource.STRATEGY_RAG, 0.95, f"Fast-path: vendor/procurement query"

    return None


def _calculate_keyword_scores(query_lower: str) -> Tuple[Dict[DataSource, float], Dict[DataSource, List[str]]]:
    """
    Calculate weighted keyword scores for each data source.
    """
    scores = {
        DataSource.INDEX_RAG: 0.0,
        DataSource.STANDARDS_RAG: 0.0,
        DataSource.STRATEGY_RAG: 0.0,
        DataSource.DEEP_AGENT: 0.0,
        DataSource.SOLUTION: 0.0,
        DataSource.WEB_SEARCH: 0.0
    }

    matches = {
        DataSource.INDEX_RAG: [],
        DataSource.STANDARDS_RAG: [],
        DataSource.STRATEGY_RAG: [],
        DataSource.DEEP_AGENT: [],
        DataSource.SOLUTION: [],
        DataSource.WEB_SEARCH: []
    }

    # Score Index RAG keywords
    for kw in INDEX_RAG_KEYWORDS:
        if kw in query_lower:
            # Higher weight for specific product models/brands
            if re.match(r'^(rosemount|yokogawa|emerson|fisher|abb|3051|ejx)', kw):
                scores[DataSource.INDEX_RAG] += 2.0
            else:
                scores[DataSource.INDEX_RAG] += 1.0
            matches[DataSource.INDEX_RAG].append(kw)

    # Score Standards RAG keywords (higher weight for explicit standard codes)
    for kw in STANDARDS_RAG_KEYWORDS:
        if kw in query_lower:
            # Highest weight for standard codes and SIL levels
            if re.match(r'^(iec|iso|api|sil|atex|61508|61511|526)', kw):
                scores[DataSource.STANDARDS_RAG] += 3.0
            elif re.match(r'^(zone|hazardous|certification|compliance)', kw):
                scores[DataSource.STANDARDS_RAG] += 2.0
            else:
                scores[DataSource.STANDARDS_RAG] += 1.0
            matches[DataSource.STANDARDS_RAG].append(kw)

    # Score Strategy RAG keywords
    for kw in STRATEGY_RAG_KEYWORDS:
        if kw in query_lower:
            # Higher weight for explicit vendor/procurement terms
            if re.match(r'^(preferred|approved|procurement|who manufactures|who makes)', kw):
                scores[DataSource.STRATEGY_RAG] += 2.5
            else:
                scores[DataSource.STRATEGY_RAG] += 1.0
            matches[DataSource.STRATEGY_RAG].append(kw)

    # Score Deep Agent keywords
    for kw in DEEP_AGENT_KEYWORDS:
        if kw in query_lower:
            # Higher weight for explicit extraction commands
            if re.match(r'^(extract|clause|section|table|annex)', kw):
                scores[DataSource.DEEP_AGENT] += 2.5
            else:
                scores[DataSource.DEEP_AGENT] += 1.0
            matches[DataSource.DEEP_AGENT].append(kw)

    # Score Solution Design keywords
    for kw in SOLUTION_KEYWORDS:
        if kw in query_lower:
            # Higher weight for explicit design/system terms
            if re.match(r'^(design|solution|system|skid|package)', kw):
                scores[DataSource.SOLUTION] += 3.0
            else:
                scores[DataSource.SOLUTION] += 1.0
            matches[DataSource.SOLUTION].append(kw)

    # Score Web Search keywords
    for kw in WEB_SEARCH_KEYWORDS:
        if kw in query_lower:
            # Higher weight for recency and market terms
            if re.match(r'^(latest|recent|current|market|industry|competitor|price)', kw):
                scores[DataSource.WEB_SEARCH] += 2.5
            else:
                scores[DataSource.WEB_SEARCH] += 1.0
            matches[DataSource.WEB_SEARCH].append(kw)

    return scores, matches


def _detect_hybrid_pattern(query_lower: str) -> bool:
    """
    Detect if query requires multiple data sources (hybrid).

    Returns True only when the query genuinely requires combining data
    from multiple sources (e.g., product specs + certification status).
    """
    # Check explicit hybrid patterns (high confidence patterns)
    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    # Check for product + standards combination
    # This is hybrid when asking about a specific product's certification
    has_specific_product = any(kw in query_lower for kw in ["rosemount", "yokogawa", "emerson", "fisher", "3051", "ejx"])
    has_standards = any(kw in query_lower for kw in ["sil", "atex", "zone", "certified", "iec", "hazardous"])
    if has_specific_product and has_standards:
        return True

    # Check for vendor + standards combination (but NOT just "vendor for X product")
    # Only hybrid if asking about vendor capabilities in hazardous/certified context
    has_vendor = any(kw in query_lower for kw in ["vendor", "supplier", "preferred", "approved"])
    if has_vendor and has_standards:
        # Additional check: is this truly asking about standards, not just mentioning a product?
        asking_about_capability = any(kw in query_lower for kw in ["suitable for", "certified for", "rated for", "approved for"])
        if asking_about_capability:
            return True

    return False


# =============================================================================
# PRODUCT INFO PAGE ROUTING (Frontend Integration)
# =============================================================================

def is_engenie_chat_intent(query: str) -> Tuple[bool, float]:
    """
    Determine if query should be routed to EnGenie Chat page in frontend.

    This function is the primary gate for deciding whether to open
    the EnGenie Chat page as a new window/tab in the frontend.

    Returns:
        Tuple of (should_route_to_engenie_chat, confidence)
    """
    query_lower = query.lower().strip()

    # Classify the query first
    data_source, classification_confidence, _ = classify_query(query_lower)

    # All RAG sources and Deep Agent should route to EnGenie Chat page
    engenie_chat_sources = [
        DataSource.INDEX_RAG,
        DataSource.STANDARDS_RAG,
        DataSource.STRATEGY_RAG,
        DataSource.DEEP_AGENT,
        DataSource.SOLUTION,
        DataSource.HYBRID
    ]

    if data_source in engenie_chat_sources:
        # High confidence - route to EnGenie Chat
        return True, classification_confidence

    # Additional pattern checks for edge cases
    engenie_chat_patterns = [
        # Product specification queries
        r"what (is|are) the (specifications?|specs|features)",
        r"tell me about.*(transmitter|sensor|valve|meter)",
        r"(datasheet|catalog|specs?) for",

        # Standards queries
        r"(iec|iso|api|sil|atex)",
        r"(hazardous|explosive) area",
        r"(installation|safety) requirement",

        # Vendor queries
        r"(preferred|approved|recommended) (vendor|supplier)",
        r"who (manufactures|makes|supplies)",
        r"procurement strategy",

        # Extraction queries
        r"extract.*(from|the)",
        r"(clause|section|table|annex)",

        # Brand/product mentions
        r"(rosemount|yokogawa|emerson|honeywell|siemens|fisher|abb)",
        r"(transmitter|sensor|valve|flowmeter|analyzer|positioner)",

        # Comparison queries
        r"compare.*(vendor|supplier|product)",
        r"difference between"
    ]

    pattern_matches = 0
    for pattern in engenie_chat_patterns:
        if re.search(pattern, query_lower):
            pattern_matches += 1

    # If multiple patterns match, likely EnGenie Chat
    if pattern_matches >= 2:
        return True, min(0.7 + (pattern_matches * 0.1), 0.95)
    elif pattern_matches == 1:
        return True, 0.65

    # Default: not a EnGenie Chat query
    return False, 0.3


def get_engenie_chat_route_decision(query: str) -> Dict:
    """
    Get detailed routing decision for EnGenie Chat page.

    Returns a structured decision object for frontend routing.

    Returns:
        Dict with routing decision details:
        - should_route: bool - whether to open EnGenie Chat page
        - confidence: float - confidence in the decision
        - data_source: str - primary data source to query
        - sources: List[str] - all relevant sources for hybrid
        - reasoning: str - explanation for the decision
    """
    # Get classification
    data_source, confidence, reasoning = classify_query(query)

    # Determine if should route to EnGenie Chat
    should_route, route_confidence = is_engenie_chat_intent(query)

    sources = [data_source.value] if data_source != DataSource.LLM else []

    return {
        "should_route": should_route,
        "confidence": max(confidence, route_confidence),
        "data_source": data_source.value,
        "sources": sources,
        "reasoning": reasoning,
        "query_type": _get_query_type_description(data_source)
    }


def _get_query_type_description(data_source: DataSource) -> str:
    """Get human-readable description of query type."""
    descriptions = {
        DataSource.INDEX_RAG: "Product Specifications Query",
        DataSource.STANDARDS_RAG: "Standards & Compliance Query",
        DataSource.STRATEGY_RAG: "Vendor & Procurement Query",
        DataSource.DEEP_AGENT: "Detailed Extraction Query",
        DataSource.SOLUTION: "Solution Design Query",
        DataSource.HYBRID: "Multi-Domain Query",
        DataSource.LLM: "General Query"
    }
    return descriptions.get(data_source, "Unknown Query Type")


# =============================================================================
# TESTING & VALIDATION
# =============================================================================

def test_classification(queries: List[str] = None) -> List[Dict]:
    """
    Test classification against sample queries from Debugs.md.

    Returns list of test results.
    """
    if queries is None:
        # Default test queries from Debugs.md
        queries = [
            # Index RAG tests
            ("What are the specifications for the Rosemount 3051S pressure transmitter?", DataSource.INDEX_RAG),
            ("Tell me about the accuracy range of the Yokogawa EJX series.", DataSource.INDEX_RAG),
            ("Find the datasheet for an Emerson magnetic flowmeter.", DataSource.INDEX_RAG),

            # Standards RAG tests
            ("What are the installation requirements for Zone 1 hazardous areas according to ATEX?", DataSource.STANDARDS_RAG),
            ("Explain the difference between SIL 2 and SIL 3 regarding IEC 61508.", DataSource.STANDARDS_RAG),
            ("What are the API 526 standard requirements for safety valves?", DataSource.STANDARDS_RAG),

            # Strategy RAG tests
            ("Who is our preferred vendor for control valves?", DataSource.STRATEGY_RAG),
            ("What is the procurement strategy for long lead time instruments?", DataSource.STRATEGY_RAG),
            ("Compare the approved suppliers for pressure transmitters.", DataSource.STRATEGY_RAG),

            # Deep Agent tests
            ("Extract the pressure limits for carbon steel flanges from the standard table.", DataSource.DEEP_AGENT),
            ("What is the specific response time requirement in the technical specification for emergency shutoff valves?", DataSource.DEEP_AGENT),

            # Hybrid tests
            ("Is the Rosemount 3051S certified for use in SIL 3 applications?", DataSource.HYBRID),
            ("Which preferred vendor supplies coriolis meters suitable for hazardous Zone 0 areas?", DataSource.HYBRID),

            # Follow-up tests (should detect as follow-up)
            ("Who manufactures the Fisher EZ valve?", DataSource.STRATEGY_RAG),
            ("What are its flow characteristics?", DataSource.INDEX_RAG),  # Follow-up
            ("Do they also make positioners?", DataSource.STRATEGY_RAG),  # Follow-up
        ]

    results = []
    for item in queries:
        if isinstance(item, tuple):
            query, expected = item
        else:
            query, expected = item, None

        source, confidence, reasoning = classify_query(query)
        should_route, route_confidence = is_engenie_chat_intent(query)
        try:
            from .engenie_chat_memory import is_follow_up_query
            is_followup = is_follow_up_query(query, session_id="test")
        except Exception:
            is_followup = False

        result = {
            "query": query,
            "expected": expected.value if expected else None,
            "actual": source.value,
            "confidence": confidence,
            "reasoning": reasoning,
            "should_route_to_engenie_chat": should_route,
            "route_confidence": route_confidence,
            "is_follow_up": is_followup,
            "passed": expected is None or source == expected
        }
        results.append(result)

        # Log result
        status = "PASS" if result["passed"] else "FAIL"
        logger.info(f"[TEST] {status}: '{query[:50]}...' -> {source.value} (expected: {expected.value if expected else 'N/A'})")

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataSource',
    'classify_query',
    'is_engenie_chat_intent',
    'get_engenie_chat_route_decision',
    'test_classification',
    # Keyword lists (for external inspection)
    'INDEX_RAG_KEYWORDS',
    'STANDARDS_RAG_KEYWORDS',
    'STRATEGY_RAG_KEYWORDS',
    'DEEP_AGENT_KEYWORDS',
    'WEB_SEARCH_KEYWORDS'
]


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("INTENT CLASSIFICATION ROUTING AGENT - TEST SUITE")
    print("=" * 80)

    results = test_classification()

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)

    # Show failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nFAILED TESTS:")
        for f in failures:
            print(f"  - Query: {f['query'][:60]}...")
            print(f"    Expected: {f['expected']}, Got: {f['actual']}")
            print(f"    Reasoning: {f['reasoning']}")
            print()
