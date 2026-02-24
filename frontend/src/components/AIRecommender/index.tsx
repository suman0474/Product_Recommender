
import { useState, useCallback, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  AppState,
  ChatMessage,
  ValidationResult,
  AnalysisResult,
  RequirementSchema,
  WorkflowStep,
  IntentClassificationResult,
  AgentResponse,
  AdvancedParametersResult,
  AdvancedParametersSelection,
} from "./types";
import LeftSidebar from "./LeftSidebar";
import ChatInterface from "./ChatInterface";
import RightPanel from "./RightPanel";
import {
  validateRequirements,
  runFinalProductAnalysis,
  getRequirementSchema,
  structureRequirements,
  additionalRequirements,
  classifyRoute,
  discoverAdvancedParameters,
  addAdvancedParameters,
  initializeNewSearch,
  clearSessionValidationState,
  callAgenticValidate,
  callAgenticAdvancedParameters,
  callAgenticSalesAgent,
  getAnalysisProductImages,
  identifyInstruments,
  callAgenticProductSearch,
  resumeProductSearch,
  modifyInstruments,
  BASE_URL,
} from "./api";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useFieldDescriptions } from "@/hooks/useFieldDescriptions";
import MainHeader from "@/components/MainHeader";

type ConversationStep = WorkflowStep;

interface AIRecommenderProps {
  initialInput?: string;
  fillParent?: boolean;
  /**
   * If true, this AIRecommender instance is for DIRECT product search (e.g., from Run button).
   * The initialInput is a sample_input and should bypass instrument identification.
   * Goes directly to Product Search Workflow.
   */
  isDirectSearch?: boolean;
  /**
   * Product type detected during instrument identification (e.g., "Temperature Sensor").
   * Passed to product search workflow for proper schema lookup.
   */
  productType?: string;
  /**
   * Item thread ID for resuming a specific item's product search workflow.
   * Provided by backend during instrument identification.
   * Ensures workflow continuation on the correct branch.
   */
  itemThreadId?: string;
  /**
   * Parent workflow thread ID for workflow context.
   * Provided by backend during instrument identification.
   */
  workflowThreadId?: string;
  /**
   * Main session thread ID (root context).
   * Provided by backend during instrument identification.
   */
  mainThreadId?: string;
  onStateChange?: (state: {
    messages: ChatMessage[];
    collectedData: { [key: string]: any };
    currentStep: ConversationStep;
    analysisResult: AnalysisResult | null;
    searchSessionId: string;
    // Extended state for complete restoration
    requirementSchema: RequirementSchema | null;
    validationResult: ValidationResult | null;
    currentProductType: string | null;
    inputValue: string;
    advancedParameters: AdvancedParametersResult | null;
    selectedAdvancedParams: { [key: string]: string };
    fieldDescriptions: Record<string, string>;
    pricingData: Record<string, any>;
    scrollPositions?: { left: number; center: number; right: number };
    dockingState?: { left: boolean; right: boolean };
  }) => void;
  // Props for restoring saved state
  savedMessages?: ChatMessage[];
  savedCollectedData?: { [key: string]: any };
  savedCurrentStep?: ConversationStep;
  savedAnalysisResult?: AnalysisResult | null;
  // Extended saved state
  savedRequirementSchema?: RequirementSchema | null;
  savedValidationResult?: ValidationResult | null;
  savedCurrentProductType?: string | null;
  savedInputValue?: string;
  savedAdvancedParameters?: AdvancedParametersResult | null;
  savedSelectedAdvancedParams?: { [key: string]: string };
  savedFieldDescriptions?: Record<string, string>;
  savedPricingData?: Record<string, any>;

  onSave?: () => void;
  // Persistence props
  savedScrollPositions?: { left: number; center: number; right: number };
  savedDockingState?: { left: boolean; right: boolean };
  savedSearchInstanceId?: string;
}

const AIRecommender = ({
  initialInput,
  fillParent,
  isDirectSearch = false,  // NEW: For direct product search from Run button
  productType: propProductType,  // NEW: Product type from instrument identification
  itemThreadId: propItemThreadId,  // NEW: Item thread ID for workflow resumption
  workflowThreadId: propWorkflowThreadId,  // NEW: Parent workflow thread ID
  mainThreadId: propMainThreadId,  // NEW: Main session thread ID
  onStateChange,
  savedMessages,
  savedCollectedData,
  savedCurrentStep,
  savedAnalysisResult,
  savedRequirementSchema,
  savedValidationResult,
  savedCurrentProductType,
  savedInputValue,
  savedAdvancedParameters,
  savedSelectedAdvancedParams,
  savedFieldDescriptions,

  savedPricingData,
  onSave,

  // New persistence props
  savedScrollPositions,
  savedDockingState,
  savedSearchInstanceId
}: AIRecommenderProps) => {
  const { toast } = useToast();
  const { logout } = useAuth();
  const [searchParams] = useSearchParams();

  // State for pricing data from RightPanel
  const [pricingData, setPricingData] = useState<Record<string, any>>(savedPricingData || {});

  // Handle pricing data updates from RightPanel
  const handlePricingDataUpdate = (priceReviewMap: Record<string, any>) => {
    console.log('[PRICING_DATA] Received pricing data update:', Object.keys(priceReviewMap).length, 'products');
    setPricingData(priceReviewMap);
  };

  // Generate unique search session ID for this component instance
  // Generate unique search session ID for this component instance
  const [searchSessionId, setSearchSessionId] = useState(() => {
    if (savedSearchInstanceId) {
      console.log(`[SEARCH_SESSION] Restoring saved session ID: ${savedSearchInstanceId}`);
      return savedSearchInstanceId;
    }
    const id = `search_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    console.log(`[SEARCH_SESSION] Created new AIRecommender with session ID: ${id}`);
    return id;
  });

  // Sync with prop if it changes (e.g. project load)
  useEffect(() => {
    if (savedSearchInstanceId && searchSessionId !== savedSearchInstanceId) {
      console.log(`[SEARCH_SESSION] Adopting restored session ID: ${savedSearchInstanceId}`);
      setSearchSessionId(savedSearchInstanceId);
    }
  }, [savedSearchInstanceId, searchSessionId]);

  // Track the last chat session for window reuse logic
  // Format: { query: string, sessionKey: string, windowName: string }
  const lastChatSessionRef = useRef<{ query: string, sessionKey: string, windowName: string } | null>(null);

  // Add session tracking for debugging
  useEffect(() => {
    console.log(`[TAB_SESSION] Tab initialized with session ID: ${searchSessionId}`);
    return () => {
      console.log(`[TAB_SESSION] Tab with session ID ${searchSessionId} is unmounting`);
      // Clean up validation tracker when component unmounts
      clearSessionValidationState(searchSessionId);
    };
  }, [searchSessionId]);

  const [collectedData, setCollectedData] = useState<{ [key: string]: any }>({});
  const [advancedParameters, setAdvancedParameters] = useState<AdvancedParametersResult | null>(null);
  const [selectedAdvancedParams, setSelectedAdvancedParams] = useState<{ [key: string]: string }>({});
  const [fieldDescriptions, setFieldDescriptions] = useState<Record<string, string>>(savedFieldDescriptions || {});
  const [state, setState] = useState<AppState>({
    messages: [],
    currentProductType: null,
    validationResult: null,
    analysisResult: null,
    identifiedItems: null,
    requirementSchema: null,
    isLoading: false,
    inputValue: "",
    productType: "",
  });
  const [currentStep, setCurrentStep] = useState<ConversationStep>("greeting");
  const [hasAutoSubmitted, setHasAutoSubmitted] = useState(false);

  // Product search workflow state - tracks active workflow for proper routing
  const [productSearchWorkflow, setProductSearchWorkflow] = useState<{
    threadId: string | null;
    currentPhase: string | null;
    awaitingUserInput: boolean;
    missingFields: string[];
  }>({
    threadId: null,
    currentPhase: null,
    awaitingUserInput: false,
    missingFields: []
  });

  // Layout states
  const [isStreaming, setIsStreaming] = useState(false);

  // Initialize docking state from saved state if available
  const [isDocked, setIsDocked] = useState(savedDockingState ? savedDockingState.left : true);
  const [isRightDocked, setIsRightDocked] = useState(savedDockingState ? savedDockingState.right : true);

  const DEFAULT_DOCKED_WIDTH = 0;
  const DEFAULT_EXPANDED_WIDTH = 16;
  const DEFAULT_RIGHT_DOCKED_WIDTH = 0;
  const DEFAULT_RIGHT_EXPANDED_WIDTH = 46;
  const [widths, setWidths] = useState({
    left: (savedDockingState?.left ?? true) ? DEFAULT_DOCKED_WIDTH : DEFAULT_EXPANDED_WIDTH,
    center: 100 - ((savedDockingState?.left ?? true) ? DEFAULT_DOCKED_WIDTH : DEFAULT_EXPANDED_WIDTH) - ((savedDockingState?.right ?? true) ? DEFAULT_RIGHT_DOCKED_WIDTH : DEFAULT_RIGHT_EXPANDED_WIDTH),
    right: (savedDockingState?.right ?? true) ? DEFAULT_RIGHT_DOCKED_WIDTH : DEFAULT_RIGHT_EXPANDED_WIDTH
  });

  const containerRef = useRef<HTMLDivElement | null>(null);
  const centerScrollRef = useRef<HTMLDivElement>(null);
  const scrollPositionsRef = useRef(savedScrollPositions || { left: 0, center: 0, right: 0 });
  const animationFrameRef = useRef<number | null>(null);
  const [draggingHandle, setDraggingHandle] = useState<"left" | "right" | null>(null);

  // Scroll persistence - throttle updates to parent
  const [scrollUpdateCounter, setScrollUpdateCounter] = useState(0);
  const scrollThrottleRef = useRef<NodeJS.Timeout | null>(null);

  // INTEGRATED: Dynamic Field Descriptions for schema fields (on-demand loading)
  const { fetchOnHover } = useFieldDescriptions(
    state.currentProductType || 'general'
  );

  // Update sidebar width when docking state changes
  useEffect(() => {
    const newLeftWidth = isDocked ? DEFAULT_DOCKED_WIDTH : DEFAULT_EXPANDED_WIDTH;
    const newRightWidth = isRightDocked ? DEFAULT_RIGHT_DOCKED_WIDTH : DEFAULT_RIGHT_EXPANDED_WIDTH;
    setWidths({
      left: newLeftWidth,
      center: 100 - newLeftWidth - newRightWidth,
      right: newRightWidth
    });
  }, [isDocked, isRightDocked, DEFAULT_DOCKED_WIDTH, DEFAULT_EXPANDED_WIDTH, DEFAULT_RIGHT_DOCKED_WIDTH, DEFAULT_RIGHT_EXPANDED_WIDTH]);

  // Initialize new search session when component mounts
  useEffect(() => {
    const initializeSearch = async () => {
      try {
        await initializeNewSearch(searchSessionId);
        console.log(`[SEARCH_SESSION] Initialized independent search session: ${searchSessionId}`);
      } catch (error) {
        console.warn(`[SEARCH_SESSION] Failed to initialize search session: ${searchSessionId}`, error);
        // Continue anyway - the backend will handle missing session ID gracefully
      }
    };

    initializeSearch();
    initializeSearch();
  }, [searchSessionId]);

  // Restore center scroll position on mount
  useEffect(() => {
    if (savedScrollPositions?.center && centerScrollRef.current) {
      // Short timeout to ensure content renders
      setTimeout(() => {
        if (centerScrollRef.current) {
          centerScrollRef.current.scrollTop = savedScrollPositions.center;
        }
      }, 100);
    }
  }, []);

  // Notify parent of state changes for persistence
  useEffect(() => {
    if (onStateChange) {
      onStateChange({
        messages: state.messages,
        collectedData,
        currentStep,
        analysisResult: state.analysisResult,
        searchSessionId, // Note: using searchSessionId which maps to searchInstanceId in parent
        requirementSchema: state.requirementSchema,
        validationResult: state.validationResult,
        currentProductType: state.productType, // or currentProductType
        inputValue: state.inputValue,
        advancedParameters,
        selectedAdvancedParams,
        fieldDescriptions,
        pricingData,
        scrollPositions: scrollPositionsRef.current,
        dockingState: { left: isDocked, right: isRightDocked }
      });
    }
  }, [state, collectedData, currentStep, advancedParameters, selectedAdvancedParams, fieldDescriptions, pricingData, isDocked, isRightDocked, searchSessionId, onStateChange, scrollUpdateCounter]);

  // Track if component has been initialized to prevent continuous re-loading
  const hasInitialized = useRef(false);

  // Initialize with saved state if provided (only once)
  useEffect(() => {
    if (hasInitialized.current) {
      console.log(`[${searchSessionId}] Already initialized, skipping`);
      return;
    }


    if (savedMessages || savedCollectedData || savedCurrentStep || savedAnalysisResult || savedRequirementSchema || savedValidationResult) {
      console.log(`[${searchSessionId}] Restoring saved state...`);

      // Restore saved state
      if (savedMessages && savedMessages.length > 0) {
        console.log(`[${searchSessionId}] Restoring ${savedMessages.length} messages`);
        setState(prev => ({
          ...prev,
          messages: savedMessages
        }));
      }

      if (savedCollectedData) {
        console.log(`[${searchSessionId}] Restoring collected data with ${Object.keys(savedCollectedData).length} keys`);
        setCollectedData(savedCollectedData);
      }

      if (savedCurrentStep) {
        console.log(`[${searchSessionId}] Restoring current step: ${savedCurrentStep}`);
        setCurrentStep(savedCurrentStep);
      }

      if (savedAnalysisResult) {
        console.log(`[${searchSessionId}] Restoring analysis result`);
        setState(prev => ({
          ...prev,
          analysisResult: savedAnalysisResult
        }));
      }

      // Restore extended state
      if (savedRequirementSchema) {
        console.log(`[${searchSessionId}] Restoring requirement schema`);
        setState(prev => ({
          ...prev,
          requirementSchema: savedRequirementSchema
        }));
      }

      if (savedValidationResult) {
        console.log(`[${searchSessionId}] Restoring validation result`);
        setState(prev => ({
          ...prev,
          validationResult: savedValidationResult,
          currentProductType: savedCurrentProductType,
          productType: savedCurrentProductType || prev.productType
        }));
      }

      // Determine input value: prefer saved draft if provided, otherwise fall back to initial input.
      // This ensures when users were mid-chat and saved a draft input, it is restored on load.
      const hasMessages = savedMessages && savedMessages.length > 0;
      // If conversation already has messages (i.e., the search was run), leave input empty.
      // Otherwise show saved draft or initial input so user can submit it.
      const inputValueToSet = hasMessages ? '' : (savedInputValue ?? initialInput ?? '');

      console.log(`[${searchSessionId}] Setting input value on restore: hasMessages=${hasMessages}, value="${inputValueToSet}"`);
      setState(prev => ({ ...prev, inputValue: inputValueToSet }));
      // Mark auto-fill as handled so other effects don't overwrite this restored input
      setHasAutoSubmitted(true);

      if (savedAdvancedParameters) {
        console.log(`[${searchSessionId}] Restoring advanced parameters`);
        setAdvancedParameters(savedAdvancedParameters);
      }

      if (savedSelectedAdvancedParams) {
        console.log(`[${searchSessionId}] Restoring selected advanced params`);
        setSelectedAdvancedParams(savedSelectedAdvancedParams);
      }

      if (savedFieldDescriptions && Object.keys(savedFieldDescriptions).length > 0) {
        console.log(`[${searchSessionId}] Restoring field descriptions:`, Object.keys(savedFieldDescriptions).length, 'fields');
        setFieldDescriptions(savedFieldDescriptions);
      }

      if (savedPricingData && Object.keys(savedPricingData).length > 0) {
        console.log(`[${searchSessionId}] Restoring pricing data:`, Object.keys(savedPricingData).length, 'products');
        setPricingData(savedPricingData);
      }

      // After restoration, trigger state notification to parent
      if (onStateChange) {
        console.log(`[${searchSessionId}] Notifying parent of restored state`);
        const hasMessages = savedMessages && savedMessages.length > 0;
        const inputValueToNotify = hasMessages ? '' : (savedInputValue ?? initialInput ?? '');
        onStateChange({
          messages: savedMessages || [],
          collectedData: savedCollectedData || {},
          currentStep: savedCurrentStep || "greeting",
          analysisResult: savedAnalysisResult || null,
          searchSessionId,
          requirementSchema: savedRequirementSchema || null,
          validationResult: savedValidationResult || null,
          currentProductType: savedCurrentProductType || null,
          inputValue: inputValueToNotify,
          advancedParameters: savedAdvancedParameters || null,
          selectedAdvancedParams: savedSelectedAdvancedParams || {},
          fieldDescriptions: savedFieldDescriptions || {},
          pricingData: savedPricingData || {}
        });
      }
    } else {
      console.log(`[${searchSessionId}] No saved state found, using defaults`);
    }

    // Mark as initialized to prevent re-initialization
    hasInitialized.current = true;
    console.log(`[${searchSessionId}] Initialization complete`);
  }, [savedMessages, savedCollectedData, savedCurrentStep, savedAnalysisResult, savedRequirementSchema, savedValidationResult, savedCurrentProductType, savedInputValue, savedAdvancedParameters, savedSelectedAdvancedParams, savedFieldDescriptions, savedPricingData, searchSessionId, initialInput]);

  // Notify parent component of state changes for project saving
  useEffect(() => {
    if (onStateChange) {
      onStateChange({
        messages: state.messages,
        collectedData,
        currentStep,
        analysisResult: state.analysisResult,
        searchSessionId,
        requirementSchema: state.requirementSchema,
        validationResult: state.validationResult,
        currentProductType: state.currentProductType,
        inputValue: state.inputValue,
        advancedParameters,
        selectedAdvancedParams,
        fieldDescriptions,
        pricingData
      });
    }
  }, [state.messages, collectedData, currentStep, state.analysisResult, searchSessionId, state.requirementSchema, state.validationResult, state.currentProductType, state.inputValue, advancedParameters, selectedAdvancedParams, fieldDescriptions, pricingData]);

  // --- Resize functionality ---
  const handleMouseDown = useCallback((e: React.MouseEvent, handle: "left" | "right") => {
    e.preventDefault();
    setDraggingHandle(handle);

    const startX = e.clientX;
    const startWidths = { ...widths };
    const containerWidth = containerRef.current?.offsetWidth || 1200;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!containerRef.current) return;

      const deltaX = moveEvent.clientX - startX;
      const deltaPercent = (deltaX / containerWidth) * 100;

      let newWidths;
      if (handle === "left") {
        // Dragging left handle (between sidebar and chat)
        const newLeft = Math.max(7, Math.min(30, startWidths.left + deltaPercent));
        const adjustment = newLeft - startWidths.left;
        newWidths = {
          left: newLeft,
          center: Math.max(10, startWidths.center - adjustment),
          right: startWidths.right
        };
      } else {
        // Dragging right handle (between chat and right panel)
        const newCenter = Math.max(10, Math.min(60, startWidths.center + deltaPercent));
        const adjustment = newCenter - startWidths.center;
        newWidths = {
          left: startWidths.left,
          center: newCenter,
          right: Math.max(10, startWidths.right - adjustment)
        };
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      animationFrameRef.current = requestAnimationFrame(() => {
        setWidths(newWidths);
      });
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      setDraggingHandle(null);

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [widths]);

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Handle field descriptions updates from LeftSidebar
  const handleFieldDescriptionsChange = useCallback((descriptions: Record<string, string>) => {
    console.log(`[${searchSessionId}] Field descriptions updated:`, Object.keys(descriptions).length, 'fields');
    setFieldDescriptions(descriptions);
  }, [searchSessionId]);

  // --- Helper functions ---
  const addMessage = useCallback(
    (message: Omit<ChatMessage, "id" | "timestamp">) => {
      const newMessage: ChatMessage = {
        ...message,
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        timestamp: new Date(),
        role: message.role,
        type: message.type,
      };
      setState((prev) => ({ ...prev, messages: [...prev.messages, newMessage] }));
      return newMessage;
    },
    []
  );

  const updateMessage = useCallback((id: string, newContent: string) => {
    setState((prev) => ({
      ...prev,
      messages: prev.messages.map((msg) =>
        msg.id === id ? { ...msg, content: newContent } : msg
      ),
    }));
  }, []);

  const streamAssistantMessage = useCallback(
    async (fullText: string) => {
      // For single-response steps we no longer stream character-by-character.
      // Add the assistant message once with the complete content.
      setIsStreaming(true);
      const msg = addMessage({ type: "assistant", content: fullText, role: undefined });
      // Slight delay to allow UI to reflect loading state briefly
      await new Promise((res) => setTimeout(res, 50));
      setIsStreaming(false);
      return msg.id;
    },
    [addMessage]
  );

  const composeUserDataString = (data: any): string => {
    const parts: string[] = [];
    if (data.productType) parts.push(`Product Type: ${data.productType}`);
    for (const key in data) {
      if (key === "productType") continue;
      const value = data[key];
      if (value != null && value !== "") {
        parts.push(
          typeof value === "object"
            ? Object.entries(value)
              .map(([k, v]) => (Array.isArray(v) ? `${k}: ${v.join(", ")}` : `${k}: ${v}`))
              .join(". ")
            : `${key}: ${value}`
        );
      }
    }
    return parts.join(". ");
  };

  const flattenRequirements = (provided: any): { [key: string]: any } => {
    const flat: { [key: string]: any } = {};
    const process = (reqs: any) => {
      if (!reqs) return;
      Object.keys(reqs).forEach((key) => {
        const value = reqs[key];
        if (value !== null && value !== "") flat[key] = value;
      });
    };
    if (provided) {
      process(provided.mandatoryRequirements);
      process(provided.optionalRequirements);
      Object.keys(provided).forEach((key) => {
        if (!["mandatoryRequirements", "optionalRequirements"].includes(key) && !(key in flat)) {
          if (provided[key] !== null && provided[key] !== "") flat[key] = provided[key];
        }
      });
    }
    return flat;
  };

  const mergeRequirementsWithSchema = (provided: { [key: string]: any }, schema: RequirementSchema) => {
    const merged: { [key: string]: any } = { ...provided };

    // Helper function to extract Deep Agent values from a schema object
    // Deep Agent populates fields with structure: { value: "...", source: "...", confidence: 0.9 }
    const extractDeepAgentValues = (obj: any, parentKey = ""): { [key: string]: any } => {
      const extracted: { [key: string]: any } = {};

      if (!obj || typeof obj !== "object") return extracted;

      Object.entries(obj).forEach(([key, val]) => {
        const fullKey = parentKey ? `${parentKey}.${key}` : key;

        if (val && typeof val === "object" && !Array.isArray(val)) {
          // Check if this is a Deep Agent value object with 'value' property
          if ("value" in val && val.value && typeof val.value === "string") {
            // This field has a Deep Agent value - use it
            extracted[fullKey] = val.value;
            // Also store by leaf key for flat-lookup compatibility
            extracted[key] = val.value;
          } else {
            // Recurse into nested objects
            const nested = extractDeepAgentValues(val, fullKey);
            Object.assign(extracted, nested);
          }
        }
      });

      return extracted;
    };

    // Extract Deep Agent values from _deep_agent_sections if available
    const deepAgentSections = (schema as any)?._deep_agent_sections || {};
    const deepAgentValues = extractDeepAgentValues(deepAgentSections);

    // Also try extracting from the main schema sections
    const mainSectionNames = ["Performance", "Electrical", "Mechanical", "Environmental",
      "Compliance", "Features", "Integration", "MechanicalOptions",
      "ServiceAndSupport", "Certifications"];

    for (const sectionName of mainSectionNames) {
      const section = (schema as any)?.[sectionName];
      if (section && typeof section === "object") {
        const sectionValues = extractDeepAgentValues(section);
        Object.assign(deepAgentValues, sectionValues);
      }
    }

    // ── Fix #3: Extract {value: '...'} objects directly from mandatory/optionalRequirements ──
    // The schema stores standards-default values in the format:
    //   { Accuracy: { value: "±0.075%", source: "standards" } }
    // or nested under sections:
    //   { Performance: { Accuracy: { value: "±0.075%", source: "standards" } } }
    // These were never read into collectedData before this fix.
    const extractFromRequirements = (reqs: any) => {
      if (!reqs || typeof reqs !== "object") return;
      Object.entries(reqs).forEach(([key, val]: [string, any]) => {
        if (val && typeof val === "object" && !Array.isArray(val)) {
          if ("value" in val && val.value && val.value !== "Not specified" && val.value !== "") {
            // Direct field with value — store by leaf key (lower priority than user-provided)
            if (!(key in deepAgentValues)) deepAgentValues[key] = val.value;
          } else {
            // Nested section — recurse one level
            Object.entries(val).forEach(([k2, v2]: [string, any]) => {
              if (v2 && typeof v2 === "object" && "value" in v2 && v2.value && v2.value !== "Not specified" && v2.value !== "") {
                if (!(k2 in deepAgentValues)) deepAgentValues[k2] = v2.value;
              }
            });
          }
        }
      });
    };
    extractFromRequirements(schema.mandatoryRequirements);
    extractFromRequirements(schema.optionalRequirements);

    // Get all schema keys
    const allKeys = [
      ...(schema.mandatoryRequirements ? Object.keys(schema.mandatoryRequirements) : []),
      ...(schema.optionalRequirements ? Object.keys(schema.optionalRequirements) : []),
    ];

    // Merge: user-provided values take priority, then Deep Agent / schema defaults, then empty
    allKeys.forEach((key) => {
      if (!(key in merged) || merged[key] === "" || merged[key] === null) {
        // Check if Deep Agent has a value for this key (try exact match and camelCase variations)
        const deepValue = deepAgentValues[key] ||
          deepAgentValues[key.charAt(0).toUpperCase() + key.slice(1)] ||
          Object.entries(deepAgentValues).find(([k]) =>
            k.toLowerCase().replace(/[._]/g, "") === key.toLowerCase().replace(/[._]/g, "")
          )?.[1];

        merged[key] = deepValue || "";
      }
    });

    console.log(`[MERGE] Merged ${Object.keys(deepAgentValues).length} Deep Agent / schema values into collectedData`);
    return merged;
  };

  // --- Core analysis and summary flow ---
  // performAnalysis can be called with optional pre-built requirements (from workflow completion)
  // or it will build them from state/collectedData (backward compatibility)
  const performAnalysis = useCallback(async (options?: {
    finalRequirements?: {
      productType?: string;
      mandatoryRequirements?: Record<string, any>;
      optionalRequirements?: Record<string, any>;
      advancedParameters?: any[];
    };
    schema?: any;
  }) => {
    setState((prev) => ({ ...prev, isLoading: true }));
    try {
      // Use provided requirements or build from state
      const effectiveProductType = options?.finalRequirements?.productType || state.productType || '';
      const effectiveSchema = options?.schema || state.requirementSchema;

      console.log(`[${searchSessionId}] [PERFORM_ANALYSIS] Starting final product search analysis`);
      console.log(`[${searchSessionId}] [PERFORM_ANALYSIS] Product Type: ${effectiveProductType}`);
      console.log(`[${searchSessionId}] [PERFORM_ANALYSIS] Using provided requirements: ${!!options?.finalRequirements}`);

      // Build structured requirements - prefer provided, fallback to state
      let structuredRequirements: {
        productType: string;
        mandatoryRequirements: Record<string, any>;
        optionalRequirements: Record<string, any>;
        selectedAdvancedParams: Record<string, any>;
      };

      if (options?.finalRequirements) {
        // Use the requirements from workflow completion directly
        structuredRequirements = {
          productType: effectiveProductType,
          mandatoryRequirements: options.finalRequirements.mandatoryRequirements || {},
          optionalRequirements: options.finalRequirements.optionalRequirements || {},
          selectedAdvancedParams: {}
        };
        // Add advanced parameters if available
        if (options.finalRequirements.advancedParameters && options.finalRequirements.advancedParameters.length > 0) {
          for (const param of options.finalRequirements.advancedParameters) {
            const key = param.name || param.key;
            if (key) {
              structuredRequirements.selectedAdvancedParams[key] = param.value || true;
            }
          }
        }
      } else {
        // Fallback: Build from state/collectedData
        structuredRequirements = {
          productType: effectiveProductType,
          mandatoryRequirements: effectiveSchema?.mandatoryRequirements || {},
          optionalRequirements: effectiveSchema?.optionalRequirements || {},
          selectedAdvancedParams: {}
        };

        // Merge collected data into requirements
        if (collectedData) {
          for (const [key, value] of Object.entries(collectedData)) {
            if (key !== 'productType' && value) {
              // Check if it's a mandatory requirement
              if (structuredRequirements.mandatoryRequirements && key in structuredRequirements.mandatoryRequirements) {
                structuredRequirements.mandatoryRequirements[key] = value;
              }
              // Check if it's an optional requirement
              else if (structuredRequirements.optionalRequirements && key in structuredRequirements.optionalRequirements) {
                structuredRequirements.optionalRequirements[key] = value;
              }
              // Otherwise treat as additional requirement
              else {
                if (!structuredRequirements.optionalRequirements) {
                  structuredRequirements.optionalRequirements = {};
                }
                structuredRequirements.optionalRequirements[key] = value;
              }
            }
          }
        }
      }

      console.log(`[${searchSessionId}] [PERFORM_ANALYSIS] Structured requirements:`, {
        productType: structuredRequirements.productType,
        mandatoryCount: Object.keys(structuredRequirements.mandatoryRequirements || {}).length,
        optionalCount: Object.keys(structuredRequirements.optionalRequirements || {}).length,
        advancedCount: Object.keys(structuredRequirements.selectedAdvancedParams || {}).length
      });

      // Validate we have required data
      if (!structuredRequirements.productType) {
        throw new Error('Product type is required for analysis');
      }

      // Call new run-analysis endpoint
      const analysis: AnalysisResult = await runFinalProductAnalysis(
        structuredRequirements,
        effectiveProductType,
        effectiveSchema,
        searchSessionId
      );

      console.log(`[${searchSessionId}] [PERFORM_ANALYSIS] Analysis complete:`, {
        totalMatches: analysis.totalMatches,
        exactMatches: analysis.exactMatchCount,
        approximateMatches: analysis.approximateMatchCount
      });

      // Thresholds for match quality
      // Exact matches: No score threshold - show ALL products where requirementsMatch === true
      // Approximate matches: Minimum score of 50% to ensure reasonable quality
      const APPROXIMATE_THRESHOLD = 50;

      // Split products into exact and approximate matches
      // Exact matches: ALL products where requirementsMatch === true (no score requirement)
      const exactMatches = analysis.overallRanking.rankedProducts.filter(
        (p) => p.requirementsMatch === true
      );

      // Approximate matches: requirementsMatch === false AND score >= 50%
      const approximateMatches = analysis.overallRanking.rankedProducts.filter(
        (p) => p.requirementsMatch === false && (p.overallScore ?? 0) >= APPROXIMATE_THRESHOLD
      );

      // Fallback logic: use exact if available, otherwise approximate
      const count = exactMatches.length > 0 ? exactMatches.length : approximateMatches.length;
      const displayMode = exactMatches.length > 0 ? 'exact' : 'approximate';


      // Check if any vendors were actually analyzed
      const vendorsAnalyzed = analysis.vendorAnalysis?.vendorsAnalyzed ?? 0;

      let message = "";

      if (vendorsAnalyzed === 0) {
        message = "No vendors found for this product type in the database.";
      } else {
        message = exactMatches.length > 0
          ? `Found ${exactMatches.length} product${exactMatches.length !== 1 ? 's' : ''} matching all requirements`
          : `No exact matches found. Found ${approximateMatches.length} close alternative${approximateMatches.length !== 1 ? 's' : ''}`;
      }

      // ✅ Fetch images ONLY for products that will be displayed
      // If exact matches exist, fetch only for exact matches
      // If no exact matches, fetch only for approximate matches
      const productsToFetchImagesFor = displayMode === 'exact' ? exactMatches : approximateMatches;

      if (productsToFetchImagesFor.length > 0) {
        try {
          console.log(`[IMAGE_FETCH] Fetching images for ${productsToFetchImagesFor.length} ${displayMode} match products`);

          const imageFetchPromises = productsToFetchImagesFor.map(async (product) => {
            if (!product.vendor || !product.productName) return product;

            try {
              const modelFamilies = product.modelFamily ? [product.modelFamily] : [];
              const imageResult = await getAnalysisProductImages(
                product.vendor,
                state.productType || "",
                product.productName,
                modelFamilies
              );

              // Update product with images and logo
              return {
                ...product,
                topImage: imageResult.topImage,
                vendorLogo: imageResult.vendorLogo,
                allImages: imageResult.allImages || [],
              };
            } catch (error) {
              console.error(`Failed to fetch images for ${product.vendor} - ${product.productName}:`, error);
              return product; // Return product without images if fetch fails
            }
          });

          const productsWithImages = await Promise.all(imageFetchPromises);

          // Update the analysis result with products that have images
          // Replace only the products we fetched images for
          analysis.overallRanking.rankedProducts = analysis.overallRanking.rankedProducts.map(product => {
            const updatedProduct = productsWithImages.find(
              p => p.vendor === product.vendor && p.productName === product.productName
            );
            return updatedProduct || product; // Use updated if found, otherwise keep original
          });

          // Also update vendorMatches if they exist
          if (analysis.vendorAnalysis?.vendorMatches) {
            for (const match of analysis.vendorAnalysis.vendorMatches) {
              const correspondingProduct = productsWithImages.find(
                p => p.vendor === match.vendor && p.productName === match.productName
              );
              if (correspondingProduct) {
                match.topImage = correspondingProduct.topImage;
                match.vendorLogo = correspondingProduct.vendorLogo;
                match.allImages = correspondingProduct.allImages;
              }
            }
          }

          console.log(`[IMAGE_FETCH] Successfully fetched images for ${productsWithImages.length} products`);
        } catch (error) {
          console.error("Error fetching product images:", error);
          // Continue even if image fetching fails
        }
      } else {
        console.log(`[IMAGE_FETCH] No products to fetch images for (displayMode: ${displayMode})`);
      }

      // =====================================================================
      // FALLBACK: Fetch generic images for products still missing top_image
      // This is the frontend fallback layer with cross-verification
      // =====================================================================
      try {
        const productsNeedingImages = analysis.overallRanking.rankedProducts.filter(
          (p: any) => !p.top_image?.url && !p.topImage?.url
        );

        if (productsNeedingImages.length > 0) {
          console.log(`[IMAGE_FALLBACK] ${productsNeedingImages.length} products need generic images`);

          // Fetch generic images sequentially for missing products (max 10 to avoid rate limits)
          for (const product of productsNeedingImages.slice(0, 10)) {
            const productType = (product as any).productType || (product as any).product_type || state.productType;
            if (!productType) continue;

            // Cross-verification: Skip if image was added by another process
            if ((product as any).top_image?.url || (product as any).topImage?.url) continue;

            try {
              const encodedType = encodeURIComponent(productType);
              const response = await fetch(`${BASE_URL}/api/generic_image/${encodedType}`, {
                credentials: 'include'
              });

              if (response.ok) {
                const data = await response.json();
                if (data.success && data.image?.url) {
                  (product as any).top_image = {
                    url: data.image.url,
                    source: 'frontend_fallback',
                    product_type: productType
                  };
                  console.log(`[IMAGE_FALLBACK] ✓ Generic image loaded for ${product.productName}`);
                }
              }
            } catch (imgErr) {
              console.warn(`[IMAGE_FALLBACK] Failed to fetch generic image for ${productType}`);
            }
          }
        } else {
          console.log(`[IMAGE_FALLBACK] All products have images, no fallback needed`);
        }
      } catch (fallbackErr) {
        console.warn(`[IMAGE_FALLBACK] Fallback image fetch failed:`, fallbackErr);
      }



      // Let the Sales Agent compute the accurate product count message from analysisResult
      const contextMessage = `Analysis complete.`;


      const llmResponse = await callAgenticSalesAgent(
        "finalAnalysis",
        contextMessage,
        {
          analysisResult: analysis,
          displayMode
        },
        searchSessionId,
        "workflow"
      );
      await streamAssistantMessage(llmResponse.content);

      setState((prev) => ({ ...prev, analysisResult: analysis, isLoading: false }));
      setCurrentStep("initialInput");
      toast({
        title: "Analysis Complete",
        description: message,
        variant: "default"  // Same variant for both exact and approximate
      });
    } catch (error) {
      console.error("Analysis error:", error);
      const llmResponse = await callAgenticSalesAgent(
        "analysisError",
        "An error occurred during final analysis.",
        {},
        searchSessionId,
        "workflow"
      );
      await streamAssistantMessage(llmResponse.content);
      setState((prev) => ({ ...prev, isLoading: false }));
      setCurrentStep("analysisError");
    }
  }, [collectedData, state.productType, toast, streamAssistantMessage, searchSessionId]);

  const handleShowSummaryAndProceed = useCallback(async (options?: { skipIntro?: boolean; introAlreadyStreamed?: boolean; skipAnalysis?: boolean }) => {
    setState((prev) => ({ ...prev, isLoading: true }));
    try {
      // Include productType in the requirements string to avoid empty input
      const requirementsString = collectedData.productType
        ? `Product Type: ${collectedData.productType}. ${composeUserDataString(collectedData)}`
        : composeUserDataString(collectedData);

      // If no requirements, still show a summary message
      let summaryContent = "";
      if (!requirementsString || requirementsString.trim().length === 0) {
        summaryContent = `**Product Type:** ${state.productType || 'Not specified'}\n\nNo additional requirements specified.`;
      } else {
        try {
          const structuredResponse = await structureRequirements(requirementsString);
          summaryContent = structuredResponse.structuredRequirements;
        } catch (structureError) {
          console.warn("Failed to structure requirements, using raw data:", structureError);
          summaryContent = `**Product Type:** ${state.productType || collectedData.productType || 'Not specified'}\n\n**Collected Requirements:**\n${requirementsString}`;
        }
      }

      if (!options?.skipIntro && !options?.introAlreadyStreamed) {
        const summaryIntro = await callAgenticSalesAgent(
          "showSummary",
          "Summary of requirements is ready.",
          collectedData,
          searchSessionId,
          "workflow"
        );
        await streamAssistantMessage(summaryIntro.content);
      }

      // Display the summary
      addMessage({ type: "assistant", content: `\n\n${summaryContent}\n\n`, role: undefined });

      // Check if we should skip analysis (wait for user confirmation)
      if (options?.skipAnalysis) {
        // Show message asking for confirmation to proceed
        await streamAssistantMessage("\nWould you like to proceed with the product analysis? Type **'yes'** or **'proceed'** to continue.");
        setState((prev) => ({ ...prev, isLoading: false }));
        setCurrentStep("showSummary");
        return;
      }

      // Run analysis if not skipping
      await performAnalysis();

      setState((prev) => ({ ...prev, isLoading: false }));
    } catch (error) {
      console.error("Summary error:", error);
      // Fallback: Show basic summary with collected data
      const fallbackSummary = `**Product Type:** ${state.productType || 'Not specified'}\n\n` +
        `I've gathered your requirements. Would you like to proceed with the product analysis?\n\n` +
        `Type **'yes'** or **'proceed'** to continue.`;
      await streamAssistantMessage(fallbackSummary);
      setState((prev) => ({ ...prev, isLoading: false }));
      setCurrentStep("showSummary");
    }
  }, [collectedData, state.productType, performAnalysis, streamAssistantMessage, addMessage, searchSessionId, isDirectSearch]);

  // Helper function to run direct product search from RightPanel.
  // parentSessionId / parentInstanceId are forwarded from the solution BOM item
  // so the Search Deep Agent can be traced back to its parent Solution run.
  // When not supplied directly, they are looked up from state.identifiedItems
  // by matching on sampleInput — avoids changing the RightPanel callback chain.
  const handleRunProductSearch = useCallback(async (
    sampleInput: string,
    parentSessionId?: string,
    parentInstanceId?: string,
  ) => {
    // Resolve parent IDs: prefer explicit args, then look up from BOM items
    let resolvedParentSessionId = parentSessionId;
    let resolvedParentInstanceId = parentInstanceId;
    if (!resolvedParentSessionId && state.identifiedItems) {
      const matchedItem = state.identifiedItems.find(
        (item: any) => (item.sampleInput || item.sample_input) === sampleInput
      );
      if (matchedItem) {
        resolvedParentSessionId = matchedItem.parent_session_id || matchedItem.parentSessionId;
        resolvedParentInstanceId = matchedItem.parent_instance_id || matchedItem.parentInstanceId;
      }
    }

    console.log(`[${searchSessionId}] ====== PRODUCT SEARCH TRIGGERED ======`);
    console.log(`[${searchSessionId}] sampleInput: ${sampleInput?.substring(0, 100)}...`);
    console.log(`[${searchSessionId}] searchSessionId: ${searchSessionId}`);
    console.log(`[${searchSessionId}] parentSessionId: ${resolvedParentSessionId?.substring(0, 16) ?? 'standalone'}`);
    console.log(`[${searchSessionId}] parentInstanceId: ${resolvedParentInstanceId?.substring(0, 16) ?? 'none'}`);

    setState((prev) => ({ ...prev, isLoading: true }));
    // Add user message to chat to reflect the action
    const displayMessage: ChatMessage = {
      role: "user",
      id: crypto.randomUUID(),
      type: "user",
      content: sampleInput,
      timestamp: new Date()
    };
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, displayMessage]
    }));

    try {
      // Call PRODUCT SEARCH endpoint directly
      // Use propProductType from parent (instrument/solution workflow) or state.productType
      const effectiveProductType = propProductType || state.productType || undefined;
      const sourceWorkflow = isDirectSearch ? 'instrument_identifier' : 'direct';

      console.log(`[${searchSessionId}] [AGENTIC_PS] Calling /api/agentic/product-search`);
      console.log(`[${searchSessionId}] [AGENTIC_PS] Payload:`, {
        message: sampleInput.substring(0, 50),
        session_id: searchSessionId,
        product_type: effectiveProductType,
        source_workflow: sourceWorkflow
      });

      const result = await callAgenticProductSearch(
        sampleInput,
        undefined, // threadId - first call
        searchSessionId,
        effectiveProductType,
        sourceWorkflow,
        propItemThreadId,              // CRITICAL: Pass item thread ID for branch resumption
        propWorkflowThreadId,          // CRITICAL: Pass workflow thread ID for context
        resolvedParentSessionId,       // Solution session that spawned this search
        resolvedParentInstanceId,      // Solution workflow_thread_id for observability
      );

      console.log(`[${searchSessionId}] [AGENTIC_PS] Response received:`, {
        success: result.success,
        hasResponse: !!result.sales_agent_response,
        hasSchema: !!result.schema,
        currentPhase: result.current_phase,
        awaitingInput: result.awaiting_user_input,
        rankedProductsCount: result.ranked_products?.length || 0
      });

      if (result.success) {
        // ============================================================
        // UPDATE LEFT SIDEBAR with schema if available
        // ============================================================
        if (result.schema) {
          console.log(`[${searchSessionId}] [AGENTIC_PS] Schema received - updating left sidebar`);
          console.log(`[${searchSessionId}] [AGENTIC_PS] Raw schema keys:`, Object.keys(result.schema));
          console.log(`[${searchSessionId}] [AGENTIC_PS] mandatoryRequirements:`, result.schema.mandatoryRequirements);
          console.log(`[${searchSessionId}] [AGENTIC_PS] optionalRequirements:`, result.schema.optionalRequirements);

          // Extract schema structure for RequirementSchema type
          const schemaForSidebar = {
            mandatoryRequirements: result.schema.mandatoryRequirements || result.schema.mandatory || {},
            optionalRequirements: result.schema.optionalRequirements || result.schema.optional || {},
            default: {
              mandatory: result.schema.mandatoryRequirements || result.schema.mandatory || {},
              optional: result.schema.optionalRequirements || result.schema.optional || {}
            }
          };
          console.log(`[${searchSessionId}] [AGENTIC_PS] schemaForSidebar:`, schemaForSidebar);

          setState((prev) => ({
            ...prev,
            requirementSchema: schemaForSidebar as any,
            currentProductType: result.product_type || prev.currentProductType,
            productType: result.product_type || prev.productType
          }));

          // ── Fix #3 complement: Immediately seed collectedData from schema field values ──
          // After Fix #5 runs on the backend, the schema's field objects contain user-provided
          // values (source="user_provided") and standards defaults (source="standards_specifications").
          // Merge those into collectedData so the sidebar shows them as green values right away.
          const initialProvidedReqs = result.provided_requirements || {};
          const initialCollectedData = mergeRequirementsWithSchema(
            { ...initialProvidedReqs },
            schemaForSidebar as any
          );
          setCollectedData((prev: any) => ({ ...prev, ...initialCollectedData }));
          console.log(`[${searchSessionId}] [AGENTIC_PS] Seeded collectedData with ${Object.keys(initialCollectedData).length} values from schema`);

          // Auto-undock sidebar to show schema
          setIsDocked(false);
        }

        // ============================================================
        // STORE THREAD_ID for follow-up calls
        // ============================================================
        if (result.thread_id) {
          console.log(`[${searchSessionId}] [AGENTIC_PS] Stored thread_id: ${result.thread_id}`);
        }

        // Display the sales agent response
        if (result.sales_agent_response) {
          console.log(`[${searchSessionId}] [AGENTIC_PS] Streaming response to chat`);
          await streamAssistantMessage(result.sales_agent_response);
        }

        // ============================================================
        // HANDLE AWAITING_USER_INPUT - workflow is paused
        // Store workflow state so handleSendMessage can route to resumeProductSearch
        // ============================================================
        if (result.awaiting_user_input) {
          console.log(`[${searchSessionId}] [AGENTIC_PS] Workflow paused - awaiting user input`);
          console.log(`[${searchSessionId}] [AGENTIC_PS] Current phase: ${result.current_phase}`);
          console.log(`[${searchSessionId}] [AGENTIC_PS] Missing fields: ${result.missing_fields?.join(', ')}`);

          // Store workflow state for routing subsequent messages
          setProductSearchWorkflow({
            threadId: result.thread_id,
            currentPhase: result.current_phase,
            awaitingUserInput: true,
            missingFields: result.missing_fields || []
          });

          // Set workflow step based on current phase - use valid WorkflowStep values
          if (
            result.current_phase === 'await_missing_fields' ||
            result.current_phase === 'collect_missing_fields' ||
            result.current_phase === 'hil_schema_review'   // HIL Checkpoint 1
          ) {
            setCurrentStep('awaitMissingInfo');
          } else if (
            result.current_phase === 'await_advanced_params' ||
            result.current_phase === 'hil_advanced_review'  // HIL Checkpoint 2
          ) {
            setCurrentStep('awaitAdditionalAndLatestSpecs');
          } else {
            // Map other phases to valid WorkflowStep values
            setCurrentStep('awaitMissingInfo');
          }
        } else {
          // Workflow not awaiting input - clear the tracking state
          setProductSearchWorkflow({
            threadId: null,
            currentPhase: null,
            awaitingUserInput: false,
            missingFields: []
          });
        }

        // Set ranked products if available
        if (result.ranked_products && result.ranked_products.length > 0) {
          console.log(`[${searchSessionId}] [AGENTIC_PS] Setting ${result.ranked_products.length} ranked products`);
          const analysisResult = {
            overallRanking: {
              rankedProducts: result.ranked_products
            }
          };
          setState((prev) => ({ ...prev, analysisResult: analysisResult as any, identifiedItems: null }));
          setIsRightDocked(false);
        }
      } else {
        console.error(`[${searchSessionId}] [AGENTIC_PS] Request failed:`, result.error);
        await streamAssistantMessage(`Sorry, I encountered an error: ${result.error}`);
      }

    } catch (error) {
      console.error(`[${searchSessionId}] [AGENTIC_PS] Direct search failed:`, error);
      await streamAssistantMessage("Sorry, I encountered an error running that search.");
    } finally {
      console.log(`[${searchSessionId}] [AGENTIC_PS] ====== PRODUCT SEARCH COMPLETE ======`);
      setState((prev) => ({ ...prev, isLoading: false }));
    }
  }, [searchSessionId, propProductType, state.productType, isDirectSearch, propItemThreadId, propWorkflowThreadId]);

  // NEW: Handler for populating input without auto-submitting
  const handlePopulateInput = useCallback((sampleInput: string) => {
    setState((prev) => ({ ...prev, inputValue: sampleInput }));
    // Optional: focus logic is handled by the textarea ref in ChatInterface if passed down, 
    // but here we just set state. The textarea value comes from props.
  }, []);

  // --- New workflow-aware message handler ---
  const handleSendMessage = useCallback(
    async (userInput: string) => {
      const trimmedInput = userInput.trim();
      if (!trimmedInput) return;

      // NEW: Intercept Direct Search — ALL messages in isDirectSearch mode bypass
      // classifyRoute and the old multi-phase workflow entirely.
      // classifyRoute must NEVER be invoked in this context — it would misroute
      // short follow-ups ("yes", "continue") to EnGenie Chat or out_of_domain.
      if (isDirectSearch) {
        // If a HIL checkpoint is active (workflow paused), resume it with the user's message.
        // Otherwise treat as a new search query.
        if (productSearchWorkflow?.awaitingUserInput && productSearchWorkflow?.threadId) {
          console.log(`[${searchSessionId}] [HIL] Resuming paused workflow — decision: "${trimmedInput}"`);
          setState((prev) => ({ ...prev, isLoading: true }));
          addMessage({ type: "user", content: trimmedInput, role: undefined });
          setState((prev) => ({ ...prev, inputValue: "" }));
          try {
            const resumeResult = await resumeProductSearch(
              productSearchWorkflow.threadId,
              trimmedInput,   // user_decision
              undefined,       // userProvidedFields (text decision only; backend merges on re-call if needed)
              trimmedInput,
              searchSessionId,
            );
            // Reuse handleRunProductSearch result processing by calling it as a pseudo-result
            // Instead, process directly like handleRunProductSearch does
            if (resumeResult?.success) {
              if (resumeResult.schema) {
                const schemaForSidebar = {
                  mandatoryRequirements: resumeResult.schema.mandatoryRequirements || resumeResult.schema.mandatory || {},
                  optionalRequirements: resumeResult.schema.optionalRequirements || resumeResult.schema.optional || {},
                  default: {
                    mandatory: resumeResult.schema.mandatoryRequirements || resumeResult.schema.mandatory || {},
                    optional: resumeResult.schema.optionalRequirements || resumeResult.schema.optional || {}
                  }
                };
                setState((prev) => ({
                  ...prev,
                  requirementSchema: schemaForSidebar as any,
                  currentProductType: resumeResult.product_type || prev.currentProductType,
                  productType: resumeResult.product_type || prev.productType
                }));
                setIsDocked(false);
              }
              if (resumeResult.awaiting_user_input) {
                // Another HIL checkpoint — update tracking
                setProductSearchWorkflow({
                  threadId: resumeResult.thread_id || productSearchWorkflow.threadId,
                  currentPhase: resumeResult.current_phase,
                  awaitingUserInput: true,
                  missingFields: resumeResult.missing_fields || []
                });
              } else {
                // Workflow completed — clear HIL tracking
                setProductSearchWorkflow({ threadId: null, currentPhase: null, awaitingUserInput: false, missingFields: [] });
              }
              if (resumeResult.sales_agent_response) {
                await streamAssistantMessage(resumeResult.sales_agent_response);
              }
              // Show ranked products if workflow completed
              if (resumeResult.ranked_products?.length > 0) {
                setState((prev) => ({
                  ...prev,
                  analysisResult: { overallRanking: { rankedProducts: resumeResult.ranked_products } } as any,
                  identifiedItems: null
                }));
                setIsRightDocked(false);
              }
            } else {
              await streamAssistantMessage(`Sorry, resume failed: ${resumeResult?.error || 'Unknown error'}`);
            }
          } catch (err) {
            console.error(`[${searchSessionId}] [HIL] Resume failed:`, err);
            await streamAssistantMessage("Sorry, I couldn't resume the search workflow. Please try your search again.");
          } finally {
            setState((prev) => ({ ...prev, isLoading: false }));
          }
          return;
        }
        // No active HIL checkpoint — treat as a fresh search query
        handleRunProductSearch(trimmedInput);
        setState((prev) => ({ ...prev, inputValue: "" }));
        return;
      }

      // NEW: Intercept /modify command for instrument modification
      if (trimmedInput.toLowerCase().startsWith("/modify")) {
        const modificationRequest = trimmedInput.replace(/^\/modify\s*/i, "").trim();

        // Add user message immediately
        addMessage({ type: "user", content: trimmedInput, role: undefined });
        setState((prev) => ({ ...prev, inputValue: "", isLoading: true }));

        if (!modificationRequest) {
          await streamAssistantMessage("Please provide a modification request, e.g., '/modify Add a thermowell'.");
          setState((prev) => ({ ...prev, isLoading: false }));
          return;
        }

        if (!state.identifiedItems || state.identifiedItems.length === 0) {
          await streamAssistantMessage("There are no identified instruments to modify. Please run instrument identification first.");
          setState((prev) => ({ ...prev, isLoading: false }));
          return;
        }

        try {
          // Filter current items
          const instruments = state.identifiedItems.filter(i => i.type === 'instrument');
          const accessories = state.identifiedItems.filter(i => i.type === 'accessory');

          // Call the modifyInstruments API
          // Note: modifyInstruments expects snake_case/backend compatible naming, which the API wrapper handles
          // We pass the raw items, the API wrapper + backend prompt handles the rest.
          const result = (await modifyInstruments(
            modificationRequest,
            instruments,
            accessories,
            searchSessionId
          )) as any;

          if (result.error) {
            await streamAssistantMessage(`Modification failed: ${result.error}`);
          } else {
            // Map results back to IdentifiedItem structure for state
            // Note: result.instruments contains IdentifiedInstrument (camelCase from convertKeysToCamelCase)
            const newItems: any[] = [
              ...(result.instruments || []).map((i: any, idx: number) => ({
                number: idx + 1,
                type: 'instrument' as const,
                name: i.productName || i.name || i.category,
                category: i.category,
                quantity: i.quantity,
                keySpecs: typeof i.specifications === 'string' ? i.specifications :
                  Object.entries(i.specifications || {}).map(([k, v]) => `${k}: ${v}`).join(', '),
                sampleInput: i.sampleInput,
                specifications: i.specifications, // Keep raw specs if needed
                imageUrl: i.imageUrl || i.image_url || '',  // Include image URL
              })),
              ...(result.accessories || []).map((a: any, idx: number) => ({
                number: (result.instruments?.length || 0) + idx + 1,
                type: 'accessory' as const,
                name: a.accessoryName || a.name || a.category,
                category: a.category,
                quantity: a.quantity,
                keySpecs: typeof a.specifications === 'string' ? a.specifications :
                  Object.entries(a.specifications || {}).map(([k, v]) => `${k}: ${v}`).join(', '),
                sampleInput: a.sampleInput,
                specifications: a.specifications,
                imageUrl: a.imageUrl || a.image_url || '',  // Include image URL
              }))
            ];

            setState(prev => ({
              ...prev,
              identifiedItems: newItems,
              // Also update collectedData if necessary (though usually for single product search)
            }));

            // Show the friendly message from the agent
            const msg = result.message || result.summary || "Modifications applied successfully.";
            await streamAssistantMessage(msg);
          }
        } catch (e: any) {
          console.error("Modification error:", e);
          await streamAssistantMessage(`An error occurred during modification: ${e.message}`);
        }
        setState(prev => ({ ...prev, isLoading: false }));
        return;
      }

      // Add user message
      addMessage({ type: "user", content: trimmedInput, role: undefined });
      setState((prev) => ({ ...prev, inputValue: "", isLoading: true }));

      try {
        // ============================================================
        // PRODUCT SEARCH WORKFLOW ROUTING
        // If we're in an active product search workflow awaiting input,
        // route the message through resumeProductSearch instead of normal flow
        // ============================================================
        if (productSearchWorkflow.awaitingUserInput && productSearchWorkflow.threadId) {
          console.log(`[${searchSessionId}] [AGENTIC_PS] Routing to resumeProductSearch`);
          console.log(`[${searchSessionId}] [AGENTIC_PS] Thread: ${productSearchWorkflow.threadId}, Phase: ${productSearchWorkflow.currentPhase}`);

          // Determine user decision based on input
          const normalizedInput = trimmedInput.toLowerCase().trim();
          let userDecision: string | undefined;
          let userProvidedFields: Record<string, any> | undefined;

          // Check for yes/no/continue patterns
          // For 'await_missing_fields': Question is "add missing details or continue anyway?"
          // - 'yes', 'continue', 'proceed', 'skip' = Continue without adding (skip missing fields)
          // - 'add', 'provide', 'no' = User wants to add the missing fields
          const wantsToContinue = /^(yes|y|yeah|yep|sure|ok|okay|continue|proceed|skip)$/i.test(normalizedInput);
          const wantsToAdd = /^(no|n|nope|add|provide)$/i.test(normalizedInput);

          if (productSearchWorkflow.currentPhase === 'await_missing_fields') {
            // User is deciding whether to add missing fields
            // Question: "Would you like to add... or continue anyway?"
            // "yes" = "yes, continue anyway" (the natural reading)
            if (wantsToContinue) {
              userDecision = 'continue';    // User wants to skip and continue to advanced params
            } else if (wantsToAdd) {
              userDecision = 'add_fields';  // User explicitly wants to add missing fields
            }
          } else if (productSearchWorkflow.currentPhase === 'collect_missing_fields') {
            // User is providing field values - parse as field data
            userProvidedFields = { userInput: trimmedInput };
          } else if (productSearchWorkflow.currentPhase === 'await_advanced_params' || productSearchWorkflow.currentPhase === 'advanced_specs_discovered') {
            // User is deciding whether to discover/add advanced specs
            // "yes" = discover/add advanced specs, "no" = skip
            if (wantsToContinue || wantsToAdd) {
              // Note: 'no' sets wantsToAdd to true
              userDecision = wantsToAdd ? 'no' : 'yes';
            }
          } else if (productSearchWorkflow.currentPhase === 'await_advanced_selection') {
            // User is selecting which advanced specs to add
            // Context: "Would you like to add any of these? Say 'all', select by name, or 'no' to skip"
            const wantsAll = /^(all|yes|everything|add all)$/i.test(normalizedInput);
            const wantsNone = /^(no|n|nope|skip|none|proceed)$/i.test(normalizedInput);

            if (wantsNone) {
              userDecision = 'no';  // Skip advanced specs, proceed to summary/analysis
            } else if (wantsAll) {
              userDecision = 'all';  // Add all discovered specs
            }
            // Otherwise, let the input pass through for spec name matching on the backend
          }

          // Call resumeProductSearch with user's decision
          const mergedRequirements = {
            ...(collectedData || {}),
            ...(userProvidedFields || {})
          };

          const result = await resumeProductSearch(
            productSearchWorkflow.threadId,
            userDecision,
            mergedRequirements, // Pass existing specs + new inputs to backend so it doesn't get lost
            userDecision ? undefined : trimmedInput, // Only pass as user_input if not a decision
            searchSessionId,
            productSearchWorkflow.currentPhase
          );

          console.log(`[${searchSessionId}] [AGENTIC_PS] Resume result:`, {
            success: result.success,
            currentPhase: result.current_phase,
            awaitingInput: result.awaiting_user_input,
            completed: result.completed
          });

          if (result.success) {
            // Display response
            if (result.sales_agent_response) {
              await streamAssistantMessage(result.sales_agent_response);
            }

            // Update workflow state
            if (result.awaiting_user_input) {
              setProductSearchWorkflow({
                threadId: result.thread_id,
                currentPhase: result.current_phase,
                awaitingUserInput: true,
                missingFields: result.missing_fields || []
              });

              // Update UI step
              if (result.current_phase === 'await_missing_fields' || result.current_phase === 'collect_missing_fields') {
                setCurrentStep('awaitMissingInfo');
              } else if (result.current_phase === 'await_advanced_params') {
                setCurrentStep('awaitAdditionalAndLatestSpecs');
              }
            } else {
              // Workflow complete or transitioned
              setProductSearchWorkflow({
                threadId: null,
                currentPhase: null,
                awaitingUserInput: false,
                missingFields: []
              });

              if (result.completed) {
                setCurrentStep('showSummary');

                // Display summary of collected requirements
                if (result.final_requirements) {
                  const reqs = result.final_requirements;
                  let summaryMessage = `\n\n**📋 Requirements Summary**\n\n`;
                  summaryMessage += `**Product Type:** ${reqs.productType || 'Not specified'}\n\n`;

                  // Mandatory requirements
                  const mandatory = reqs.mandatoryRequirements || {};
                  if (Object.keys(mandatory).length > 0) {
                    summaryMessage += `**Mandatory Requirements:**\n`;
                    for (const [key, value] of Object.entries(mandatory)) {
                      summaryMessage += `• ${key}: ${value}\n`;
                    }
                    summaryMessage += `\n`;
                  }

                  // Optional requirements
                  const optional = reqs.optionalRequirements || {};
                  if (Object.keys(optional).length > 0) {
                    summaryMessage += `**Optional Requirements:**\n`;
                    for (const [key, value] of Object.entries(optional)) {
                      summaryMessage += `• ${key}: ${value}\n`;
                    }
                    summaryMessage += `\n`;
                  }

                  // Advanced parameters
                  const advanced = reqs.advancedParameters || [];
                  if (advanced.length > 0) {
                    summaryMessage += `**Advanced Specifications:** ${advanced.length} discovered\n`;
                    for (const spec of advanced.slice(0, 5)) {
                      summaryMessage += `• ${spec.name || spec.key}\n`;
                    }
                    if (advanced.length > 5) {
                      summaryMessage += `• ... and ${advanced.length - 5} more\n`;
                    }
                  }

                  summaryMessage += `\n🔍 **Starting vendor analysis now...**`;
                  await streamAssistantMessage(summaryMessage);
                }

                // Auto-trigger final analysis with the collected requirements
                console.log(`[${searchSessionId}] [AGENTIC_PS] Workflow complete - auto-triggering performAnalysis()`);
                console.log(`[${searchSessionId}] [AGENTIC_PS] Passing final_requirements:`, result.final_requirements);

                await performAnalysis({
                  finalRequirements: result.final_requirements,
                  schema: result.schema
                });
              }
            }

            // Update schema display if updated
            if (result.schema) {
              const schemaForSidebar = {
                mandatoryRequirements: result.schema.mandatoryRequirements || {},
                optionalRequirements: result.schema.optionalRequirements || {},
                default: {
                  mandatory: result.schema.mandatoryRequirements || {},
                  optional: result.schema.optionalRequirements || {}
                }
              };
              setState((prev) => ({
                ...prev,
                requirementSchema: schemaForSidebar as any
              }));
            }

            // Handle ranked products if workflow is complete
            if (result.ranked_products && result.ranked_products.length > 0) {
              const analysisResult = {
                overallRanking: {
                  rankedProducts: result.ranked_products
                }
              };
              setState((prev) => ({ ...prev, analysisResult: analysisResult as any }));
              setIsRightDocked(false);
            }
          } else {
            // Error - show message
            await streamAssistantMessage(result.error || "Sorry, I encountered an error processing your response.");
          }

          setState((prev) => ({ ...prev, isLoading: false }));
          return; // Exit - we've handled the message through product search workflow
        }

        // ============================================================
        // NORMAL WORKFLOW FLOW - Intent classification + Flask endpoints
        // ============================================================

        // Step 0: Check workflow routing - determine if this should go to ProductInfo page
        // This handles chat-based queries (questions, greetings) by routing to ProductInfo
        const routingResult = await classifyRoute(trimmedInput, {
          current_step: currentStep,
          context: state.productType || undefined
        }, searchSessionId);  // Pass session ID for workflow isolation

        console.log('[WORKFLOW_ROUTING] Classification result:', {
          target_workflow: routingResult.target_workflow,
          intent: routingResult.intent,
          confidence: routingResult.confidence
        });

        // Route to EnGenie Chat page for chat-based queries (questions, greetings, product info)
        if (routingResult.target_workflow === 'engenie_chat') {
          console.log('[WORKFLOW_ROUTING] Routing to EnGenie Chat page for chat-based query');

          // Logic for window reuse based on query content
          let sessionKey: string;
          let windowName: string;

          if (lastChatSessionRef.current && lastChatSessionRef.current.query === trimmedInput) {
            // REUSE: Query is identical -> Reuse the existing session/window
            console.log('[WORKFLOW_ROUTING] Reusing existing chat window context');
            sessionKey = lastChatSessionRef.current.sessionKey;
            windowName = lastChatSessionRef.current.windowName;
          } else {
            // NEW: Different query -> New session, New window
            console.log('[WORKFLOW_ROUTING] Query changed -> Creating new chat window context');
            sessionKey = `engenie_context_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            windowName = `engenie_chat_window_${sessionKey}`; // Unique name

            // Update ref
            lastChatSessionRef.current = {
              query: trimmedInput,
              sessionKey,
              windowName
            };

            // Store context
            try {
              localStorage.setItem(sessionKey, JSON.stringify({ query: trimmedInput }));
              console.log('[WORKFLOW_ROUTING] Stored context for new window:', sessionKey);
            } catch (e) {
              console.error('[WORKFLOW_ROUTING] Failed to store context:', e);
            }
          }

          const productInfoUrl = `/chat?sessionKey=${sessionKey}`;
          const fullUrl = `${window.location.origin}${productInfoUrl}`;

          // Show a message with a clickable link that opens in a new tab
          await streamAssistantMessage(
            `This looks like a knowledge question about: **"${trimmedInput.substring(0, 80)}${trimmedInput.length > 80 ? '...' : ''}"**\n\n` +
            `I can help you with this in our **Chat** knowledge base.\n\n` +
            `<a href="${productInfoUrl}" target="${windowName}" rel="noopener noreferrer" style="display: inline-block; background: #0F6CBD; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 500; margin: 10px 0; box-shadow: 0 4px 15px rgba(15, 108, 189, 0.4);">💬 Open Chat Page</a>\n\n` +
            `_Alternatively, if you'd like to search for a specific product, please describe your requirements (e.g., "I need a pressure transmitter 0-100 PSI")._`
          );

          setState((prev) => ({ ...prev, isLoading: false }));
          return;
        }

        // Handle out-of-domain queries with rejection message
        if (routingResult.target_workflow === 'out_of_domain') {
          console.log('[WORKFLOW_ROUTING] Out-of-domain query detected');

          const rejectMessage = routingResult.reject_message ||
            "I'm EnGenie, your industrial automation assistant. I can help with instrument identification, product search, and standards compliance. Please ask a question related to industrial automation.";

          await streamAssistantMessage(rejectMessage);
          setState((prev) => ({ ...prev, isLoading: false }));
          return;
        }

        // ============================================================
        // SMART ROUTING: Solution ↔ Search Page Navigation
        // Using backend routing classifier's target_workflow
        // ============================================================

        const getCurrentPage = (): 'solution' | 'search' => {
          const path = window.location.pathname;
          if (path.includes('/solution')) return 'solution';
          if (path.includes('/search')) return 'search';
          return 'solution'; // default
        };

        // Map backend target_workflow to page names
        const targetPageMap: Record<string, 'solution' | 'search'> = {
          'solution': 'solution',
          'instrument_identifier': 'search'
        };

        const currentPage = getCurrentPage();
        const targetWorkflow = routingResult.target_workflow;
        const targetPage = targetPageMap[targetWorkflow];

        // Check if backend suggests different page than current
        if (targetPage && targetPage !== currentPage) {
          console.log(`[SMART_ROUTING] Backend detected target=${targetWorkflow}, current=${currentPage} → Suggesting ${targetPage}`);

          const sessionKey = `nav_context_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
          const targetUrl = `/${targetPage}?sessionKey=${sessionKey}`;
          const windowName = `${targetPage}_window_${sessionKey}`;

          // Store context
          try {
            localStorage.setItem(sessionKey, JSON.stringify({ query: trimmedInput }));
          } catch (e) {
            console.error('[SMART_ROUTING] Failed to store context:', e);
          }

          const message = targetPage === 'search'
            ? `This looks like a **simple product search** query.\n\n` +
            `For better results with single-product searches, I recommend using our **Search** page.\n\n` +
            `<a href="${targetUrl}" target="${windowName}" rel="noopener noreferrer" style="display: inline-block; background: #0F6CBD; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 500; margin: 10px 0; box-shadow: 0 4px 15px rgba(15, 108, 189, 0.4);">🔍 Open Search Page</a>\n\n` +
            `_Or, if you'd like to continue with a complete solution here, please describe your full system requirements._`
            : `This looks like a **complex solution** requiring multiple instruments.\n\n` +
            `For better results with multi-instrument systems, I recommend using our **Solution** page.\n\n` +
            `<a href="${targetUrl}" target="${windowName}" rel="noopener noreferrer" style="display: inline-block; background: #0F6CBD; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 500; margin: 10px 0; box-shadow: 0 4px 15px rgba(15, 108, 189, 0.4);">📋 Open Solution Page</a>\n\n` +
            `_Or, if you'd like a simple search instead, please specify a single product requirement._`;

          await streamAssistantMessage(message);
          setState((prev) => ({ ...prev, isLoading: false }));
          return;
        }

        // Step 1: Classify user intent (with session ID for isolation)
        // Refactored to use result from classifyRoute called above
        const intentResult: IntentClassificationResult = {
          intent: routingResult.intent as IntentClassificationResult['intent'],
          nextStep: routingResult.extracted_info?.next_step || null,
          resumeWorkflow: false,
          isSolution: routingResult.target_workflow === 'solution'
        };
        console.log('Intent classification result (mapped):', intentResult);

        // ════════════════════════════════════════════════════════════════════════
        // INSTRUMENT IDENTIFIER WORKFLOW - Direct Product Search
        // When target_workflow is 'instrument_identifier', call product-search API
        // directly instead of going through sales-agent conversation flow.
        // ════════════════════════════════════════════════════════════════════════
        if (routingResult.target_workflow === 'instrument_identifier') {
          console.log('[PRODUCT_SEARCH] Routing to /api/agentic/product-search for instrument_identifier');

          try {
            // Call product search workflow directly
            const searchResult = await callAgenticProductSearch(
              trimmedInput,
              undefined,           // threadId - new search
              searchSessionId,     // session ID
              undefined,           // productType - let backend extract
              'direct'             // source workflow
            );

            console.log('[PRODUCT_SEARCH] Result received:', {
              productType: searchResult.product_type,
              schemaFields: searchResult.schema ? Object.keys(searchResult.schema).length : 0,
              rankedProducts: searchResult.ranked_products?.length || 0,
              awaitingUserInput: searchResult.awaiting_user_input,
              currentPhase: searchResult.current_phase
            });

            // Update state with search results
            if (searchResult.product_type) {
              setState((prev) => ({
                ...prev,
                productType: searchResult.product_type,
                currentProductType: searchResult.product_type,
                requirementSchema: searchResult.schema || {}
              }));
            }

            // Update collected data with provided requirements
            if (searchResult.provided_requirements) {
              const flatRequirements = flattenRequirements(searchResult.provided_requirements);
              const mergedData = mergeRequirementsWithSchema(flatRequirements, searchResult.schema || {});
              setCollectedData(mergedData);
            }

            // Display the response message
            const responseMessage = searchResult.sales_agent_response ||
              `I found information about ${searchResult.product_type || 'your product'}. Let me analyze the requirements.`;
            await streamAssistantMessage(responseMessage);

            // Handle awaiting user input (HITL checkpoint)
            if (searchResult.awaiting_user_input) {
              setCurrentStep(searchResult.current_phase || 'awaitMissingInfo');
              // Store thread ID for resumption
              if (searchResult.thread_id) {
                setProductSearchWorkflow({
                  threadId: searchResult.thread_id,
                  awaitingUserInput: true,
                  currentPhase: searchResult.current_phase
                });
              }
            } else if (searchResult.completed || (searchResult.ranked_products && searchResult.ranked_products.length > 0)) {
              // Search complete - show results
              setCurrentStep('finalAnalysis');
              if (searchResult.ranked_products) {
                // Build AnalysisResult object in correct format for RightPanel
                const analysisResult = {
                  productType: searchResult.product_type || '',
                  vendorAnalysis: {
                    vendorMatches: searchResult.vendorAnalysis?.vendorMatches || [],
                    totalMatches: searchResult.vendorAnalysis?.totalMatches || searchResult.ranked_products.length
                  },
                  overallRanking: {
                    markdownAnalysis: null,
                    rankedProducts: searchResult.ranked_products
                  },
                  topRecommendation: searchResult.topRecommendation || searchResult.ranked_products[0]
                };

                console.log('[PRODUCT_SEARCH] Setting analysisResult for RightPanel:', {
                  productType: analysisResult.productType,
                  rankedProductsCount: analysisResult.overallRanking.rankedProducts.length,
                  vendorMatchesCount: analysisResult.vendorAnalysis.vendorMatches.length
                });

                setState((prev) => ({
                  ...prev,
                  analysisResult: analysisResult  // Use correct field name (singular)
                }));
              }
            }

            setState((prev) => ({ ...prev, isLoading: false }));
            return;

          } catch (error: any) {
            console.error('[PRODUCT_SEARCH] Error:', error);
            // Fall through to sales-agent flow as fallback
            await streamAssistantMessage(
              "I encountered an issue processing your search. Let me help you step by step."
            );
          }
        }

        // Handle knowledge questions (interrupts workflow)
        if (intentResult.intent === "knowledgeQuestion") {
          const agentResponse: AgentResponse = await callAgenticSalesAgent(
            currentStep,
            trimmedInput,
            {
              productType: state.productType,
              collectedData: collectedData
            },
            searchSessionId,
            "knowledgeQuestion"
          );

          await streamAssistantMessage(agentResponse.content);
          setState((prev) => ({ ...prev, isLoading: false }));
          // Keep current step unchanged for workflow resumption
          return;
        }

        // Step 2: Handle workflow based on intent and current step
        let targetStep = intentResult.nextStep || currentStep;
        let agentResponse: AgentResponse;

        // Special case: When intent API identifies nextStep as "showSummary",
        // directly call structure_requirements and then analyze, skipping sales agent response
        if (intentResult.nextStep === "showSummary") {
          setCurrentStep("showSummary");
          setState((prev) => ({ ...prev, isLoading: true }));
          try {
            // Directly call structure_requirements API
            if (!collectedData || Object.keys(collectedData).length === 0) {
              throw new Error("No collected data available for summary");
            }

            const requirementsString = collectedData.productType
              ? `Product Type: ${collectedData.productType}. ${composeUserDataString(collectedData)}`
              : composeUserDataString(collectedData);

            // Validate that we have something to send
            if (!requirementsString || requirementsString.trim().length === 0) {
              throw new Error("No requirements data to structure");
            }

            const structuredResponse = await structureRequirements(requirementsString);
            const summaryContent = structuredResponse.structuredRequirements;

            // Display the structured summary
            addMessage({ type: "assistant", content: `\n\n${summaryContent}\n\n`, role: undefined });

            // Directly call analyze API after structure_requirements
            await performAnalysis();

            setState((prev) => ({ ...prev, isLoading: false }));
            return;
          } catch (error) {
            console.error("Error in direct structure and analysis flow:", error);
            setState((prev) => ({ ...prev, isLoading: false }));
            // Fall through to normal flow if error occurs
          }
        }

        // Force initialInput if user provides product requirements, but only when
        // the conversation is starting or in a neutral state. Do NOT force a reset
        // when the user is already in `awaitAdditionalAndLatestSpecs`, `awaitAdvancedSpecs`, or
        // `awaitMissingInfo` — that causes an extra /validate call.
        if (intentResult.intent === "productRequirements") {
          const forceableSteps = ["greeting", "initialInput", "default"];
          if (forceableSteps.includes(currentStep)) {
            targetStep = "initialInput";
          } else {
            // Keep the current step to continue collecting optional/advanced info
            targetStep = currentStep;
          }
        }

        console.log('Target step:', targetStep);

        // Note: Do NOT shortcut to `showSummary` from the client.
        // Let the backend sales-agent workflow decide next steps for awaitAdvancedSpecs.

        switch (targetStep) {
          case "greeting": {
            agentResponse = await callAgenticSalesAgent(
              "greeting",
              trimmedInput,
              {},
              searchSessionId,
              "workflow"
            );
            await streamAssistantMessage(agentResponse.content);
            setCurrentStep("initialInput");
            break;
          }

          case "initialInput": {
            // Process product requirements using AGENTIC VALIDATION TOOL
            console.log(`[${searchSessionId}] [AGENTIC] Processing initialInput with validation tool`);
            try {
              // STEP 1: Call Agentic Validation Tool
              console.log(`[${searchSessionId}] [VALIDATION] Calling callAgenticValidate...`);
              const validationResult = await callAgenticValidate(
                trimmedInput,
                undefined, // no product type hint
                searchSessionId,
                true // enable PPI
              );

              if (!validationResult.productType) {
                console.error(`[${searchSessionId}] [VALIDATION] No product type detected`);
                agentResponse = await callAgenticSalesAgent("initialInput", "No product type detected.", {}, searchSessionId, "workflow");
                await streamAssistantMessage(agentResponse.content);
                setCurrentStep("initialInput");
                break;
              }

              console.log(`[${searchSessionId}] [VALIDATION] Product type: ${validationResult.productType}`);
              console.log(`[${searchSessionId}] [VALIDATION] Valid: ${validationResult.isValid}`);
              console.log(`[${searchSessionId}] [VALIDATION] Missing fields: ${validationResult.missingFields?.length || 0}`);

              // STEP 2: Generate AI Response using Sales Agent Tool
              console.log(`[${searchSessionId}] [SALES_AGENT] Generating response for validation result...`);
              const salesAgentResponse = await callAgenticSalesAgent(
                "initialInput",
                trimmedInput,
                {
                  product_type: validationResult.productType,
                  schema: validationResult.schema,
                  provided_requirements: validationResult.providedRequirements,
                  missing_fields: validationResult.missingFields,
                  is_valid: validationResult.isValid,
                  ppi_workflow_used: validationResult.ppiWorkflowUsed,
                  schema_source: validationResult.schemaSource
                },
                searchSessionId,
                "workflow"
              );

              console.log(`[${searchSessionId}] [SALES_AGENT] Response generated, nextStep: ${salesAgentResponse.nextStep}`);

              // Convert agentic format to legacy format for compatibility
              const schema = validationResult.schema;
              const flatRequirements = flattenRequirements(validationResult.providedRequirements);
              const mergedData = mergeRequirementsWithSchema(flatRequirements, schema);


              // STEP 3: Update state with validation results
              setCollectedData(mergedData);
              setState((prev) => ({
                ...prev,
                requirementSchema: schema,
                productType: validationResult.productType,
                currentProductType: validationResult.productType,
                validationResult: {
                  ...validationResult,
                  validationAlert: validationResult.missingFields && validationResult.missingFields.length > 0 ? {
                    message: salesAgentResponse.content,
                    missingFields: validationResult.missingFields
                  } : undefined
                },
              }));

              // STEP 3.5: Use field descriptions from validation result for on-hover tooltips
              // This avoids separate API calls as descriptions are now included in the validation response
              if (validationResult.fieldDescriptions && Object.keys(validationResult.fieldDescriptions).length > 0) {
                console.log(`[${searchSessionId}] [VALIDATION] Setting ${Object.keys(validationResult.fieldDescriptions).length} field descriptions for tooltips`);
                setFieldDescriptions(validationResult.fieldDescriptions);
              }

              // STEP 4: Display AI-generated response
              await streamAssistantMessage(salesAgentResponse.content);

              // Undock the sidebar to show schema
              setIsDocked(false);

              // STEP 5: Transition to next step based on validation result
              if (salesAgentResponse.nextStep) {
                setCurrentStep(salesAgentResponse.nextStep as any);
              } else if (!validationResult.isValid && validationResult.missingFields && validationResult.missingFields.length > 0) {
                // Missing fields - wait for user to provide them
                setCurrentStep("awaitMissingInfo");
              } else {
                // All fields provided - ask about advanced parameters
                setCurrentStep("awaitAdditionalAndLatestSpecs");
              }

              console.log(`[${searchSessionId}] [VALIDATION] Complete. Next step: ${salesAgentResponse.nextStep || currentStep}`);

            } catch (error) {
              console.error(`[${searchSessionId}] [VALIDATION] Error:`, error);
              agentResponse = await callAgenticSalesAgent("default", "Error during validation. Please try again.", {}, searchSessionId, "workflow");
              await streamAssistantMessage(agentResponse.content);
              setCurrentStep("initialInput");
            }
            break;
          }

          case "awaitMissingInfo": {
            // Handle missing mandatory fields collection step
            console.log(`[${searchSessionId}] [MISSING_INFO] Processing user response for missing fields`);
            try {
              // Call sales agent to handle missing info response
              const missingInfoResponse = await callAgenticSalesAgent(
                "awaitMissingInfo",
                trimmedInput,
                {
                  productType: state.productType,
                  schema: state.requirementSchema,
                  missingFields: state.validationResult?.missingFields || [],
                  collectedData: collectedData
                },
                searchSessionId,
                "workflow"
              );

              console.log(`[${searchSessionId}] [MISSING_INFO] Response received, nextStep: ${missingInfoResponse.nextStep}`);

              // Check if re-validation is needed (user provided new info)
              if (missingInfoResponse.requiresRevalidation && trimmedInput.trim()) {
                console.log(`[${searchSessionId}] [MISSING_INFO] Re-validating with new user input`);

                // Re-run validation with combined input
                const originalInput = state.validationResult?.originalInput || '';
                const combinedInput = `${originalInput} ${trimmedInput}`;

                const revalidationResult = await callAgenticValidate(
                  combinedInput,
                  state.productType || undefined,
                  searchSessionId,
                  true
                );

                console.log(`[${searchSessionId}] [MISSING_INFO] Re-validation complete. Missing fields: ${revalidationResult.missingFields?.length || 0}`);

                // Update state with new validation result
                const schema = revalidationResult.schema;
                const flatRequirements = flattenRequirements(revalidationResult.providedRequirements);
                const mergedData = mergeRequirementsWithSchema(flatRequirements, schema);

                setCollectedData(mergedData);
                setState((prev) => ({
                  ...prev,
                  requirementSchema: schema,
                  validationResult: {
                    ...revalidationResult,
                    originalInput: combinedInput
                  },
                }));

                // Update field descriptions from re-validation result for on-hover tooltips
                if (revalidationResult.fieldDescriptions && Object.keys(revalidationResult.fieldDescriptions).length > 0) {
                  console.log(`[${searchSessionId}] [MISSING_INFO] Updating ${Object.keys(revalidationResult.fieldDescriptions).length} field descriptions`);
                  setFieldDescriptions(revalidationResult.fieldDescriptions);
                }

                // Check if still missing fields
                if (revalidationResult.missingFields && revalidationResult.missingFields.length > 0) {
                  // Still missing fields - ask again
                  const formattedFields = revalidationResult.missingFields.slice(0, 5).join(", ");
                  const remaining = revalidationResult.missingFields.length > 5
                    ? ` and ${revalidationResult.missingFields.length - 5} more`
                    : "";

                  await streamAssistantMessage(
                    `Thank you! I've captured those details.\n\n` +
                    `There are still some specifications we need: **${formattedFields}${remaining}**\n\n` +
                    `Would you like to provide these, or type 'proceed' to continue?`
                  );
                  setCurrentStep("awaitMissingInfo");
                } else {
                  // All fields provided now
                  await streamAssistantMessage(
                    `Excellent! I've captured all the required specifications.\n\n` +
                    `Would you like to add any additional or advanced specifications?`
                  );
                  setCurrentStep("awaitAdditionalAndLatestSpecs");
                }
              } else {
                // Display response and follow nextStep
                await streamAssistantMessage(missingInfoResponse.content);

                if (missingInfoResponse.nextStep) {
                  setCurrentStep(missingInfoResponse.nextStep as any);
                }
              }

            } catch (error) {
              console.error(`[${searchSessionId}] [MISSING_INFO] Error:`, error);
              await streamAssistantMessage("I encountered an error processing your response. Please try again.");
              setCurrentStep("awaitMissingInfo");
            }
            break;
          }

          case "awaitAdditionalAndLatestSpecs": {
            // Handle the combined "Additional and Latest Specs" step
            try {
              const normalizedInput = trimmedInput.toLowerCase().trim();
              const isYes = /^(yes|y|yeah|yep|sure|ok|okay)$/i.test(normalizedInput);
              const isNo = /^(no|n|nope|skip)$/i.test(normalizedInput);

              // Always send to backend to handle yes/no logic and state tracking
              agentResponse = await callAgenticSalesAgent(
                "awaitAdditionalAndLatestSpecs",
                trimmedInput,
                { productType: state.productType, collectedData },
                searchSessionId,
                "workflow"
              );

              // Follow backend's nextStep decision
              if (agentResponse.nextStep) {
                if (agentResponse.nextStep === "awaitAdvancedSpecs") {
                  // User said YES - discover advanced parameters using AGENTIC WRAPPER
                  await streamAssistantMessage(agentResponse.content);

                  // STEP 1: Discover advanced parameters using the agentic wrapper API
                  console.log(`[${searchSessionId}] [AGENTIC] Discovering advanced parameters for: ${state.productType}`);
                  try {
                    const advancedParamsResult = await callAgenticAdvancedParameters(
                      state.productType!,
                      searchSessionId
                    );

                    console.log(`[${searchSessionId}] [ADVANCED_PARAMS] Discovered ${advancedParamsResult.totalUniqueSpecifications} parameters`);

                    // Store discovered parameters in state
                    setAdvancedParameters(advancedParamsResult);

                    // STEP 2: Generate AI response using Sales Agent Tool
                    console.log(`[${searchSessionId}] [SALES_AGENT] Generating response for advanced parameters...`);
                    const paramsResponse = await callAgenticSalesAgent(
                      "awaitAdvancedSpecs",
                      "Yes, show me advanced options",
                      {
                        product_type: state.productType,
                        schema: state.requirementSchema,
                        provided_requirements: collectedData,
                        discovered_parameters: advancedParamsResult.uniqueSpecifications || []
                      },
                      searchSessionId,
                      "workflow"
                    );

                    // STEP 3: Display AI-generated response with parameters
                    await streamAssistantMessage(paramsResponse.content);

                    // If no AI response, fall back to formatted list
                    if (!paramsResponse.content && advancedParamsResult.uniqueSpecifications && advancedParamsResult.uniqueSpecifications.length > 0) {
                      const paramsList = advancedParamsResult.uniqueSpecifications
                        .map((spec: any, idx: number) => `${idx + 1}. ${spec.name}`)
                        .join('\n');

                      const fallbackMessage = `\n\nI've discovered the following latest advanced specifications:\n\n${paramsList}\n\nWould you like to add any of these to your requirements? You can say "all", select specific ones by name, or say "no" to skip.`;
                      await streamAssistantMessage(fallbackMessage);
                    }

                  } catch (error) {
                    console.error(`[${searchSessionId}] [ADVANCED_PARAMS] Discovery failed:`, error);
                    await streamAssistantMessage("\n\nI had trouble discovering advanced parameters. Let's continue without them.");
                  }

                  setCurrentStep("awaitAdvancedSpecs");
                } else if (agentResponse.nextStep === "showSummary") {
                  // User said "no" - stream the transition message first
                  await streamAssistantMessage(agentResponse.content);
                  setCurrentStep("showSummary");
                  // Then proceed with summary - intro was already streamed
                  await handleShowSummaryAndProceed({ introAlreadyStreamed: true, skipAnalysis: true });
                } else {
                  // Other transitions - stream the message
                  await streamAssistantMessage(agentResponse.content);
                  setCurrentStep(agentResponse.nextStep as any);
                }
              } else {
                // Stay in current step if no nextStep provided - stream the message
                await streamAssistantMessage(agentResponse.content);
                setCurrentStep("awaitAdditionalAndLatestSpecs");
              }
            } catch (error) {
              console.error("Additional and latest specs error:", error);
              agentResponse = await callAgenticSalesAgent("awaitAdditionalAndLatestSpecs", "Error processing additional specifications.", {}, searchSessionId, "workflow");
              await streamAssistantMessage(agentResponse.content);
              setCurrentStep("awaitAdditionalAndLatestSpecs");
            }
            setState((prev) => ({ ...prev, isLoading: false }));
            break;
          }

          case "awaitAdvancedSpecs": {
            try {
              // Do not short-circuit skip/summary behavior on the client.
              // Always send the user's input to the backend sales-agent handler
              // and follow the `nextStep` it returns.
              const normalizedInput = trimmedInput.toLowerCase().replace(/\s/g, "");

              if (advancedParameters) {
                // If we already discovered parameters locally, attempt to parse selections
                // Convert the agentic format to the expected format
                const availableParams = advancedParameters.uniqueSpecifications?.map((spec: any) => spec.name || spec.key) ||
                  advancedParameters.uniqueParameters || [];

                const selectionResult = await addAdvancedParameters(
                  state.productType!,
                  trimmedInput,
                  availableParams
                );

                if (selectionResult.totalSelected > 0) {
                  const updatedData = { ...collectedData, ...selectionResult.selectedParameters };
                  setCollectedData(updatedData);
                  setSelectedAdvancedParams({ ...selectedAdvancedParams, ...selectionResult.selectedParameters });
                  console.log(`[${searchSessionId}] [ADVANCED_PARAMS] Selected ${selectionResult.totalSelected} parameters`);
                }

                // Let backend generate the response (including parameter list display)
                agentResponse = await callAgenticSalesAgent(
                  "awaitAdvancedSpecs",
                  trimmedInput,
                  {
                    productType: state.productType,
                    selectedParameters: selectionResult?.selectedParameters || {},
                    totalSelected: selectionResult?.totalSelected || 0,
                    availableParameters: availableParams
                  },
                  searchSessionId,
                  "workflow"
                );
                await streamAssistantMessage(agentResponse.content);

                if (agentResponse.nextStep === "showSummary") {
                  setCurrentStep("showSummary");
                  // If backend returned assistant content that we've streamed, set introAlreadyStreamed to true
                  const introAlreadyStreamed = !!agentResponse.content && agentResponse.content.trim().length > 0;
                  await handleShowSummaryAndProceed({ introAlreadyStreamed });
                } else if (agentResponse.nextStep) {
                  setCurrentStep(agentResponse.nextStep as any);
                }
              } else {
                // No parameters discovered yet, always ask backend what to do next
                agentResponse = await callAgenticSalesAgent(
                  "awaitAdvancedSpecs",
                  trimmedInput,
                  { productType: state.productType },
                  searchSessionId,
                  "workflow"
                );
                await streamAssistantMessage(agentResponse.content);

                if (agentResponse.nextStep === "showSummary") {
                  setCurrentStep("showSummary");
                  // If backend returned assistant content that we've streamed, set introAlreadyStreamed to true
                  const introAlreadyStreamed = !!agentResponse.content && agentResponse.content.trim().length > 0;
                  await handleShowSummaryAndProceed({ introAlreadyStreamed });
                  setState((prev) => ({ ...prev, isLoading: false }));
                  return;
                } else if (agentResponse.nextStep) {
                  setCurrentStep(agentResponse.nextStep as any);
                }
              }
            } catch (error) {
              console.error("Advanced parameters error:", error);
              agentResponse = await callAgenticSalesAgent("awaitAdvancedSpecs", "Error processing advanced parameters.", {}, searchSessionId, "workflow");
              await streamAssistantMessage(agentResponse.content);
            }
            setState((prev) => ({ ...prev, isLoading: false }));
            break;
          }

          case "showSummary": {
            // User is confirming to proceed with analysis after seeing summary
            console.log(`[${searchSessionId}] [SHOW_SUMMARY] User input: ${trimmedInput}`);
            try {
              // Call backend to handle the response
              const summaryResponse = await callAgenticSalesAgent(
                "showSummary",
                trimmedInput,
                {
                  productType: state.productType,
                  collectedData: collectedData,
                  schema: state.requirementSchema
                },
                searchSessionId,
                "workflow"
              );

              // Display the response
              await streamAssistantMessage(summaryResponse.content);

              // Check if backend says to trigger analysis
              if (summaryResponse.triggerAnalysis || summaryResponse.nextStep === "finalAnalysis") {
                console.log(`[${searchSessionId}] [SHOW_SUMMARY] Triggering final analysis`);
                setCurrentStep("finalAnalysis");
                await performAnalysis();
              } else if (summaryResponse.nextStep) {
                setCurrentStep(summaryResponse.nextStep as any);
              }
            } catch (error) {
              console.error(`[${searchSessionId}] [SHOW_SUMMARY] Error:`, error);
              // Fallback: Check if user confirmed with simple keywords
              const normalizedInput = trimmedInput.toLowerCase().replace(/\s/g, "");
              if (["yes", "proceed", "continue", "run", "analyze", "ok", "start"].some(cmd => normalizedInput.includes(cmd))) {
                await streamAssistantMessage("Starting product analysis...");
                setCurrentStep("finalAnalysis");
                await performAnalysis();
              } else {
                await streamAssistantMessage("Would you like to proceed with the product analysis? Type 'yes' or 'proceed' to continue.");
              }
            }
            break;
          }

          case "finalAnalysis": {
            // Handle rerun requests after analysis
            const normalizedInput = trimmedInput.toLowerCase().replace(/\s/g, "");
            if (["rerun", "run", "runagain"].some(cmd => normalizedInput.includes(cmd))) {
              await performAnalysis();
            }
            break;
          }

          case "analysisError": {
            const normalizedInput = trimmedInput.toLowerCase().replace(/\s/g, "");
            if (["rerun", "run", "runagain"].includes(normalizedInput)) {
              await performAnalysis();
            } else {
              agentResponse = await callAgenticSalesAgent("analysisError", "Please type 'rerun' to try again.", {}, searchSessionId, "workflow");
              await streamAssistantMessage(agentResponse.content);
            }
            break;
          }

          default: {
            // Handle missing info or general conversation
            if (currentStep === "awaitMissingInfo") {
              try {
                // Short-circuit confirmations like "yes", "y", "skip", "proceed", "continue" to allow skipping missing fields
                const shortConfirm = /^(yes|y|skip|proceed|continue)$/i.test(trimmedInput);
                // Check for "no" response
                const shortNo = /^(no|n|nope)$/i.test(trimmedInput);

                if (shortConfirm) {
                  // Let backend and LLM know user chose to skip providing missing mandatory info
                  const confirmationResponse = await callAgenticSalesAgent(
                    "confirmAfterMissingInfo",
                    "User confirmed to proceed without providing missing mandatory fields.",
                    { productType: state.productType, collectedData },
                    searchSessionId,
                    "workflow"
                  );
                  await streamAssistantMessage(confirmationResponse.content);
                  // Move to additional and latest specs step
                  setCurrentStep("awaitAdditionalAndLatestSpecs");
                } else if (shortNo) {
                  // User said "no" - they want to provide missing fields
                  // Get the missing fields from validation result
                  const missingFields = state.validationResult?.validationAlert?.missingFields || [];
                  const missingFieldsFormatted = missingFields
                    .map((f: string) => f.replace(/([A-Z])/g, ' $1').replace(/^./, (str: string) => str.toUpperCase()).trim())
                    .join(", ");

                  // Generate a response asking what they want to add
                  const noResponse = await callAgenticSalesAgent(
                    "askForMissingFields",
                    `User wants to provide missing fields: ${missingFieldsFormatted}`,
                    {
                      productType: state.productType,
                      missingFields: missingFieldsFormatted,
                      missingFieldsList: missingFields
                    },
                    searchSessionId,
                    "workflow"
                  );
                  await streamAssistantMessage(noResponse.content);
                  // Stay at awaitMissingInfo step - loop until user says yes
                  setCurrentStep("awaitMissingInfo");
                } else {
                  // User provided additional data - RE-VALIDATE using AGENTIC VALIDATION TOOL
                  console.log(`[${searchSessionId}] [AGENTIC] Re-validating with updated fields`);
                  const combinedInput = `${composeUserDataString(collectedData)} ${trimmedInput}`;

                  // STEP 1: Re-validate with agentic tool
                  const revalidationResult = await callAgenticValidate(
                    combinedInput,
                    state.validationResult?.productType,
                    searchSessionId,
                    true // enable PPI
                  );

                  console.log(`[${searchSessionId}] [REVALIDATION] Valid: ${revalidationResult.isValid}`);
                  console.log(`[${searchSessionId}] [REVALIDATION] Missing fields: ${revalidationResult.missingFields?.length || 0}`);

                  // STEP 2: Generate AI response for re-validation
                  const salesAgentResponse = await callAgenticSalesAgent(
                    revalidationResult.isValid ? "awaitAdditionalAndLatestSpecs" : "awaitMissingInfo",
                    trimmedInput,
                    {
                      product_type: revalidationResult.productType,
                      schema: revalidationResult.schema,
                      provided_requirements: revalidationResult.providedRequirements,
                      missing_fields: revalidationResult.missingFields,
                      is_valid: revalidationResult.isValid
                    },
                    searchSessionId,
                    "workflow"
                  );

                  // Update state
                  const newFlatRequirements = flattenRequirements(revalidationResult.providedRequirements);
                  const updatedData = mergeRequirementsWithSchema({ ...collectedData, ...newFlatRequirements }, state.requirementSchema!);

                  setCollectedData(updatedData);
                  setState((prev) => ({
                    ...prev,
                    validationResult: {
                      ...revalidationResult,
                      validationAlert: revalidationResult.missingFields && revalidationResult.missingFields.length > 0 ? {
                        message: salesAgentResponse.content,
                        missingFields: revalidationResult.missingFields
                      } : undefined
                    }
                  }));

                  // Update field descriptions from re-validation for on-hover tooltips
                  if (revalidationResult.fieldDescriptions && Object.keys(revalidationResult.fieldDescriptions).length > 0) {
                    console.log(`[${searchSessionId}] [REVALIDATION] Updating ${Object.keys(revalidationResult.fieldDescriptions).length} field descriptions`);
                    setFieldDescriptions(revalidationResult.fieldDescriptions);
                  }

                  // STEP 3: Display AI-generated response
                  await streamAssistantMessage(salesAgentResponse.content);

                  // STEP 4: Transition to next step
                  if (revalidationResult.isValid) {
                    // All required info is now provided
                    setCurrentStep(salesAgentResponse.nextStep as any || "awaitAdditionalAndLatestSpecs");
                  } else {
                    // Still missing some required info
                    setCurrentStep("awaitMissingInfo");
                  }
                }
              } catch (error) {
                console.error("Missing info processing error:", error);
                agentResponse = await callAgenticSalesAgent("default", "Error processing your input.", {}, searchSessionId, "workflow");
                await streamAssistantMessage(agentResponse.content);
              }
            } else {
              // Default conversation handling
              agentResponse = await callAgenticSalesAgent("default", trimmedInput, {}, searchSessionId, "workflow");
              await streamAssistantMessage(agentResponse.content);
            }
          }
        }

        setState((prev) => ({ ...prev, isLoading: false }));

      } catch (error) {
        console.error("Message handling error:", error);
        await streamAssistantMessage("I'm sorry, there was an error processing your message. Please try again.");
        setState((prev) => ({ ...prev, isLoading: false }));
      }
    },
    [currentStep, collectedData, state.productType, state.validationResult, state.requirementSchema, addMessage, performAnalysis, handleShowSummaryAndProceed, streamAssistantMessage, searchSessionId, productSearchWorkflow, isDirectSearch, handleRunProductSearch]
  );

  const setInputValue = useCallback((value: string) => {
    setState((prev) => ({ ...prev, inputValue: value }));
  }, []);

  const handleRetry = useCallback(() => performAnalysis(), [performAnalysis]);

  // Handle URL parameters for auto-filling input (but not auto-submitting)
  useEffect(() => {
    // Check for sessionStorage key first (for large inputs from Requirements page)
    const inputKey = searchParams.get('inputKey');
    let inputParam = searchParams.get('input');

    if (inputKey) {
      // Retrieve from sessionStorage and clean up
      const storedInput = sessionStorage.getItem(inputKey);
      if (storedInput) {
        inputParam = storedInput;
        sessionStorage.removeItem(inputKey); // Clean up after reading
      }
    }

    // Prefer prop-based initial input when provided (for embedded tabs)
    if (!inputParam && initialInput) {
      inputParam = initialInput;
    }

    if (inputParam && !hasAutoSubmitted) {
      // If a saved conversation already has messages (either from props or current state), do not auto-fill
      const hasSavedMessages = (savedMessages && savedMessages.length > 0) || (state.messages && state.messages.length > 0);

      if (hasSavedMessages) {
        // Avoid overwriting input or re-running search if we already have context
        console.log(`[${searchSessionId}] Skipping auto-run/fill because messages occur:`, {
          saved: savedMessages?.length,
          current: state.messages?.length
        });
        // Ensure we don't try again for this session
        setHasAutoSubmitted(true);
      } else {
        console.log(`[${searchSessionId}] ====== INITIAL INPUT PROCESSING ======`);
        console.log(`[${searchSessionId}] isDirectSearch: ${isDirectSearch}`);
        console.log(`[${searchSessionId}] inputParam: ${inputParam?.substring(0, 100)}...`);

        if (isDirectSearch && inputParam) {
          // DIRECT SEARCH: Auto-run the search immediately
          console.log(`[${searchSessionId}] ⚡ DIRECT SEARCH MODE - Auto-running search for: ${inputParam}`);

          handleRunProductSearch(inputParam);
          // Set state to clean after running
          setState((prev) => ({ ...prev, inputValue: "" }));
          setHasAutoSubmitted(true);
        } else {
          // NORMAL MODE: Just set the input value, let user decide when to submit
          console.log(`[${searchSessionId}] Normal mode - setting input value for user to submit`);
          setState((prev) => ({ ...prev, inputValue: inputParam }));
          setHasAutoSubmitted(true);
        }
      }
    }
  }, [searchParams, hasAutoSubmitted, initialInput, savedMessages, searchSessionId, isDirectSearch, handleRunProductSearch]);


  const handleScrollUpdate = useCallback(({ section, value }: { section: 'left' | 'center' | 'right'; value: number }) => {
    // 1. Update ref immediately (so component reads correct value if re-rendered)
    if (scrollPositionsRef.current) {
      scrollPositionsRef.current[section] = value;
    }

    // 2. Throttle notification to parent (to persist to DB)
    if (!scrollThrottleRef.current) {
      scrollThrottleRef.current = setTimeout(() => {
        scrollThrottleRef.current = null;
        // Trigger effect to notify parent
        setScrollUpdateCounter(c => c + 1);
      }, 1000); // 1 second throttle
    }
  }, []);

  return (

    <div
      className={`flex flex-col ${fillParent ? 'h-full' : 'h-screen'} text-foreground app-glass-gradient !min-h-0 ${fillParent ? '' : 'pt-24'}`}
      ref={containerRef}
    >
      {/* MAIN HEADER - Only show when not embedded in another page */}
      {!fillParent && <MainHeader onSave={onSave} />}


      {/* Left corner dock button - positioned below header */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-32 left-0 z-20 btn-glass-secondary border-0 shadow-lg rounded-l-none"
        onClick={() => setIsDocked(!isDocked)}
        aria-label={isDocked ? "Expand left panel" : "Collapse left panel"}
      >
        {isDocked ? <ChevronRight /> : <ChevronLeft />}
      </Button>

      {/* Right corner dock button - positioned below header (raise above panel content) */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-32 right-0 z-50 btn-glass-secondary border-0 shadow-lg rounded-r-none"
        onClick={() => setIsRightDocked(!isRightDocked)}
        aria-label={isRightDocked ? "Expand right panel" : "Collapse right panel"}
      >
        {isRightDocked ? <ChevronLeft /> : <ChevronRight />}
      </Button>

      <div className="flex flex-1 overflow-hidden">
        <div
          className="h-full flex flex-col relative transition-all duration-500 ease-in-out"
          style={{
            width: `${widths.left}%`,
            minWidth: widths.left === 0 ? "0%" : "7%",
            willChange: "width, opacity",
            overflow: "hidden",
            opacity: isDocked ? 0 : 1
          }}
        >
          <LeftSidebar
            validationResult={state.validationResult}
            requirementSchema={state.requirementSchema}
            currentProductType={state.currentProductType}
            collectedData={collectedData}
            logout={logout}
            isDocked={isDocked}
            setIsDocked={setIsDocked}
            hideProfile={true}
            fieldDescriptions={fieldDescriptions}
            onFieldDescriptionsChange={handleFieldDescriptionsChange}
            fetchOnHover={fetchOnHover}
            onScroll={(val) => handleScrollUpdate({ section: 'left', value: val })}
            initialScrollTop={scrollPositionsRef.current.left}
          />
        </div>

        {widths.left > 0 && (
          <div
            className={`w-1.5 cursor-col-resize transition-colors duration-500 ease-in-out ${draggingHandle === "left"
              ? "bg-[#5FB3E6]"
              : "bg-border hover:bg-[#5FB3E6]"
              }`}
            style={{ height: "100%", zIndex: 20 }}
            onMouseDown={(e) => handleMouseDown(e, "left")}
          />
        )}

        <div
          ref={centerScrollRef}
          onScroll={(e) => {
            scrollPositionsRef.current.center = e.currentTarget.scrollTop;
          }}
          className="h-full transition-all duration-500 ease-in-out overflow-auto flex flex-col glass-sidebar"
          style={{ width: `${100 - (widths.left > 0 ? widths.left : 0) - (widths.right > 0 ? widths.right : 0)}%`, minWidth: "5%", willChange: "width" }}
        >
          <ChatInterface
            messages={state.messages}
            onSendMessage={handleSendMessage}
            isLoading={state.isLoading}
            isStreaming={isStreaming}
            inputValue={state.inputValue}
            setInputValue={setInputValue}
            currentStep={currentStep}
            isValidationComplete={!!state.validationResult}
            productType={state.currentProductType}
            collectedData={collectedData}
            vendorAnalysisComplete={!!state.analysisResult}
            onRetry={handleRetry}
            searchSessionId={searchSessionId}
          />
        </div>

        {widths.right > 0 && (
          <div
            className={`w-1.5 cursor-col-resize transition-colors duration-500 ease-in-out ${draggingHandle === "right"
              ? "bg-[#5FB3E6]"
              : "bg-border hover:bg-[#5FB3E6]"
              }`}
            style={{ height: "100%", zIndex: 20 }}
            onMouseDown={(e) => handleMouseDown(e, "right")}
          />
        )}

        <div
          className="h-full transition-all duration-500 ease-in-out"
          style={{
            width: `${widths.right}%`,
            minWidth: widths.right === 0 ? "0%" : "7%",
            willChange: "width, opacity",
            overflow: "hidden",
            opacity: isRightDocked ? 0 : 1
          }}
        >
          <RightPanel
            analysisResult={state.analysisResult}
            productType={""}
            validationResult={undefined}
            requirementSchema={undefined}
            isDocked={isRightDocked}
            setIsDocked={setIsRightDocked}
            onPricingDataUpdate={handlePricingDataUpdate}
            identifiedItems={state.identifiedItems}
            onRunSearch={handleRunProductSearch}
            onPopulateInput={handlePopulateInput}
            onScroll={(val) => handleScrollUpdate({ section: 'right', value: val })}
            initialScrollTop={scrollPositionsRef.current.right}
          />
        </div>
      </div>
    </div>
  );
};

export default AIRecommender;
