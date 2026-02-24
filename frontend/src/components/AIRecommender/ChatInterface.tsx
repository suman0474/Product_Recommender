import { useRef, useEffect, useState, useCallback, memo } from "react";
import remarkGfm from 'remark-gfm';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Bot, User, CheckCircle, Loader2, AlertCircle, Info } from "lucide-react";
import { ChatMessage, AgenticCheckpointState, WorkflowType } from "./types";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import BouncingDots from './BouncingDots';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  isStreaming: boolean;
  inputValue: string;
  setInputValue: (value: string) => void;
  currentStep:
  | "greeting"
  | "initialInput"
  | "awaitMissingInfo"
  | "awaitAdditionalAndLatestSpecs"
  | "awaitAdvancedSpecs"
  | "confirmAfterMissingInfo"
  | "showSummary"
  | "finalConfirmation"
  | "finalAnalysis"
  | "analysisError"
  | "default";
  isValidationComplete: boolean;
  productType: string | null;
  collectedData: { [key: string]: string };
  vendorAnalysisComplete: boolean;
  onRetry: () => void;
  searchSessionId?: string; // Optional session ID for debugging

  // NEW: Agentic workflow support
  agenticState?: AgenticCheckpointState;
  workflowType?: WorkflowType;
  onRetryError?: () => void;
}

interface MessageRowProps {
  message: ChatMessage;
  isHistory: boolean;
  renderVendorAnalysisStatus: (message: ChatMessage) => React.ReactNode;
  formatTimestamp: (ts: any) => string;
}

// Memoized MessageRow component - only re-renders when message.id changes
// This prevents unnecessary re-renders when parent state updates
const MessageRow = memo(({ message, isHistory, renderVendorAnalysisStatus, formatTimestamp }: MessageRowProps) => {
  const [isVisible, setIsVisible] = useState(isHistory);

  useEffect(() => {
    if (!isHistory) {
      const delay = message.type === "user" ? 200 : 0;
      const timer = setTimeout(() => {
        setIsVisible(true);
      }, delay);
      return () => clearTimeout(timer);
    }
  }, [isHistory, message.type]);

  // Handler for action buttons (routing to other screens)
  const handleActionClick = async (action: any) => {
    if (action.action === 'openNewWindow' && action.url) {
      let targetUrl = action.url;
      let targetWindowName = '_blank';

      if (action.contextData && action.contextData.query) {
        const queryHash = btoa(action.contextData.query).substring(0, 16);
        const persistentSessionKey = `engenie_query_${queryHash}`;
        const existingMappingStr = localStorage.getItem(`window_mapping_${persistentSessionKey}`);

        let windowHandled = false;

        // ─── STEP 1: Check if an existing window is alive ───
        if (existingMappingStr) {
          try {
            const mapping = JSON.parse(existingMappingStr);
            const channel = new BroadcastChannel('engenie_channel');

            const status = await new Promise<{ isAlive: boolean; messageCount: number }>((resolve) => {
              let resolved = false;

              channel.onmessage = (event: MessageEvent) => {
                if (event.data.type === 'pong' && event.data.sessionKey === mapping.sessionKey) {
                  if (!resolved) {
                    resolved = true;
                    channel.close();
                    resolve({ isAlive: true, messageCount: event.data.messageCount ?? 0 });
                  }
                }
              };

              channel.postMessage({ type: 'ping', sessionKey: mapping.sessionKey });

              setTimeout(() => {
                if (!resolved) {
                  resolved = true;
                  channel.close();
                  resolve({ isAlive: false, messageCount: 0 });
                }
              }, 2000);
            });

            if (status.isAlive) {
              console.log(`[SEARCH] Window is ALIVE with ${status.messageCount} messages. Just focusing.`);
              targetUrl = mapping.url.split('?')[0] + '?sessionKey=' + encodeURIComponent(mapping.sessionKey);
              targetWindowName = mapping.windowName;
              windowHandled = true;
            } else {
              console.log('[SEARCH] Window is CLOSED. Will create fresh instance.');
            }
          } catch (e) {
            console.error('[SEARCH] Error checking window status:', e);
          }
        }

        // ─── STEP 2: No existing window → Open fresh & run ───
        if (!windowHandled) {
          const sessionKey = `${persistentSessionKey}_${Date.now()}`;
          const windowName = `engenie_window_${sessionKey}`;

          try {
            localStorage.setItem(sessionKey, JSON.stringify(action.contextData));

            const urlObj = new URL(action.url, window.location.origin);
            urlObj.searchParams.delete('query');
            urlObj.searchParams.set('sessionKey', sessionKey);
            targetUrl = urlObj.toString();

            localStorage.setItem(`window_mapping_${persistentSessionKey}`, JSON.stringify({
              url: targetUrl,
              windowName: windowName,
              query: action.contextData.query,
              sessionKey: sessionKey
            }));

            targetWindowName = windowName;
            console.log('[SEARCH] Opening NEW window:', targetUrl);
          } catch (e) {
            console.error('[SEARCH] Failed to store context:', e);
          }
        }
      }

      window.open(targetUrl, targetWindowName);
    } else if (action.action === 'navigate' && action.url) {
      window.location.href = action.url;
    }
  };

  return (
    <div
      className={`flex ${message.type === "user" ? "justify-end" : "justify-start"
        }`}
    >
      <div
        className={`max-w-[80%] flex items-start space-x-2 ${message.type === "user" ? "flex-row-reverse space-x-reverse" : ""
          }`}
      >
        <div
          className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center ${message.type === "user"
            ? "bg-transparent text-white"
            : "bg-transparent"
            }`}
        >
          {message.type === "user" ? (
            <img
              src="/icon-user-3d.png"
              alt="User"
              className="w-10 h-10 object-contain"
              loading="lazy"
            />
          ) : (
            <img
              src="/icon-engenie.png"
              alt="Assistant"
              className="w-14 h-14 object-contain"
              loading="lazy"
            />
          )}
        </div>
        <div className="flex-1">
          <div
            className={`break-words ${message.type === "user"
              ? "glass-bubble-user"
              : "glass-bubble-assistant"
              }`}
            style={{
              opacity: isVisible ? 1 : 0,
              transform: isVisible ? "scale(1)" : "scale(0.8)",
              transformOrigin: message.type === "user" ? "top right" : "top left",
              transition: "opacity 0.8s ease-out, transform 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275)"
            }}
          >
            <div  className="prose prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]} 
                components={{
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline hover:text-primary/80 font-medium"
                      onClick={(e) => {
                        // For internal links, use navigation instead of new tab
                        if (href && href.startsWith('/')) {
                          e.preventDefault();
                          window.open(href, '_blank');
                        }
                      }}
                    >
                      {children}
                    </a>
                  )
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
            {renderVendorAnalysisStatus(message)}

            {/* Render action buttons if present */}
            {message.actionButtons && message.actionButtons.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {message.actionButtons.map((btn: any, idx: number) => (
                  <button
                    key={idx}
                    onClick={() => handleActionClick(btn)}
                    className="px-4 py-2.5 rounded-lg font-semibold text-white text-sm transition-all duration-300 hover:scale-105 hover:shadow-lg active:scale-95 flex items-center gap-2"
                    style={{
                      background: '#0F6CBD',
                      boxShadow: '0 4px 15px rgba(15, 108, 189, 0.4)'
                    }}
                  >
                    {btn.icon && <span>{btn.icon}</span>}
                    {btn.label}
                  </button>
                ))}
              </div>
            )}
          </div>
          <p
            className={`text-xs text-muted-foreground mt-1 px-1 ${message.type === "user" ? "text-right" : ""
              }`}
            style={{
              opacity: isVisible ? 1 : 0,
              transition: "opacity 0.8s ease 0.3s"
            }}
          >
            {formatTimestamp(message.timestamp)}
          </p>
        </div>
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison: only re-render if message ID or content changes
  return prevProps.message.id === nextProps.message.id &&
    prevProps.message.content === nextProps.message.content &&
    prevProps.isHistory === nextProps.isHistory;
});

const ChatInterface = ({
  messages,
  onSendMessage,
  isLoading,
  isStreaming,
  inputValue,
  setInputValue,
  currentStep,
  isValidationComplete,
  productType,
  collectedData,
  vendorAnalysisComplete,
  onRetry,
  searchSessionId,
  // NEW: Agentic workflow props
  agenticState,
  workflowType = "flask",
  onRetryError,
}: ChatInterfaceProps) => {
  const { toast } = useToast();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isHistoryRef = useRef(true);

  // Set isHistory to false after initial mount so new messages animate
  useEffect(() => {
    // Small timeout to ensure first render completes before we switch mode
    const timer = setTimeout(() => {
      isHistoryRef.current = false;
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  const [activeDescription, setActiveDescription] = useState<string | null>(null);
  const [showThinking, setShowThinking] = useState(false);

  useEffect(() => {
    if (isLoading) {
      const timer = setTimeout(() => {
        setShowThinking(true);
      }, 600);
      return () => clearTimeout(timer);
    } else {
      setShowThinking(false);
    }
  }, [isLoading]);

  // Optimized scroll - only scroll on new messages, use 'auto' for performance
  useEffect(() => {
    if (messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [messages.length]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "40px";
      textareaRef.current.style.height = `${Math.max(40, textareaRef.current.scrollHeight)}px`;
    }
  }, [inputValue]);

  // Memoized utility functions to prevent unnecessary re-renders
  const prettifyRequirement = useCallback((req: string) =>
    req
      .replace(/\_/g, " ")
      .replace(/-/g, " ")
      .replace(/\b\w/g, (l) => l.toUpperCase()),
    []);
  // ... existing code ...
  {
    showThinking && !isStreaming && (
      <div className="flex justify-start">
        <div className={`max-w-[80%] flex items-start space-x-2`}>
          <div className="flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center bg-transparent">
            <img
              src="/icon-engenie.png"
              alt="Assistant"
              className="w-14 h-14 object-contain"
            />
          </div>
          <div className="p-3 rounded-lg">
            <BouncingDots />
          </div>
        </div>
      </div>
    )
  }

  // Memoized timestamp formatter
  const formatTimestamp = useCallback((ts: any) => {
    if (!ts) return "";
    // If already a Date
    if (ts instanceof Date) {
      try {
        return ts.toLocaleTimeString();
      } catch (e) {
        return ts.toString();
      }
    }

    // If numeric (epoch ms)
    if (typeof ts === "number") {
      const d = new Date(ts);
      return isNaN(d.getTime()) ? String(ts) : d.toLocaleTimeString();
    }

    // If ISO/string, try to parse
    if (typeof ts === "string") {
      const parsed = Date.parse(ts);
      if (!isNaN(parsed)) {
        return new Date(parsed).toLocaleTimeString();
      }

      // If string isn't parseable, return as-is (fallback)
      return ts;
    }

    // Unknown type: fallback to string
    try {
      return String(ts);
    } catch (e) {
      return "";
    }
  }, []);

  const handleSend = () => {
    const trimmedInput = inputValue.trim();
    if (!trimmedInput) {
      toast({
        title: "Input required",
        description: "Please enter your data before sending.",
        variant: "destructive",
      });
      return;
    }
    if (isLoading) return;

    // Reset textarea height to initial height before sending
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    onSendMessage(trimmedInput);
    setInputValue("");
    setActiveDescription(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };



  const handleSampleClick = (sampleText: string) => {
    setInputValue(sampleText);
    textareaRef.current?.focus();
    setActiveDescription(null);
  };

  const handleInteractiveButtonClick = (label: string) => {
    setActiveDescription((current) => (current === label ? null : label));
  };

  // Memoized vendor analysis status renderer
  const renderVendorAnalysisStatus = useCallback((message: ChatMessage) => {
    if (message.metadata?.vendorAnalysisComplete) {
      return (
        <div className="mt-3 p-4 rounded-lg bg-ai-primary/5 border border-ai-primary/20 space-y-2 shadow-inner">
          <h4 className="font-semibold text-ai-primary mb-1 flex items-center">
            <CheckCircle className="h-4 w-4 mr-2" /> Vendor Analysis Complete
          </h4>
          <p className="text-sm text-muted-foreground">
            Detailed results are displayed in the right panel.
          </p>
        </div>
      );
    }
    return null;
  }, []);

  // NEW: Render agentic checkpoint-specific UI
  const renderAgenticCheckpoint = (checkpoint: string) => {
    if (workflowType !== "agentic" || !agenticState) return null;

    switch (checkpoint) {
      case "greeting":
        return (
          <div className="mt-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-xs text-blue-700 flex items-center">
              <Info className="h-3 w-3 mr-1" />
              Welcome! Let's find the perfect industrial product for you.
            </p>
          </div>
        );

      case "initialInput":
        return agenticState.productType ? (
          <div className="mt-2 p-3 bg-green-50 rounded-lg border border-green-200">
            <p className="text-xs text-green-700 flex items-center">
              <CheckCircle className="h-3 w-3 mr-1" />
              Product type detected: <strong className="ml-1">{agenticState.productType}</strong>
            </p>
          </div>
        ) : null;

      case "awaitMissingInfo":
        return (
          <div className="mt-2 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
            <p className="text-xs font-medium text-yellow-800 mb-1">
              📋 Missing Information
            </p>
            <p className="text-xs text-yellow-700">
              A few more details will help me find the best products.
            </p>
          </div>
        );

      case "awaitAdvancedSpecs":
        return agenticState.availableAdvancedParams && agenticState.availableAdvancedParams.length > 0 ? (
          <div className="mt-2 p-3 bg-purple-50 rounded-lg border border-purple-200">
            <p className="text-xs font-medium text-purple-800 mb-2">
              🔍 Advanced Parameters Available ({agenticState.availableAdvancedParams.length})
            </p>
            <div className="space-y-0.5">
              {agenticState.availableAdvancedParams.slice(0, 5).map((param: any, idx: number) => (
                <div key={idx} className="text-xs text-purple-600">
                  • {param.name || param}
                </div>
              ))}
              {agenticState.availableAdvancedParams.length > 5 && (
                <div className="text-xs text-purple-500 italic mt-1">
                  +{agenticState.availableAdvancedParams.length - 5} more parameters...
                </div>
              )}
            </div>
          </div>
        ) : null;

      case "showSummary":
        return (
          <div className="mt-2 p-3 bg-indigo-50 rounded-lg border border-indigo-200">
            <p className="text-xs font-medium text-indigo-800 mb-1">
              📊 Requirements Summary
            </p>
            <p className="text-xs text-indigo-700">
              Please review and confirm before I search for products.
            </p>
          </div>
        );

      case "analysisError":
        return (
          <div className="mt-2 p-3 bg-red-50 rounded-lg border border-red-200">
            <p className="text-xs font-medium text-red-800 mb-2 flex items-center">
              <AlertCircle className="h-3 w-3 mr-1" />
              Analysis Error
            </p>
            {agenticState.errorMessage && (
              <p className="text-xs text-red-700 mb-2">{agenticState.errorMessage}</p>
            )}
            {onRetryError && (
              <Button
                onClick={onRetryError}
                size="sm"
                variant="outline"
                className="h-6 text-xs border-red-300 text-red-700 hover:bg-red-100"
              >
                Retry Analysis
              </Button>
            )}
          </div>
        );

      case "knowledgeQuestion":
        return (
          <div className="mt-2 p-3 bg-teal-50 rounded-lg border border-teal-200">
            <p className="text-xs font-medium text-teal-800 flex items-center">
              <Info className="h-3 w-3 mr-1" />
              Question Answered - Let's continue...
            </p>
          </div>
        );

      default:
        return null;
    }
  };

  // NEW: Render workflow status bar
  const renderWorkflowStatus = () => {
    if (workflowType === "agentic" && agenticState) {
      return (
        <div className="p-2 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg mb-3">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2">
              <div className={`w-1.5 h-1.5 rounded-full ${agenticState.awaitingUserInput ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'
                }`} />
              <span className="font-medium text-gray-700">
                {agenticState.awaitingUserInput ? 'Awaiting Your Response' : 'Processing...'}
              </span>
            </div>
            <span className="text-gray-500">
              {agenticState.currentStep}
            </span>
          </div>
          {agenticState.threadId && (
            <div className="mt-1 text-xs text-gray-400">
              Thread: {agenticState.threadId.substring(0, 12)}...
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  const getPlaceholderText = () => {
    if (isLoading) {
      return "Thinking...";
    }
    switch (currentStep) {
      case "initialInput":
        return "";
      case "awaitMissingInfo":
        return "";
      case "awaitAdditionalAndLatestSpecs":
        return "";
      case "awaitAdvancedSpecs":
        return "";
      case "showSummary":
      case "analysisError":
        return "";
      case "finalAnalysis":
        return "";
      default:
        return "Send a message...";
    }
  };

  const sampleInputs = [
    {
      label: "Pressure Transmitter",
      text: "I am looking for a very specific pressure transmitter. The required performance includes a tight pressure range of -10 to 10 inH2O and a high standard accuracy of 0.035% of span. For system integration, the device must provide a 4-20mA with HART output signal. In terms of materials, the process-wetted parts must be compatible with Hastelloy C-276, and it should feature a 1/4-18 NPT process connection.",
    },
    {
      label: "Temperature Transmitter",
      text: "We are looking for a high-performance temperature transmitter suitable for a critical process monitoring application. The unit must be compatible with a Pt100 RTD sensor and provide a high degree of accuracy, specifically ±0.10 °C. For integration with our current system, it needs to have a 4-20 mA output signal with HART protocol. The physical installation requires a rugged stainless steel housing and the ability to be pipe-mounted. Most importantly, the transmitter must meet our stringent safety standards, which requires both a SIL 3 certification and an ATEX rating for use in potentially hazardous areas.",
    },
    {
      label: "Humidity Transmitter",
      text: "I am looking for a humidity transmitter with a 0-10V output. The measurement range should be 0-100% RH and it needs to be wall-mountable.",
    },
  ];

  return (
    <div className="flex-1 flex flex-col h-full bg-transparent relative">
      {/* Debug session indicator - can be removed in production
      {searchSessionId && (
        <div className="text-xs text-gray-500 px-4 py-1 bg-gray-50 border-b">
          Session: {searchSessionId.slice(-8)}
        </div>
      )} */}
      <div className="flex-none py-0 border-b border-white/10 bg-transparent z-20 flex justify-center items-center">
        <div className="flex items-center gap-1">
          <div className="flex items-center justify-center">
            <img
              src="/icon-engenie.png"
              alt="EnGenie"
              className="w-16 h-16 object-contain"
            />
          </div>
          <h1 className="text-3xl font-bold text-[#0f172a] inline-flex items-center gap-2 whitespace-nowrap">
            EnGenie <span className="text-sky-500">♦</span> Search
          </h1>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-no-scrollbar pb-32">
        {/* NEW: Workflow status bar for agentic workflow */}
        {renderWorkflowStatus()}

        {messages.length === 0 ? (
          <div className="text-center p-6">
            {/* Empty state */}
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={message.id}>
              <MessageRow
                message={message}
                isHistory={isHistoryRef.current}
                renderVendorAnalysisStatus={renderVendorAnalysisStatus}
                formatTimestamp={formatTimestamp}
              />

              {/* NEW: Render agentic checkpoint UI after assistant messages */}
              {message.type === "assistant" &&
                workflowType === "agentic" &&
                agenticState &&
                index === messages.length - 1 && (
                  <div className="ml-14">
                    {renderAgenticCheckpoint(agenticState.currentStep)}
                  </div>
                )}
            </div>
          ))
        )}

        {showThinking && !isStreaming && (
          <div className="flex justify-start">
            <div className={`max-w-[80%] flex items-start space-x-2`}>
              <div className="flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center bg-transparent">
                <img
                  src="/icon-engenie.png"
                  alt="Assistant"
                  className="w-14 h-14 object-contain"
                />
              </div>
              <div className="p-3 rounded-lg">
                <BouncingDots />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {activeDescription && (
        <div
          className="p-4 bg-secondary/30 rounded border border-border text-sm text-muted-foreground max-w-2xl mx-auto mb-4 cursor-pointer hover:bg-secondary/50 transition"
          onClick={() =>
            handleSampleClick(
              sampleInputs.find(({ label }) => label === activeDescription)
                ?.text || ""
            )
          }
        >
          <p>
            {sampleInputs.find(({ label }) => label === activeDescription)?.text}
          </p>
        </div>
      )}

      {/*
      {messages.length === 0 && (
        <div className="flex flex-wrap justify-center items-center gap-2 space-x-2 p-2 border-t border-border bg-background">
          {sampleInputs.map(({ label }) => (
            <Button
              key={label}
              variant={activeDescription === label ? "default" : "outline"}
              onClick={() => handleInteractiveButtonClick(label)}
              className="min-w-[150px]"
              type="button"
            >
              {label}
            </Button>
          ))}
        </div>
      )}
      */}

      <div className="fixed bottom-0 left-0 right-0 p-4 bg-transparent z-30 pointer-events-none">
        <div className="max-w-4xl mx-auto px-2 md:px-8 pointer-events-auto">
          <form onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
            <div className="relative group">
              <div className={`relative w-full rounded-[26px] transition-all duration-300 focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-transparent hover:scale-[1.02]`}
                style={{
                  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
                  WebkitBackdropFilter: 'blur(12px)',
                  backdropFilter: 'blur(12px)',
                  backgroundColor: '#ffffff',
                  border: '1px solid rgba(255, 255, 255, 0.4)',
                  color: 'rgba(0, 0, 0, 0.8)'
                }}>
                <textarea
                  ref={textareaRef}
                  placeholder="Type your message here..."
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement;
                    target.style.height = 'auto';
                    target.style.height = `${Math.min(target.scrollHeight, 150)}px`;
                  }}
                  className="w-full bg-transparent border-0 focus:ring-0 focus:outline-none px-4 py-4 pr-20 text-sm resize-none min-h-[64px] max-h-[150px] leading-relaxed flex items-center custom-no-scrollbar"
                  style={{
                    fontSize: '16px',
                    fontFamily: 'inherit',
                    boxShadow: 'none',
                    overflowY: 'auto'
                  }}
                  disabled={isLoading}
                />

                {/* Action Button - positioned like Solution page */}
                <div className="absolute bottom-1.5 right-1.5 flex items-center gap-0.5">
                  <Button
                    type="submit"
                    disabled={!inputValue.trim() || isLoading}
                    className={`w-8 h-8 p-0 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-transparent ${!inputValue.trim() ? 'text-muted-foreground' : 'text-primary hover:scale-110'}`}
                    variant="ghost"
                    size="icon"
                    title="Submit"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
