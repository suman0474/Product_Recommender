import { useRef, useState, useEffect, KeyboardEvent, FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Send, Loader2, Database, Sparkles, AlertCircle, X, Save, LogOut, User, FileText, FolderOpen } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useNavigate, useSearchParams } from "react-router-dom";
import axios from "axios";
import ReactMarkdown from 'react-markdown';
import BouncingDots from '@/components/AIRecommender/BouncingDots';
import { BASE_URL, classifyRoute } from "@/components/AIRecommender/api";
import { useAuth } from '@/contexts/AuthContext';
import MainHeader from "@/components/MainHeader";
import { useMemo } from 'react';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
    DropdownMenu,
    DropdownMenuTrigger,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuItem,
    DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// Helper to extract first N words from input
const extractFirstWords = (input: string, count: number = 2): string => {
    if (!input || typeof input !== 'string') return 'Chat';

    // Clean and split the input
    const words = input.trim().split(/\s+/).filter(word => word.length > 0);

    if (words.length === 0) return 'Chat';

    // Take first N words and capitalize first letter of each
    const firstWords = words.slice(0, count).map(word =>
        word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');

    return firstWords || 'Chat';
};

// Escape string for use in RegExp
const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Compute next available duplicate name
const computeNextDuplicateName = (base: string, projects: any[]) => {
    if (!base) return `${base} (1)`;
    const baseTrim = base.trim();

    // Extract the actual base name without any numbering
    const baseNameMatch = baseTrim.match(/^(.*?)(?:\s*\(\d+\))?$/);
    const actualBaseName = baseNameMatch ? baseNameMatch[1].trim() : baseTrim;

    // Create regex to match all variations of the base name with numbers
    const regex = new RegExp(`^${escapeRegExp(actualBaseName)}(?:\\s*\\((\\d+)\\))?$`, 'i');
    let maxNum = 0;

    projects.forEach((p: any) => {
        const pName = (p.projectName || p.project_name || '').trim();
        if (!pName) return;

        const match = pName.match(regex);
        if (match) {
            const num = match[1] ? parseInt(match[1], 10) : 0;
            if (num > maxNum) maxNum = num;
        }
    });

    return `${actualBaseName} (${maxNum + 1})`;
};

// Types for RAG response
interface RAGResponse {
    success: boolean;
    answer: string;
    source: "database" | "llm" | "pending_confirmation" | "user_declined" | "unknown";
    foundInDatabase: boolean;
    awaitingConfirmation: boolean;
    sourcesUsed: string[]
    resultsCount?: number;
    note?: string;
    error?: string;
}

// Action button interface for chat messages
interface ChatActionButton {
    label: string;
    action: 'openNewWindow' | 'navigate' | 'custom';
    url?: string;
    icon?: string;
    contextData?: any;
}

interface ChatMessage {
    id: string;
    type: "user" | "assistant";
    content: string;
    source?: string;
    sourcesUsed?: string[];
    awaitingConfirmation?: boolean;
    timestamp: Date;
    actionButtons?: ChatActionButton[];  // NEW: Routing buttons
}

// UI Labels
interface UILabels {
    loadingText: string;
    confirmationHint: string;
    inputPlaceholder: string;
    sourceDatabase: string;
    sourceLlm: string;
    sourcePending: string;
    errorMessage: string;
}

// MessageRow component with animations
interface MessageRowProps {
    message: ChatMessage;
    isHistory: boolean;
    uiLabels: UILabels;
}

const MessageRow = ({ message, isHistory, uiLabels }: MessageRowProps) => {
    const [isVisible, setIsVisible] = useState(isHistory);

    useEffect(() => {
        if (!isHistory) {
            const delay = message.type === 'user' ? 200 : 0;
            const timer = setTimeout(() => {
                setIsVisible(true);
            }, delay);
            return () => clearTimeout(timer);
        }
    }, [isHistory, message.type]);

    const formatTimestamp = (ts: Date) => {
        try {
            return ts.toLocaleTimeString();
        } catch {
            return '';
        }
    };

    // Handler for action buttons (routing to other screens)
    const handleActionClick = async (action: ChatActionButton) => {
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
                            console.log(`[CHAT] Window is ALIVE with ${status.messageCount} messages. Just focusing.`);
                            targetUrl = mapping.url.split('?')[0] + '?sessionKey=' + encodeURIComponent(mapping.sessionKey);
                            targetWindowName = mapping.windowName;
                            windowHandled = true;
                        } else {
                            console.log('[CHAT] Window is CLOSED. Will create fresh instance.');
                        }
                    } catch (e) {
                        console.error('[CHAT] Error checking window status:', e);
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
                        console.log('[CHAT] Opening NEW window:', targetUrl);
                    } catch (e) {
                        console.error('[CHAT] Failed to store context:', e);
                    }
                }
            }

            window.open(targetUrl, targetWindowName);
        } else if (action.action === 'navigate' && action.url) {
            window.location.href = action.url;
        }
    };

    return (
        <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] flex items-start space-x-2 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center ${message.type === 'user' ? 'bg-transparent text-white' : 'bg-transparent'}`}>
                    {message.type === 'user' ? (
                        <img src="/icon-user-3d.png" alt="User" className="w-10 h-10 object-contain" />
                    ) : (
                        <img src="/icon-engenie.png" alt="Assistant" className="w-14 h-14 object-contain" />
                    )}
                </div>

                <div className="flex-1">
                    <div
                        className={`break-words ${message.type === 'user' ? 'glass-bubble-user' : 'glass-bubble-assistant'}`}
                        style={{
                            opacity: isVisible ? 1 : 0,
                            transform: isVisible ? 'scale(1)' : 'scale(0.8)',
                            transformOrigin: message.type === 'user' ? 'top right' : 'top left',
                            transition: 'opacity 0.8s ease-out, transform 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
                        }}
                    >
                        <div>
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>

                        {message.awaitingConfirmation && (
                            <div className="mt-2 pt-2 border-t border-yellow-200/50 text-xs text-yellow-700">
                                💡 {uiLabels.confirmationHint}
                            </div>
                        )}

                        {/* Render action buttons if present */}
                        {message.actionButtons && message.actionButtons.length > 0 && (
                            <div className="mt-4 flex flex-wrap gap-2">
                                {message.actionButtons.map((btn, idx) => (
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
                        className={`text-xs text-muted-foreground mt-1 px-1 ${message.type === 'user' ? 'text-right' : ''}`}
                        style={{
                            opacity: isVisible ? 1 : 0,
                            transition: 'opacity 0.8s ease 0.3s'
                        }}
                    >
                        {formatTimestamp(message.timestamp)}
                    </p>
                </div>
            </div>
        </div>
    );
};

// Default UI labels
const DEFAULT_UI_LABELS: UILabels = {
    loadingText: "Searching database...",
    confirmationHint: "Type 'Yes' for AI answer, or 'No' to skip",
    inputPlaceholder: "Type your message here...",
    sourceDatabase: "From Database",
    sourceLlm: "From AI Knowledge",
    sourcePending: "Awaiting Your Response",
    errorMessage: "Sorry, something went wrong. Please try again."
};

// Persistent storage setup happens via hook now

const EnGenieChat = () => {
    const { toast } = useToast();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { user, logout } = useAuth();

    const [inputValue, setInputValue] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [showThinking, setShowThinking] = useState(false);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [sessionId, setSessionId] = useState(() => `engenie_chat_${Date.now()}`);
    const [hasAutoSubmitted, setHasAutoSubmitted] = useState(false);
    const [isRestoring, setIsRestoring] = useState(true); // Track restoration status
    const [isHistory, setIsHistory] = useState(false);
    const [isDataReady, setIsDataReady] = useState(false); // Explicit data ready flag
    const [uiLabels] = useState<UILabels>(DEFAULT_UI_LABELS);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Project saving states
    const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
    const [duplicateNameDialogOpen, setDuplicateNameDialogOpen] = useState(false);
    const [duplicateProjectName, setDuplicateProjectName] = useState<string | null>(null);
    const [autoRenameSuggestion, setAutoRenameSuggestion] = useState<string | null>(null);
    const [duplicateDialogNameInput, setDuplicateDialogNameInput] = useState('');
    const [duplicateDialogError, setDuplicateDialogError] = useState<string | null>(null);
    const handleSaveProjectRef = useRef<any>(null);

    const stateRef = useRef({ messages: [] as ChatMessage[], sessionId: '' });
    const lastSessionKeyRef = useRef<string | null>(null);
    const lastProcessedTsRef = useRef<string | null>(null); // Track last processed timestamp for re-click detection

    useEffect(() => {
        stateRef.current = { messages, sessionId };
    }, [messages, sessionId]);

    // IndexedDB configuration for persisting Chat state
    const CHAT_DB_NAME = 'chat_db';
    const CHAT_STORE_NAME = 'chat_state';
    const CHAT_BACKUP_KEY = 'chat_state_backup';
    // CHAT_STATE_KEY is now dynamic based on sessionKey

    // Helper function to open IndexedDB
    const openChatDB = (): Promise<IDBDatabase> => {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(CHAT_DB_NAME, 1);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);

            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(CHAT_STORE_NAME)) {
                    db.createObjectStore(CHAT_STORE_NAME, { keyPath: 'id' });
                }
            };
        });
    };

    // Helper function to get the correct storage key
    // Uses the sessionKey from URL as the primary unique identifier for this window's context
    const getStorageKey = (): string => {
        const params = new URLSearchParams(window.location.search);
        const urlSessionKey = params.get('sessionKey');
        return urlSessionKey || 'current_session';
    };

    // Helper function to save state to IndexedDB
    const saveStateToChatDB = async (state: any): Promise<void> => {
        try {
            const db = await openChatDB();
            const transaction = db.transaction(CHAT_STORE_NAME, 'readwrite');
            const store = transaction.objectStore(CHAT_STORE_NAME);
            const key = getStorageKey();

            await new Promise<void>((resolve, reject) => {
                const request = store.put({ id: key, ...state });
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            db.close();
            // console.log(`[CHAT] Saved state to DB with key: ${key}`);
        } catch (e) {
            console.warn('[CHAT] Failed to save to IndexedDB:', e);
        }
    };

    // Helper function to load state from IndexedDB
    const loadStateFromChatDB = async (): Promise<any | null> => {
        try {
            const db = await openChatDB();
            const transaction = db.transaction(CHAT_STORE_NAME, 'readonly');
            const store = transaction.objectStore(CHAT_STORE_NAME);
            const key = getStorageKey();

            const result = await new Promise<any>((resolve, reject) => {
                const request = store.get(key);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            db.close();

            if (result) {
                console.log(`[CHAT] Loaded state for key: ${key}`);
                // Restore Date objects for chat messages if needed
                if (result.messages) {
                    result.messages = result.messages.map((msg: any) => ({
                        ...msg,
                        timestamp: msg.timestamp ? new Date(msg.timestamp) : undefined
                    }));
                }
                return result;
            }
            return null;
        } catch (e) {
            console.warn('[CHAT] Failed to load from IndexedDB:', e);
            return null;
        }
    };

    // Helper function to clear IndexedDB state
    const clearChatDBState = async (): Promise<void> => {
        try {
            const db = await openChatDB();
            const transaction = db.transaction(CHAT_STORE_NAME, 'readwrite');
            const store = transaction.objectStore(CHAT_STORE_NAME);
            const key = getStorageKey();

            await new Promise<void>((resolve, reject) => {
                const request = store.delete(key);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            db.close();
            // Also clear localStorage backup (global backup might still be issue for multi-tab, but less critical)
            localStorage.removeItem(CHAT_BACKUP_KEY);
            console.log('[CHAT] IndexedDB state cleared');
        } catch (e) {
            console.warn('[CHAT] Failed to clear IndexedDB:', e);
        }
    };

    // SAVE ON PAGE CLOSE/REFRESH: Save state immediately
    useEffect(() => {
        const handleBeforeUnload = () => {
            const stateToSave = {
                messages: stateRef.current.messages,
                sessionId: stateRef.current.sessionId,
                savedAt: new Date().toISOString()
            };

            // Use synchronous localStorage as fallback for immediate save
            try {
                localStorage.setItem(CHAT_BACKUP_KEY, JSON.stringify(stateToSave));
                console.log('[CHAT] Saved state to localStorage backup on page close');
            } catch (e) {
                console.warn('[CHAT] Failed to save backup state:', e);
            }

            // Also try to save to IndexedDB (might not complete)
            saveStateToChatDB(stateToSave);
        };

        window.addEventListener('beforeunload', handleBeforeUnload);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);

    // LOAD PROJECT FROM URL: Load existing project from backend
    useEffect(() => {
        const projectId = searchParams.get('projectId');
        if (!projectId) return;

        const loadProject = async () => {
            setIsRestoring(true);
            try {
                console.log('[CHAT] Loading project:', projectId);
                const response = await fetch(`${BASE_URL}/api/projects/${projectId}`, {
                    credentials: 'include'
                });

                if (!response.ok) {
                    throw new Error('Failed to load project');
                }

                const data = await response.json();
                const project = data.project || data;

                // ✅ Store project ID for updates
                const loadedProjectId = project.id || project._id || project.project_id || projectId;
                if (loadedProjectId) {
                    setCurrentProjectId(loadedProjectId);
                    console.log('[CHAT] Set project ID for updates:', loadedProjectId);
                }

                // Extract chat history
                const convHistories = project.conversationHistories || project.conversation_histories || {};
                const chatHistory = convHistories['engenie_chat'];

                if (chatHistory && chatHistory.messages) {
                    // Restore messages with proper Date objects
                    const restoredMessages = chatHistory.messages.map((msg: any) => ({
                        id: msg.id || `${Date.now()}_${Math.random()}`,
                        type: msg.type,
                        content: msg.content,
                        source: msg.source,
                        sourcesUsed: msg.sourcesUsed,
                        awaitingConfirmation: msg.awaitingConfirmation,
                        timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
                        actionButtons: msg.actionButtons
                    }));

                    setMessages(restoredMessages);
                    setSessionId(chatHistory.sessionId || sessionId);
                    setIsHistory(true);

                    console.log('[CHAT] Restored', restoredMessages.length, 'messages');

                    toast({
                        title: "Project Loaded",
                        description: `Loaded "${project.projectName || project.project_name}"`,
                    });
                }
            } catch (error: any) {
                console.error('[CHAT] Error loading project:', error);
                toast({
                    title: "Load Failed",
                    description: error.message || "Failed to load chat project",
                    variant: "destructive",
                });
            } finally {
                setIsRestoring(false);
            }
        };

        loadProject();
    }, [searchParams, toast]);

    // LOAD FROM INDEXEDDB: Restore state on mount (unless projectId is present)
    useEffect(() => {
        // If loading a specific project, don't restore session state
        if (searchParams.get('projectId')) {
            setIsRestoring(false);
            return;
        }

        const loadState = async () => {
            // Check if we need to clear state (triggered by New button)
            if (sessionStorage.getItem('clear_chat_state') === 'true') {
                console.log('[CHAT] Clearing state as requested by New button');
                sessionStorage.removeItem('clear_chat_state');
                await clearChatDBState();
                setIsRestoring(false);
                return; // Don't restore anything
            }

            try {
                // First check localStorage backup (faster/synchronous)
                let restoredState: any = null;
                try {
                    const backup = localStorage.getItem(CHAT_BACKUP_KEY);
                    if (backup) {
                        restoredState = JSON.parse(backup);
                        console.log('[CHAT] Loaded state from localStorage backup');
                    }
                } catch (e) {
                    console.warn('[CHAT] Failed to load backup:', e);
                }

                // If no backup, try IndexedDB
                if (!restoredState) {
                    restoredState = await loadStateFromChatDB();
                    if (restoredState) {
                        console.log('[CHAT] Loaded state from IndexedDB');
                    }
                }

                if (restoredState && restoredState.messages && restoredState.messages.length > 0) {
                    // Restore messages with proper Date objects
                    const restoredMessages = restoredState.messages.map((msg: any) => ({
                        ...msg,
                        timestamp: msg.timestamp ? new Date(msg.timestamp) : undefined
                    }));

                    setMessages(restoredMessages);

                    // Restore the saved sessionId to maintain backend conversation context
                    if (restoredState.sessionId) {
                        console.log('[CHAT] Restoring sessionId:', restoredState.sessionId);
                        setSessionId(restoredState.sessionId);
                    }

                    setHasAutoSubmitted(true); // Don't re-submit initial query
                    setIsHistory(true); // Disable animations

                    // Scroll to bottom
                    setTimeout(() => {
                        if (chatContainerRef.current) {
                            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
                        }
                    }, 100);
                }
            } catch (e) {
                console.warn('[CHAT] Error restoring state:', e);
            } finally {
                setIsRestoring(false);
            }
        };

        loadState();
    }, [searchParams]);

    // Auto-scroll
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages, showThinking]);

    // Convert snake_case to camelCase
    const toCamelCase = (obj: any): any => {
        if (Array.isArray(obj)) return obj.map(v => toCamelCase(v));
        if (obj !== null && typeof obj === 'object') {
            return Object.keys(obj).reduce((acc: Record<string, any>, key: string) => {
                const camelKey = key.replace(/([-_][a-z])/g, (g) => g.toUpperCase().replace("-", "").replace("_", ""));
                acc[camelKey] = toCamelCase(obj[key]);
                return acc;
            }, {});
        }
        return obj;
    };

    // HEARTBEAT for Window Presence Detection
    // This allows the parent Project page to know if this window is still open
    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const currentSessionKey = params.get('sessionKey'); // Use raw global sessionKey

        if (!currentSessionKey) return;

        const heartbeatKey = `engenie_heartbeat_${currentSessionKey}`;

        // Initial write
        localStorage.setItem(heartbeatKey, Date.now().toString());

        const interval = setInterval(() => {
            localStorage.setItem(heartbeatKey, Date.now().toString());
        }, 1000);

        const cleanup = () => {
            clearInterval(interval);
            localStorage.removeItem(heartbeatKey);
        };

        window.addEventListener('beforeunload', cleanup);

        return () => {
            window.removeEventListener('beforeunload', cleanup);
            cleanup();
        };
    }, []);

    // Refs for accessing latest state inside BroadcastChannel handler without restarting the effect
    const messagesRef = useRef(messages);
    const sessionIdRef = useRef(sessionId);
    const lastSessionKeyRefForChannel = useRef(lastSessionKeyRef.current);

    useEffect(() => {
        messagesRef.current = messages;
        sessionIdRef.current = sessionId;
        lastSessionKeyRefForChannel.current = lastSessionKeyRef.current;
    }, [messages, sessionId]);

    // BROADCAST CHANNEL for Project.tsx communication
    // Responds to pings to confirm window is open and report message count
    useEffect(() => {
        const channel = new BroadcastChannel('engenie_channel');

        channel.onmessage = (event) => {
            if (event.data.type === 'ping') {
                const requestedKey = event.data.sessionKey;
                // Use Refs to get latest values without re-binding
                const currentSessionId = sessionIdRef.current;
                const lastKey = lastSessionKeyRefForChannel.current;
                const currentMessages = messagesRef.current;

                // Only respond if we match the session key OR if we are just checking generic window existence
                if (requestedKey === currentSessionId || requestedKey === lastKey || window.location.search.includes(requestedKey)) {
                    console.log('[ENGENIE_CHAT] Received ping, sending pong with count:', currentMessages.length);
                    channel.postMessage({
                        type: 'pong',
                        sessionKey: requestedKey,
                        messageCount: currentMessages.length,
                        timestamp: Date.now()
                    });
                }
            }
        };

        return () => {
            channel.close();
        };
    }, []);




    // Query API - defined first so it can be used by auto-submit
    const queryEnGenieChat = async (query: string): Promise<RAGResponse> => {
        try {
            const response = await axios.post("/api/engenie-chat/query", {
                query,
                session_id: sessionId
            }, { withCredentials: true });

            const data = response.data;

            // Check for validation rejection in success response (200 OK with rejected: true)
            if (data.rejected) {
                console.log('[VALIDATION] Query rejected:', data.answer);
                return {
                    success: false,
                    answer: data.answer || "Your query appears to be outside my expertise area.",
                    source: data.source || "validation",
                    foundInDatabase: false,
                    awaitingConfirmation: false,
                    sourcesUsed: [],
                    error: data.error
                };
            }

            return toCamelCase(data) as RAGResponse;
        } catch (error: any) {
            // Other errors
            return {
                success: false,
                answer: error.response?.data?.answer || uiLabels.errorMessage,
                source: "unknown",
                foundInDatabase: false,
                awaitingConfirmation: false,
                sourcesUsed: [],
                error: error.message
            };
        }
    };

    // Load state from DB on mount
    // Load state from DB on mount
    useEffect(() => {
        const load = async () => {
            const params = new URLSearchParams(window.location.search);
            const urlSessionKey = params.get('sessionKey');

            // Always attempt to load state first
            let state = null;
            try {
                state = await loadStateFromChatDB();
            } catch (e) {
                console.warn('[ENGENIE_CHAT] Failed to load initial state:', e);
            }

            if (state) {
                // If we have state, restore it
                // We assume for now that if state exists in DB, it belongs to this user/session context
                // In a multi-tab scenario, this might cross-pollinate if not keyed by sessionKey, 
                // but given the requirement to "come back", this is better than clearing it.

                setMessages(state.messages || []);
                setSessionId(state.sessionId || `engenie_chat_${Date.now()}`);
                console.log('[ENGENIE_CHAT] Restored session from DB:', state.sessionId);

                // CRITICAL: If we restored messages, we must prevent auto-submit from running again
                if (state.messages && state.messages.length > 0) {
                    setHasAutoSubmitted(true);
                    setIsHistory(true);
                }
            } else {
                console.log('[ENGENIE_CHAT] No previous state found in DB');
            }

            // If URL session key provided, ensure we track it
            if (urlSessionKey) {
                lastSessionKeyRef.current = urlSessionKey;
                console.log('[ENGENIE_CHAT] Initialized session key ref:', urlSessionKey);
            }

            setIsRestoring(false); // Mark restoration as complete
            setIsDataReady(true); // Signal that data is fully loaded and safe to check
        };
        load();
    }, []);

    // Handle incoming query from URL parameter (from workflow routing)
    useEffect(() => {
        // Only run after state restoration attempt is complete
        if (isRestoring || !isDataReady) return;

        let queryToSubmit = searchParams.get('query');
        const sessionKey = searchParams.get('sessionKey');

        // Handle session/window reuse: If sessionKey changes, reset state for new conversation
        if (sessionKey && sessionKey !== lastSessionKeyRef.current) {
            console.log('[ENGENIE_CHAT] New session key detected (window reuse with different query) - resetting state');
            setMessages([]);
            setHasAutoSubmitted(false); // CRITICAL: Reset flag to allow new query to submit
            setSessionId(`engenie_chat_${Date.now()}`); // Generate new session ID for the new context
            setIsHistory(false);
            lastSessionKeyRef.current = sessionKey;
            // Don't return - continue to process the new query below
        }

        // Handle re-click on same query button: Check if timestamp changed (indicates new button click)
        const urlTimestamp = searchParams.get('ts');
        if (urlTimestamp && urlTimestamp !== lastProcessedTsRef.current) {
            console.log('[ENGENIE_CHAT] New timestamp detected (re-click on same query button)');
            lastProcessedTsRef.current = urlTimestamp;
            // If same sessionKey but new timestamp, DO NOT RESET. Just preserve state.
            if (sessionKey === lastSessionKeyRef.current && messages.length > 0) {
                console.log('[ENGENIE_CHAT] Same session, new click - preserving existing state (no re-run)');
                return;
            }
        }


        // Check for session key context if query param is missing or to override
        if (sessionKey) {
            try {
                const storedContext = localStorage.getItem(sessionKey);
                console.log('[ENGENIE_CHAT] Looking for context with sessionKey:', sessionKey);
                console.log('[ENGENIE_CHAT] Found stored context:', storedContext ? 'YES' : 'NO');

                if (storedContext) {
                    const parsed = JSON.parse(storedContext);
                    console.log('[ENGENIE_CHAT] Parsed context:', parsed);
                    if (parsed.query) {
                        console.log('[ENGENIE_CHAT] Found query in session context:', parsed.query.substring(0, 50) + '...');
                        queryToSubmit = parsed.query;
                    }
                }
            } catch (e) {
                console.error('[ENGENIE_CHAT] Failed to read session context:', e);
            }
        }

        // Auto-submit only if:
        // 1. We have a query
        // 2. It hasn't been submitted in this memory session
        // 3. We have NO existing messages (fresh session), OR it's a new window explicit request
        // 4. We are NOT currently restoring state (prevent race condition)
        // NOTE: If messages exist, we assume the user is revisiting/refreshing and we shouldn't re-run the query
        console.log('[ENGENIE_CHAT] Auto-submit check:', {
            hasQuery: !!queryToSubmit,
            hasAutoSubmitted,
            messagesLength: messages.length,
            isRestoring,
            willAutoSubmit: !!(queryToSubmit && !hasAutoSubmitted && messages.length === 0 && !isRestoring)
        });

        if (queryToSubmit && !hasAutoSubmitted && messages.length === 0 && !isRestoring) {
            console.log('[ENGENIE_CHAT] Auto-submitting query:', queryToSubmit.substring(0, 50) + '...');
            console.log('[ENGENIE_CHAT] New Session ID created:', sessionId);

            setHasAutoSubmitted(true);


            // Define and execute auto-submit inline
            const autoSubmit = async () => {
                const userMessage: ChatMessage = {
                    id: `user_${Date.now()}`,
                    type: "user",
                    content: queryToSubmit!, // Use the resolved query
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, userMessage]);
                setIsLoading(true);
                setShowThinking(true);

                try {
                    // ============================================================================
                    // PHASE 1: OUT_OF_DOMAIN BLOCKING (Auto-submit from URL)
                    // Classify query before processing to block invalid queries
                    // ============================================================================
                    const routingResult = await classifyRoute(queryToSubmit!);
                    const targetWorkflow = routingResult.target_workflow;

                    console.log('[CHAT_AUTO_SUBMIT] Classification:', {
                        target: targetWorkflow,
                        intent: routingResult.intent
                    });

                    // Block OUT_OF_DOMAIN queries
                    if (targetWorkflow === 'out_of_domain') {
                        console.log('[CHAT_AUTO_SUBMIT] OUT_OF_DOMAIN blocked');
                        setShowThinking(false);

                        const rejectMessage = routingResult.reject_message ||
                            "I'm EnGenie, your industrial automation assistant. I can help with:\n\n" +
                            "• **Instrument Identification** - Finding the right products for your needs\n\n" +
                            "• **Product Search** - Searching for specific industrial instruments\n\n" +
                            "• **Standards & Compliance** - Questions about industrial standards (ISA, IEC, etc.)\n\n" +
                            "• **Technical Knowledge** - Industrial automation concepts and best practices\n\n" +
                            "Please ask a question related to industrial automation, instrumentation, or process control.";

                        const assistantMessage: ChatMessage = {
                            id: `assistant_${Date.now()}`,
                            type: "assistant",
                            content: rejectMessage,
                            timestamp: new Date()
                        };
                        setMessages(prev => [...prev, assistantMessage]);
                        setIsLoading(false);
                        return; // ✅ BLOCKED
                    }

                    // Continue with normal query processing
                    const response = await queryEnGenieChat(queryToSubmit!);
                    setShowThinking(false);

                    const assistantMessage: ChatMessage = {
                        id: `assistant_${Date.now()}`,
                        type: "assistant",
                        content: response.answer,
                        source: response.source,
                        sourcesUsed: response.sourcesUsed,
                        awaitingConfirmation: response.awaitingConfirmation,
                        timestamp: new Date()
                    };
                    setMessages(prev => [...prev, assistantMessage]);

                    if (response.source === "database") {
                        toast({
                            title: uiLabels.sourceDatabase,
                            description: response.sourcesUsed?.join(", ") || "database",
                        });
                    } else if (response.source === "llm") {
                        toast({
                            title: uiLabels.sourceLlm,
                            description: "AI knowledge",
                        });
                    }
                } catch (error) {
                    setShowThinking(false);
                    toast({
                        title: "Error",
                        description: uiLabels.errorMessage,
                        variant: "destructive",
                    });
                } finally {
                    setIsLoading(false);
                }
            };

            // Execute after short delay to ensure component is mounted
            setTimeout(autoSubmit, 100);
        }
    }, [searchParams, hasAutoSubmitted, messages.length, sessionId, isRestoring, isDataReady]);

    const submitQuery = async (query: string) => {
        const userMessage: ChatMessage = {
            id: `user_${Date.now()}`,
            type: "user",
            content: query,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);
        setInputValue("");
        setIsLoading(true);
        setShowThinking(true);

        try {
            // ============================================================================
            // SMART ROUTING: Use backend routing classifier to detect target workflow
            // If query belongs to different workflow (Solution/Search), show navigation button
            // ============================================================================
            const routingResult = await classifyRoute(query);
            const targetWorkflow = routingResult.target_workflow;

            console.log('[CHAT_ROUTING] Backend classification:', {
                target: targetWorkflow,
                intent: routingResult.intent,
                confidence: routingResult.confidence
            });

            // ============================================================================
            // ============================================================================
            // PHASE 1: OUT_OF_DOMAIN BLOCKING - DISABLED
            // Blocking has been disabled to allow all queries through to the backend
            // ============================================================================
            // if (targetWorkflow === 'out_of_domain') {
            //     console.log('[CHAT_ROUTING] OUT_OF_DOMAIN query detected - blocking');
            //     setShowThinking(false);
            //
            //     const rejectMessage = routingResult.reject_message ||
            //         "I'm EnGenie, your industrial automation assistant. I can help with:\n\n" +
            //         "• **Instrument Identification** - Finding the right products for your needs\n\n" +
            //         "• **Product Search** - Searching for specific industrial instruments\n\n" +
            //         "• **Standards & Compliance** - Questions about industrial standards (ISA, IEC, etc.)\n\n" +
            //         "• **Technical Knowledge** - Industrial automation concepts and best practices\n\n" +
            //         "Please ask a question related to industrial automation, instrumentation, or process control.";
            //
            //     const assistantMessage: ChatMessage = {
            //         id: `assistant_${Date.now()}`,
            //         type: "assistant",
            //         content: rejectMessage,
            //         timestamp: new Date()
            //     };
            //     setMessages(prev => [...prev, assistantMessage]);
            //     setIsLoading(false);
            //     return; // ✅ BLOCKED - No API call made
            // }

            // Only redirect to the Solution page when the query is explicitly a
            // solution-design request. The classify-route classifier is tuned for
            // page routing (not RAG routing) and produces too many false positives.
            // Chat's own classify_query() handles all RAG routing correctly once
            // a query reaches the backend — so the default is: send to chat.
            //
            // A query redirects to Solution ONLY when it matches an explicit
            // design-intent verb + system/package noun pattern.
            const SOLUTION_DESIGN_RE = /\b(design|build|create|develop|implement|engineer|specify|commission)\b.{0,60}\b(system|skid|package|unit|loop|solution|plant|facility|network|panel|cabinet)\b/i;
            const isSolutionDesign = SOLUTION_DESIGN_RE.test(query);

            const targetInfo = isSolutionDesign
                ? {
                    page: 'solution',
                    label: '📋 Open Solution Page',
                    icon: '🔧',
                    description: 'This looks like a **complex solution** requiring multiple instruments.\n\nFor better results, I recommend using our **Solution** page.'
                  }
                : null;

            // If backend suggests different page, show navigation button with actionButtons
            if (targetInfo) {
                setShowThinking(false);

                const sessionKey = `chat_nav_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                const targetUrl = `/${targetInfo.page}?sessionKey=${sessionKey}`;

                // Store context for the target page
                try {
                    localStorage.setItem(sessionKey, JSON.stringify({ query: query }));
                } catch (e) {
                    console.error('[CHAT_ROUTING] Failed to store context:', e);
                }

                // Create assistant message with action button
                const assistantMessage: ChatMessage = {
                    id: `assistant_${Date.now()}`,
                    type: "assistant",
                    content: `${targetInfo.description}\n\n_Or, if you'd like to ask a knowledge question instead, please rephrase your query._`,
                    timestamp: new Date(),
                    actionButtons: [
                        {
                            label: targetInfo.label,
                            action: 'openNewWindow',
                            url: targetUrl,
                            icon: targetInfo.icon,
                            contextData: { query: query }
                        }
                    ]
                };
                setMessages(prev => [...prev, assistantMessage]);
                setIsLoading(false);
                return;
            }

            // Target is chat - continue with normal EnGenie Chat query
            const response = await queryEnGenieChat(query);
            setShowThinking(false);

            const assistantMessage: ChatMessage = {
                id: `assistant_${Date.now()}`,
                type: "assistant",
                content: response.answer,
                source: response.source,
                sourcesUsed: response.sourcesUsed,
                awaitingConfirmation: response.awaitingConfirmation,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, assistantMessage]);

            if (response.source === "database") {
                toast({
                    title: uiLabels.sourceDatabase,
                    description: response.sourcesUsed?.join(", ") || "database",
                });
            } else if (response.source === "llm") {
                toast({
                    title: uiLabels.sourceLlm,
                    description: "AI knowledge",
                });
            }
        } catch (error) {
            setShowThinking(false);
            toast({
                title: "Error",
                description: uiLabels.errorMessage,
                variant: "destructive",
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleSend = async () => {
        const trimmedInput = inputValue.trim();
        if (!trimmedInput) return;

        // Reset textarea height to initial height before sending
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }

        await submitQuery(trimmedInput);
    };

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        handleSend();
    };

    const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleNewSession = async () => {
        await clearChatDBState();
        setMessages([]);
        setSessionId(`engenie_chat_${Date.now()}`);
        setHasAutoSubmitted(false);
        setIsHistory(false);
        setCurrentProjectId(null); // Reset project ID for new session
        toast({ title: "New Session", description: "Started fresh session" });
    };

    const handleLogout = async () => {
        try {
            await logout();
            navigate('/login');
        } catch (error) {
            toast({ title: "Logout Failed", variant: "destructive" });
        }
    };

    const handleSaveProject = async (
        overrideName?: string,
        options?: { skipDuplicateDialog?: boolean }
    ) => {
        if (!messages || messages.length === 0) {
            toast({
                title: "Nothing to Save",
                description: "Start a conversation first before saving.",
                variant: "destructive",
            });
            return;
        }

        try {
            // Extract first 2 words from the first user message for project name
            let projectNameBase = 'Chat';
            const firstUserMessage = messages.find((m) => m.type === 'user');
            if (firstUserMessage && firstUserMessage.content) {
                projectNameBase = extractFirstWords(firstUserMessage.content, 2);
            }

            // Use override name if provided, otherwise add "(Chat)" suffix
            const effectiveProjectName = overrideName
                ? overrideName.trim()
                : `${projectNameBase} (Chat)`;

            // Check for duplicate project name
            if (!options?.skipDuplicateDialog) {
                try {
                    const listResponse = await fetch(`${BASE_URL}/api/projects`, {
                        credentials: 'include'
                    });

                    if (listResponse.ok) {
                        const data = await listResponse.json();
                        const projects: any[] = data.projects || [];

                        const nameLower = effectiveProjectName.toLowerCase();
                        const hasDuplicate = projects.some((p: any) => {
                            const pName = (p.projectName || p.project_name || '').trim();
                            const pId = p.id || p._id || null;
                            if (!pName) return false;
                            // Same name (case-insensitive) and not the very same project we are updating
                            const isSameName = pName.toLowerCase() === nameLower;
                            const isSameProject = currentProjectId && pId === currentProjectId;
                            return isSameName && !isSameProject;
                        });

                        if (hasDuplicate) {
                            const suggested = computeNextDuplicateName(effectiveProjectName, projects);
                            setDuplicateProjectName(effectiveProjectName);
                            setAutoRenameSuggestion(suggested);
                            setDuplicateDialogNameInput(effectiveProjectName);
                            setDuplicateNameDialogOpen(true);
                            return;
                        }
                    }
                } catch (e) {
                    // If duplicate check fails, continue with normal save flow.
                }
            }

            // Get the first user message as initial requirements
            const initialRequirements = firstUserMessage?.content || 'EnGenie chat session';

            // Build project data
            const projectData = {
                project_id: currentProjectId || undefined,
                project_name: effectiveProjectName,
                project_description: `EnGenie Chat - Created on ${new Date().toLocaleDateString()}`,
                initial_requirements: initialRequirements,
                source_page: 'engenie_chat',
                project_instance_id: sessionId,
                conversation_histories: {
                    'engenie_chat': {
                        messages: messages.map(msg => ({
                            id: msg.id,
                            type: msg.type,
                            content: msg.content,
                            source: msg.source,
                            sourcesUsed: msg.sourcesUsed,
                            awaitingConfirmation: msg.awaitingConfirmation,
                            timestamp: msg.timestamp,
                            actionButtons: msg.actionButtons
                        })),
                        sessionId: sessionId,
                        currentStep: 'chat',
                        hasAutoSubmitted: hasAutoSubmitted,
                        isHistory: isHistory
                    }
                },
                current_step: 'chat',
                workflow_position: {
                    current_tab: 'engenie_chat',
                    has_results: messages.some(m => m.type === 'assistant'),
                    last_interaction: new Date().toISOString(),
                    project_phase: 'chat_conversation'
                },
                user_interactions: {
                    conversations_count: 1,
                    messages_count: messages.length,
                    has_assistant_responses: messages.some(m => m.type === 'assistant'),
                    last_save: new Date().toISOString()
                },
                field_descriptions: {
                    project_name: 'Name/title of the EnGenie chat session',
                    project_description: 'Description of the chat session purpose',
                    initial_requirements: 'First user message in the conversation',
                    source_page: 'Page where the project was created (engenie_chat)',
                    conversation_histories: 'Complete chat conversation with RAG responses',
                    workflow_position: 'Current position in the chat workflow',
                    user_interactions: 'Summary of user interactions and message counts'
                }
            };

            console.log('[ENGENIE_CHAT_SAVE] Saving project:', effectiveProjectName);

            const response = await fetch(`${BASE_URL}/api/projects/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(projectData),
            });

            if (!response.ok) {
                let errorData: any = null;
                try {
                    errorData = await response.json();
                } catch (e) {
                    // ignore JSON parse errors
                }

                const errorMessage = errorData?.error || 'Failed to save project';
                const errorCode = errorData?.code || errorData?.errorCode;

                const looksLikeDuplicateNameError =
                    response.status === 409 ||
                    errorCode === 'DUPLICATE_PROJECT_NAME' ||
                    /already exists|already present|duplicate project name/i.test(errorMessage);

                if (!options?.skipDuplicateDialog && looksLikeDuplicateNameError) {
                    const nameInErrorMatch = errorMessage.match(/"([^"]+)"/);
                    const nameFromError = nameInErrorMatch ? nameInErrorMatch[1] : effectiveProjectName;

                    let suggested = `${nameFromError} (1)`;
                    try {
                        const listResp = await fetch(`${BASE_URL}/api/projects`, { credentials: 'include' });
                        if (listResp.ok) {
                            const listData = await listResp.json();
                            suggested = computeNextDuplicateName(nameFromError, listData.projects || []);
                        }
                    } catch (e) {
                        // fallback remains
                    }

                    setDuplicateProjectName(nameFromError);
                    setAutoRenameSuggestion(suggested);
                    setDuplicateDialogNameInput(nameFromError);
                    setDuplicateDialogError(null);
                    setDuplicateNameDialogOpen(true);
                    return;
                }

                throw new Error(errorMessage);
            }

            const result = await response.json();
            console.log('[ENGENIE_CHAT_SAVE] Project saved successfully:', result);

            // Store project ID for future updates
            if (result.project_id || result.projectId || result.id) {
                const savedProjectId = result.project_id || result.projectId || result.id;
                setCurrentProjectId(savedProjectId);
                console.log('[ENGENIE_CHAT_SAVE] Project ID stored:', savedProjectId);
            }

            // Also save to IndexedDB for quick restore
            const stateToSave = {
                messages: messages,
                sessionId: sessionId,
                currentProjectId: currentProjectId,
                savedAt: new Date().toISOString()
            };
            saveStateToChatDB(stateToSave);

            toast({
                title: "Project Saved",
                description: `"${effectiveProjectName}" has been saved successfully.`,
            });

        } catch (error: any) {
            console.error('[ENGENIE_CHAT_SAVE] Error saving project:', error);
            toast({
                title: "Save Failed",
                description: error.message || "Failed to save project",
                variant: "destructive",
            });
        }
    };

    // Keep ref updated for auto-save
    useEffect(() => {
        handleSaveProjectRef.current = handleSaveProject;
    }, [messages, sessionId, currentProjectId, hasAutoSubmitted, isHistory]);

    // Helper functions for duplicate name dialog
    const resetDuplicateDialog = () => {
        setDuplicateNameDialogOpen(false);
        setDuplicateProjectName(null);
        setAutoRenameSuggestion(null);
        setDuplicateDialogError(null);
        setDuplicateDialogNameInput('');
    };

    const handleDuplicateNameChangeConfirm = () => {
        const trimmed = (duplicateDialogNameInput || '').trim();
        if (!trimmed) {
            setDuplicateDialogError('Project name is required');
            return;
        }

        resetDuplicateDialog();
        handleSaveProject(trimmed, { skipDuplicateDialog: false });
    };

    const handleDuplicateNameAutoRename = async () => {
        const baseName = (duplicateProjectName || '').trim() || 'Chat';
        let suggested = autoRenameSuggestion || `${baseName} (1)`;
        try {
            const listResp = await fetch(`${BASE_URL}/api/projects`, { credentials: 'include' });
            if (listResp.ok) {
                const listData = await listResp.json();
                suggested = computeNextDuplicateName(baseName, listData.projects || []);
            }
        } catch (e) {
            // ignore and use fallback
        }

        resetDuplicateDialog();
        handleSaveProject(suggested, { skipDuplicateDialog: true });
    };

    // NOTE: Backend save is MANUAL ONLY (when user clicks Save button)
    // IndexedDB auto-save already handled by existing beforeunload and state change handlers

    const handleExportChat = () => {
        // Export chat as text/markdown
        const chatText = messages.map(msg =>
            `[${msg.timestamp.toLocaleString()}] ${msg.type.toUpperCase()}: ${msg.content}`
        ).join('\n\n');

        const blob = new Blob([chatText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `engenie-chat-${sessionId}.txt`;
        a.click();
        URL.revokeObjectURL(url);

        toast({ title: "Chat Exported", description: "Downloaded as text file" });
    };

    const handleLoadSessions = () => {
        // Navigate to sessions list or show modal
        toast({ title: "Load Sessions", description: "Feature coming soon!" });
    };

    const profileButtonLabel = user?.name || user?.username || "User";

    return (
        <div className="h-screen w-full app-glass-gradient flex flex-col overflow-hidden relative">
            <MainHeader
                onSave={() => handleSaveProject()}
                onNew={handleNewSession}
            />


            {/* Sticky Header Title */}
            {/* Sticky Header Title */}
            <div className="flex-none pt-24 pb-0">
                <div className="py-0 border-b border-white/10 bg-transparent flex justify-center items-center">
                    <div className="flex items-center gap-1">
                        <div className="flex items-center justify-center">
                            <img
                                src="/icon-engenie.png"
                                alt="EnGenie"
                                className="w-16 h-16 object-contain"
                            />
                        </div>
                        <h1 className="text-3xl font-bold text-[#0f172a] inline-flex items-center gap-2 whitespace-nowrap">
                            EnGenie <span className="text-sky-500">♦</span> Chat
                        </h1>
                    </div>
                </div>
            </div>

            {/* Chat Messages */}
            <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 custom-no-scrollbar pb-24">


                {messages.map((message) => (
                    <MessageRow
                        key={message.id}
                        message={message}
                        isHistory={isHistory}
                        uiLabels={uiLabels}
                    />
                ))}

                {showThinking && (
                    <div className="flex justify-start">
                        <div className="max-w-[80%] flex items-start space-x-2">
                            <div className="flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center bg-transparent">
                                <img src="/icon-engenie.png" alt="Assistant" className="w-14 h-14 object-contain" />
                            </div>
                            <div className="p-3 rounded-lg">
                                <BouncingDots />
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="fixed bottom-0 left-0 right-0 p-4 bg-transparent z-30 pointer-events-none">
                <div className="max-w-4xl mx-auto px-2 md:px-8 pointer-events-auto">
                    <form onSubmit={handleSubmit}>
                        <div className="relative group">
                            <div
                                className="relative w-full rounded-[26px] transition-all duration-300 focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-transparent hover:scale-[1.02]"
                                style={{
                                    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
                                    WebkitBackdropFilter: 'blur(12px)',
                                    backdropFilter: 'blur(12px)',
                                    backgroundColor: '#ffffff',
                                    border: '1px solid rgba(255, 255, 255, 0.4)',
                                    color: 'rgba(0, 0, 0, 0.8)'
                                }}
                            >
                                <textarea
                                    ref={textareaRef}
                                    value={inputValue}
                                    onChange={(e) => setInputValue(e.target.value)}
                                    onKeyDown={handleKeyPress}
                                    onInput={(e) => {
                                        const target = e.target as HTMLTextAreaElement;
                                        target.style.height = 'auto';
                                        target.style.height = `${Math.min(target.scrollHeight, 150)}px`;
                                    }}
                                    className="w-full bg-transparent border-0 focus:ring-0 focus:outline-none px-4 py-2.5 pr-20 text-sm resize-none min-h-[40px] max-h-[150px] leading-relaxed flex items-center custom-no-scrollbar"
                                    style={{
                                        fontSize: '16px',
                                        fontFamily: 'inherit',
                                        boxShadow: 'none',
                                        overflowY: 'auto'
                                    }}
                                    placeholder={uiLabels.inputPlaceholder}
                                    disabled={isLoading}
                                />
                                <div className="absolute bottom-1.5 right-1.5 flex items-center gap-0.5">
                                    <Button
                                        type="submit"
                                        disabled={!inputValue.trim() || isLoading}
                                        className={`w-8 h-8 p-0 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-transparent flex-shrink-0 ${!inputValue.trim() ? 'text-muted-foreground' : 'text-primary hover:scale-110'}`}
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

            {/* Duplicate Name Dialog */}
            <Dialog open={duplicateNameDialogOpen} onOpenChange={setDuplicateNameDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Project Name Already Exists</DialogTitle>
                        <DialogDescription>
                            A project named "{duplicateProjectName}" already exists. Please choose a different name or use the suggested name.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <div className="grid gap-2">
                            <Label htmlFor="project-name">Project Name</Label>
                            <Input
                                id="project-name"
                                value={duplicateDialogNameInput}
                                onChange={(e) => {
                                    setDuplicateDialogNameInput(e.target.value);
                                    setDuplicateDialogError(null);
                                }}
                                placeholder="Enter new project name"
                                className={duplicateDialogError ? 'border-red-500' : ''}
                            />
                            {duplicateDialogError && (
                                <p className="text-sm text-red-500">{duplicateDialogError}</p>
                            )}
                        </div>
                        {autoRenameSuggestion && (
                            <div className="text-sm text-muted-foreground">
                                Suggested name: <span className="font-medium text-foreground">{autoRenameSuggestion}</span>
                            </div>
                        )}
                    </div>
                    <DialogFooter className="gap-2">
                        <Button
                            variant="outline"
                            onClick={resetDuplicateDialog}
                        >
                            Cancel
                        </Button>
                        {autoRenameSuggestion && (
                            <Button
                                variant="secondary"
                                onClick={handleDuplicateNameAutoRename}
                            >
                                Use Suggested Name
                            </Button>
                        )}
                        <Button
                            onClick={handleDuplicateNameChangeConfirm}
                        >
                            Save with This Name
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div >
    );
};

export default EnGenieChat;
