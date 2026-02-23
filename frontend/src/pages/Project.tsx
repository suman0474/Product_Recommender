import { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Loader2, Play, Bot, LogOut, User, Upload, Save, FolderOpen, FileText, X, ChevronLeft, ChevronRight, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import { BASE_URL } from '../components/AIRecommender/api';
import { routeUserInputByIntent, validateRequirements } from '@/components/AIRecommender/api';

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ReactMarkdown from 'react-markdown';
import BouncingDots from '@/components/AIRecommender/BouncingDots';

import {
    DropdownMenu,
    DropdownMenuTrigger,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuItem,
    DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import AIRecommender from "@/components/AIRecommender";
import { useAuth } from '@/contexts/AuthContext';
import { instanceOrchestrationService } from '@/services/InstanceOrchestrationService';
import { SessionManager } from '@/services/SessionManager';


import '../components/TabsLayout.css';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useFieldDescriptions } from '@/hooks/useFieldDescriptions';
import MainHeader from '@/components/MainHeader';

// Helper for generating unique tab IDs
const generateTabId = () => `search_${Date.now().toString(36)}`;

interface IdentifiedInstrument {
    category: string;
    quantity?: number;
    productName: string;
    specifications: Record<string, string>;
    sampleInput: string;
    item_thread_id?: string;
    workflow_thread_id?: string;
    main_thread_id?: string;
}

interface IdentifiedAccessory {
    category: string;
    quantity?: number;
    accessoryName: string;
    specifications: Record<string, string>;
    sampleInput: string;
    item_thread_id?: string;
    workflow_thread_id?: string;
    main_thread_id?: string;
}

// Action button interface for chat messages
interface ChatActionButton {
    label: string;
    action: 'openNewWindow' | 'navigate' | 'custom';
    url?: string;
    icon?: string;
    contextData?: any;
    // Internal state for window reuse
    sessionKey?: string;
    windowName?: string;
    generatedUrl?: string;
}

// Chat message interface for Project page
interface ProjectChatMessage {
    id: string;
    type: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    actionButtons?: ChatActionButton[];  // Optional action buttons
}

// MessageRow component with animations (same as Dashboard)
interface MessageRowProps {
    message: ProjectChatMessage;
    isHistory: boolean;
}

const MessageRow = ({ message, isHistory }: MessageRowProps) => {
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

    // Handler for action buttons
    const handleActionClick = async (action: ChatActionButton) => {
        if (action.action === 'openNewWindow' && action.url) {
            let targetUrl = action.url;
            let targetWindowName = '_blank';

            if (action.contextData && action.contextData.query) {
                console.log('[PROJECT] Action button clicked with contextData:', action.contextData);

                // Generate deterministic key from query so same query = same lookup key
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
                            // ─── CASE A: Window is open ───
                            // Just focus it. Don't change URL. Don't re-run query.
                            console.log(`[PROJECT] Window is ALIVE with ${status.messageCount} messages. Just focusing.`);
                            targetUrl = mapping.url.split('?')[0] + '?sessionKey=' + encodeURIComponent(mapping.sessionKey);
                            targetWindowName = mapping.windowName;
                            windowHandled = true;
                        } else {
                            console.log('[PROJECT] Window is CLOSED. Will create fresh instance.');
                        }
                    } catch (e) {
                        console.error('[PROJECT] Error checking window status:', e);
                    }
                }

                // ─── STEP 2: No existing window (or closed) → Open fresh & run ───
                if (!windowHandled) {
                    const sessionKey = `${persistentSessionKey}_${Date.now()}`;
                    const windowName = `engenie_window_${sessionKey}`;

                    try {
                        // Store query context so EnGenieChat can read it and auto-run
                        localStorage.setItem(sessionKey, JSON.stringify(action.contextData));
                        console.log('[PROJECT] Stored context for NEW session:', sessionKey);

                        // Build clean URL with only sessionKey (no query param in URL)
                        const urlObj = new URL(action.url);
                        urlObj.searchParams.delete('query');
                        urlObj.searchParams.set('sessionKey', sessionKey);
                        targetUrl = urlObj.toString();

                        // Save mapping so next click can find this window
                        localStorage.setItem(`window_mapping_${persistentSessionKey}`, JSON.stringify({
                            url: targetUrl,
                            windowName: windowName,
                            query: action.contextData.query,
                            sessionKey: sessionKey
                        }));

                        targetWindowName = windowName;
                        console.log('[PROJECT] Opening NEW window:', targetUrl);
                    } catch (e) {
                        console.error('[PROJECT] Failed to store context:', e);
                    }
                }
            }

            // Open/focus the window
            window.open(targetUrl, targetWindowName);
        }

        if (action.action === 'navigate' && action.url) {
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

// IndexedDB configuration for persisting project state (no size limits like localStorage)
const PROJECT_DB_NAME = 'project_page_db';
const PROJECT_STORE_NAME = 'project_state';
const PROJECT_STATE_KEY = 'current_session';

// Helper function to open IndexedDB
const openProjectDB = (): Promise<IDBDatabase> => {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(PROJECT_DB_NAME, 1);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
            const db = (event.target as IDBOpenDBRequest).result;
            if (!db.objectStoreNames.contains(PROJECT_STORE_NAME)) {
                db.createObjectStore(PROJECT_STORE_NAME, { keyPath: 'id' });
            }
        };
    });
};

// Helper function to save state to IndexedDB
const saveStateToIndexedDB = async (state: any): Promise<void> => {
    try {
        const db = await openProjectDB();
        const transaction = db.transaction(PROJECT_STORE_NAME, 'readwrite');
        const store = transaction.objectStore(PROJECT_STORE_NAME);

        await new Promise<void>((resolve, reject) => {
            const request = store.put({ id: PROJECT_STATE_KEY, ...state });
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });

        db.close();
    } catch (e) {
        console.warn('[PROJECT] Failed to save to IndexedDB:', e);
    }
};

// Helper function to load state from IndexedDB
const loadStateFromIndexedDB = async (): Promise<any | null> => {
    try {
        const db = await openProjectDB();
        const transaction = db.transaction(PROJECT_STORE_NAME, 'readonly');
        const store = transaction.objectStore(PROJECT_STORE_NAME);

        const result = await new Promise<any>((resolve, reject) => {
            const request = store.get(PROJECT_STATE_KEY);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });

        db.close();

        if (result) {
            // Restore Date objects for chat messages
            if (result.chatMessages) {
                result.chatMessages = result.chatMessages.map((msg: any) => ({
                    ...msg,
                    timestamp: new Date(msg.timestamp)
                }));
            }
            // Restore Date objects in tabStates messages
            if (result.tabStates) {
                Object.values(result.tabStates).forEach((state: any) => {
                    if (state?.messages) {
                        state.messages = state.messages.map((msg: any) => ({
                            ...msg,
                            timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date()
                        }));
                    }
                });
            }
            return result;
        }
        return null;
    } catch (e) {
        console.warn('[PROJECT] Failed to load from IndexedDB:', e);
        return null;
    }
};

// Helper function to clear IndexedDB state
const clearIndexedDBState = async (): Promise<void> => {
    try {
        const db = await openProjectDB();
        const transaction = db.transaction(PROJECT_STORE_NAME, 'readwrite');
        const store = transaction.objectStore(PROJECT_STORE_NAME);

        await new Promise<void>((resolve, reject) => {
            const request = store.delete(PROJECT_STATE_KEY);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });

        db.close();
        console.log('[PROJECT] IndexedDB state cleared');
    } catch (e) {
        console.warn('[PROJECT] Failed to clear IndexedDB:', e);
    }
};

// Synchronous loader that returns cached state (populated by async load on mount)
let cachedSavedState: any = null;

// Helper function to load state synchronously from cache (for initial render)
const loadSavedState = () => {
    return cachedSavedState;
};

const Project = () => {
    const [requirements, setRequirements] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [instruments, setInstruments] = useState<IdentifiedInstrument[]>([]);
    const [accessories, setAccessories] = useState<IdentifiedAccessory[]>([]);
    const [showResults, setShowResults] = useState(false);
    const [activeTab, setActiveTab] = useState<string>('project');
    const [previousTab, setPreviousTab] = useState<string>('project');
    const [searchTabs, setSearchTabs] = useState<{ id: string; title: string; input: string; isDirectSearch?: boolean; productType?: string; itemThreadId?: string; workflowThreadId?: string; mainThreadId?: string }[]>([]);
    const [currentWorkflowType, setCurrentWorkflowType] = useState<'solution' | 'search' | null>(null); // Track current workflow
    // Session Manager instance
    const sessionManager = SessionManager.getInstance();

    const [projectName, setProjectName] = useState<string>('Project');
    const [editingProjectName, setEditingProjectName] = useState<boolean>(false);
    const [editProjectNameValue, setEditProjectNameValue] = useState<string>(projectName);
    const editNameInputRef = useRef<HTMLInputElement | null>(null);
    const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
    const [duplicateNameDialogOpen, setDuplicateNameDialogOpen] = useState(false);
    const [duplicateProjectName, setDuplicateProjectName] = useState<string | null>(null);
    const [autoRenameSuggestion, setAutoRenameSuggestion] = useState<string | null>(null);
    const [duplicateDialogNameInput, setDuplicateDialogNameInput] = useState<string>('');
    const [duplicateDialogError, setDuplicateDialogError] = useState<string | null>(null);


    // Right panel dock state
    const [isRightDocked, setIsRightDocked] = useState(true);
    // Right panel tab state (instruments/accessories)
    const [rightPanelTab, setRightPanelTab] = useState<'instruments' | 'accessories'>('instruments');
    // Collapse state for instrument/accessory cards
    const [collapsedInstruments, setCollapsedInstruments] = useState<Set<number>>(new Set());
    const [collapsedAccessories, setCollapsedAccessories] = useState<Set<number>>(new Set());

    // NEW: Field descriptions for tooltips (Managed by hook below)
    // const [fieldDescriptions, setFieldDescriptions] = useState<Record<string, string>>({});

    // Track conversation states for each search tab
    const [tabStates, setTabStates] = useState<Record<string, any>>({});
    const navigate = useNavigate();
    const { toast } = useToast();
    const { user, logout } = useAuth(); // Get user info and logout function

    // INTEGRATED: Dynamic Field Descriptions
    const { descriptions: fieldDescriptions, fetchOnHover } = useFieldDescriptions(
        tabStates['project']?.currentProductType || 'general'
    );

    // NEW: For scroll position handling
    const projectScrollRef = useRef<HTMLDivElement | null>(null);
    const rightPanelScrollRef = useRef<HTMLDivElement | null>(null); // For right panel scroll
    const [savedScrollPosition, setSavedScrollPosition] = useState(0);
    // Track current scroll positions continuously (updated on scroll events)
    // const currentScrollPositionRef = useRef(0); // Using stateRef.current.savedScrollPosition
    const rightPanelScrollPositionRef = useRef(0); // Right panel scroll position
    const [isRestoringState, setIsRestoringState] = useState(true);
    const [pendingScrollPosition, setPendingScrollPosition] = useState<number | null>(null);
    const [pendingRightPanelScroll, setPendingRightPanelScroll] = useState<number | null>(null);

    // NEW: For generic product type images
    const [genericImages, setGenericImages] = useState<Record<string, string>>({});

    // Flag to prevent refetching images during project load
    const [isLoadingProject, setIsLoadingProject] = useState(false);

    // NEW: Chat messages for Project page (replaces responseMessage)
    const [chatMessages, setChatMessages] = useState<ProjectChatMessage[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const isHistoryRef = useRef(true);
    const [showThinking, setShowThinking] = useState(false);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages, showThinking]);

    // Set isHistory to false after initial mount so new messages animate
    useEffect(() => {
        const timer = setTimeout(() => {
            isHistoryRef.current = false;
        }, 1000);
        return () => clearTimeout(timer);
    }, []);

    // Show thinking indicator with delay
    useEffect(() => {
        if (isLoading) {
            const timer = setTimeout(() => setShowThinking(true), 600);
            return () => clearTimeout(timer);
        } else {
            setShowThinking(false);
        }
    }, [isLoading]);

    // Helper to add a message to chat (with optional action buttons)
    const addChatMessage = (type: 'user' | 'assistant', content: string, actionButtons?: ChatActionButton[]) => {
        const newMessage: ProjectChatMessage = {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            type,
            content,
            timestamp: new Date(),
            actionButtons
        };
        setChatMessages(prev => [...prev, newMessage]);
    };

    // Helper to parse specification values (handles both string and structured formats)
    const parseSpecValue = (value: any): { displayValue: string; source: string | null; confidence: number | null } => {
        // Handle structured Deep Agent format: { value: "...", source: "...", confidence: 0.9 }
        if (value && typeof value === 'object' && !Array.isArray(value) && 'value' in value) {
            return {
                displayValue: value.value || 'Not specified',
                source: value.source || null,
                confidence: value.confidence || null
            };
        }
        // Handle simple string/number values
        return {
            displayValue: value || 'Not specified',
            source: null,
            confidence: null
        };
    };

    // Helper to get source label for display
    const getSourceLabel = (source: string | null): string | null => {
        if (!source) return null;
        if (source.toLowerCase().includes('standard')) return 'Standards';
        if (source.toLowerCase().includes('infer')) return 'Inferred';
        if (source.toLowerCase().includes('rag')) return 'Knowledge Base';
        if (source.toLowerCase().includes('user')) return 'User Input';
        return source;
    };



    // State to track failed image fetches for regeneration
    const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
    const [regeneratingImages, setRegeneratingImages] = useState<Set<string>>(new Set());
    const [loadingImages, setLoadingImages] = useState<Set<string>>(new Set());

    // Regenerate a single image - called automatically when image is not found
    const regenerateImage = async (productName: string) => {
        if (regeneratingImages.has(productName)) return; // Already regenerating

        setRegeneratingImages(prev => new Set(prev).add(productName));

        // Remove from failed images while regenerating
        setFailedImages(prev => {
            const next = new Set(prev);
            next.delete(productName);
            return next;
        });

        try {
            const response = await fetch(`${BASE_URL}/api/generic_image/regenerate/${encodeURIComponent(productName)}`, {
                method: 'POST',
                credentials: 'include'
            });

            const data = await response.json();

            if (response.ok && data.success && data.image?.url) {
                // Successfully regenerated
                setGenericImages(prev => ({
                    ...prev,
                    [productName]: data.image.url
                }));
                // Remove from failed set (already done at start but good to ensure)
                setFailedImages(prev => {
                    const next = new Set(prev);
                    next.delete(productName);
                    return next;
                });
            } else if (response.status === 429) {
                // Rate limited - show wait time
                const waitSeconds = data.wait_seconds || 30;
                console.warn(`Rate limited. Please wait ${waitSeconds} seconds before retrying.`);
                setFailedImages(prev => new Set(prev).add(productName));
            } else {
                console.error('Image regeneration failed:', data.error || 'Unknown error');
                setFailedImages(prev => new Set(prev).add(productName));
            }
        } catch (error) {
            console.error('Error regenerating image:', error);
            setFailedImages(prev => new Set(prev).add(productName));
        } finally {
            setRegeneratingImages(prev => {
                const next = new Set(prev);
                next.delete(productName);
                return next;
            });
        }
    };

    // NEW: For file upload
    const [attachedFile, setAttachedFile] = useState<File | null>(null);
    const [extractedText, setExtractedText] = useState<string>('');
    const [isExtracting, setIsExtracting] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // =================================================================================================
    // PERSISTENCE LOGIC (IndexedDB + LocalStorage)
    // =================================================================================================

    // State ref to access latest state in event listeners (like beforeunload)
    const stateRef = useRef({
        requirements,
        instruments: [] as IdentifiedInstrument[],
        accessories: [] as IdentifiedAccessory[],
        showResults,
        activeTab: 'project',
        searchTabs: [] as any[],
        projectName: 'Project',
        currentProjectId: null as string | null,
        chatMessages: [] as ProjectChatMessage[],
        isRightDocked: true,
        genericImages: {} as Record<string, string>,
        tabStates: {} as Record<string, any>,
        savedScrollPosition: 0,
        rightPanelScroll: 0,
        fieldDescriptions: {} as Record<string, string>
    });

    // Update stateRef whenever relevant state changes
    useEffect(() => {
        stateRef.current = {
            requirements,
            instruments,
            accessories,
            showResults,
            activeTab,
            searchTabs,
            projectName,
            currentProjectId,
            chatMessages,
            isRightDocked,
            genericImages,
            tabStates,
            savedScrollPosition: projectScrollRef.current ? projectScrollRef.current.scrollTop : 0,
            rightPanelScroll: rightPanelScrollRef.current ? rightPanelScrollRef.current.scrollTop : 0,
            fieldDescriptions
        };
    }, [
        requirements, instruments, accessories, showResults, activeTab,
        searchTabs, projectName, currentProjectId, chatMessages,
        isRightDocked, genericImages, tabStates, fieldDescriptions
    ]);

    // RESTORE SCROLL POSITION: After loading screen is gone and DOM is ready
    useEffect(() => {
        if (!isRestoringState && pendingScrollPosition !== null && pendingScrollPosition > 0) {
            // Use requestAnimationFrame for instant restore
            requestAnimationFrame(() => {
                let scrollApplied = false;

                // Try element scroll first
                if (projectScrollRef.current) {
                    projectScrollRef.current.scrollTop = pendingScrollPosition;
                    scrollApplied = projectScrollRef.current.scrollTop > 0;
                    console.log('[PROJECT] Applied left scroll:', pendingScrollPosition);
                }

                // If element scroll didn't work, try window scroll
                if (!scrollApplied) {
                    window.scrollTo(0, pendingScrollPosition);
                }

                // Clear the pending position
                setPendingScrollPosition(null);
            });
        }
    }, [isRestoringState, pendingScrollPosition]);

    // RESTORE RIGHT PANEL SCROLL POSITION: After right panel is mounted
    useEffect(() => {
        if (!isRestoringState && pendingRightPanelScroll !== null && pendingRightPanelScroll > 0 && showResults) {
            // Use requestAnimationFrame for instant restore
            requestAnimationFrame(() => {
                if (rightPanelScrollRef.current) {
                    rightPanelScrollRef.current.scrollTop = pendingRightPanelScroll;
                    console.log('[PROJECT] Applied right panel scroll:', pendingRightPanelScroll);
                }
                // Clear the pending position
                setPendingRightPanelScroll(null);
            });
        }
    }, [isRestoringState, pendingRightPanelScroll, showResults]);

    // AUTO-SAVE: Persist COMPLETE state to IndexedDB whenever data changes
    // Debounced to avoid too many writes
    const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        // Skip saving during initial load/restore
        if (isRestoringState) return;

        // Skip saving if nothing meaningful to save
        if (!chatMessages.length && !instruments.length && !accessories.length && projectName === 'Project') {
            return;
        }

        // Debounce saves to avoid too many writes
        if (saveTimeoutRef.current) {
            clearTimeout(saveTimeoutRef.current);
        }

        saveTimeoutRef.current = setTimeout(async () => {
            // Use current scroll position from ref/elements
            const scrollPosition = projectScrollRef.current?.scrollTop || 0;
            const rightPanelScroll = rightPanelScrollRef.current?.scrollTop || 0;

            const stateToSave = {
                chatMessages,
                instruments,
                accessories,
                showResults,
                activeTab,
                previousTab,
                searchTabs,
                projectName,
                currentProjectId,
                isRightDocked,
                tabStates, // Save COMPLETE tabStates
                genericImages, // Save ALL images
                fieldDescriptions,
                scrollPosition, // Save left panel scroll position
                rightPanelScroll, // Save right panel scroll position
                savedAt: new Date().toISOString()
            };

            await saveStateToIndexedDB(stateToSave);

            // Calculate size for logging
            const dataSize = new Blob([JSON.stringify(stateToSave)]).size / 1024;
            console.log(`[PROJECT] Complete state auto-saved to IndexedDB (${dataSize.toFixed(1)}KB)`);
        }, 1000); // Debounce for 1 second

        return () => {
            if (saveTimeoutRef.current) {
                clearTimeout(saveTimeoutRef.current);
            }
        };
    }, [
        chatMessages, instruments, accessories, showResults, activeTab, previousTab,
        searchTabs, projectName, currentProjectId, isRightDocked, tabStates,
        genericImages, fieldDescriptions, isRestoringState
    ]);

    // SAVE ON PAGE CLOSE/REFRESH: Capture scroll position and save immediately
    useEffect(() => {
        const handleBeforeUnload = () => {
            // Get current scroll positions
            const scrollPosition = projectScrollRef.current?.scrollTop || 0;
            const rightPanelScroll = rightPanelScrollRef.current?.scrollTop || 0;

            console.log('[PROJECT] beforeunload - leftScroll:', scrollPosition, 'rightPanelScroll:', rightPanelScroll);

            const stateToSave = {
                ...stateRef.current,
                scrollPosition,
                rightPanelScroll,
                savedAt: new Date().toISOString()
            };

            // Use synchronous localStorage as fallback for immediate save
            try {
                localStorage.setItem('project_page_state_backup', JSON.stringify(stateToSave));
                // Lighter backup for session metadata if needed
                localStorage.setItem('project_page_state_meta', JSON.stringify({
                    projectName: stateToSave.projectName,
                    timestamp: new Date().toISOString()
                }));
                console.log('[PROJECT] Saved state to localStorage backup on page close');
            } catch (e) {
                console.warn('[PROJECT] Failed to save backup state:', e);
            }

            // Also try to save to IndexedDB (might not complete)
            saveStateToIndexedDB(stateToSave);
        };

        window.addEventListener('beforeunload', handleBeforeUnload);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);

    // LOAD FROM INDEXEDDB: Restore complete state on mount
    useEffect(() => {
        const loadState = async () => {
            try {
                // Check if this is a fresh window opened via navigation popup
                const urlParams = new URLSearchParams(window.location.search);
                if (urlParams.get('fresh') === 'true') {
                    console.log('[PROJECT] Fresh window detected - skipping state restoration');
                    // Clear stored state so it doesn't persist
                    await clearIndexedDBState();
                    localStorage.removeItem('project_page_state_backup');
                    localStorage.removeItem('project_page_state_meta');
                    // Remove fresh param from URL to prevent re-clearing on manual refresh
                    urlParams.delete('fresh');
                    const newUrl = urlParams.toString()
                        ? `${window.location.pathname}?${urlParams.toString()}`
                        : window.location.pathname;
                    window.history.replaceState({}, '', newUrl);
                    setIsRestoringState(false);
                    return;
                }

                // First check if there's a localStorage backup (more reliable for scroll position on refresh)
                let backupData: any = null;
                try {
                    const backup = localStorage.getItem('project_page_state_backup');
                    if (backup) {
                        backupData = JSON.parse(backup);
                        console.log('[PROJECT] Found localStorage backup');
                        // Clear the backup after reading (optional, but good cleanup)
                        // localStorage.removeItem('project_page_state_backup');
                    }
                } catch (e) {
                    console.warn('[PROJECT] Failed to read localStorage backup:', e);
                }

                // Load main state from IndexedDB
                const saved = await loadStateFromIndexedDB();

                // Use backup scroll positions if available (they're more recent)
                const scrollToRestore = backupData?.scrollPosition || saved?.scrollPosition || 0;
                const rightPanelScrollToRestore = backupData?.rightPanelScroll || saved?.rightPanelScroll || 0;

                if (saved || backupData) {
                    // Prefer backup data if available (most recent), otherwise IndexedDB
                    const dataToUse = backupData || saved;

                    console.log('[PROJECT] Restoring complete state');

                    // Restore all state - use batch updates where possible
                    if (dataToUse.chatMessages?.length) setChatMessages(dataToUse.chatMessages);
                    if (dataToUse.instruments?.length) setInstruments(dataToUse.instruments);
                    if (dataToUse.accessories?.length) setAccessories(dataToUse.accessories);
                    if (dataToUse.showResults !== undefined) setShowResults(dataToUse.showResults);

                    // Restore project name
                    if (dataToUse.projectName && dataToUse.projectName !== 'Project') {
                        setProjectName(dataToUse.projectName);
                        setEditProjectNameValue(dataToUse.projectName);
                    }

                    if (dataToUse.currentProjectId) setCurrentProjectId(dataToUse.currentProjectId);
                    if (dataToUse.isRightDocked !== undefined) setIsRightDocked(dataToUse.isRightDocked);
                    if (dataToUse.genericImages && Object.keys(dataToUse.genericImages).length) setGenericImages(dataToUse.genericImages);

                    // Restore tabStates
                    if (dataToUse.tabStates && Object.keys(dataToUse.tabStates).length) {
                        setTabStates(dataToUse.tabStates);
                    }

                    // Restore search tabs and active tab
                    if (dataToUse.searchTabs?.length) {
                        setSearchTabs(dataToUse.searchTabs);
                    }
                    if (dataToUse.previousTab) setPreviousTab(dataToUse.previousTab);

                    // Restore active tab - check URL first
                    const issearchUrl = location.pathname === '/solution/search';
                    if (issearchUrl && dataToUse.searchTabs?.length > 0) {
                        // Let the URL logic handle it, or restore if it matches a tab ID
                        if (dataToUse.activeTab && dataToUse.activeTab !== 'project') {
                            setActiveTab(dataToUse.activeTab);
                        }
                    } else if (dataToUse.activeTab) {
                        setActiveTab(dataToUse.activeTab);
                    }

                    // Restore pending scroll positions
                    if (scrollToRestore > 0) {
                        setPendingScrollPosition(scrollToRestore);
                    }
                    if (rightPanelScrollToRestore > 0) {
                        setPendingRightPanelScroll(rightPanelScrollToRestore);
                    }
                }
            } catch (e) {
                console.warn('[PROJECT] Error loading state:', e);
            } finally {
                setIsRestoringState(false);
            }
        };

        loadState();
    }, []);

    // Track scroll position changes for the active tab (debounced)
    const handleScroll = () => {
        if (projectScrollRef.current && activeTab === 'project') {
            // We don't need to do anything complex here as we read from ref on save
            // But we could update state if needed for other features
        }
    };

    const capitalizeFirstLetter = (str?: string): string => {
        if (!str) return "";
        return str.charAt(0).toUpperCase() + str.slice(1);
    };



    // Helper to convert relative image URLs to absolute URLs
    const getAbsoluteImageUrl = (url: string | undefined | null): string | undefined => {
        if (!url) return undefined;

        // Already absolute URL
        if (url.startsWith('http') || url.startsWith('data:')) {
            return url;
        }

        // Convert relative URL to absolute
        const baseUrl = BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL;
        const path = url.startsWith('/') ? url : `/${url}`;
        return `${baseUrl}${path}`;
    };

    // Helper to format keys: snake_case/space separated -> Title_Case_With_Underscores
    const prettifyKey = (key: string) => {
        // Handle undefined or empty keys
        if (!key) return "";

        // 1. Split by underscores, spaces, or camelCase boundaries
        // 2. Filter empty parts
        // 3. Capitalize first letter of each part
        // 4. Join with underscores
        return key
            .replace(/([A-Z])/g, ' $1') // Split camelCase
            .replace(/_/g, ' ')         // Normalize underscores to spaces first
            .trim()
            .split(/\s+/)               // Split by whitespace
            .map(part => part.charAt(0).toUpperCase() + part.slice(1)) // Capitalize
            .join('_');                 // Join with underscores
    };

    /**
     * Format sample input by prettifying field names (removing underscores, etc.)
     * Input: "range: 0-100 bar, Body_Material: 316SS [STANDARD]"
     * Output: "Range: 0-100 bar, Body Material: 316SS [STANDARD]"
     */
    const formatSampleInput = (sampleInput: string) => {
        if (!sampleInput) return "";
        const parts = sampleInput.split(',').map(part => part.trim());
        return parts.map(part => {
            const colonIndex = part.indexOf(':');
            if (colonIndex === -1) return part;
            const key = part.substring(0, colonIndex).trim();
            const value = part.substring(colonIndex + 1).trim();
            return `${prettifyKey(key)}: ${value}`;
        }).join(', ');
    };

    const toggleInstrumentCollapse = (index: number) => {
        setCollapsedInstruments(prev => {
            const newSet = new Set(prev);
            if (newSet.has(index)) {
                newSet.delete(index);
            } else {
                newSet.add(index);
            }
            return newSet;
        });
    };

    const toggleAccessoryCollapse = (index: number) => {
        setCollapsedAccessories(prev => {
            const newSet = new Set(prev);
            if (newSet.has(index)) {
                newSet.delete(index);
            } else {
                newSet.add(index);
            }
            return newSet;
        });
    };

    /**
     * Extract the base product type from a product name.
     * Handles common naming patterns like:
     * - "X with Y" (e.g., "Temperature Transmitter with RTD Sensor" → "Temperature Transmitter")
     * - "X for Y" (e.g., "Connection Head for Temperature Transmitter" → "Connection Head")
     * - "X including Y", "X featuring Y"
     * This ensures we get the correct generic image for the base product type.
     */
    const extractBaseProductType = (productName: string): string => {
        if (!productName) return productName;

        // Pattern 1: "X for Y" - extract X (common for accessories)
        if (productName.includes(' for ')) {
            return productName.split(' for ')[0].trim();
        }

        // Pattern 2: "X with Y" - extract X (common for instruments with modifiers)
        if (productName.includes(' with ')) {
            return productName.split(' with ')[0].trim();
        }

        // Pattern 3: "X including Y" - extract X
        if (productName.includes(' including ')) {
            return productName.split(' including ')[0].trim();
        }

        // Pattern 4: "X featuring Y" - extract X
        if (productName.includes(' featuring ')) {
            return productName.split(' featuring ')[0].trim();
        }

        // No pattern found, return original
        return productName;
    };

    /**
     * Extract the proper image key from an accessory.
     * For accessories named "X for Y" (e.g., "Connection Head for Temperature Transmitter"),
     * we want to use "X" (the actual accessory type) as the image key, not the full name.
     * This ensures we get the correct generic image (e.g., Connection Head image, not Transmitter image).
     */
    const getAccessoryImageKey = (accessory: any): string => {
        const accessoryName = accessory.accessoryName || '';
        const category = accessory.category || '';

        // If category is specific (not generic "Accessories"), prefer it
        const isGenericCategory = !category ||
            category.toLowerCase() === 'accessories' ||
            category.toLowerCase() === 'accessory' ||
            category.toLowerCase() === 'unknown accessory';

        if (!isGenericCategory && category) {
            return category;  // Use specific category like "Connection Head"
        }

        // Use the common extraction function
        return extractBaseProductType(accessoryName);
    };

    /**
     * Extract the proper image key from an instrument.
     * For instruments with modifiers like "Temperature Transmitter with RTD Sensor",
     * we want to use the base type "Temperature Transmitter" for image lookup.
     */
    const getInstrumentImageKey = (instrument: any): string => {
        const productName = instrument.productName || '';

        // Use the common extraction function
        return extractBaseProductType(productName);
    };

    // Cached images load instantly (no delay), only uncached ones are slow
    const fetchGenericImagesLazy = async (productTypes: string[]) => {
        const uniqueTypes = [...new Set(productTypes)]; // Remove duplicates

        console.log(`[SEQUENTIAL_LOAD] Starting sequential load for ${uniqueTypes.length} images...`);

        // Mark all images as loading initially
        setLoadingImages(prev => {
            const next = new Set(prev);
            uniqueTypes.forEach(pt => {
                if (!genericImages[pt] && !failedImages.has(pt)) {
                    next.add(pt);
                }
            });
            return next;
        });

        // Load images one by one - cached ones are instant, uncached trigger LLM
        for (let i = 0; i < uniqueTypes.length; i++) {
            const productType = uniqueTypes[i];

            // Skip if already loaded
            if (genericImages[productType]) {
                console.log(`[SEQUENTIAL_LOAD] [${i + 1}/${uniqueTypes.length}] Already loaded: ${productType}`);
                setLoadingImages(prev => {
                    const next = new Set(prev);
                    next.delete(productType);
                    return next;
                });
                continue;
            }

            try {
                const encodedType = encodeURIComponent(productType);
                console.log(`[SEQUENTIAL_LOAD] [${i + 1}/${uniqueTypes.length}] Fetching: ${productType}`);

                const response = await fetch(`${BASE_URL}/api/generic_image/${encodedType}`, {
                    credentials: 'include'
                });

                if (response.ok) {
                    const data = await response.json();
                    if (data.success && data.image) {
                        const absoluteUrl = getAbsoluteImageUrl(data.image.url);
                        if (absoluteUrl) {
                            // Update state immediately for each image (shows as soon as loaded)
                            setGenericImages(prev => ({
                                ...prev,
                                [productType]: absoluteUrl
                            }));
                            // Remove from failed set if previously failed
                            setFailedImages(prev => {
                                const next = new Set(prev);
                                next.delete(productType);
                                return next;
                            });
                            console.log(`[SEQUENTIAL_LOAD] ✓ Loaded ${i + 1}/${uniqueTypes.length}: ${productType}`);
                        } else {
                            // No URL returned - mark as failed
                            setFailedImages(prev => new Set(prev).add(productType));
                            console.warn(`[SEQUENTIAL_LOAD] ✗ No URL for ${productType}`);
                        }
                    } else {
                        // Backend returned success=false or no image - mark as failed
                        setFailedImages(prev => new Set(prev).add(productType));
                        console.warn(`[SEQUENTIAL_LOAD] ✗ No image data for ${productType}`);
                    }
                } else if (response.status === 404) {
                    // Image not found in backend or DB - explicitly mark as failed so UI shows placeholder instantly
                    setFailedImages(prev => new Set(prev).add(productType));
                    console.log(`[SEQUENTIAL_LOAD] Not found: ${productType}. Displaying placeholder.`);
                } else {
                    // Other HTTP errors - mark as failed
                    setFailedImages(prev => new Set(prev).add(productType));
                    console.warn(`[SEQUENTIAL_LOAD] ✗ Failed (${response.status}): ${productType}`);
                }
            } catch (error) {
                // Network/parsing errors - mark as failed
                setFailedImages(prev => new Set(prev).add(productType));
                console.error(`[SEQUENTIAL_LOAD] ✗ Error fetching ${productType}:`, error);
            } finally {
                // Always remove from loading set when done
                setLoadingImages(prev => {
                    const next = new Set(prev);
                    next.delete(productType);
                    return next;
                });
            }
        }

        console.log(`[SEQUENTIAL_LOAD] All ${uniqueTypes.length} images processed.`);
    };

    // Escape string for use in RegExp
    const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // Compute next available duplicate name: e.g., if 'Name' exists, suggest 'Name (1)';
    // if 'Name (1)' exists, suggest 'Name (2)', etc.
    const computeNextDuplicateName = (base: string, projects: any[]) => {
        if (!base) return `${base} (1)`;
        const baseTrim = base.trim();

        // Extract the actual base name without any numbering
        // If base is "Distillation Column (1)", extract "Distillation Column"
        const baseNameMatch = baseTrim.match(/^(.*?)(?:\s*\(\d+\))?$/);
        const actualBaseName = baseNameMatch ? baseNameMatch[1].trim() : baseTrim;

        // Create regex to match all variations of the base name with numbers
        const regex = new RegExp(`^${escapeRegExp(actualBaseName)}(?:\\s*\\((\\d+)\\))?$`, 'i');
        let maxNum = 0;
        let foundBase = false;

        for (const p of projects) {
            const pName = (p.projectName || p.project_name || '').trim();
            if (!pName) continue;
            const m = pName.match(regex);
            if (m) {
                if (!m[1]) {
                    foundBase = true;
                } else {
                    const n = parseInt(m[1], 10);
                    if (!isNaN(n) && n > maxNum) maxNum = n;
                }
            }
        }

        if (maxNum > 0) {
            return `${actualBaseName} (${maxNum + 1})`;
        }

        if (foundBase) return `${actualBaseName} (1)`;

        // fallback
        return `${actualBaseName} (1)`;
    };

    useEffect(() => {
        setEditProjectNameValue(projectName);
    }, [projectName]);

    const profileButtonLabel = capitalizeFirstLetter(user?.name || user?.username || "User");
    const profileFullName = user?.name || `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || user?.username || "User";

    const handleSubmit = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();

        // Check if we have either text requirements or extracted text from file
        if (!requirements.trim() && !extractedText.trim()) {
            toast({
                title: "Input Required",
                description: "Please enter your requirements or attach a file",
                variant: "destructive",
            });
            return;
        }

        // Combine manual requirements with extracted text from file
        const finalRequirements = requirements.trim() && extractedText.trim()
            ? `${requirements}\n\n${extractedText}`
            : requirements.trim() || extractedText.trim();

        // Create display message for chat (show file name instead of extracted text)
        const displayMessage = attachedFile
            ? requirements.trim()
                ? `${requirements.trim()}\n\n📎 ${attachedFile.name}`
                : `📎 ${attachedFile.name}`
            : requirements.trim();

        // Add user message to chat (show file name, not extracted text)
        addChatMessage('user', displayMessage);

        // Clear the input immediately after adding the message
        setRequirements('');

        // Reset textarea height to initial height
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }

        // Clear attached file immediately after capturing the display message
        setAttachedFile(null);
        setExtractedText('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }

        setIsLoading(true);

        try {
            // ====================================================================
            // UNIFIED INTENT ROUTING - Routes based on intent classification
            // NO direct workflow calls - intent classification determines the route
            // ====================================================================
            console.log('[INTENT_ROUTER] Routing user input with', instruments.length, 'instruments and', accessories.length, 'accessories');

            // Use unified intent-based routing instead of direct workflow calls
            const response = await routeUserInputByIntent(
                finalRequirements,
                instruments.length > 0 ? instruments : undefined,
                accessories.length > 0 ? accessories : undefined
            );

            // Check response type and intent
            const responseType = response.responseType;
            const isSolution = response.isSolution;

            console.log('[INTENT_ROUTER] Response:', { intent: response.intent, responseType, isSolution });

            // CASE 1: Greeting response - Show message in chat
            if (responseType === 'greeting') {
                addChatMessage('assistant', response.message || "Hello! How can I help you find industrial instruments today?");
                setShowResults(true);
                // Keep existing data on greeting
                if (instruments.length === 0 && accessories.length === 0) {
                    setInstruments([]);
                    setAccessories([]);
                }
                return;
            }

            // CASE 2: Question response - Show message in chat
            if (responseType === 'question') {
                addChatMessage('assistant', response.message || '');
                setShowResults(true);
                // Keep existing data on question
                if (instruments.length === 0 && accessories.length === 0) {
                    setInstruments([]);
                    setAccessories([]);
                }
                return;
            }

            // CASE 2.5: Workflow Suggestion - Show clickable option to open EnGenie Chat in new window
            if (responseType === 'workflowSuggestion') {
                if (response.suggestWorkflow?.workflow_id === 'engenie_chat') {
                    console.log('[PROJECT] User asked a general question -> EnGenie Chat');
                }
                const queryEncoded = encodeURIComponent(finalRequirements);
                const enGenieChatUrl = `${window.location.origin}/chat?query=${queryEncoded}`;

                // Show message with action button that opens in new window
                addChatMessage(
                    'assistant',
                    response.message || 'This looks like a knowledge question. Click the button below to get detailed answers from our knowledge base.',
                    [
                        {
                            label: '🚀 Open Chat',
                            action: 'openNewWindow',
                            url: enGenieChatUrl,
                            icon: '💬',
                            contextData: { query: finalRequirements }
                        }
                    ]
                );

                setShowResults(true);
                return;
            }

            // CASE 2.6: Cross-workflow navigation (Search ↔ Solution)
            // If user is in one workflow and asks for another, show navigation button
            const incomingWorkflowType = isSolution ? 'solution' : (responseType === 'requirements' ? 'search' : null);

            if (currentWorkflowType && incomingWorkflowType && currentWorkflowType !== incomingWorkflowType) {
                console.log(`[WORKFLOW_SWITCH] Detected switch from ${currentWorkflowType} to ${incomingWorkflowType}`);

                const sessionKey = `workflow_nav_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                const targetPage = incomingWorkflowType === 'solution' ? 'solution' : 'search';
                const targetUrl = `/${targetPage}?sessionKey=${sessionKey}`;

                // Store context for the target page
                try {
                    localStorage.setItem(sessionKey, JSON.stringify({ query: finalRequirements }));
                } catch (e) {
                    console.error('[WORKFLOW_SWITCH] Failed to store context:', e);
                }

                const workflowMessages: Record<string, { label: string; icon: string; description: string }> = {
                    'solution': {
                        label: '📋 Open Solution Page',
                        icon: '🔧',
                        description: 'This looks like a **complex solution** requiring multiple instruments.\n\nFor better results, I recommend opening this in our **Solution** page.'
                    },
                    'search': {
                        label: '🔍 Open Search Page',
                        icon: '📦',
                        description: 'This looks like a **single product search** query.\n\nFor better results, I recommend opening this in our **Search** page.'
                    }
                };

                const targetInfo = workflowMessages[incomingWorkflowType];

                // Show message with action button
                addChatMessage(
                    'assistant',
                    `${targetInfo.description}\n\n_Or, continue here to work with both workflows._`,
                    [
                        {
                            label: targetInfo.label,
                            action: 'openNewWindow',
                            url: targetUrl,
                            icon: targetInfo.icon,
                            contextData: { query: finalRequirements }
                        }
                    ]
                );

                setShowResults(true);
                return;
            }

            // CASE 3: Modification response - Update the list with changes
            if (responseType === 'modification') {
                const modMessage = response.message || 'I\'ve updated your instrument list based on your request.';
                addChatMessage('assistant', modMessage);

                // Update instruments and accessories with modified list
                setInstruments(response.instruments || []);
                setAccessories(response.accessories || []);
                setShowResults(true);

                // Lazy load generic images for any new items
                const productNames: string[] = [];
                (response.instruments || []).forEach((inst: any) => {
                    // Use extracted base type for image key (e.g., "Temperature Transmitter" not "Temperature Transmitter with RTD Sensor")
                    const imageKey = getInstrumentImageKey(inst);
                    if (imageKey) productNames.push(imageKey);
                });
                (response.accessories || []).forEach((acc: any) => {
                    // Use extracted accessory type for image key (e.g., "Connection Head" not "Connection Head for Temp Transmitter")
                    const imageKey = getAccessoryImageKey(acc);
                    if (imageKey) productNames.push(imageKey);
                });

                // Only fetch images for items not already loaded
                const newProductNames = productNames.filter(name => !genericImages[name]);
                if (newProductNames.length > 0) {
                    fetchGenericImagesLazy(newProductNames);
                }

                const changeCount = response.changesMade?.length || 0;
                toast({
                    title: "List Updated",
                    description: `Applied ${changeCount} change(s). You now have ${response.instruments?.length || 0} instruments and ${response.accessories?.length || 0} accessories.`,
                });
                return;
            }

            // CASE 4: Solution response (complex engineering challenge)
            // Also handles regular requirements response
            if (responseType === 'solution' || responseType === 'requirements') {
                // Log solution-specific handling
                if (isSolution) {
                    console.log('[SOLUTION] Processing solution workflow response');
                }

                // Update current workflow type
                setCurrentWorkflowType(isSolution ? 'solution' : 'search');

                setInstruments(response.instruments || []);
                setAccessories(response.accessories || []);
                setShowResults(true);

                // Automatically undock the right panel when results arrive
                setIsRightDocked(false);

                // Set the project name from the API response
                if (response.projectName) {
                    setProjectName(response.projectName);
                }

                // Capture field descriptions if available
                // Capture field descriptions if available
                // Deprecated: fieldDescriptions are now lazy-loaded
                /* if (response.fieldDescriptions || response.field_descriptions) {
                    const loadedDescriptions = response.fieldDescriptions || response.field_descriptions;
                    console.log('Loaded field descriptions from solution response:', Object.keys(loadedDescriptions).length);
                    setFieldDescriptions(loadedDescriptions);
                } */

                // Lazy load generic images in BACKGROUND (non-blocking)
                const productNames: string[] = [];
                (response.instruments || []).forEach((inst: any) => {
                    // Use extracted base type for image key (e.g., "Temperature Transmitter" not "Temperature Transmitter with RTD Sensor")
                    const imageKey = getInstrumentImageKey(inst);
                    if (imageKey) productNames.push(imageKey);
                });
                (response.accessories || []).forEach((acc: any) => {
                    // Use extracted accessory type for image key (e.g., "Connection Head" not "Connection Head for Temp Transmitter")
                    const imageKey = getAccessoryImageKey(acc);
                    if (imageKey) productNames.push(imageKey);
                });

                if (productNames.length > 0) {
                    fetchGenericImagesLazy(productNames);
                }

                // Solution-specific toast message
                const toastTitle = isSolution ? "Solution Identified" : "Success";
                const toastDesc = isSolution
                    ? `Identified ${response.instruments?.length || 0} instruments and ${response.accessories?.length || 0} accessories for your engineering challenge`
                    : `Identified ${response.instruments?.length || 0} instruments and ${response.accessories?.length || 0} accessories`;

                toast({
                    title: toastTitle,
                    description: toastDesc,
                });
            }

        } catch (error: any) {
            addChatMessage('assistant', `I couldn't process your request. ${error.message || 'Please try again.'}`);
            toast({
                title: "Error",
                description: error.message || "Failed to process request",
                variant: "destructive",
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    // Handle file selection and immediately extract text
    const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // Attach the file and start extraction immediately
        setAttachedFile(file);
        setIsExtracting(true);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${BASE_URL}/api/upload-requirements`, {
                method: 'POST',
                credentials: 'include',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to extract text from file');
            }

            const data = await response.json();

            if (data.success && data.extracted_text) {
                // Store the extracted text for later use on submit
                setExtractedText(data.extracted_text);

            } else {
                throw new Error(data.error || 'No text extracted from file');
            }
        } catch (error: any) {
            toast({
                title: "Extraction Failed",
                description: error.message || "Failed to extract text from file",
                variant: "destructive",
            });
            // Clear the file if extraction failed
            setAttachedFile(null);
            setExtractedText('');
        } finally {
            setIsExtracting(false);
            // Reset file input so the same file can be selected again if needed
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    // Handle removing attached file and its extracted text
    const handleRemoveFile = () => {
        setAttachedFile(null);
        setExtractedText('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const addSearchTab = (
        input: string,
        categoryName?: string,
        isDirectSearch: boolean = false,
        productType?: string,
        itemThreadId?: string,
        workflowThreadId?: string,
        mainThreadId?: string
    ) => {
        // Save current scroll position before switching tabs
        if (activeTab === 'project' && projectScrollRef.current) {
            setSavedScrollPosition(projectScrollRef.current.scrollTop);
        }

        const title = categoryName || `Search ${searchTabs.length + 1}`;
        const existingTabIndex = searchTabs.findIndex(tab => tab.title === title);

        if (existingTabIndex !== -1) {
            const updatedTabs = [...searchTabs];

            // Check if we should run the search based on existing messages
            // User Requirement 3: "incase it was found then it need to check is that was having message in case having the messgae then the query no need to run just show the screen thats it"
            const existingTabId = updatedTabs[existingTabIndex].id;
            const tabState = tabStates[existingTabId];
            const hasMessages = tabState?.messages && tabState.messages.length > 0;

            // If existing window has messages, DO NOT run again (just show it).
            // If existing window has NO messages (e.g. fresh or cleared), allow run if requested.
            const shouldRun = isDirectSearch ? !hasMessages : isDirectSearch;

            // Update existing tab with new input and direct search flag
            updatedTabs[existingTabIndex] = {
                ...updatedTabs[existingTabIndex],
                input,
                isDirectSearch: shouldRun,
                productType,
                itemThreadId,
                workflowThreadId,
                mainThreadId
            };
            setSearchTabs(updatedTabs);

            setTimeout(() => {
                setPreviousTab(activeTab);
                setActiveTab(updatedTabs[existingTabIndex].id);
            }, 0);

            return;
        }

        // User Requirement 2: "incase it was not there then it need to open the window and then it need to run the query"
        // User Requirement 4: "user open the new window and close... again clicking thenbutton then it will check it was not there then new window open... here the instance need to be diffrent"
        // Implementation: We generate a NEW unique ID based on timestamp.
        // Since AIRecommender key={tab.id}, a new ID forces a fresh component mount.
        // Since tabStates[newId] is undefined, no savedSearchInstanceId is passed.
        // Result: AIRecommender generates a brand new session ID.
        const nextIndex = searchTabs.length + 1;
        const id = `search-${Date.now()}-${nextIndex}`;
        const newTabs = [
            ...searchTabs,
            {
                id,
                title,
                input,
                isDirectSearch, // This triggers auto-run in new component
                productType,
                itemThreadId,
                workflowThreadId,
                mainThreadId
            }
        ];
        setSearchTabs(newTabs);

        setTimeout(() => {
            setPreviousTab(activeTab);
            setActiveTab(id);
            navigate('/solution/search');
        }, 0);
    };

    const closeSearchTab = (id: string) => {
        const remaining = searchTabs.filter(t => t.id !== id);
        setSearchTabs(remaining);
        if (activeTab === id) {
            const targetTab = remaining.find(t => t.id === previousTab)
                ? previousTab
                : remaining.length > 0
                    ? remaining[remaining.length - 1].id
                    : 'project';

            // If returning to project tab, restore scroll position
            setPreviousTab(activeTab);
            setActiveTab(targetTab);

            // Sync URL with tab state - if closing active tab
            if (targetTab === 'project') {
                navigate('/solution');
            } else {
                navigate('/solution/search');
            }
        } else if (remaining.length === 0) {
            // If we closed a background tab and now there are none, ensure URL is clean
            navigate('/solution');
        }
    };

    const handleRun = async (instrument: IdentifiedInstrument, index: number) => {
        const qty = instrument.quantity ? ` (${instrument.quantity})` : '';

        // Pass instrument.category as productType for proper schema lookup
        // Pass thread IDs received from backend for workflow resumption
        addSearchTab(
            instrument.sampleInput,
            `${index + 1}. ${instrument.category}${qty}`,
            true,
            instrument.category,
            instrument.item_thread_id,
            instrument.workflow_thread_id,
            instrument.main_thread_id
        );
    };

    const handleRunAccessory = async (accessory: IdentifiedAccessory, index: number) => {
        const qty = accessory.quantity ? ` (${accessory.quantity})` : '';

        // Smart category extraction: If category is generic "Accessories", extract the type from accessoryName
        let smartCategory = accessory.category || '';
        const accessoryName = accessory.accessoryName || '';
        const isGeneric = smartCategory.toLowerCase() === 'accessories' || smartCategory.toLowerCase() === 'accessory';

        if (isGeneric && accessoryName) {
            // Extract product type from accessoryName (before " for ")
            // e.g., "Thermowell for Process Temperature Transmitter" -> "Thermowell"
            const parts = accessoryName.split(' for ');
            smartCategory = parts[0] || accessoryName;
        }

        // Pass smart category for both tab title and productType for schema lookup
        // Pass thread IDs received from backend for workflow resumption
        addSearchTab(
            accessory.sampleInput,
            `${index + 1}. ${smartCategory}${qty}`,
            true,
            smartCategory,
            accessory.item_thread_id,
            accessory.workflow_thread_id,
            accessory.main_thread_id
        );
    };

    const handleNewProject = () => {
        // Clear current project ID to create a new project instead of updating
        setCurrentProjectId(null);

        // Reset all project state
        setShowResults(false);
        setInstruments([]);
        setAccessories([]);
        setRequirements('');
        setChatMessages([]); // Clear chat messages
        setSearchTabs([]);
        setPreviousTab('project');
        setActiveTab('project');
        setProjectName('Project'); // Reset project name to default
        setTabStates({});

        console.log('Started new project - cleared project ID');

        toast({
            title: "New Project Started",
            description: "You can now create a fresh project",
        });
    };

    // Handle state updates from AIRecommender instances
    const handleTabStateChange = (tabId: string, state: any) => {
        setTabStates(prev => {
            // Only update if state has actually changed
            if (JSON.stringify(prev[tabId]) !== JSON.stringify(state)) {
                return {
                    ...prev,
                    [tabId]: state
                };
            }
            return prev;
        });
    };

    const handleSaveProject = async (
        overrideName?: string,
        options?: { skipDuplicateDialog?: boolean }
    ) => {
        // Use detected product type if available; do NOT fallback to projectName.
        // Do not call validation during Save to avoid blocking the save operation.
        let detectedProductType = tabStates['project']?.currentProductType;

        // Fallback: Try to detect product type from identified instruments or accessories if not in tab state
        if (!detectedProductType && instruments.length > 0) {
            detectedProductType = instruments[0].category || instruments[0].productName;
        }
        if (!detectedProductType && accessories.length > 0) {
            detectedProductType = accessories[0].category || accessories[0].accessoryName;
        }
        detectedProductType = detectedProductType || '';

        // Smart Name Generation: If project name is generic "Project", try to use the detected product type
        let nameToUse = overrideName || projectName;
        if ((!nameToUse || nameToUse.trim() === 'Project') && detectedProductType) {
            nameToUse = capitalizeFirstLetter(detectedProductType);
        }

        const effectiveProjectName = (nameToUse || '').trim() || 'Project';

        try {
            // Collect all current project data including chat states
            const conversationHistories: Record<string, any> = {};
            const collectedDataAll: Record<string, any> = {};
            const analysisResults: Record<string, any> = {};

            // Collect data from each search tab
            const allFieldDescriptions: Record<string, string> = {};
            Object.entries(tabStates).forEach(([tabId, state]) => {
                if (state) {
                    conversationHistories[tabId] = {
                        messages: state.messages || [],
                        currentStep: state.currentStep || 'greeting',
                        searchSessionId: state.searchSessionId,
                        // Extended state for complete restoration
                        requirementSchema: state.requirementSchema || null,
                        validationResult: state.validationResult || null,
                        currentProductType: state.currentProductType || null,
                        inputValue: state.inputValue || '',
                        advancedParameters: state.advancedParameters || null,
                        selectedAdvancedParams: state.selectedAdvancedParams || {},
                        fieldDescriptions: state.fieldDescriptions || {}
                    };

                    if (state.collectedData) {
                        collectedDataAll[tabId] = state.collectedData;
                    }

                    if (state.analysisResult) {
                        analysisResults[tabId] = state.analysisResult;
                    }

                    // Merge field descriptions from all tabs
                    if (state.fieldDescriptions) {
                        Object.assign(allFieldDescriptions, state.fieldDescriptions);
                    }
                }
            });

            // Create field descriptions for better data understanding
            const baseFieldDescriptions = {
                project_name: 'Name/title of the project',
                project_description: 'Detailed description of the project purpose and scope',
                initial_requirements: 'Original user requirements and specifications provided at project start',
                product_type: 'Type/category of product being developed or analyzed',
                identified_instruments: 'List of instruments identified as suitable for the project requirements',
                identified_accessories: 'List of accessories and supporting equipment identified for the project',
                search_tabs: 'Individual search sessions created by user for different aspects of the project',
                conversation_histories: 'Complete conversation threads for each search tab including AI interactions',
                collected_data: 'Data collected during conversations and analysis for each search tab',
                current_step: 'Current workflow step in the project (greeting, requirements, analysis, etc.)',
                active_tab: 'The tab that was active when the project was last saved',
                analysis_results: 'Results from AI analysis and recommendations for each search tab',
                workflow_position: 'Detailed position in workflow to enable exact continuation',
                user_interactions: 'Summary of user actions and decisions made during the project',
                project_metadata: 'Additional metadata about project creation, updates, and usage patterns'
            };

            // Merge field descriptions from tabs with base descriptions
            const fieldDescriptions = { ...baseFieldDescriptions, ...allFieldDescriptions };

            // Check for duplicate project name on the client by looking at existing projects.
            // This ensures we can prompt even if the backend does not enforce unique names.
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
                            const projectsList: any[] = data.projects || [];
                            const suggested = computeNextDuplicateName(effectiveProjectName, projectsList);
                            setDuplicateProjectName(effectiveProjectName);
                            setAutoRenameSuggestion(suggested);
                            setDuplicateNameDialogOpen(true);
                            return;
                        }
                    }
                } catch (e) {
                    // If duplicate check fails, continue with normal save flow.
                }
            }

            // Combine manual requirements with extracted text from file (same logic as submit)
            const finalRequirements = requirements.trim() && extractedText.trim()
                ? `${requirements}\n\n${extractedText}`
                : requirements.trim() || extractedText.trim();

            // Get session context from SessionManager
            const currentSession = sessionManager.getCurrentSession();
            const mainThreadId = sessionManager.getMainThreadId();

            const projectData: any = {
                project_id: currentProjectId || undefined,
                project_name: overrideName || projectName,
                initial_requirements: finalRequirements,
                product_type: detectedProductType || 'UnknownProduct', // Removed `productType` as it's not defined in scope
                identified_instruments: instruments,
                identified_accessories: accessories,
                generic_images: genericImages,

                // ✅ ADD: Session context preservation
                session_context: {
                    session_id: currentSession?.sessionId || null,
                    instance_id: sessionManager.getTabInstanceId(),
                    main_thread_id: mainThreadId,
                    saved_at: new Date().toISOString(),
                    user_id: currentSession?.userId || (user as any)?.id || null
                },

                // ✅ ADD: Workflow state preservation
                workflow_state: {
                    awaiting_selection: showResults,
                    current_step: showResults ? 'results' : 'initial',
                    has_results: instruments.length > 0 || accessories.length > 0,
                    results_timestamp: showResults ? new Date().toISOString() : null,
                    workflow_type: 'solution'
                },

                // ✅ ADD: Intent data preservation
                intent_data: {
                    project_intent: 'solution',  // Always 'solution' for Project page
                    workflow_type: 'solution',
                    saved_timestamp: new Date().toISOString()
                },
                conversation_histories: conversationHistories,
                collected_data: collectedDataAll,
                project_chat_messages: chatMessages, // Save Project page chat messages
                current_step: activeTab === 'project' ? (showResults ? 'showSummary' : 'initialInput') : 'search',
                active_tab: activeTab === 'project' ? 'Project' : (searchTabs.find(t => t.id === activeTab)?.title || activeTab), // Save tab name instead of ID
                analysis_results: analysisResults,
                field_descriptions: fieldDescriptions,
                workflow_position: {
                    current_tab: activeTab,
                    has_results: showResults,
                    total_search_tabs: searchTabs.length,
                    last_interaction: new Date().toISOString(),
                    project_phase: showResults ? 'results_review' : 'requirements_gathering'
                },
                user_interactions: {
                    tabs_created: searchTabs.length,
                    conversations_count: Object.keys(conversationHistories).length,
                    has_analysis: Object.keys(analysisResults).length > 0,
                    last_save: new Date().toISOString()
                }
            };

            // Include client-side pricing and feedback entries if present in local state
            // `pricing` may be assembled by the frontend or analysisResult; include if available
            if ((analysisResults && Object.keys(analysisResults).length > 0) || (tabStates && Object.keys(tabStates).length > 0)) {
                try {
                    // Try to collect pricing info from tab states (from RightPanel)
                    const pricingDataFromTabs: any = {};
                    Object.entries(tabStates).forEach(([tabId, tabState]: [string, any]) => {
                        if (tabState && tabState.pricingData && Object.keys(tabState.pricingData).length > 0) {
                            console.log(`[SAVE_PROJECT] Collecting pricing data from tab ${tabId}:`, Object.keys(tabState.pricingData).length, 'products');
                            pricingDataFromTabs[tabId] = tabState.pricingData;
                        }
                    });

                    if (Object.keys(pricingDataFromTabs).length > 0) {
                        projectData.pricing = pricingDataFromTabs;
                        console.log(`[SAVE_PROJECT] Included pricing data from`, Object.keys(pricingDataFromTabs).length, 'tabs');
                    }

                    // Also try to collect pricing info embedded in analysisResults for the active tab (fallback)
                    const activeAnalysis = analysisResults[activeTab] || analysisResults['project'] || null;
                    if (activeAnalysis && activeAnalysis.pricing && !projectData.pricing) {
                        projectData.pricing = activeAnalysis.pricing;
                    }
                } catch (e) {
                    console.error('[SAVE_PROJECT] Error collecting pricing data:', e);
                }
            }

            // If UI has any feedback objects (from RightPanel interactions), include them
            // We expect feedback entries to be stored in `tabStates` under each tab's user interactions
            try {
                const feedbackEntries: any[] = [];
                Object.values(tabStates).forEach((s: any) => {
                    if (s && s.feedbackEntries && Array.isArray(s.feedbackEntries)) {
                        feedbackEntries.push(...s.feedbackEntries);
                    }
                });
                if (feedbackEntries.length > 0) projectData.feedback_entries = feedbackEntries;
            } catch (e) {
                // ignore
            }

            // If we have a current project ID, include it to update the existing project
            if (currentProjectId) {
                projectData.project_id = currentProjectId;
                console.log('Updating existing project:', currentProjectId);
            } else {
                console.log('Creating new project');
            }
            console.log('Saving project with comprehensive data and descriptions:', {
                fieldCount: Object.keys(projectData).length,
                hasFieldDescriptions: !!projectData.field_descriptions,
                descriptionsCount: projectData.field_descriptions ? Object.keys(projectData.field_descriptions).length : 0,
                hasWorkflowPosition: !!projectData.workflow_position,
                hasUserInteractions: !!projectData.user_interactions
            });

            const response = await fetch(`${BASE_URL}/api/projects/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Include only media currently displayed on the frontend (reduce unnecessary downloads/storage)
                body: JSON.stringify({
                    ...projectData,
                    displayed_media_map: (() => {
                        try {
                            const map: Record<string, any> = {};
                            const activeState = tabStates[activeTab];
                            if (!activeState || !activeState.analysisResult) return map;

                            const ranked = (activeState.analysisResult?.overallRanking?.rankedProducts) || [];
                            ranked.forEach((product: any) => {
                                try {
                                    // Save ALL products (both exact and approximate matches)
                                    if (!product) return;
                                    const vendor = product.vendor || product.vendorName || product.vendor_name || '';
                                    const pname = product.productName || product.product_name || product.name || '';
                                    if (!vendor && !pname) return;
                                    const key = `${vendor}-${pname}`.trim();
                                    const entry: any = {};

                                    const top = product.topImage || product.top_image || product.top_image_url || product.topImageUrl || null;
                                    const vendorLogo = product.vendorLogo || product.vendor_logo || product.logo || null;

                                    const resolveUrl = (obj: any) => {
                                        if (!obj) return null;
                                        if (typeof obj === 'string') return obj;
                                        return obj.url || obj.src || null;
                                    };

                                    const topUrl = resolveUrl(top);
                                    const vLogoUrl = resolveUrl(vendorLogo);

                                    if (topUrl) entry.top_image = { url: topUrl };
                                    if (vLogoUrl) entry.vendor_logo = { url: vLogoUrl };

                                    // Add matchType metadata
                                    entry.matchType = product.requirementsMatch ? 'exact' : 'approximate';

                                    if (Object.keys(entry).length > 0) map[key] = entry;
                                } catch (e) {
                                    // Continue on minor errors
                                }
                            });
                            return map;
                        } catch (e) {
                            return {};
                        }
                    })()
                }),
                credentials: 'include'
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
                    // Compute smarter suggestion based on existing projects
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

                    // Do not show generic error toast here; the dialog will guide the user.
                    return;
                }

                throw new Error(errorMessage);
            }

            const result = await response.json();

            // Extract project_id from the response
            // Backend returns: { message: "...", project: { project_id: "...", ... } }
            const savedProjectId = result.project?.project_id || result.project_id;

            // If we didn't have a project ID before, set it now for future updates
            if (!currentProjectId && savedProjectId) {
                setCurrentProjectId(savedProjectId);
                console.log('Set currentProjectId for future updates:', savedProjectId);
            }

            // Ensure local state reflects the name we actually saved
            if (overrideName && overrideName.trim()) {
                setProjectName(overrideName.trim());
            }

            toast({
                title: currentProjectId ? "Project Updated" : "Project Saved",
                description: currentProjectId
                    ? `"${effectiveProjectName}" has been updated successfully`
                    : `"${effectiveProjectName}" has been saved successfully`,
            });

        } catch (error: any) {
            // Check if this is a duplicate name error from backend
            const errorMessage = error.message || "";
            if (errorMessage.includes("already exists") && errorMessage.includes("Please choose a different name")) {
                // Extract the project name from the error message
                const nameMatch = errorMessage.match(/Project name '([^']+)' already exists/);
                const duplicateName = nameMatch ? nameMatch[1] : effectiveProjectName;

                // Get current projects to compute suggestion
                try {
                    const listResp = await fetch(`${BASE_URL}/api/projects`, { credentials: 'include' });
                    if (listResp.ok) {
                        const listData = await listResp.json();
                        const suggested = computeNextDuplicateName(duplicateName, listData.projects || []);
                        setDuplicateProjectName(duplicateName);
                        setAutoRenameSuggestion(suggested);
                        setDuplicateDialogNameInput(duplicateName);
                        setDuplicateDialogError(null);
                        setDuplicateNameDialogOpen(true);
                        return; // Don't show the generic error toast
                    }
                } catch (e) {
                    // If we can't get projects list, fall back to default behavior
                }
            }

            toast({
                title: "Save Failed",
                description: error.message || "Failed to save project",
                variant: "destructive",
            });
        }
    };

    const handleProjectDelete = (deletedProjectId: string) => {
        // Check if the deleted project was the currently active one
        if (currentProjectId === deletedProjectId) {
            console.log('Current project was deleted, starting new project...');
            handleNewProject();
        }
    };

    // =================================================================================================
    // =================================================================================================
    // NOTE: Backend save is MANUAL ONLY (when user clicks Save button)
    // IndexedDB auto-save already handled by existing state change useEffect above
    // =================================================================================================

    const handleOpenProject = async (projectId: string) => {
        setIsLoadingProject(true);  // Prevent image refetch during load
        try {
            const response = await fetch(`${BASE_URL}/api/projects/${projectId}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to load project');
            }

            const data = await response.json();
            const project = data.project;
            console.log('Loading project data:', project);

            // ========================================================================
            // STEP 1: Restore Session Context to SessionManager
            // ========================================================================
            if (project.session_context) {
                console.log('[LOAD] Restoring session context to SessionManager', project.session_context);
                // Note: We need to add restoreFromProject to SessionManager first
                if (sessionManager) {
                    sessionManager.restoreFromProject(project.session_context);
                } else {
                    console.warn('[LOAD] SessionManager not available');
                }
            } else {
                console.warn('[LOAD] ⚠️ No session_context in saved project - session continuity will be broken');
            }

            // ========================================================================
            // STEP 2: Restore Workflow State
            // ========================================================================
            if (project.workflow_state) {
                console.log('[LOAD] Restoring workflow state:', project.workflow_state);
                // We will use this momentarily to set showResults
            } else {
                console.warn('[LOAD] ⚠️ No workflow_state in saved project');
            }

            // ========================================================================
            // STEP 3: Restore Intent Data
            // ========================================================================
            if (project.intent_data) {
                console.log('[LOAD] Restoring intent data:', project.intent_data);
                // Intent is informational, no action needed for Project page
            } else {
                console.warn('[LOAD] ⚠️ No intent_data in saved project');
            }

            // Do not clear existing session state before loading project

            // Restore project state with debugging
            // Restore product type from loaded project
            const restoredProductType = project.productType || project.product_type || projectName;
            const savedInstruments = project.identifiedInstruments || project.identified_instruments || [];
            const savedAccessories = project.identifiedAccessories || project.identified_accessories || [];

            console.log('Restoring project name:', project.projectName || project.project_name);
            setProjectName(project.projectName || project.project_name || 'Project');

            console.log('Restoring requirements:', (project.initialRequirements || project.initial_requirements || '').substring(0, 100));
            setRequirements(project.initialRequirements || project.initial_requirements || '');

            console.log('Restoring instruments count:', savedInstruments.length);
            setInstruments(savedInstruments);

            console.log('Restoring accessories count:', savedAccessories.length);
            setAccessories(savedAccessories);

            // Set product type in tabStates for use in API calls
            setTabStates(prev => ({
                ...prev,
                project: {
                    ...(prev.project || {}),
                    currentProductType: restoredProductType
                }
            }));

            // Restore generic images
            const savedGenericImages = project.genericImages || project.generic_images || {};
            if (Object.keys(savedGenericImages).length > 0) {
                console.log('Restoring generic images:', Object.keys(savedGenericImages).length);
                // Convert all relative URLs to absolute URLs for deployment compatibility
                const absoluteGenericImages: Record<string, string> = {};
                Object.entries(savedGenericImages).forEach(([key, url]) => {
                    const absoluteUrl = getAbsoluteImageUrl(url as string);
                    if (absoluteUrl) {
                        absoluteGenericImages[key] = absoluteUrl;
                    }
                });
                setGenericImages(absoluteGenericImages);
                console.log('Generic images restored from saved project, skipping refetch');
            } else {
                setGenericImages({});
            }

            // Restore Project page chat messages
            const savedChatMessages = project.projectChatMessages || project.project_chat_messages || [];
            if (savedChatMessages.length > 0) {
                console.log('Restoring Project page chat messages:', savedChatMessages.length);
                // Convert timestamp strings back to Date objects
                const restoredMessages = savedChatMessages.map((msg: any) => ({
                    ...msg,
                    timestamp: new Date(msg.timestamp)
                }));
                setChatMessages(restoredMessages);
                // Mark as history so messages don't animate on load
                isHistoryRef.current = true;

                // Allow new messages to animate after load
                setTimeout(() => {
                    isHistoryRef.current = false;
                }, 1000);
            } else {
                setChatMessages([]);
            }

            // Show results if we have instruments/accessories
            // Use savedInstruments and savedAccessories that were already extracted above
            console.log('Checking results - instruments:', savedInstruments.length, 'accessories:', savedAccessories.length);
            // ========================================================================
            // STEP 4: Restore showResults from workflow_state
            // ========================================================================
            const hasResults = savedInstruments.length > 0 || savedAccessories.length > 0;

            // Prefer workflow_state.awaiting_selection if available, otherwise use hasResults
            let shouldShowResults = hasResults;

            if (project.workflow_state?.awaiting_selection !== undefined) {
                shouldShowResults = project.workflow_state.awaiting_selection;
                console.log(`[LOAD] Using workflow_state.awaiting_selection: ${shouldShowResults}`);
            } else {
                console.log(`[LOAD] No workflow_state.awaiting_selection, using hasResults: ${shouldShowResults}`);
            }

            if (shouldShowResults) {
                console.log(`[LOAD] ✅ Showing results panel (${savedInstruments.length} instruments, ${savedAccessories.length} accessories)`);
                setShowResults(true);
            } else {
                console.log(`[LOAD] Results panel will remain hidden`);
                setShowResults(false);
            }

            // Restore search tabs and conversation states
            const savedSearchTabs = project.searchTabs || project.search_tabs || [];
            console.log('Saved search tabs:', savedSearchTabs);

            if (savedSearchTabs.length > 0) {
                console.log('Restoring search tabs...');
                // Convert search tabs to include all metadata
                const restoredTabs = savedSearchTabs.map((tab: any) => ({
                    id: tab.id || tab.tabId || generateTabId(),
                    title: tab.title || tab.name || 'Untitled',
                    input: tab.input || tab.searchInput || '',
                    isDirectSearch: tab.isDirectSearch || false,
                    productType: tab.productType,
                    itemThreadId: tab.itemThreadId || tab.item_thread_id,
                    workflowThreadId: tab.workflowThreadId || tab.workflow_thread_id,
                    mainThreadId: tab.mainThreadId || tab.main_thread_id
                }));

                setSearchTabs(restoredTabs);

                // Restore conversation states for each tab
                const conversationHistories = project.conversationHistories || project.conversation_histories || {};
                const collectedDataAll = project.collectedData || project.collected_data || {};
                const analysisResults = project.analysisResults || project.analysis_results || {};

                const restoredTabStates: Record<string, any> = {};

                for (const tab of restoredTabs) {
                    const tabId = tab.id;

                    // Get conversation history for this tab
                    const tabMessages = conversationHistories[tabId] || [];

                    // Get collected data for this tab
                    const tabCollectedData = collectedDataAll[tabId] || {};

                    // Get analysis result for this tab
                    const tabAnalysisResult = analysisResults[tabId];

                    // Get field descriptions for this tab (if available)
                    const tabFieldDescriptionsKey = `${tabId}_field_descriptions`;
                    const tabFieldDescriptions = project[tabFieldDescriptionsKey] || {};

                    restoredTabStates[tabId] = {
                        messages: tabMessages.map((msg: any) => ({
                            role: msg.role,
                            content: msg.content,
                            timestamp: new Date(msg.timestamp || Date.now())
                        })),
                        currentStep: tabMessages.length > 0 ? 'conversation' : 'initial',
                        searchSessionId: tabId,
                        collectedData: tabCollectedData,
                        analysisResult: tabAnalysisResult,
                        requirementSchema: tabAnalysisResult?.requirementSchema || null,
                        validationResult: tabAnalysisResult?.validationResult || null,
                        currentProductType: tab.productType || '',
                        fieldDescriptions: tabFieldDescriptions,
                        advancedParameters: tabAnalysisResult?.advancedParameters || {}
                    };

                    // Inject pricing data into ranked_products if it exists
                    if (tabAnalysisResult?.ranked_products) {
                        const pricingData = project.pricing?.[tabId];
                        if (pricingData) {
                            console.log(`Restoring pricing data for tab ${tabId}`);
                            tabAnalysisResult.ranked_products = tabAnalysisResult.ranked_products.map((product: any) => {
                                const productKey = `${product.vendor}-${product.productName || product.product_name}`;
                                const productPricing = pricingData[productKey];
                                if (productPricing) {
                                    return {
                                        ...product,
                                        pricing: productPricing
                                    };
                                }
                                return product;
                            });
                        }
                    }
                }

                setTabStates(restoredTabStates);
                console.log('Restored tab states for', Object.keys(restoredTabStates).length, 'tabs');

                // Restore active tab
                const savedActiveTab = project.activeTab || project.active_tab;
                if (savedActiveTab && savedActiveTab !== 'Project') {
                    // Find matching tab by title
                    const matchingTab = restoredTabs.find((t: any) => t.title === savedActiveTab);
                    if (matchingTab) {
                        console.log('Restoring active tab by title match:', matchingTab.id);
                        setActiveTab(matchingTab.id);
                        setPreviousTab('project');
                    } else {
                        console.log('Could not find tab with title:', savedActiveTab, ', setting to first search tab');
                        setActiveTab(restoredTabs[0].id);
                        setPreviousTab('project');
                    }
                } else if (savedSearchTabs.length > 0) {
                    console.log('No saved active tab, setting to first search tab:', restoredTabs[0].id);
                    setActiveTab(restoredTabs[0].id);
                    setPreviousTab('project');
                }
            } else {
                console.log('No search tabs to restore');
                // Clear tab states if no search tabs
                setTabStates({});
            }

            // Only reset to project tab if no search tabs were restored and no active tab was saved
            const savedActiveTab = project.activeTab || project.active_tab;
            if (savedSearchTabs.length === 0 && !savedActiveTab) {
                console.log('No search tabs and no saved active tab, setting active tab to project');
                setActiveTab('project');
                setPreviousTab('project');
            } else {
                console.log('Search tabs or saved active tab found, active tab should be restored above');
            }

            console.log('Project loading completed successfully');

            // Log field descriptions and metadata if available
            const fieldDescriptions = project.fieldDescriptions || project.field_descriptions;
            if (fieldDescriptions) {
                console.log('Project field descriptions loaded:', Object.keys(fieldDescriptions).length, 'fields documented');
                // setFieldDescriptions(fieldDescriptions); // DEPRECATED: Lazy loaded now
            }

            const workflowPosition = project.workflowPosition || project.workflow_position;
            if (workflowPosition) {
                console.log('Project workflow position:', workflowPosition);
            }

            const userInteractions = project.userInteractions || project.user_interactions;
            if (userInteractions) {
                console.log('Project user interactions summary:', userInteractions);
            }

            const projectMetadata = project.projectMetadata || project.project_metadata;
            if (projectMetadata) {
                console.log('Project metadata loaded:', projectMetadata);
            }

            // Set the current project ID for future saves (so it updates instead of creating new)
            console.log('Setting current project ID for updates:', projectId);
            setCurrentProjectId(projectId);

            // Also restore the project's current step if available
            const projectCurrentStep = project.currentStep || project.current_step;
            if (projectCurrentStep) {
                console.log('Project was at step:', projectCurrentStep);
            }

            toast({
                title: "Project Loaded",
                description: `"${project.projectName || project.project_name}" has been loaded successfully. ${savedSearchTabs.length} search tabs restored.`,
            });
        } catch (error: any) {
            toast({
                title: "Load Failed",
                description: error.message || "Failed to load project",
                variant: "destructive",
            });
        } finally {
            setIsLoadingProject(false);
        }
    };


    // ✅ Save scroll position before leaving Project tab
    const handleTabChange = (newTab: string) => {
        if (activeTab === 'project' && projectScrollRef.current) {
            setSavedScrollPosition(projectScrollRef.current.scrollTop);
        }
        setPreviousTab(activeTab);
        setActiveTab(newTab);

        // Sync URL with tab state
        if (newTab === 'project') {
            navigate('/solution');
        } else {
            navigate('/solution/search');
        }
    };

    // ✅ Restore scroll position when returning to Project tab
    useEffect(() => {
        if (activeTab === 'project' && projectScrollRef.current && savedScrollPosition > 0) {
            // Use requestAnimationFrame for more reliable DOM timing
            requestAnimationFrame(() => {
                if (projectScrollRef.current) {
                    projectScrollRef.current.scrollTop = savedScrollPosition;
                }
                // Double-check with a small delay as fallback
                setTimeout(() => {
                    if (projectScrollRef.current && projectScrollRef.current.scrollTop !== savedScrollPosition) {
                        projectScrollRef.current.scrollTop = savedScrollPosition;
                    }
                }, 50);
            });
        }
    }, [activeTab, savedScrollPosition]);

    // Additional effect to handle scroll position restoration after content changes
    useEffect(() => {
        if (activeTab === 'project' && projectScrollRef.current && savedScrollPosition > 0) {
            const timer = setTimeout(() => {
                if (projectScrollRef.current) {
                    projectScrollRef.current.scrollTop = savedScrollPosition;
                }
            }, 150); // Longer delay to ensure content including images is loaded

            return () => clearTimeout(timer);
        }
    }, [activeTab, showResults, instruments, accessories]);

    // Sync URL with active tab
    useEffect(() => {
        // Skip URL manipulation if we are initially loading a search route
        // This allows the initial search tab creation logic (below) to run first
        const path = window.location.pathname;
        if (activeTab === 'project' && (path.includes('/solution/search') || path === '/search')) {
            return;
        }

        if (activeTab === 'project') {
            // Only update if not already correct to minimize history noise
            if (!path.endsWith('/solution') && path !== '/' && !path.includes('/search')) {
                navigate('/solution', { replace: true });
            }
        } else {
            if (!path.includes('/solution/search')) {
                navigate('/solution/search', { replace: true });
            }
        }
    }, [activeTab, navigate]);

    // Handle initial route for /search or /solution/search
    useEffect(() => {
        const path = window.location.pathname;
        // If user lands on search route but has no search tabs, create one
        if ((path.includes('/solution/search') || path === '/search') && searchTabs.length === 0) {
            const newTabId = `search_${Date.now()}`;
            const newTab = {
                id: newTabId,
                title: 'Product Search',
                input: '',
                isDirectSearch: true
            };
            setSearchTabs([newTab]);
            setActiveTab(newTabId);
        }
    }, []); // Run once on mount

    const resetDuplicateDialog = () => {
        setDuplicateNameDialogOpen(false);
        setDuplicateProjectName(null);
        setAutoRenameSuggestion(null);
        setDuplicateDialogError(null);
    };

    const handleDuplicateNameChangeConfirm = () => {
        const trimmed = (duplicateDialogNameInput || '').trim();
        if (!trimmed) {
            setDuplicateDialogError('Project name is required');
            return;
        }

        resetDuplicateDialog();
        handleSaveProject(trimmed);
    };

    const handleDuplicateNameAutoRename = async () => {
        const baseName = (duplicateProjectName || projectName || '').trim() || 'Project';
        let suggested = autoRenameSuggestion || `${baseName} (1)`;
        try {
            // Try to compute next available suggestion based on existing projects
            const listResp = await fetch(`${BASE_URL}/api/projects`, { credentials: 'include' });
            if (listResp.ok) {
                const listData = await listResp.json();
                suggested = computeNextDuplicateName(baseName, listData.projects || []);
            }
        } catch (e) {
            // ignore and use fallback
        }

        resetDuplicateDialog();

        // Save immediately with the suggested name, and avoid showing the duplicate dialog again for this attempt
        handleSaveProject(suggested, { skipDuplicateDialog: true });
    };

    return (
        <div className="min-h-screen w-full app-glass-gradient flex flex-col animate-in fade-in duration-300">
            {/* Header is now MainHeader */}
            {/* Header is now MainHeader */}
            <MainHeader
                onSave={() => handleSaveProject()}
                onNew={handleNewProject}
                onProjectSelect={handleOpenProject}
                onProjectDelete={handleProjectDelete}
            >
                {searchTabs.length > 0 && (
                    <div className="max-w-[calc(100vw-330px)] min-w-0">
                        <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
                            <TabsList className="w-full bg-transparent p-0 h-auto">
                                <div className="flex items-center gap-2 overflow-x-auto whitespace-nowrap">
                                    <div className="flex items-center gap-2">
                                        <TabsTrigger
                                            value="project"
                                            className="rounded-lg px-4 py-2 text-base font-bold text-foreground border-2 border-transparent bg-transparent data-[state=active]:border-white/30 data-[state=active]:bg-white/20 data-[state=active]:backdrop-blur-md whitespace-nowrap flex-shrink-0"
                                        >
                                            {!editingProjectName ? (
                                                <span className="inline-flex items-center gap-2">
                                                    <span className="block">{projectName}</span>
                                                    {currentProjectId && (
                                                        <span className="ml-2 text-[10px] text-[#0F6CBD] uppercase tracking-wider font-medium">
                                                            Saved
                                                        </span>
                                                    )}
                                                    <span
                                                        onClick={(e) => {
                                                            // Prevent tab switch when clicking edit
                                                            e.stopPropagation();
                                                            e.preventDefault();
                                                            setEditingProjectName(true);
                                                            setTimeout(() => editNameInputRef.current?.focus(), 0);
                                                        }}
                                                        title="Edit project name"
                                                        className="ml-2 text-muted-foreground hover:text-foreground text-sm px-2 py-1 rounded cursor-pointer"
                                                        role="button"
                                                        tabIndex={0}
                                                        aria-label="Edit project name"
                                                        onKeyDown={(e) => {
                                                            if (e.key === 'Enter' || e.key === ' ') {
                                                                e.preventDefault();
                                                                setEditingProjectName(true);
                                                                setTimeout(() => editNameInputRef.current?.focus(), 0);
                                                            }
                                                        }}
                                                    >
                                                        ✎
                                                    </span>
                                                </span>
                                            ) : (
                                                <input
                                                    ref={editNameInputRef}
                                                    value={editProjectNameValue}
                                                    onChange={(e) => setEditProjectNameValue(e.target.value)}
                                                    onBlur={() => {
                                                        const v = (editProjectNameValue || '').trim() || 'Project';
                                                        setProjectName(v);
                                                        setEditingProjectName(false);
                                                    }}
                                                    onKeyDown={(e) => {
                                                        if (e.key === 'Enter') {
                                                            e.preventDefault();
                                                            const v = (editProjectNameValue || '').trim() || 'Project';
                                                            setProjectName(v);
                                                            setEditingProjectName(false);
                                                        } else if (e.key === 'Escape') {
                                                            setEditProjectNameValue(projectName);
                                                            setEditingProjectName(false);
                                                        }
                                                    }}
                                                    className="text-sm px-2 py-1 rounded-md border border-border bg-background min-w-[160px]"
                                                    autoFocus
                                                />
                                            )}
                                        </TabsTrigger>
                                    </div>
                                    {searchTabs.map((tab, index) => (
                                        <div key={tab.id} className="flex items-center min-w-0 flex-shrink">
                                            <TabsTrigger
                                                value={tab.id}
                                                className="rounded-lg px-3 py-1 text-sm data-[state=active]:bg-secondary data-[state=active]:text-secondary-foreground min-w-0"
                                            >
                                                <span className="truncate block w-full">{tab.title}</span>
                                            </TabsTrigger>
                                            <button
                                                onClick={() => closeSearchTab(tab.id)}
                                                className="ml-1 text-muted-foreground hover:text-foreground text-lg flex-shrink-0"
                                                aria-label={`Close ${tab.title}`}
                                            >
                                                ×
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </TabsList>
                        </Tabs>
                    </div>
                )}
            </MainHeader>



            {/* Duplicate project name dialog */}
            <AlertDialog
                open={duplicateNameDialogOpen}
                onOpenChange={(open) => {
                    if (!open) {
                        resetDuplicateDialog();
                    } else {
                        setDuplicateNameDialogOpen(open);
                    }
                }}
            >
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Project name already exists</AlertDialogTitle>
                        <AlertDialogDescription>
                            {duplicateProjectName
                                ? `"${duplicateProjectName}" is already present. Do you want to change the project name, or save it as "${(autoRenameSuggestion || `${duplicateProjectName} (1)`)}"?`
                                : 'A project with this name is already present. Do you want to change the project name, or save it with a default suffix (1)?'}
                        </AlertDialogDescription>
                        <div className="mt-4 space-y-2">
                            <label htmlFor="duplicate-project-name-input" className="text-sm font-medium">
                                New project name
                            </label>
                            <input
                                id="duplicate-project-name-input"
                                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                                value={duplicateDialogNameInput}
                                onChange={(e) => {
                                    setDuplicateDialogNameInput(e.target.value);
                                    if (duplicateDialogError) {
                                        setDuplicateDialogError(null);
                                    }
                                }}
                                autoFocus
                            />
                            {duplicateDialogError && (
                                <p className="text-xs text-destructive">{duplicateDialogError}</p>
                            )}
                        </div>
                    </AlertDialogHeader>
                    <button
                        type="button"
                        onClick={resetDuplicateDialog}
                        className="absolute right-3 top-3 rounded-full p-1 text-muted-foreground hover:text-foreground hover:bg-muted"
                        aria-label="Close duplicate name dialog"
                    >
                        <X className="h-4 w-4" />
                    </button>
                    <AlertDialogFooter>
                        <AlertDialogAction
                            onClick={handleDuplicateNameAutoRename}
                        >
                            Use suggested name
                        </AlertDialogAction>
                        <AlertDialogAction onClick={handleDuplicateNameChangeConfirm}>
                            Save new name
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* Main Content */}
            <div className="flex-1 relative">
                {/* Right corner dock button - only show on project tab when results are available */}
                {activeTab === 'project' && showResults && (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="fixed top-36 right-0 z-50 btn-glass-secondary border-0 shadow-lg rounded-r-none"
                        onClick={() => setIsRightDocked(!isRightDocked)}
                        aria-label={isRightDocked ? "Expand right panel" : "Collapse right panel"}
                    >
                        {isRightDocked ? <ChevronLeft /> : <ChevronRight />}
                    </Button>
                )}

                <div className="w-full h-full flex">
                    {/* Main Content Area - Always mounted, use CSS to show/hide */}
                    <div className={`${activeTab === 'project' ? 'contents' : 'hidden'}`}>
                        <>
                            {/* Centered Input Section */}
                            <div className={`transition-all duration-500 ease-in-out ${!isRightDocked && showResults ? 'w-1/2' : 'w-full'} h-screen overflow-y-auto custom-no-scrollbar pt-24`} ref={projectScrollRef}>
                                {/* Initial Welcome Screen - with glass-card wrapper */}
                                {chatMessages.length === 0 ? (
                                    <div className="mx-auto max-w-[900px] px-4 md:px-6 min-h-full flex items-center justify-center">
                                        <div className="w-full p-4 md:p-6 glass-card animate-in fade-in duration-500 my-6">

                                            {/* Header - Only show when no chat messages */}
                                            {chatMessages.length === 0 && (
                                                <>
                                                    <div className="text-center mb-6">
                                                        <div className="flex items-center justify-center gap-4 mb-4">
                                                            <div className="w-16 h-16 rounded-full overflow-hidden shadow">
                                                                <video
                                                                    src="/animation.mp4"
                                                                    autoPlay
                                                                    muted
                                                                    playsInline
                                                                    disablePictureInPicture
                                                                    controls={false}
                                                                    onContextMenu={(e) => e.preventDefault()}
                                                                    onError={(e) => {
                                                                        // Retry loading on error (handles 304 cache issues)
                                                                        const video = e.currentTarget;
                                                                        video.load();
                                                                        video.play().catch(() => { });
                                                                    }}
                                                                    className="w-full h-full object-cover pointer-events-none"
                                                                />
                                                            </div>
                                                            <h1 className="text-4xl font-bold text-foreground">
                                                                EnGenie
                                                            </h1>
                                                        </div>
                                                    </div>

                                                    {!showResults && (
                                                        <div className="text-center space-y-4 mb-8">
                                                            <h2 className="text-3xl font-normal text-muted-foreground">
                                                                Welcome, {user?.firstName || user?.username || 'User'}! what are your requirements
                                                            </h2>
                                                        </div>
                                                    )}
                                                </>
                                            )}


                                            {/* Bouncing Dots when loading */}
                                            {showThinking && (
                                                <div className="mb-4">
                                                    <div className="flex justify-start">
                                                        <div className="max-w-[80%] flex items-start space-x-2">
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
                                                </div>
                                            )}

                                            {/* Input Form */}
                                            <form onSubmit={handleSubmit}>
                                                <div className="relative group">
                                                    <div className={`relative w-full rounded-[26px] transition-all duration-300 focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-transparent hover:scale-[1.01] flex flex-col`}
                                                        style={{
                                                            boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
                                                            WebkitBackdropFilter: 'blur(12px)',
                                                            backdropFilter: 'blur(12px)',
                                                            backgroundColor: 'rgba(255, 255, 255, 0.3)',
                                                            border: '1px solid rgba(255, 255, 255, 0.4)',
                                                            color: 'rgba(0, 0, 0, 0.8)'
                                                        }}>
                                                        <Textarea
                                                            ref={textareaRef}
                                                            value={requirements}
                                                            onChange={(e) => setRequirements(e.target.value)}
                                                            onKeyDown={handleKeyPress}
                                                            className={`w-full bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/70 resize-none text-base p-4 md:p-6 text-lg leading-relaxed shadow-none custom-no-scrollbar ${showResults ? 'min-h-[80px]' : 'min-h-[120px]'}`}
                                                            style={{
                                                                backgroundColor: 'transparent',
                                                                boxShadow: 'none',
                                                                color: 'inherit'
                                                            }}
                                                            placeholder="Describe the product you are looking for..."
                                                            disabled={isLoading}
                                                        />

                                                        {/* File Display & Buttons Bar - Footer inside the glass box */}
                                                        <div className="flex items-center justify-between px-4 pb-4 md:px-6 md:pb-6 pt-2">
                                                            <div className="flex items-center gap-2">
                                                                {/* Attached File Badge - visible before submit */}
                                                                {attachedFile && (
                                                                    <div className="flex items-center gap-2 p-1.5 px-3 glass-card bg-primary/10 border-0 rounded-full text-xs">
                                                                        {isExtracting ? (
                                                                            <Loader2 className="h-3 w-3 text-primary animate-spin" />
                                                                        ) : (
                                                                            <FileText className="h-3 w-3 text-primary" />
                                                                        )}
                                                                        <span className="text-primary truncate max-w-[100px]">{attachedFile.name}</span>
                                                                        <button
                                                                            type="button"
                                                                            onClick={handleRemoveFile}
                                                                            className="text-primary/70 hover:text-primary"
                                                                            title="Remove file"
                                                                        >
                                                                            <X className="h-3 w-3" />
                                                                        </button>
                                                                    </div>
                                                                )}

                                                                {/* Hidden file input */}
                                                                <input
                                                                    ref={fileInputRef}
                                                                    type="file"
                                                                    accept=".pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.bmp,.tiff"
                                                                    onChange={handleFileSelect}
                                                                    className="hidden"
                                                                />
                                                            </div>

                                                            {/* Action Buttons */}
                                                            <div className="flex items-center gap-1">
                                                                {/* Attach Button */}
                                                                <Button
                                                                    type="button"
                                                                    onClick={() => fileInputRef.current?.click()}
                                                                    disabled={isLoading || isExtracting}
                                                                    className="w-8 h-8 rounded-full hover:bg-transparent transition-all duration-300 flex-shrink-0 text-muted-foreground hover:text-primary hover:scale-110"
                                                                    variant="ghost"
                                                                    size="icon"
                                                                    title="Attach file"
                                                                >
                                                                    <Upload className="h-4 w-4" />
                                                                </Button>

                                                                {/* Submit Button */}
                                                                <Button
                                                                    type="submit"
                                                                    disabled={isLoading || isExtracting || (!requirements.trim() && !extractedText.trim())}
                                                                    className={`w-8 h-8 p-0 rounded-full transition-all duration-300 flex-shrink-0 hover:bg-transparent ${(!requirements.trim() && !extractedText.trim()) ? 'text-muted-foreground' : 'text-primary hover:scale-110'}`}
                                                                    variant="ghost"
                                                                    size="icon"
                                                                    title="Submit"
                                                                >
                                                                    {isLoading || isExtracting ? (
                                                                        <Loader2 className="h-4 w-4 animate-spin text-primary" />
                                                                    ) : (
                                                                        <Send className="h-4 w-4" />
                                                                    )}
                                                                </Button>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </form>
                                        </div>
                                    </div>
                                ) : (
                                    /* Full Screen Chat Mode - Matching ChatInterface layout exactly */
                                    <div className="flex-1 flex flex-col h-full bg-transparent relative">
                                        {/* Header with Logo and EnGenie name - Same as ChatInterface */}
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
                                                    EnGenie <span className="text-sky-500">♦</span> Solution
                                                </h1>
                                            </div>
                                        </div>

                                        {/* Chat Messages Area - Same padding as ChatInterface */}
                                        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-no-scrollbar pb-32">
                                            {chatMessages.map((message) => (
                                                <MessageRow
                                                    key={message.id}
                                                    message={message}
                                                    isHistory={isHistoryRef.current}
                                                />
                                            ))}

                                            {/* Bouncing Dots Loading Indicator */}
                                            {showThinking && (
                                                <div className="flex justify-start">
                                                    <div className="max-w-[80%] flex items-start space-x-2">
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

                                        {/* Input Form - Fixed at bottom like ChatInterface */}
                                        <div className="absolute bottom-0 left-0 right-0 p-4 bg-transparent">
                                            <div className="max-w-4xl mx-auto px-2 md:px-8">
                                                <form onSubmit={handleSubmit}>
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
                                                                value={requirements}
                                                                onChange={(e) => setRequirements(e.target.value)}
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
                                                                placeholder="Type your message here..."
                                                                disabled={isLoading}
                                                            />

                                                            {/* Attached File Badge - visible before submit */}
                                                            {attachedFile && (
                                                                <div className="absolute bottom-1.5 left-3 flex items-center gap-2 p-1 px-2 bg-primary/10 rounded-full text-xs">
                                                                    {isExtracting ? (
                                                                        <Loader2 className="h-3 w-3 text-primary animate-spin" />
                                                                    ) : (
                                                                        <FileText className="h-3 w-3 text-primary" />
                                                                    )}
                                                                    <span className="text-primary truncate max-w-[80px]">{attachedFile.name}</span>
                                                                    <button
                                                                        type="button"
                                                                        onClick={handleRemoveFile}
                                                                        className="text-primary/70 hover:text-primary"
                                                                        title="Remove file"
                                                                    >
                                                                        <X className="h-3 w-3" />
                                                                    </button>
                                                                </div>
                                                            )}

                                                            {/* Action Buttons - positioned like ChatInterface */}
                                                            <div className="absolute bottom-1.5 right-1.5 flex items-center gap-0.5">
                                                                {/* Attach Button */}
                                                                <Button
                                                                    type="button"
                                                                    onClick={() => fileInputRef.current?.click()}
                                                                    disabled={isLoading || isExtracting}
                                                                    className="w-8 h-8 rounded-full hover:bg-transparent transition-all duration-300 flex-shrink-0 text-muted-foreground hover:text-primary hover:scale-110"
                                                                    variant="ghost"
                                                                    size="icon"
                                                                    title="Attach file"
                                                                >
                                                                    <Upload className="h-4 w-4" />
                                                                </Button>

                                                                {/* Submit Button */}
                                                                <Button
                                                                    type="submit"
                                                                    disabled={isLoading || isExtracting || (!requirements.trim() && !extractedText.trim())}
                                                                    className={`w-8 h-8 p-0 rounded-full transition-all duration-300 flex-shrink-0 hover:bg-transparent ${(!requirements.trim() && !extractedText.trim()) ? 'text-muted-foreground' : 'text-primary hover:scale-110'}`}
                                                                    variant="ghost"
                                                                    size="icon"
                                                                    title="Submit"
                                                                >
                                                                    {isLoading || isExtracting ? (
                                                                        <Loader2 className="h-4 w-4 animate-spin text-primary" />
                                                                    ) : (
                                                                        <Send className="h-4 w-4" />
                                                                    )}
                                                                </Button>
                                                            </div>
                                                        </div>

                                                    </div>

                                                    {/* Hidden file input */}
                                                    <input
                                                        ref={fileInputRef}
                                                        type="file"
                                                        accept=".pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.bmp,.tiff"
                                                        onChange={handleFileSelect}
                                                        className="hidden"
                                                    />
                                                </form>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Separator Line between Chat and Right Panel */}
                            {showResults && !isRightDocked && (
                                <div
                                    className="w-1.5 h-screen bg-border hover:bg-[#5FB3E6] transition-colors duration-500 ease-in-out flex-shrink-0"
                                    style={{ zIndex: 20 }}
                                />
                            )}

                            {/* Right Panel - Results */}
                            {showResults && (
                                <div
                                    ref={rightPanelScrollRef}
                                    className={`
                                    h-[calc(100vh-8rem)] mt-32 mr-1
                                    overflow-y-auto custom-visible-scrollbar
                                    transition-all duration-500 ease-in-out origin-right
                                    ${!isRightDocked ? 'w-[47%] opacity-100' : 'w-0 opacity-0 overflow-hidden'}
                                `}>
                                    <div className="h-fit ml-4 mb-2
                                        bg-white/60 dark:bg-slate-900/60
                                        backdrop-blur-md border-2 border-[#45A4DE] shadow-xl rounded-2xl
                                        p-6">
                                        <div className="w-full">
                                            {/* Tabs for Instruments and Accessories */}
                                            <Tabs value={rightPanelTab} onValueChange={(value) => setRightPanelTab(value as 'instruments' | 'accessories')} className="w-full">
                                                <TabsList className="grid w-full grid-cols-2 mb-10">
                                                    <TabsTrigger value="instruments" className="data-[state=active]:bg-primary data-[state=active]:text-white">
                                                        Instruments ({instruments.length})
                                                    </TabsTrigger>
                                                    <TabsTrigger value="accessories" className="data-[state=active]:bg-primary data-[state=active]:text-white">
                                                        Accessories ({accessories.length})
                                                    </TabsTrigger>
                                                </TabsList>

                                                {/* Instruments Tab Content */}
                                                {rightPanelTab === 'instruments' && (
                                                    <div className="space-y-6">
                                                        {instruments.length > 0 ? (
                                                            <div className="space-y-8">
                                                                {instruments.map((instrument, index) => (
                                                                    <div
                                                                        key={index}
                                                                        className="rounded-xl bg-gradient-to-br from-[#F5FAFC]/90 to-[#EAF6FB]/90 dark:from-slate-900/90 dark:to-slate-900/50 backdrop-blur-2xl border border-white/20 dark:border-slate-700/30 shadow-2xl transition-all duration-300 ease-in-out hover:shadow-3xl hover:scale-[1.01] p-8 space-y-6"
                                                                    >
                                                                        {/* Category (primary) and Product Name (secondary) - smart category display */}
                                                                        <div className="flex items-start justify-between">
                                                                            <div className="space-y-1">
                                                                                <div className="flex items-center gap-3">
                                                                                    <h3 className="text-xl font-semibold">
                                                                                        {index + 1}. {(() => {
                                                                                            const cat = instrument.category || '';
                                                                                            const name = instrument.productName || '';
                                                                                            const isGeneric = cat.toLowerCase() === 'accessories' || cat.toLowerCase() === 'accessory';
                                                                                            if (isGeneric && name) {
                                                                                                const parts = name.split(' for ');
                                                                                                return parts[0] || name;
                                                                                            }
                                                                                            return cat || name;
                                                                                        })()}{instrument.quantity ? ` (${instrument.quantity})` : ''}
                                                                                    </h3>
                                                                                    {(() => {
                                                                                        const imageKey = getInstrumentImageKey(instrument);
                                                                                        const imageUrl = genericImages[imageKey];

                                                                                        if (imageUrl) {
                                                                                            return (
                                                                                                <Tooltip>
                                                                                                    <TooltipTrigger asChild>
                                                                                                        <div className="w-10 h-10 rounded-md overflow-hidden border border-border/50 bg-white flex-shrink-0 cursor-pointer hover:border-primary/50 transition-colors shadow-sm">
                                                                                                            <img
                                                                                                                src={imageUrl}
                                                                                                                alt={imageKey}
                                                                                                                className="w-full h-full object-contain p-1 mix-blend-multiply"
                                                                                                            />
                                                                                                        </div>
                                                                                                    </TooltipTrigger>
                                                                                                    <TooltipContent side="bottom" align="center" className="p-0 border-none bg-transparent shadow-none" sideOffset={10}>
                                                                                                        <div className="w-64 h-64 bg-white rounded-xl shadow-2xl border border-border/50 p-4 overflow-hidden flex items-center justify-center animate-in fade-in zoom-in-95 duration-200">
                                                                                                            <img
                                                                                                                src={imageUrl}
                                                                                                                alt={imageKey}
                                                                                                                className="max-w-full max-h-full object-contain mix-blend-multiply"
                                                                                                            />
                                                                                                        </div>
                                                                                                    </TooltipContent>
                                                                                                </Tooltip>
                                                                                            );
                                                                                        }

                                                                                        if (loadingImages.has(imageKey) || regeneratingImages.has(imageKey)) {
                                                                                            return (
                                                                                                <div className="w-10 h-10 rounded-md flex items-center justify-center bg-muted/20 border border-border/50 flex-shrink-0 shadow-sm">
                                                                                                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                                                                                                </div>
                                                                                            );
                                                                                        }

                                                                                        return (
                                                                                            <div className="w-10 h-10 rounded-md flex items-center justify-center bg-muted/30 border border-border/50 flex-shrink-0 text-muted-foreground/30 shadow-sm">
                                                                                                <span className="text-[10px]">Img</span>
                                                                                            </div>
                                                                                        );
                                                                                    })()}
                                                                                </div>

                                                                                <p className="text-muted-foreground">
                                                                                    {instrument.productName}
                                                                                </p>
                                                                            </div>
                                                                            <div className="flex flex-col gap-2">
                                                                                <Button
                                                                                    onClick={() => handleRun(instrument, index)}
                                                                                    className="rounded-xl w-10 h-10 p-0 flex items-center justify-center bg-primary/40 hover:bg-primary text-primary hover:text-white transition-all duration-300 hover:scale-110"
                                                                                    variant="ghost"
                                                                                >
                                                                                    <Play className="h-4 w-4" />
                                                                                </Button>
                                                                                <button
                                                                                    onClick={() => toggleInstrumentCollapse(index)}
                                                                                    className="w-10 h-10 flex items-center justify-center text-muted-foreground hover:text-foreground transition-all duration-200 cursor-pointer group"
                                                                                >
                                                                                    {collapsedInstruments.has(index) ? (
                                                                                        <ChevronDown className="h-4 w-4 group-hover:scale-125 transition-transform" />
                                                                                    ) : (
                                                                                        <ChevronUp className="h-4 w-4 group-hover:scale-125 transition-transform" />
                                                                                    )}
                                                                                </button>
                                                                            </div>
                                                                        </div>

                                                                        {/* Specifications */}
                                                                        {!collapsedInstruments.has(index) && Object.keys(instrument.specifications).length > 0 && (
                                                                            <div className="space-y-2">
                                                                                <h4 className="font-medium text-sm text-muted-foreground">
                                                                                    Specifications:
                                                                                </h4>

                                                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                                                    {Object.entries(instrument.specifications).map(([key, value]) => {
                                                                                        const prettyKey = prettifyKey(key);
                                                                                        const { displayValue, source, confidence } = parseSpecValue(value);
                                                                                        const sourceLabel = getSourceLabel(source);

                                                                                        return (
                                                                                            <div key={key} className="text-sm group break-words">
                                                                                                <span className="font-medium">{prettyKey}:</span>{' '}
                                                                                                <span className="text-muted-foreground">{displayValue}</span>
                                                                                                {sourceLabel && (
                                                                                                    <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                                                        {sourceLabel}
                                                                                                    </span>
                                                                                                )}
                                                                                                {confidence && confidence < 0.7 && (
                                                                                                    <span className="ml-1 text-xs text-amber-600" title={`Confidence: ${Math.round(confidence * 100)}%`}>
                                                                                                        ⚠️
                                                                                                    </span>
                                                                                                )}
                                                                                            </div>
                                                                                        );
                                                                                    })}
                                                                                </div>
                                                                            </div>
                                                                        )}

                                                                        {/* Sample Input Preview */}
                                                                        {!collapsedInstruments.has(index) && (
                                                                            <div className="pt-3 border-t">
                                                                                <p className="text-xs text-muted-foreground mb-2">Sample Input:</p>
                                                                                <p className="text-sm bg-muted p-3 rounded-lg font-mono">
                                                                                    {formatSampleInput(instrument.sampleInput)}
                                                                                </p>
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ) : (
                                                            <div className="text-center py-8 text-muted-foreground">
                                                                No instruments identified
                                                            </div>
                                                        )}
                                                    </div>
                                                )}

                                                {/* Accessories Tab Content */}
                                                {rightPanelTab === 'accessories' && (
                                                    <div className="space-y-6">
                                                        {accessories.length > 0 ? (
                                                            <div className="space-y-8">
                                                                {accessories.map((accessory, index) => (
                                                                    <div
                                                                        key={index}
                                                                        className="rounded-xl bg-gradient-to-br from-[#F5FAFC]/90 to-[#EAF6FB]/90 dark:from-slate-900/90 dark:to-slate-900/50 backdrop-blur-2xl border border-white/20 dark:border-slate-700/30 shadow-2xl transition-all duration-300 ease-in-out hover:shadow-3xl hover:scale-[1.01] p-6 space-y-4"
                                                                    >
                                                                        {/* Accessory Category (primary) and Name (secondary) - extract type from name if category is generic */}
                                                                        <div className="flex items-start justify-between">
                                                                            <div className="space-y-1">
                                                                                <div className="flex items-center gap-3">
                                                                                    <h3 className="text-xl font-semibold">
                                                                                        {index + 1}. {(() => {
                                                                                            const cat = accessory.category || '';
                                                                                            const name = accessory.accessoryName || '';
                                                                                            const isGeneric = cat.toLowerCase() === 'accessories' || cat.toLowerCase() === 'accessory';
                                                                                            if (isGeneric && name) {
                                                                                                const parts = name.split(' for ');
                                                                                                return parts[0] || name;
                                                                                            }
                                                                                            return cat || name;
                                                                                        })()}{accessory.quantity ? ` (${accessory.quantity})` : ''}
                                                                                    </h3>
                                                                                    {(() => {
                                                                                        const imageKey = getAccessoryImageKey(accessory);
                                                                                        const imageUrl = genericImages[imageKey];

                                                                                        if (imageUrl) {
                                                                                            return (
                                                                                                <Tooltip>
                                                                                                    <TooltipTrigger asChild>
                                                                                                        <div className="w-10 h-10 rounded-md overflow-hidden border border-border/50 bg-white flex-shrink-0 cursor-pointer hover:border-primary/50 transition-colors shadow-sm">
                                                                                                            <img
                                                                                                                src={imageUrl}
                                                                                                                alt={imageKey}
                                                                                                                className="w-full h-full object-contain p-1 mix-blend-multiply"
                                                                                                            />
                                                                                                        </div>
                                                                                                    </TooltipTrigger>
                                                                                                    <TooltipContent side="bottom" align="center" className="p-0 border-none bg-transparent shadow-none" sideOffset={10}>
                                                                                                        <div className="w-64 h-64 bg-white rounded-xl shadow-2xl border border-border/50 p-4 overflow-hidden flex items-center justify-center animate-in fade-in zoom-in-95 duration-200">
                                                                                                            <img
                                                                                                                src={imageUrl}
                                                                                                                alt={imageKey}
                                                                                                                className="max-w-full max-h-full object-contain mix-blend-multiply"
                                                                                                            />
                                                                                                        </div>
                                                                                                    </TooltipContent>
                                                                                                </Tooltip>
                                                                                            );
                                                                                        }

                                                                                        if (loadingImages.has(imageKey) || regeneratingImages.has(imageKey)) {
                                                                                            return (
                                                                                                <div className="w-10 h-10 rounded-md flex items-center justify-center bg-muted/20 border border-border/50 flex-shrink-0 shadow-sm">
                                                                                                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                                                                                                </div>
                                                                                            );
                                                                                        }

                                                                                        return (
                                                                                            <div className="w-10 h-10 rounded-md flex items-center justify-center bg-muted/30 border border-border/50 flex-shrink-0 text-muted-foreground/30 shadow-sm">
                                                                                                <span className="text-[10px]">Img</span>
                                                                                            </div>
                                                                                        );
                                                                                    })()}
                                                                                </div>

                                                                                <p className="text-muted-foreground">
                                                                                    {accessory.accessoryName}
                                                                                </p>
                                                                            </div>
                                                                            <div className="flex flex-col gap-2">
                                                                                <Button
                                                                                    onClick={() => handleRunAccessory(accessory, index)}
                                                                                    className="rounded-xl w-10 h-10 p-0 flex items-center justify-center bg-primary/40 hover:bg-primary text-primary hover:text-white transition-all duration-300 hover:scale-110"
                                                                                    variant="ghost"
                                                                                >
                                                                                    <Play className="h-4 w-4" />
                                                                                </Button>
                                                                                <button
                                                                                    onClick={() => toggleAccessoryCollapse(index)}
                                                                                    className="w-10 h-10 flex items-center justify-center text-muted-foreground hover:text-foreground transition-all duration-200 cursor-pointer group"
                                                                                >
                                                                                    {collapsedAccessories.has(index) ? (
                                                                                        <ChevronDown className="h-4 w-4 group-hover:scale-125 transition-transform" />
                                                                                    ) : (
                                                                                        <ChevronUp className="h-4 w-4 group-hover:scale-125 transition-transform" />
                                                                                    )}
                                                                                </button>
                                                                            </div>
                                                                        </div>

                                                                        {/* Specifications */}
                                                                        {!collapsedAccessories.has(index) && Object.keys(accessory.specifications).length > 0 && (
                                                                            <div className="space-y-2">
                                                                                <h4 className="font-medium text-sm text-muted-foreground">
                                                                                    Specifications:
                                                                                </h4>

                                                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                                                    {Object.entries(accessory.specifications).map(([key, value]) => {
                                                                                        const prettyKey = prettifyKey(key);
                                                                                        const { displayValue, source, confidence } = parseSpecValue(value);
                                                                                        const sourceLabel = getSourceLabel(source);

                                                                                        return (
                                                                                            <div key={key} className="text-sm group break-words">
                                                                                                <span className="font-medium">{prettyKey}:</span>{' '}
                                                                                                <span className="text-muted-foreground">{displayValue}</span>
                                                                                                {sourceLabel && (
                                                                                                    <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                                                        {sourceLabel}
                                                                                                    </span>
                                                                                                )}
                                                                                                {confidence && confidence < 0.7 && (
                                                                                                    <span className="ml-1 text-xs text-amber-600" title={`Confidence: ${Math.round(confidence * 100)}%`}>
                                                                                                        ⚠️
                                                                                                    </span>
                                                                                                )}
                                                                                            </div>
                                                                                        );
                                                                                    })}
                                                                                </div>
                                                                            </div>
                                                                        )}

                                                                        {/* Sample Input Preview */}
                                                                        {!collapsedAccessories.has(index) && (
                                                                            <div className="pt-3 border-t">
                                                                                <p className="text-xs text-muted-foreground mb-2">Sample Input:</p>
                                                                                <p className="text-sm bg-muted p-3 rounded-lg font-mono">
                                                                                    {formatSampleInput(accessory.sampleInput)}
                                                                                </p>
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ) : (
                                                            <div className="text-center py-8 text-muted-foreground">
                                                                No accessories identified
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </Tabs>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    </div>
                </div>

                {/* Search Tabs - positioned absolutely relative to the flex-1 relative container */}
                {
                    searchTabs.map((tab) => {
                        const savedState = tabStates[tab.id];
                        console.log(`Rendering AIRecommender for tab ${tab.id} with saved state:`, savedState);
                        return (
                            <div
                                key={tab.id}
                                className={`absolute inset-0 top-24 ${activeTab === tab.id ? 'block' : 'hidden'}`}
                            >
                                <AIRecommender
                                    key={tab.id}
                                    initialInput={tab.input}
                                    isDirectSearch={tab.isDirectSearch}
                                    productType={tab.productType}
                                    itemThreadId={tab.itemThreadId}
                                    workflowThreadId={tab.workflowThreadId}
                                    mainThreadId={tab.mainThreadId}
                                    fillParent
                                    onStateChange={(state) => handleTabStateChange(tab.id, state)}
                                    savedMessages={savedState?.messages}
                                    savedCollectedData={savedState?.collectedData}
                                    savedCurrentStep={savedState?.currentStep}
                                    savedAnalysisResult={savedState?.analysisResult}
                                    savedRequirementSchema={savedState?.requirementSchema}
                                    savedValidationResult={savedState?.validationResult}
                                    savedCurrentProductType={savedState?.currentProductType}
                                    savedInputValue={savedState?.inputValue}
                                    savedAdvancedParameters={savedState?.advancedParameters}
                                    savedSelectedAdvancedParams={savedState?.selectedAdvancedParams}
                                    savedFieldDescriptions={savedState?.fieldDescriptions}
                                    savedPricingData={savedState?.pricingData}
                                    savedScrollPositions={savedState?.scrollPositions}
                                    savedDockingState={savedState?.dockingState}
                                    savedSearchInstanceId={savedState?.searchSessionId}
                                />
                            </div>
                        );
                    })
                }
            </div >
        </div >
    );
};

export default Project;
