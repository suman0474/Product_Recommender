import AIRecommender from "@/components/AIRecommender";
import { useState, useCallback, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useToast } from "@/components/ui/use-toast";
import { BASE_URL } from "@/components/AIRecommender/api";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { X } from 'lucide-react';

const DOCKING_STATE_KEY = "ai_recommender_docking_state";

// Helper to extract first N words from input
const extractFirstWords = (input: string, count: number = 2): string => {
  if (!input || typeof input !== 'string') return 'Search';

  // Clean and split the input
  const words = input.trim().split(/\s+/).filter(word => word.length > 0);

  if (words.length === 0) return 'Search';

  // Take first N words and capitalize first letter of each
  const firstWords = words.slice(0, count).map(word =>
    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
  ).join(' ');

  return firstWords || 'Search';
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

  return `${actualBaseName} (1)`;
};

// IndexedDB configuration for persisting Search state
const search_DB_NAME = 'search_db';
const search_STORE_NAME = 'search_state';
const search_STATE_KEY = 'current_session';
const search_BACKUP_KEY = 'search_state_backup';

// Helper function to open IndexedDB
const opensearchDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(search_DB_NAME, 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(search_STORE_NAME)) {
        db.createObjectStore(search_STORE_NAME, { keyPath: 'id' });
      }
    };
  });
};

// Helper function to save state to IndexedDB
const saveStateTosearchDB = async (state: any): Promise<void> => {
  try {
    const db = await opensearchDB();
    const transaction = db.transaction(search_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(search_STORE_NAME);

    await new Promise<void>((resolve, reject) => {
      const request = store.put({ id: search_STATE_KEY, ...state });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    db.close();
  } catch (e) {
    console.warn('[search] Failed to save to IndexedDB:', e);
  }
};

// Helper function to load state from IndexedDB
const loadStateFromsearchDB = async (): Promise<any | null> => {
  try {
    const db = await opensearchDB();
    const transaction = db.transaction(search_STORE_NAME, 'readonly');
    const store = transaction.objectStore(search_STORE_NAME);

    const result = await new Promise<any>((resolve, reject) => {
      const request = store.get(search_STATE_KEY);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    db.close();

    if (result) {
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
    console.warn('[search] Failed to load from IndexedDB:', e);
    return null;
  }
};

// Helper function to clear IndexedDB state
const clearsearchDBState = async (): Promise<void> => {
  try {
    const db = await opensearchDB();
    const transaction = db.transaction(search_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(search_STORE_NAME);

    await new Promise<void>((resolve, reject) => {
      const request = store.delete(search_STATE_KEY);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    db.close();
    // Also clear localStorage backup
    localStorage.removeItem(search_BACKUP_KEY);
    console.log('[search] IndexedDB state cleared');
  } catch (e) {
    console.warn('[search] Failed to clear IndexedDB:', e);
  }
};

const Index = () => {
  const { toast } = useToast();
  const [searchParams] = useSearchParams();

  // Track the latest AIRecommender state for saving
  const currentStateRef = useRef<any>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Project ID tracking for updates
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);

  // Duplicate name dialog states
  const [duplicateNameDialogOpen, setDuplicateNameDialogOpen] = useState(false);
  const [duplicateProjectName, setDuplicateProjectName] = useState<string | null>(null);
  const [autoRenameSuggestion, setAutoRenameSuggestion] = useState<string | null>(null);
  const [duplicateDialogNameInput, setDuplicateDialogNameInput] = useState<string>('');
  const [duplicateDialogError, setDuplicateDialogError] = useState<string | null>(null);

  // State for loaded project data
  const [loadedProjectData, setLoadedProjectData] = useState<any>(() => {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.get('projectId')) return null;
      if (urlParams.get('fresh') === 'true') return null; // Fresh window - start clean
      if (sessionStorage.getItem('clear_search_state') === 'true') return null;

      const backup = localStorage.getItem(search_BACKUP_KEY);
      if (!backup) return null;

      const restoredState = JSON.parse(backup);
      if (restoredState?.messages) {
        restoredState.messages = restoredState.messages.map((msg: any) => ({
          ...msg,
          timestamp: msg.timestamp ? new Date(msg.timestamp) : undefined
        }));
      }

      return {
        messages: restoredState.messages || [],
        currentStep: restoredState.currentStep || 'greeting',
        searchInstanceId: restoredState.searchInstanceId || restoredState.searchSessionId,
        requirementSchema: restoredState.requirementSchema || null,
        validationResult: restoredState.validationResult || null,
        currentProductType: restoredState.currentProductType || null,
        inputValue: restoredState.inputValue || '',
        advancedParameters: restoredState.advancedParameters || null,
        selectedAdvancedParams: restoredState.selectedAdvancedParams || {},
        fieldDescriptions: restoredState.fieldDescriptions || {},
        collectedData: restoredState.collectedData || {},
        analysisResult: restoredState.analysisResult || null,
        pricingData: restoredState.pricingData || {},
        scrollPositions: restoredState.scrollPositions || { left: 0, center: 0, right: 0 },
        dockingState: restoredState.dockingState || { left: true, right: true },
      };
    } catch (e) {
      return null;
    }
  });

  // Load saved docking state from localStorage
  const [savedDockingState] = useState<{ left: boolean; right: boolean } | undefined>(() => {
    try {
      const saved = localStorage.getItem(DOCKING_STATE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.warn("Failed to load saved docking state:", e);
    }
    return undefined;
  });

  // SAVE ON PAGE CLOSE/REFRESH: Save state immediately
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (!currentStateRef.current) return;

      const stateToSave = {
        ...currentStateRef.current,
        savedAt: new Date().toISOString()
      };

      // Use synchronous localStorage as fallback for immediate save
      try {
        localStorage.setItem(search_BACKUP_KEY, JSON.stringify(stateToSave));
        console.log('[search] Saved state to localStorage backup on page close');
      } catch (e) {
        console.warn('[search] Failed to save backup state:', e);
      }

      // Also try to save to IndexedDB (might not complete)
      saveStateTosearchDB(stateToSave);
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      // Also save state on component unmount (e.g. navigation within app)
      handleBeforeUnload();
    };
  }, []);

  // LOAD FROM INDEXEDDB: Restore state on mount (unless projectId is present)
  useEffect(() => {
    // If loading a specific project, don't restore session state
    if (searchParams.get('projectId')) return;

    const loadState = async () => {
      // Check if this is a fresh window opened via navigation popup
      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.get('fresh') === 'true') {
        console.log('[SEARCH] Fresh window detected - skipping state restoration');
        await clearsearchDBState();
        setLoadedProjectData(null);
        // Remove fresh param from URL to prevent re-clearing on manual refresh
        urlParams.delete('fresh');
        const newUrl = urlParams.toString()
          ? `${window.location.pathname}?${urlParams.toString()}`
          : window.location.pathname;
        window.history.replaceState({}, '', newUrl);
        return;
      }

      // Check if we need to clear state (triggered by New button)
      if (sessionStorage.getItem('clear_search_state') === 'true') {
        console.log('[search] Clearing state as requested by New button');
        sessionStorage.removeItem('clear_search_state');
        await clearsearchDBState();
        return; // Don't restore anything
      }

      try {
        // First check localStorage backup (faster/synchronous)
        let restoredState: any = null;
        try {
          const backup = localStorage.getItem(search_BACKUP_KEY);
          if (backup) {
            restoredState = JSON.parse(backup);
            console.log('[search] Loaded state from localStorage backup');
          }
        } catch (e) {
          console.warn('[search] Failed to load backup:', e);
        }

        // If no backup, try IndexedDB
        if (!restoredState) {
          restoredState = await loadStateFromsearchDB();
          if (restoredState) {
            console.log('[search] Loaded state from IndexedDB');
          }
        }

        if (restoredState) {
          // Restore messages with proper Date objects
          if (restoredState.messages) {
            restoredState.messages = restoredState.messages.map((msg: any) => ({
              ...msg,
              timestamp: msg.timestamp ? new Date(msg.timestamp) : undefined
            }));
          }

          // Ensure all required fields are present for AIRecommender restoration
          const completeState = {
            messages: restoredState.messages || [],
            currentStep: restoredState.currentStep || 'greeting',
            searchInstanceId: restoredState.searchInstanceId || restoredState.searchSessionId,
            requirementSchema: restoredState.requirementSchema || null,
            validationResult: restoredState.validationResult || null,
            currentProductType: restoredState.currentProductType || null,
            inputValue: restoredState.inputValue || '',
            advancedParameters: restoredState.advancedParameters || null,
            selectedAdvancedParams: restoredState.selectedAdvancedParams || {},
            fieldDescriptions: restoredState.fieldDescriptions || {},
            collectedData: restoredState.collectedData || {},
            analysisResult: restoredState.analysisResult || null,
            pricingData: restoredState.pricingData || {},
            scrollPositions: restoredState.scrollPositions || { left: 0, center: 0, right: 0 },
            dockingState: restoredState.dockingState || { left: true, right: true },
          };

          // Set loaded project data to restore the state in AIRecommender
          setLoadedProjectData(completeState);
          console.log('[SEARCH] Restored complete state from IndexedDB:', {
            messages: completeState.messages.length,
            currentStep: completeState.currentStep,
            hasCollectedData: !!completeState.collectedData && Object.keys(completeState.collectedData).length > 0,
            hasAnalysisResult: !!completeState.analysisResult,
          });
        }
      } catch (e) {
        console.warn('[search] Error restoring state:', e);
      }
    };

    loadState();
  }, [searchParams]);

  // Load project if projectId is in the URL
  useEffect(() => {
    const projectId = searchParams.get('projectId');
    if (!projectId) return;

    const loadProject = async () => {
      try {
        console.log('[search] Loading project:', projectId);
        const response = await fetch(`${BASE_URL}/api/projects/${projectId}`, {
          credentials: 'include'
        });

        if (!response.ok) {
          throw new Error('Failed to load project');
        }

        const data = await response.json();
        const project = data.project || data;

        // ✅ FIX: Store project ID for updates
        const loadedProjectId = project.id || project._id || project.project_id || projectId;
        if (loadedProjectId) {
          setCurrentProjectId(loadedProjectId);
          console.log('[SEARCH] Set project ID for updates:', loadedProjectId);
        }

        // Load conversation history from the project
        const convHistories = project.conversationHistories || project.conversation_histories || {};
        const searchHistory = convHistories['search'];

        if (searchHistory) {
          // Restore instance ID from top level or nested location
          const restoredInstanceId = project.projectInstanceId || project.project_instance_id ||
            searchHistory.searchInstanceId || searchHistory.searchSessionId;

          // Pass the loaded data to be used by AIRecommender
          setLoadedProjectData({
            messages: searchHistory.messages || [],
            currentStep: searchHistory.currentStep || project.currentStep || project.current_step || 'greeting',
            searchInstanceId: restoredInstanceId,
            requirementSchema: searchHistory.requirementSchema,
            validationResult: searchHistory.validationResult,
            currentProductType: searchHistory.currentProductType || project.productType || project.product_type,
            advancedParameters: searchHistory.advancedParameters,
            selectedAdvancedParams: searchHistory.selectedAdvancedParams,
            fieldDescriptions: searchHistory.fieldDescriptions || project.fieldDescriptions || project.field_descriptions,
            collectedData: (project.collectedData || project.collected_data || {})['search'],
            analysisResult: (project.analysisResults || project.analysis_results || {})['search'],
          });

          if (restoredInstanceId) {
            console.log('[SEARCH] Restoring searchInstanceId:', restoredInstanceId);
          }

          console.log('[search] Loaded project data');

          toast({
            title: "Project Loaded",
            description: `Loaded "${project.projectName || project.project_name}"`,
          });
        }
      } catch (error: any) {
        console.error('[search] Error loading project:', error);
        toast({
          title: "Load Failed",
          description: error.message || "Failed to load project",
          variant: "destructive",
        });
      }
    };

    loadProject();
  }, [searchParams, toast]);

  // Handle state changes from AIRecommender to persist docking state and track current state
  const handleStateChange = useCallback((state: any) => {
    // Store the latest state for save functionality
    currentStateRef.current = state;

    if (state.dockingState) {
      try {
        localStorage.setItem(DOCKING_STATE_KEY, JSON.stringify(state.dockingState));
      } catch (e) {
        console.warn("Failed to save docking state:", e);
      }
    }

    // Debounced auto-save to IndexedDB
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveStateTosearchDB({
        ...state,
        savedAt: new Date().toISOString()
      });
    }, 1000);
  }, []);

  const resetDuplicateDialog = () => {
    setDuplicateNameDialogOpen(false);
    setDuplicateProjectName(null);
    setAutoRenameSuggestion(null);
    setDuplicateDialogError(null);
    setDuplicateDialogNameInput('');
  };

  // Save project functionality
  const handleSaveProject = useCallback(async (
    overrideName?: string,
    options?: { skipDuplicateDialog?: boolean }
  ) => {
    const state = currentStateRef.current;

    if (!state) {
      toast({
        title: "Nothing to Save",
        description: "Start a conversation first before saving.",
        variant: "destructive",
      });
      return;
    }

    try {
      // Extract first 2 words from the first user message for project name
      let projectNameBase = 'Search';
      if (state.messages && state.messages.length > 0) {
        const firstUserMessage = state.messages.find((m: any) => m.type === 'user');
        if (firstUserMessage && firstUserMessage.content) {
          projectNameBase = extractFirstWords(firstUserMessage.content, 2);
        }
      }

      // Use override name if provided, otherwise add "(Search)" suffix
      const effectiveProjectName = overrideName
        ? overrideName.trim()
        : `${projectNameBase} (Search)`;

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
              // ✅ FIX: Same name (case-insensitive) and not the very same project we are updating
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
      const firstUserMsg = state.messages?.find((m: any) => m.type === 'user');
      const initialRequirements = firstUserMsg?.content || 'Search query';

      const projectData = {
        project_id: currentProjectId || undefined, // ✅ FIX: Include project ID for updates
        project_name: effectiveProjectName,
        project_description: `Search project - Created on ${new Date().toLocaleDateString()}`,
        initial_requirements: initialRequirements,
        source_page: 'search',
        project_instance_id: state.searchInstanceId || '', // Save the search instance ID at top level
        conversation_histories: {
          'search': {
            messages: state.messages || [],
            currentStep: state.currentStep || 'greeting',
            searchInstanceId: state.searchInstanceId,
            requirementSchema: state.requirementSchema || null,
            validationResult: state.validationResult || null,
            currentProductType: state.currentProductType || null,
            inputValue: state.inputValue || '',
            advancedParameters: state.advancedParameters || null,
            selectedAdvancedParams: state.selectedAdvancedParams || {},
            fieldDescriptions: state.fieldDescriptions || {}
          }
        },
        collected_data: state.collectedData ? { 'search': state.collectedData } : {},
        analysis_results: state.analysisResult ? { 'search': state.analysisResult } : {},
        field_descriptions: state.fieldDescriptions || {},
        pricing: state.pricingData ? { 'search': state.pricingData } : {},
        product_type: state.currentProductType || '',
        current_step: state.currentStep || 'greeting',
        workflow_position: {
          current_tab: 'search',
          has_results: !!state.analysisResult,
          last_interaction: new Date().toISOString(),
          project_phase: state.analysisResult ? 'results_review' : 'requirements_gathering'
        },
        user_interactions: {
          conversations_count: 1,
          has_analysis: !!state.analysisResult,
          last_save: new Date().toISOString()
        }
      };

      console.log('[search_SAVE] Saving project:', effectiveProjectName);

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
      console.log('[search_SAVE] Project saved successfully:', result);

      // ✅ FIX: Store project ID for future updates
      if (result.project_id || result.projectId || result.id) {
        const savedProjectId = result.project_id || result.projectId || result.id;
        setCurrentProjectId(savedProjectId);
        console.log('[search_SAVE] Project ID stored for updates:', savedProjectId);
      }

      toast({
        title: "Project Saved",
        description: `"${effectiveProjectName}" has been saved successfully.`,
      });

    } catch (error: any) {
      console.error('[search_SAVE] Error saving project:', error);
      toast({
        title: "Save Failed",
        description: error.message || "Failed to save project",
        variant: "destructive",
      });
    }
  }, [toast]);

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
    const baseName = (duplicateProjectName || '').trim() || 'Search';
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

  return (
    <>
      <AIRecommender
        savedDockingState={loadedProjectData?.dockingState || savedDockingState}
        onStateChange={handleStateChange}
        onSave={() => handleSaveProject()}
        // Pass loaded project data if available
        savedMessages={loadedProjectData?.messages}
        savedCollectedData={loadedProjectData?.collectedData}
        savedCurrentStep={loadedProjectData?.currentStep}
        savedAnalysisResult={loadedProjectData?.analysisResult}
        savedRequirementSchema={loadedProjectData?.requirementSchema}
        savedValidationResult={loadedProjectData?.validationResult}
        savedCurrentProductType={loadedProjectData?.currentProductType}
        savedAdvancedParameters={loadedProjectData?.advancedParameters}
        savedSelectedAdvancedParams={loadedProjectData?.selectedAdvancedParams}
        savedFieldDescriptions={loadedProjectData?.fieldDescriptions}
        savedScrollPositions={loadedProjectData?.scrollPositions}
        savedPricingData={loadedProjectData?.pricingData}
        savedSearchInstanceId={loadedProjectData?.searchInstanceId || loadedProjectData?.searchSessionId}
      />

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
            <AlertDialogAction onClick={handleDuplicateNameAutoRename}>
              Use suggested name
            </AlertDialogAction>
            <AlertDialogAction onClick={handleDuplicateNameChangeConfirm}>
              Save new name
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default Index;
