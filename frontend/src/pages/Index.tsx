import React, { useState, useEffect, useCallback } from 'react';
import { Brain, FileText, Sparkles, CheckCircle, ArrowRight, AlertTriangle, Loader2 } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ChatInterface } from '@/components/ChatInterface';
import { DocumentUpload } from '@/components/DocumentUpload';
import { CompactSessionManager } from '@/components/CompactSessionManager';
import { ApiStatus } from '@/components/ApiStatus';
import { ApiKeysSetup } from '@/components/ApiKeysSetup';

import { useApiKeys } from '@/hooks/useApiKeys';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api';

interface Document {
  id: string;
  filename: string;
  type: string;
  size: number;
  upload_date: string;
  category: 'personal' | 'job-specific';
  tags: string[];
}

interface DocumentListResponseRaw {
  documents: any[];
}

interface ClearResponse {
  success: boolean;
  message?: string;
}

interface DocumentUploadResponse {
  document_id: string;
  success: boolean;
  message: string;
  file_name?: string;
  error?: string;
  metadata?: {
    tags?: string[];
    [key: string]: unknown;
  };
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [activeTab, setActiveTab] = useState('api-keys');
  const [showApiKeysModal, setShowApiKeysModal] = useState(false);
  
  const [existingDocuments, setExistingDocuments] = useState<Document[]>([]);
  const [sessionData, setSessionData] = useState({
    personalDocsCount: 0,
    jobSpecificDocsCount: 0,
    chatMessagesCount: 0,
    lastActivity: new Date(),
    currentJobTitle: undefined as string | undefined
  });

  // Normalize backend -> frontend document shape
  const normalizeDocument = (raw: any): Document => {
    const filename = raw?.filename ?? raw?.original_filename ?? 'untitled';
    const size = typeof raw?.file_size === 'number' ? raw.file_size : (typeof raw?.size === 'number' ? raw.size : 0);
    const uploadDate = raw?.upload_timestamp ?? raw?.upload_date ?? new Date().toISOString();
    const category = (raw?.category === 'job-specific' || raw?.category === 'personal') ? raw.category : 'personal';
    const type = raw?.type ?? raw?.document_type ?? 'unknown';
    return {
      id: String(raw?.id ?? raw?.document_id ?? crypto.randomUUID()),
      filename,
      type,
      size,
      upload_date: String(uploadDate),
      category,
      tags: Array.isArray(raw?.tags) ? raw.tags : []
    };
  };

  // Fetch existing documents from the backend
  const fetchExistingDocuments = useCallback(async () => {
    try {
      const response = await apiClient.get<DocumentListResponseRaw>('/documents/list');
      if (response.success && response.data) {
        const normalized = (response.data.documents || []).map(normalizeDocument);
        setExistingDocuments(normalized);
        
        // Update session data with document counts
        const personalCount = normalized.filter((doc: Document) => doc.category === 'personal').length;
        const jobCount = normalized.filter((doc: Document) => doc.category === 'job-specific').length;
        
        setSessionData(prev => ({
          ...prev,
          personalDocsCount: personalCount,
          jobSpecificDocsCount: jobCount
        }));
      }
    } catch (error) {
      console.error('Error fetching existing documents:', error);
    }
  }, []);
  
  // Use the API keys hook
  const { apiKeyStatus, isLoading: apiKeysLoading, refreshStatus } = useApiKeys();
  const hasApiKeys = apiKeyStatus?.has_any_configured || false;
  const hasEnvConfigured = apiKeyStatus?.has_env_configured || false;
  
  // Check if any provider (OpenAI, Mistral, Novita, or Ollama) is configured
  const hasRequiredApiKeys = (apiKeyStatus?.providers?.openai?.configured || apiKeyStatus?.providers?.mistral?.configured || apiKeyStatus?.providers?.novita?.configured || apiKeyStatus?.providers?.ollama?.configured) || false;
  
  // Check if only cloud providers are configured (for informational purposes)
  const hasCloudProviders = (apiKeyStatus?.providers?.openai?.configured || apiKeyStatus?.providers?.mistral?.configured || apiKeyStatus?.providers?.novita?.configured) || false;
  const hasOnlyLocalProviders = apiKeyStatus?.providers?.ollama?.configured && !hasCloudProviders;
  


  // Fetch existing documents when API keys are available
  useEffect(() => {
    if (hasRequiredApiKeys) {
      fetchExistingDocuments();
    }
  }, [hasRequiredApiKeys, fetchExistingDocuments]);
  
  // Determine if we should show API key setup
  // Show setup if OpenAI is not configured (required provider)
  const shouldShowApiKeySetup = !hasRequiredApiKeys;

  const handleSendMessage = async (message: string): Promise<void> => {
    // ChatInterface handles its own message sending with streaming
    // This is just a stub to satisfy the prop interface
    return Promise.resolve();
  };

  const handleDocumentUpload = async (file: File, category: 'personal' | 'job-specific') => {
    setIsLoading(true);
    try {
      const response = await apiClient.uploadFile('/documents/upload', file, { category });

      if (!response.success) {
        throw new Error(response.error || 'Failed to upload document');
      }

      // Immediately add the document to the UI using the upload response
      const uploadData = response.data as DocumentUploadResponse;
      if (uploadData && uploadData.document_id) {
        const newDocument: Document = {
          id: uploadData.document_id,
          filename: uploadData.file_name || file.name,
          type: category === 'personal' ? 'cv' : 'job_description', // Default types based on category
          size: file.size,
          upload_date: new Date().toISOString(),
          category: category,
          tags: uploadData.metadata?.tags || []
        };
        
        // Add to existing documents immediately
        setExistingDocuments(prev => [newDocument, ...prev]);
        
        // Update session data counts
        setSessionData(prev => ({
          ...prev,
          personalDocsCount: category === 'personal' ? prev.personalDocsCount + 1 : prev.personalDocsCount,
          jobSpecificDocsCount: category === 'job-specific' ? prev.jobSpecificDocsCount + 1 : prev.jobSpecificDocsCount,
          lastActivity: new Date()
        }));
      }

      // Also refresh from backend after a longer delay to ensure consistency
      // This serves as a backup and handles any edge cases
      // nosemgrep: unsafe-eval
      setTimeout(async () => {
        await fetchExistingDocuments();
      }, 2000);

    } catch (error) {
      console.error('Error uploading document:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentDelete = async (documentId: string) => {
    setIsLoading(true);
    try {
      const response = await apiClient.delete(`/documents/delete/${documentId}`);

      if (!response.success) {
        throw new Error(response.error || 'Failed to delete document');
      }

      // Refresh the document list after deletion
      await fetchExistingDocuments();

    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };



  const handleClearPersonalDocs = async () => {
    setIsLoading(true);
    try {
      const response = await apiClient.delete<ClearResponse>('/documents/clear/personal');

      if (!response.success) {
        throw new Error(response.error || 'Failed to clear personal documents');
      }

      if (response.data && response.data.success) {
        // Refresh the document list after clearing
        await fetchExistingDocuments();
      } else {
        throw new Error(response.data?.message || 'Failed to clear personal documents');
      }
    } catch (error) {
      console.error('Error clearing personal documents:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearJobDocs = async () => {
    setIsLoading(true);
    try {
      const response = await apiClient.delete<ClearResponse>('/documents/clear/job');

      if (!response.success) {
        throw new Error(response.error || 'Failed to clear job documents');
      }

      if (response.data && response.data.success) {
        // Refresh the document list after clearing
        await fetchExistingDocuments();
      } else {
        throw new Error(response.data?.message || 'Failed to clear job documents');
      }
    } catch (error) {
      console.error('Error clearing job documents:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleShowApiKeys = () => {
    setShowApiKeysModal(true);
  };

  const handleApiKeysSuccess = async () => {
    // Refresh the API key status to ensure we have the latest state
    await refreshStatus();
    
    setShowApiKeysModal(false);
    setShowSuccess(true);
    
    // Show success message briefly, then show initialization
    // nosemgrep: unsafe-eval
    setTimeout(() => {
      setShowSuccess(false);
      setIsInitializing(true);
      
      // Show initialization message for a moment before switching tabs
      // nosemgrep: unsafe-eval
      setTimeout(() => {
        setIsInitializing(false);
        setActiveTab('documents');
      }, 1000);
    }, 1500);
  };

  const handleApiKeysStateChange = async () => {
    // Refresh the API key status when state changes (add/remove keys)
    await refreshStatus();
  };


  return (
    <div className="min-h-screen gradient-secondary">
      {/* Modern Header */}
      <header className="glass border-b border-glass-border sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 gradient-primary rounded-xl flex items-center justify-center shadow-glow animate-pulse-slow">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gradient">Job Application AI</h1>
                <p className="text-sm text-muted-foreground hidden sm:block">Smart assistant for your career</p>
              </div>
            </div>
            
            <div className="flex items-center">
              <CompactSessionManager
                sessionData={sessionData}
                onClearPersonalDocs={handleClearPersonalDocs}
                onClearJobDocs={handleClearJobDocs}
                onShowApiKeys={handleShowApiKeys}
                isLoading={isLoading}
                hasApiKeys={hasRequiredApiKeys}
              />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Success Overlay */}
        {showSuccess && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="glass rounded-2xl p-8 border-glass-border text-center">
              <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
              <h3 className="text-xl font-bold text-gradient mb-2">API Keys Saved Successfully!</h3>
              <p className="text-muted-foreground">
                Your API keys have been securely stored and tested.
              </p>
            </div>
          </div>
        )}
        
        {/* Initialization Overlay */}
        {isInitializing && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="glass rounded-2xl p-8 border-glass-border text-center">
              <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
              </div>
              <h3 className="text-xl font-bold text-gradient mb-2">Initializing System</h3>
              <p className="text-muted-foreground">
                Setting up your AI assistant. This will only take a moment...
              </p>
            </div>
          </div>
        )}
        
        <div className="space-y-8">
          {/* Main Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-col">
            <TabsList className="grid w-full grid-cols-3 mb-6 glass border-glass-border bg-transparent backdrop-blur-md">
              <TabsTrigger 
                value="api-keys" 
                className="flex items-center gap-2 data-[state=active]:bg-white/20 data-[state=active]:text-primary hover:bg-white/10 transition-all relative"
              >
                <div className="flex items-center gap-1">
                  {hasRequiredApiKeys ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <Brain className="h-4 w-4" />
                  )}
                  API Keys
                  {hasRequiredApiKeys && (
                    <div className="w-2 h-2 bg-green-500 rounded-full ml-1"></div>
                  )}
                </div>
              </TabsTrigger>
              <TabsTrigger 
                value="documents" 
                className="flex items-center gap-2 data-[state=active]:bg-white/20 data-[state=active]:text-primary hover:bg-white/10 transition-all relative"
              >
                <div className="flex items-center gap-1">
                  {(sessionData.personalDocsCount > 0 || sessionData.jobSpecificDocsCount > 0) ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <FileText className="h-4 w-4" />
                  )}
                  Documents
                  {(sessionData.personalDocsCount > 0 || sessionData.jobSpecificDocsCount > 0) && (
                    <div className="w-2 h-2 bg-green-500 rounded-full ml-1"></div>
                  )}
                </div>
              </TabsTrigger>
              <TabsTrigger 
                value="chat" 
                className="flex items-center gap-2 data-[state=active]:bg-white/20 data-[state=active]:text-primary hover:bg-white/10 transition-all relative"
              >
                <div className="flex items-center gap-1">
                  {hasRequiredApiKeys && (sessionData.personalDocsCount > 0 || sessionData.jobSpecificDocsCount > 0) ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <Brain className="h-4 w-4" />
                  )}
                  AI Chat
                  {hasRequiredApiKeys && (sessionData.personalDocsCount > 0 || sessionData.jobSpecificDocsCount > 0) && (
                    <div className="w-2 h-2 bg-green-500 rounded-full ml-1"></div>
                  )}
                </div>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="api-keys" className="mt-0">
              <div className="space-y-6">
                <div className="glass rounded-2xl p-6 border-glass-border">
                  <ApiKeysSetup
                    isOpen={true}
                    onClose={() => {}} // No close for embedded version
                    onSuccess={handleApiKeysSuccess}
                    onStateChange={handleApiKeysStateChange}
                    embedded={true}
                  />
                </div>
                

              </div>
            </TabsContent>

            <TabsContent value="documents" className="mt-0">
              <div className="glass rounded-2xl p-6 border-glass-border">
                {!hasRequiredApiKeys ? (
                  <div className="text-center space-y-6">
                    <div className="w-16 h-16 bg-yellow-500/20 rounded-full flex items-center justify-center mx-auto">
                      <AlertTriangle className="h-8 w-8 text-yellow-500" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gradient mb-2">AI Provider Required</h3>
                      <p className="text-muted-foreground">
                        Please configure at least one AI provider (OpenAI, Mistral, Novita, or Ollama) before uploading documents.
                      </p>
                      <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                        <p className="text-sm text-yellow-800">
                          <strong>Next Step:</strong> Go to the API Keys tab to configure an AI provider
                        </p>
                        <p className="text-xs text-yellow-700 mt-1">
                          Choose from cloud providers (OpenAI, Mistral, Novita) or local (Ollama)
                        </p>
                      </div>
                    </div>
                    <div className="flex justify-center">
                      <Button
                        onClick={() => setActiveTab('api-keys')}
                        variant="outline"
                        className="border-yellow-200 text-yellow-700 hover:bg-yellow-50"
                      >
                        Configure API Keys First
                        <ArrowRight className="h-4 w-4 ml-2" />
                      </Button>
                    </div>
                  </div>
                ) : (
                  <DocumentUpload
                    onUpload={handleDocumentUpload}
                    onDelete={handleDocumentDelete}
                    isUploading={isLoading}
                    existingDocuments={existingDocuments}
                    onDocumentsLoaded={setExistingDocuments}
                    onNext={() => setActiveTab('chat')}
                  />
                )}
              </div>
            </TabsContent>

            <TabsContent value="chat" className="mt-0">
              <div className="glass rounded-2xl border-glass-border">
                {!hasRequiredApiKeys ? (
                  <div className="text-center space-y-6">
                    <div className="w-16 h-16 bg-yellow-500/20 rounded-full flex items-center justify-center mx-auto">
                      <AlertTriangle className="h-8 w-8 text-yellow-500" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gradient mb-2">API Keys Required</h3>
                      <p className="text-muted-foreground">
                        Please configure at least one AI provider (OpenAI or Mistral) before chatting with the AI assistant.
                      </p>
                      <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                        <p className="text-sm text-yellow-800">
                          <strong>Next Step:</strong> Go to the API Keys tab to configure an AI provider
                        </p>
                      </div>
                    </div>
                    <div className="flex justify-center">
                      <Button
                        onClick={() => setActiveTab('api-keys')}
                        variant="outline"
                        className="border-yellow-200 text-yellow-700 hover:bg-yellow-50"
                      >
                        Configure API Keys First
                        <ArrowRight className="h-4 w-4 ml-2" />
                      </Button>
                    </div>
                  </div>
                ) : (
                  <ChatInterface
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading}
                    hasApiKeys={hasRequiredApiKeys}
                  />
                )}
              </div>
            </TabsContent>
          </Tabs>

          {/* API Keys Modal */}
          {showApiKeysModal && (
            <ApiKeysSetup
              isOpen={showApiKeysModal}
              onClose={() => setShowApiKeysModal(false)}
              onSuccess={handleApiKeysSuccess}
              onStateChange={handleApiKeysStateChange}
            />
          )}
        </div>
      </main>

      {/* Floating API Status */}
      <ApiStatus />
    </div>
  );
};

export default Index;
