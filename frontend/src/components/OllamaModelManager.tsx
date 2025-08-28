import React, { useState, useEffect } from 'react';
import { Download, CheckCircle, XCircle, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface ModelInfo {
  name: string;
  size?: number;
  modified_at?: string;
  digest?: string;
}

interface ModelStatus {
  model: string;
  available: boolean;
  message: string;
}

interface OllamaStatus {
  status: 'available' | 'unavailable' | 'error';
  message: string;
  available_models: number;
  models: string[];
}

export const OllamaModelManager: React.FC = () => {
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus>>({});
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const supportedModels = ['gemma3:1b', 'llama3.2:1b'];

  const fetchOllamaStatus = async () => {
    try {
      const response = await apiClient.get<OllamaStatus>('/ollama/status');
      if (response.success && response.data) {
        setOllamaStatus(response.data);
      }
    } catch (error) {
      console.error('Failed to fetch Ollama status:', error);
      setOllamaStatus({
        status: 'error',
        message: 'Failed to connect to Ollama service',
        available_models: 0,
        models: []
      });
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await apiClient.get<ModelInfo[]>('/ollama/models');
      if (response.success && response.data) {
        setAvailableModels(response.data);
      }
    } catch (error) {
      console.error('Failed to fetch available models:', error);
    }
  };

  const checkModelStatus = async (modelName: string) => {
    try {
      const response = await apiClient.get<ModelStatus>(`/ollama/models/${modelName}/status`);
      if (response.success && response.data) {
        setModelStatuses(prev => ({
          ...prev,
          [modelName]: response.data
        }));
      }
    } catch (error) {
      console.error('Failed to check status for', modelName, ':', error);
    }
  };

  const downloadModel = async (modelName: string) => {
    setDownloadingModels(prev => new Set(prev).add(modelName));
    
    try {
      const response = await apiClient.post('/ollama/models/download', {
        model: modelName
      });
      
      if (response.success && response.data) {
        toast({
          title: "Model Downloaded",
          description: `Successfully downloaded ${modelName}`,
        });
        
        // Refresh the model lists
        await fetchAvailableModels();
        await checkModelStatus(modelName);
      } else {
        toast({
          title: "Download Failed",
          description: response.error || `Failed to download ${modelName}`,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Failed to download', modelName, ':', error);
      toast({
        title: "Download Failed",
        description: `Failed to download ${modelName}. Please try again.`,
        variant: "destructive",
      });
    } finally {
      setDownloadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(modelName);
        return newSet;
      });
    }
  };

  const refreshAll = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        fetchOllamaStatus(),
        fetchAvailableModels(),
        ...supportedModels.map(checkModelStatus)
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
  }, []);

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'Unknown';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const isModelAvailable = (modelName: string) => {
    return availableModels.some(model => model.name === modelName);
  };

  const isModelDownloading = (modelName: string) => {
    return downloadingModels.has(modelName);
  };

  if (!ollamaStatus) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Ollama Model Manager</CardTitle>
          <CardDescription>Manage your local Ollama models</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="ml-2">Loading...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Ollama Model Manager</CardTitle>
            <CardDescription>Manage your local Ollama models</CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={refreshAll}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span className="ml-2">Refresh</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Ollama Service Status */}
        <div className="flex items-center gap-2">
          <Badge variant={ollamaStatus.status === 'available' ? 'default' : 'destructive'}>
            {ollamaStatus.status === 'available' ? (
              <CheckCircle className="h-3 w-3 mr-1" />
            ) : (
              <XCircle className="h-3 w-3 mr-1" />
            )}
            {ollamaStatus.status}
          </Badge>
          <span className="text-sm text-muted-foreground">{ollamaStatus.message}</span>
        </div>

        {ollamaStatus.status === 'available' && (
          <>
            {/* Available Models Summary */}
            <div className="text-sm text-muted-foreground">
              {ollamaStatus.available_models} model{ollamaStatus.available_models !== 1 ? 's' : ''} available locally
            </div>

            {/* Supported Models */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Supported Models</h3>
              <div className="grid gap-4">
                {supportedModels.map(modelName => {
                  const isAvailable = isModelAvailable(modelName);
                  const isDownloading = isModelDownloading(modelName);
                  const modelInfo = availableModels.find(m => m.name === modelName);

                  return (
                    <div
                      key={modelName}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div>
                          <div className="font-medium">{modelName}</div>
                          <div className="text-sm text-muted-foreground">
                            {isAvailable ? (
                              <>
                                Available • {modelInfo && formatFileSize(modelInfo.size)}
                              </>
                            ) : (
                              'Not downloaded'
                            )}
                          </div>
                        </div>
                        <Badge variant={isAvailable ? 'default' : 'secondary'}>
                          {isAvailable ? (
                            <CheckCircle className="h-3 w-3 mr-1" />
                          ) : (
                            <XCircle className="h-3 w-3 mr-1" />
                          )}
                          {isAvailable ? 'Available' : 'Not Available'}
                        </Badge>
                      </div>
                      
                      <Button
                        variant={isAvailable ? 'outline' : 'default'}
                        size="sm"
                        onClick={() => downloadModel(modelName)}
                        disabled={isDownloading}
                      >
                        {isDownloading ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : isAvailable ? (
                          'Re-download'
                        ) : (
                          <Download className="h-4 w-4 mr-1" />
                        )}
                        {isDownloading ? 'Downloading...' : isAvailable ? 'Re-download' : 'Download'}
                      </Button>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* All Available Models */}
            {availableModels.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-medium">All Available Models</h3>
                <div className="grid gap-2">
                  {availableModels.map(model => (
                    <div
                      key={model.name}
                      className="flex items-center justify-between p-3 border rounded-lg"
                    >
                      <div>
                        <div className="font-medium">{model.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {formatFileSize(model.size)}
                          {model.modified_at && ` • Modified: ${new Date(model.modified_at).toLocaleDateString()}`}
                        </div>
                      </div>
                      <Badge variant="outline">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Available
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {ollamaStatus.status === 'unavailable' && (
          <div className="text-center p-8">
            <XCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">Ollama Service Unavailable</h3>
            <p className="text-muted-foreground mb-4">
              Please ensure Ollama is installed and running:
            </p>
            <div className="text-sm text-muted-foreground space-y-1">
              <p>1. Install Ollama from <a href="https://ollama.ai" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">ollama.ai</a></p>
              <p>2. Run <code className="bg-muted px-1 py-0.5 rounded">ollama serve</code> in your terminal</p>
              <p>3. Refresh this page</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
