import React, { useState, useEffect, useRef } from 'react';
import { Key, Shield, Eye, EyeOff, CheckCircle, AlertTriangle, Loader2, Info, RefreshCw, X, Trash2, Check, Save } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useApiKeys } from '@/hooks/useApiKeys';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface ApiKeysSetupProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onStateChange?: () => void; // New callback for state changes
  embedded?: boolean;
}

export const ApiKeysSetup: React.FC<ApiKeysSetupProps> = ({ isOpen, onClose, onSuccess, onStateChange, embedded = false }) => {
  const { apiKeyStatus, providerInfo, setApiKey, removeApiKey, refreshStatus, isLoading: apiKeysLoading, error: apiKeysError } = useApiKeys();
  const [openaiKey, setOpenaiKey] = useState('');
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [isValid, setIsValid] = useState({ openai: false });
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>(undefined);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  // Removed testingKeys state

  // Track previous configuration state to detect changes
  const prevConfigRef = useRef<boolean>(false);

  const providers = [
    { key: 'openai', name: 'OpenAI', icon: <Key className="h-4 w-4" />, required: true },
  ];

  const apiKeys: { [key: string]: string } = {
    openai: openaiKey,
  };

  // Check if a provider is configured
  const isProviderConfigured = (providerKey: string) => {
    const status = getProviderStatus(providerKey);
    return status?.configured || false;
  };

  // Check if OpenAI (required provider) is configured
  const isRequiredProviderConfigured = () => {
    return isProviderConfigured('openai');
  };

  // Use the hook's status directly instead of local state
  const currentStatus = apiKeyStatus;

  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !embedded) {
        onClose();
      }
    };

    if (isOpen && !embedded) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, embedded, onClose]);

  // Monitor when required provider becomes configured in embedded mode
  useEffect(() => {
    if (embedded) {
      const isCurrentlyConfigured = isRequiredProviderConfigured();
      const wasConfigured = prevConfigRef.current;
      
      // Only trigger if we just became configured (transition from false to true)
      if (isCurrentlyConfigured && !wasConfigured) {
        // Small delay to ensure the status is fully updated
        const timer = setTimeout(() => {
          onSuccess();
        }, 1000);
        return () => clearTimeout(timer);
      }
      
      // Update the previous state
      prevConfigRef.current = isCurrentlyConfigured;
    }
  }, [embedded, apiKeyStatus?.providers?.openai?.configured, onSuccess]);

  const validateOpenaiKey = (key: string) => {
    const isValidFormat = key.startsWith('sk-') && key.length > 20;
    setIsValid(prev => ({ ...prev, openai: isValidFormat }));
    return isValidFormat;
  };

  const handleOpenaiKeyChange = (value: string) => {
    setOpenaiKey(value);
    validateOpenaiKey(value);
  };

  const handleRefreshStatus = async () => {
    await refreshStatus();
  };

  const handleRemoveKey = async (provider: string) => {
    try {
      await removeApiKey(provider);
      await refreshStatus();
      // Clear the input field
      if (provider === 'openai') {
        setOpenaiKey('');
        setIsValid(prev => ({ ...prev, openai: false }));
      }
      setMessage({ type: 'success', text: `${provider} key removed successfully.` });
      onStateChange?.(); // Call the new callback
    } catch (error) {
      console.error(`Error removing ${provider} key:`, error);
      setMessage({ type: 'error', text: `Failed to remove ${provider} key.` });
    }
  };

  const handleSaveKey = async () => {
    if (!apiKeyInput.trim()) {
      setMessage({ type: 'error', text: 'API key cannot be empty.' });
      return;
    }

    setIsSaving(true);
    setMessage(null);

    try {
      await setApiKey(selectedProvider || '', apiKeyInput);
      setMessage({ type: 'success', text: `${providers.find(p => p.key === selectedProvider)?.name} key saved successfully.` });
      setApiKeyInput('');
      setSelectedProvider(undefined);
      setShowApiKey(false);
      // Clear testing state for this provider
      // Removed testingKeys state
      onStateChange?.(); // Call the new callback
    } catch (error) {
      setMessage({ type: 'error', text: error instanceof Error ? error.message : 'Failed to save API key' });
    } finally {
      setIsSaving(false);
    }
  };

  // Removed handleTestKey function

  const getProviderStatus = (provider: string) => {
    if (!currentStatus?.providers) return null;
    return currentStatus.providers[provider];
  };

  const getStatusBadge = (provider: string) => {
    const status = getProviderStatus(provider);
    if (!status) return null;

    if (status.configured) {
      if (status.source === 'env') {
        return (
          <Badge variant="secondary" className="bg-blue-100 text-blue-800 border-blue-200">
            <CheckCircle className="h-3 w-3 mr-1" />
            From Environment
          </Badge>
        );
      } else {
        return (
          <Badge variant="default" className="bg-green-100 text-green-800 border-green-200">
            <CheckCircle className="h-3 w-3 mr-1" />
            Stored Locally
          </Badge>
        );
      }
    } else {
      return (
        <Badge variant="secondary" className="bg-gray-100 text-gray-600">
          Not Configured
        </Badge>
      );
    }
  };

  const canUpdateKey = (provider: string) => {
    const status = getProviderStatus(provider);
    // Can only update if the key is stored locally (not from environment)
    return status?.configured && status?.source === 'stored';
  };

  const getProviderDescription = (provider: string) => {
    switch (provider) {
      case 'openai':
        return 'GPT-4.1-mini - AI language model for intelligent responses';
      default:
        return '';
    }
  };

  const getSystemStatusMessage = () => {
    const openaiConfigured = isProviderConfigured('openai');
    
    if (openaiConfigured) {
      return {
        type: 'complete' as const,
        title: 'All Systems Ready',
        description: 'OpenAI is configured. You have access to all features.'
      };
    } else {
      return {
        type: 'incomplete' as const,
        title: 'Setup Required',
        description: 'OpenAI API key is required to use AI features.'
      };
    }
  };

  const hasChangesToSave = () => {
    const openaiStatus = getProviderStatus('openai');
    
    // Check if we have new keys to save (not from environment)
    const hasOpenaiChange = openaiKey.trim() && openaiStatus?.source !== 'env';
    
    return hasOpenaiChange;
  };

  if (!isOpen && !embedded) return null;

  const systemStatus = getSystemStatusMessage();

  const content = (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center">
          <Key className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gradient">API Keys Configuration</h2>
          <p className="text-sm text-muted-foreground">
            Configure your API keys to enable AI-powered features
          </p>
        </div>
      </div>

      {/* System Status Overview */}
      <div className={`p-4 rounded-lg border ${
        systemStatus.type === 'complete' 
          ? 'bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-800'
          : systemStatus.type === 'incomplete'
          ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-950/20 dark:border-yellow-800'
          : 'bg-muted/50 border-border'
      }`}>
        <div className="flex items-start gap-3">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            systemStatus.type === 'complete' 
              ? 'bg-green-100 dark:bg-green-900/30'
              : systemStatus.type === 'incomplete'
              ? 'bg-yellow-100 dark:bg-yellow-900/30'
              : 'bg-muted'
          }`}>
            {systemStatus.type === 'complete' ? (
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
            )}
          </div>
          <div className="flex-1">
            <h3 className={`font-medium text-sm ${
              systemStatus.type === 'complete' 
                ? 'text-green-800 dark:text-green-200'
                : 'text-yellow-800 dark:text-yellow-200'
            }`}>
              {systemStatus.title}
            </h3>
            <p className={`text-xs mt-1 ${
              systemStatus.type === 'complete' 
                ? 'text-green-600 dark:text-green-300'
                : 'text-yellow-600 dark:text-yellow-300'
            }`}>
              {systemStatus.description}
            </p>
          </div>
        </div>
      </div>

      {/* Provider Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {providers.map((provider) => {
          const isConfigured = isProviderConfigured(provider.key);
          // Removed isTesting state
          
          return (
            <div
              key={provider.key}
              className={`relative p-4 rounded-lg border transition-all duration-200 ${
                isConfigured
                  ? 'bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-800'
                  : 'bg-muted/50 border-border'
              }`}
            >
              {/* Remove button - only show when configured and can be updated */}
              {isConfigured && canUpdateKey(provider.key) && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleRemoveKey(provider.key)}
                  className="absolute top-2 right-2 h-7 w-7 p-0 hover:bg-destructive/10 hover:text-destructive border border-destructive/20"
                  // Removed disabled={isTesting}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}

              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  isConfigured ? 'bg-green-100 dark:bg-green-900/30' : 'bg-muted'
                }`}>
                  {isConfigured ? (
                    <Check className="h-4 w-4 text-green-600 dark:text-green-400" />
                  ) : (
                    <X className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="font-medium text-sm">{provider.name}</h3>
                    {provider.required ? (
                      <Badge variant="destructive" className="text-xs">Required</Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">Optional</Badge>
                    )}
                    {/* Removed isTesting && (
                      <div className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                        <span className="text-xs text-blue-600 dark:text-blue-400">Testing...</span>
                      </div>
                    ) */}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {getProviderDescription(provider.key)}
                  </p>
                  <div className="mt-2">
                    {getStatusBadge(provider.key)}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* API Key Entry Sections */}
      <div className="space-y-6">
        {/* OpenAI API Key Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">OpenAI API Key</h3>
            <Badge variant="destructive" className="text-xs">Required</Badge>
            <div className="flex-1 h-px bg-border" />
          </div>

          {isProviderConfigured('openai') ? (
            <div className="space-y-3">
              {/* Removed: Configured status box for OpenAI API key */}
              {canUpdateKey('openai') && (
                <Button
                  variant="outline"
                  onClick={() => {
                    setSelectedProvider('openai');
                    setApiKeyInput('');
                    setShowApiKey(false);
                  }}
                  className="w-full"
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Update OpenAI Key
                </Button>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <Input
                  type={showApiKey ? "text" : "password"}
                  placeholder="Enter your OpenAI API key (sk-...)"
                  value={selectedProvider === 'openai' ? apiKeyInput : ''}
                  onChange={(e) => {
                    setSelectedProvider('openai');
                    setApiKeyInput(e.target.value);
                  }}
                  className="pr-10"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-8 w-8 p-0"
                >
                  {showApiKey ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <div className="flex gap-3">
                <Button
                  onClick={handleSaveKey}
                  disabled={!apiKeyInput.trim() || selectedProvider !== 'openai' || isSaving}
                  className="flex-1"
                >
                  {isSaving && selectedProvider === 'openai' ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="mr-2 h-4 w-4" />
                      Save Key
                    </>
                  )}
                </Button>
                {/* Removed Test Key button */}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Success/Error Messages */}
      {message && (
        <Alert className={message.type === 'success' ? 'border-green-200 bg-green-50 dark:bg-green-950/20' : 'border-red-200 bg-red-50 dark:bg-red-950/20'}>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{message.text}</AlertDescription>
        </Alert>
      )}

      {/* Security Note */}
      <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Shield className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100">
              Security & Privacy
            </h4>
            <p className="text-xs text-blue-700 dark:text-blue-300">
              Your API keys are encrypted using industry-standard encryption and stored locally. 
              They are never transmitted to our servers and are only used to make API calls to the respective providers.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  if (embedded) {
    return content;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] glass animate-scale-in relative">
        {/* Fixed close button - always visible */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="absolute top-4 right-4 h-8 w-8 p-0 hover:bg-muted z-10"
        >
          <X className="h-4 w-4" />
        </Button>
        
        <CardContent className="p-6 overflow-hidden">
          <div className="space-y-6 max-h-[calc(90vh-3rem)] overflow-y-auto pr-2 custom-scrollbar">
            {content}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};