import { useState, useEffect, useCallback } from 'react';
import { apiClient, ApiResponse } from '@/lib/api';

interface ProviderStatus {
  has_env_key: boolean;
  has_stored_key: boolean;
  configured: boolean;
  source: 'env' | 'stored' | 'none';
}

interface ApiKeyStatus {
  providers: {
    openai: ProviderStatus;
    mistral: ProviderStatus;
  };
  has_any_configured: boolean;
  has_env_configured: boolean;
}

interface ProviderInfo {
  type: string;
  name: string;
  available: boolean;
  configured: boolean;
  key_source: string;
  capabilities?: string[];
  default_model?: string;
  description?: string;
  error?: string;
}

export const useApiKeys = () => {
  const [apiKeyStatus, setApiKeyStatus] = useState<ApiKeyStatus | null>(null);
  const [providerInfo, setProviderInfo] = useState<ProviderInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchApiKeyStatus = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await apiClient.get<ApiKeyStatus>('/api/keys/status');
      if (!response.success) {
        throw new Error(response.error || 'Failed to fetch API key status');
      }
      
      setApiKeyStatus(response.data);
      
      // Also fetch provider info
      const providerResponse = await apiClient.get<ProviderInfo[]>('/api/keys/providers');
      if (providerResponse.success) {
        setProviderInfo(providerResponse.data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching API key status:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const setApiKey = async (provider: string, apiKey: string) => {
    try {
      setError(null);
      
      const response = await apiClient.post('/api/keys/set', {
        provider,
        api_key: apiKey,
      });

      if (!response.success) {
        throw new Error(response.error || 'Failed to set API key');
      }

      // Small delay to ensure backend has processed the change
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Refresh status after setting key
      await fetchApiKeyStatus();
      
      return response.data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error setting API key:', err);
      throw err;
    }
  };

  const removeApiKey = async (provider: string) => {
    try {
      setError(null);
      
      const response = await apiClient.delete(`/api/keys/${provider}`);

      if (!response.success) {
        throw new Error(response.error || 'Failed to remove API key');
      }

      // Small delay to ensure backend has processed the change
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Refresh status after removing key
      await fetchApiKeyStatus();
      
      return response.data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error removing API key:', err);
      throw err;
    }
  };



  useEffect(() => {
    fetchApiKeyStatus();
  }, [fetchApiKeyStatus]);

  return {
    apiKeyStatus,
    providerInfo,
    isLoading,
    error,
    setApiKey,
    removeApiKey,
    refreshStatus: fetchApiKeyStatus,
  };
}; 