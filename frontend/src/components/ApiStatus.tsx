
import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Loader2, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/lib/api';

interface ApiStatusProps {
  onStatusChange?: (isOnline: boolean) => void;
}

interface HealthResponse {
  status: string;
  timestamp: string;
  version?: string;
}

export const ApiStatus: React.FC<ApiStatusProps> = ({ onStatusChange }) => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const checkApiStatus = async () => {
    setIsChecking(true);
    setError(null);

    try {
      const response = await apiClient.get<HealthResponse>('/health/');
      
      if (response.success) {
        setIsOnline(true);
        setLastCheck(new Date());
        onStatusChange?.(true);
      } else {
        setIsOnline(false);
        setError(response.error || 'API is not responding');
        onStatusChange?.(false);
      }
    } catch (err) {
      setIsOnline(false);
      setError(err instanceof Error ? err.message : 'Failed to connect to API');
      onStatusChange?.(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkApiStatus();
    
    // Check every 30 seconds
    // nosemgrep: unsafe-eval
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    if (isChecking) {
      return <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />;
    }
    return isOnline ? (
      <Wifi className="h-4 w-4 text-green-500" />
    ) : (
      <WifiOff className="h-4 w-4 text-red-500" />
    );
  };

  const getStatusText = () => {
    if (isChecking) return 'Checking...';
    return isOnline ? 'Connected' : 'Disconnected';
  };

  const getStatusColor = () => {
    if (isChecking) return 'bg-yellow-500/20 border-yellow-500/30';
    return isOnline ? 'bg-green-500/20 border-green-500/30' : 'bg-red-500/20 border-red-500/30';
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Compact floating status */}
      <div 
        className={`glass border-glass-border rounded-full px-4 py-2 shadow-lg backdrop-blur-md transition-all duration-300 hover:scale-105 cursor-pointer ${getStatusColor()}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className="text-sm font-medium text-foreground">
            {getStatusText()}
          </span>
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="absolute bottom-12 right-0 w-64 glass border-glass-border rounded-lg p-4 shadow-xl backdrop-blur-md animate-in slide-in-from-bottom-2">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">API Status</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  checkApiStatus();
                }}
                disabled={isChecking}
                className="h-6 w-6 p-0"
              >
                <RefreshCw className={`h-3 w-3 ${isChecking ? 'animate-spin' : ''}`} />
              </Button>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Status:</span>
                <Badge 
                  variant={isOnline ? "default" : "destructive"} 
                  className="text-xs"
                >
                  {getStatusText()}
                </Badge>
              </div>
              
              {error && (
                <div className="text-xs text-red-600 bg-red-50 p-2 rounded border border-red-200">
                  {error}
                </div>
              )}
              
              {lastCheck && (
                <div className="text-xs text-muted-foreground">
                  Last checked: {lastCheck.toLocaleTimeString()}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
