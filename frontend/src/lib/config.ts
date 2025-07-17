/**
 * Frontend Configuration
 * 
 * Security-focused configuration management for environment variables
 * and API endpoints. Never expose sensitive data in client-side code.
 */

interface AppConfig {
  apiBaseUrl: string;
  isDevelopment: boolean;
  isProduction: boolean;
  requestTimeout: number;
  maxRetries: number;
}

// Environment-based configuration
const getApiBaseUrl = (): string => {
  // Check for environment variable first (production)
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // Development fallback
  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }
  
  // Production fallback - should be set via environment
  throw new Error('API_BASE_URL must be configured in production');
};

export const config: AppConfig = {
  apiBaseUrl: getApiBaseUrl(),
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD,
  requestTimeout: 30000, // 30 seconds
  maxRetries: 3
};

// Security headers for API requests
export const getSecureHeaders = (): Record<string, string> => ({
  'Content-Type': 'application/json',
  'X-Requested-With': 'XMLHttpRequest', // Basic CSRF protection
  'Accept': 'application/json'
});

// Error sanitization for production
export const sanitizeError = (error: unknown): string => {
  if (config.isDevelopment) {
    return error instanceof Error ? error.message : String(error);
  }
  
  // In production, return generic error messages
  return 'An error occurred. Please try again.';
}; 