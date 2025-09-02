
import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Download, ChevronDown, Sparkles, MessageSquare, Trash2, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { apiClient } from '@/lib/api';
import { useApiKeys } from '@/hooks/useApiKeys';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ClearSessionResponse {
  success: boolean;
  message: string;
  session_id: string | null;
  messages_deleted: number;
  error?: string;
}

interface ChatInterfaceProps {
  onSendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
  hasApiKeys?: boolean;
}

// Generate a unique session ID
const generateSessionId = (): string => {
  return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now().toString(36);
};

// Pass through content as-is - let the model output naturally
const cleanBoxedContent = (content: string): string => {
  return content;
};

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ onSendMessage, isLoading, hasApiKeys = true }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [thinkingContent, setThinkingContent] = useState('');
  const [showThinking, setShowThinking] = useState(false);
  const [thinkingExpanded, setThinkingExpanded] = useState(true);
  const [currentAiMessageId, setCurrentAiMessageId] = useState<string>('');
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [reasoningEffort, setReasoningEffort] = useState<string>('medium');
  
  // Check if current model supports reasoning effort
  const supportsReasoningEffort = (model: string) => {
    const reasoningModels = ['gpt-5-mini']; // Only OpenAI models support reasoning
    return reasoningModels.includes(model);
  };

  // Check if model supports reasoning levels (OpenAI)
  const supportsReasoningLevels = (model: string) => {
    const levelModels = ['gpt-5-mini']; // OpenAI models with levels
    return levelModels.includes(model);
  };
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const thinkingContentRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef<boolean>(true);
  const scrollRafRef = useRef<number | null>(null);

  // Get API keys status to show available providers
  const { apiKeyStatus } = useApiKeys();

  // Cleanup any pending animation frame on unmount
  useEffect(() => {
    return () => {
      if (scrollRafRef.current) {
        cancelAnimationFrame(scrollRafRef.current);
      }
    };
  }, []);

  // Get available providers (only configured ones)
  const getAvailableProviders = () => {
    if (!apiKeyStatus?.providers) return [];
    
    const providers = [
      { key: 'openai', name: 'OpenAI', models: ['gpt-5-mini'] },
      { key: 'mistral', name: 'Mistral', models: ['mistral-small-latest', 'mistral-medium-latest'] },
      { key: 'novita', name: 'Novita', models: ['openai/gpt-oss-20b', 'qwen/qwen3-32b-fp8', 'zai-org/glm-4.5'] },
      { key: 'ollama', name: 'Ollama (Local)', models: ['gemma3:1b', 'llama3.2:1b', 'hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M'] },
    ];

    return providers.filter(provider => 
      apiKeyStatus.providers[provider.key as keyof typeof apiKeyStatus.providers]?.configured
    );
  };

  const availableProviders = getAvailableProviders();

  // Initialize default provider/model selection
  useEffect(() => {
    if (availableProviders.length > 0 && !selectedProvider) {
      const firstProvider = availableProviders[0];
      setSelectedProvider(firstProvider.key);
      setSelectedModel(firstProvider.models[0]);
    }
  }, [availableProviders.length, selectedProvider]);

  // Update available models when provider changes
  const getAvailableModels = () => {
    const provider = availableProviders.find(p => p.key === selectedProvider);
    return provider?.models || [];
  };

  // Initialize session ID on component mount
  useEffect(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    console.log('Generated new session ID:', newSessionId);
  }, []);

  const isNearBottom = () => {
    if (!messagesContainerRef.current) return true;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const threshold = 150; // Increased threshold for better UX
    return scrollHeight - scrollTop - clientHeight < threshold;
  };

  const scrollToBottom = () => {
    if (!messagesContainerRef.current) return;
    const container = messagesContainerRef.current;
    // cancel any pending frame
    if (scrollRafRef.current) cancelAnimationFrame(scrollRafRef.current);
    scrollRafRef.current = requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    });
  };

  const handleScroll = () => {
    const atBottom = isNearBottom();
    setShowScrollButton(!atBottom);
    // If user scrolls up, disable auto-scroll; re-enable when near bottom
    autoScrollRef.current = atBottom;
  };

  useEffect(() => {
    // Only auto-scroll if user is already near the bottom or if it's a new message
    if (autoScrollRef.current || messages.length === 0) {
      scrollToBottom();
    }
  }, [messages]);

  // Auto-scroll thinking content to bottom as it's generated
  useEffect(() => {
    if (thinkingContentRef.current && thinkingExpanded && isStreaming) {
      thinkingContentRef.current.scrollTop = thinkingContentRef.current.scrollHeight;
    }
  }, [thinkingContent, thinkingExpanded, isStreaming]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || isStreaming || !sessionId) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const messageToSend = inputMessage.trim();
    setInputMessage('');

    // Create a placeholder AI message for streaming
    const aiMessageId = (Date.now() + 1).toString();
    const aiMessage: Message = {
      id: aiMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, aiMessage]);
    setIsStreaming(true);
    setCurrentAiMessageId(aiMessageId);
    setThinkingContent('');
    setShowThinking(false);
    setThinkingExpanded(true);

    try {
      // Use secure API client for streaming
      const requestData: Record<string, unknown> = {
        message: messageToSend,
        session_id: sessionId, // Use the generated session ID
        conversation_history: messages.slice(-10), // Send last 10 messages for context
        provider: selectedProvider, // Send selected provider
        model: selectedModel, // Send selected model
      };
      
      // Send reasoning_effort for models that support it
      if (supportsReasoningEffort(selectedModel)) {
        requestData.reasoning_effort = reasoningEffort;
      }
      
      const reader = await apiClient.stream('/chat/stream', requestData);

      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();
      let accumulatedContent = '';

      let streamComplete = false;
      // Buffer to handle partial lines across chunks
      let leftover = '';
      while (true) {
        const { done, value } = await reader.read();
        if (streamComplete) break;
        if (done) {
          // Process any remaining buffered content
          if (leftover) {
            const lastLine = leftover;
            leftover = '';
            if (lastLine.startsWith('data: ')) {
              try {
                const data = JSON.parse(lastLine.slice(6));
                if (data.type === 'chunk' || typeof data === 'string') {
                  const content = typeof data === 'string' ? data : data.content || '';
                  accumulatedContent += content;
                  const cleanedContent = cleanBoxedContent(accumulatedContent);
                  setMessages(prev => prev.map(msg => msg.id === aiMessageId ? { ...msg, content: cleanedContent } : msg));
                }
              } catch {
                // ignore invalid trailing JSON fragment
              }
            }
          }
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        const combined = leftover + chunk;
        const lines = combined.split('\n');
        // Keep the last line in buffer if it may be partial
        leftover = lines.pop() || '';

        for (const rawLine of lines) {
          const line = rawLine.trimEnd();
          if (!line) continue;
          if (line.startsWith('data: ')) {
            const payload = line.slice(6).trim();
            // Handle the [DONE] signal
            if (payload === '[DONE]') {
              streamComplete = true;
              break;
            }
            try {
              const data = JSON.parse(payload);
              
              // Handle structured format (Mistral reasoning model)
              if (typeof data === 'string') {
                try {
                  const structuredData = JSON.parse(data);
                  
                  if (structuredData.type === 'thinking') {
                    // Show thinking content
                    setShowThinking(true);
                    setThinkingExpanded(true);
                    setThinkingContent(prev => prev + structuredData.content);
                  } else if (structuredData.type === 'thinking_complete') {
                    // Keep thinking but collapse it, prepare for final answer
                    setThinkingExpanded(false);
                  } else if (structuredData.type === 'answer') {
                    // Add to final answer content
                    accumulatedContent += structuredData.content;
                    setMessages(prev => prev.map(msg => 
                      msg.id === aiMessageId 
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    ));
                  } else if (structuredData.type === 'final_answer') {
                    // This is the final answer from \boxed{} - display it prominently
                    accumulatedContent += `\n\n**Final Answer:** ${structuredData.content}`;
                    setMessages(prev => prev.map(msg => 
                      msg.id === aiMessageId 
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    ));
                  }
                } catch (parseError) {
                  // If not valid structured JSON, treat as regular content
                  accumulatedContent += data;
                  const cleanedContent = cleanBoxedContent(accumulatedContent);
                  setMessages(prev => prev.map(msg => 
                    msg.id === aiMessageId 
                      ? { ...msg, content: cleanedContent }
                      : msg
                  ));
                }
              }
              // Handle direct structured format (if already an object with type)
              else if (data.type) {
                if (data.type === 'thinking') {
                  setShowThinking(true);
                  setThinkingExpanded(true);
                  setThinkingContent(prev => prev + data.content);
                } else if (data.type === 'thinking_complete') {
                  setThinkingExpanded(false);
                } else if (data.type === 'chunk') {
                  // Handle streaming chunks
                  accumulatedContent += data.content;
                  const cleanedContent = cleanBoxedContent(accumulatedContent);
                  setMessages(prev => prev.map(msg => 
                    msg.id === aiMessageId 
                      ? { ...msg, content: cleanedContent }
                      : msg
                  ));
                } else if (data.type === 'answer') {
                  accumulatedContent += data.content;
                  const cleanedContent = cleanBoxedContent(accumulatedContent);
                  setMessages(prev => prev.map(msg => 
                    msg.id === aiMessageId 
                      ? { ...msg, content: cleanedContent }
                      : msg
                  ));
                }
              }
              // Handle legacy OpenAI-style format
              else if (data.chunk) {
                accumulatedContent += data.chunk;
                const cleanedContent = cleanBoxedContent(accumulatedContent);
                setMessages(prev => prev.map(msg => 
                  msg.id === aiMessageId 
                    ? { ...msg, content: cleanedContent }
                    : msg
                ));
              }
              
              if (data.done) {
                // Streaming is complete
                break;
              }
              
              if (data.error) {
                throw new Error(data.error);
              }
            } catch (e) {
              // Skip invalid JSON lines (likely partial), accumulate to content if plain text
              try {
                // Some providers may send raw text chunks
                accumulatedContent += payload;
                const cleanedContent = cleanBoxedContent(accumulatedContent);
                setMessages(prev => prev.map(msg => 
                  msg.id === aiMessageId 
                    ? { ...msg, content: cleanedContent }
                    : msg
                ));
              } catch {
                // ignore parse fallback errors
              }
              continue;
            }
          }
        }
      }

      // Ensure final content is set when streaming completes
      if (accumulatedContent && aiMessageId) {
        const cleanedContent = cleanBoxedContent(accumulatedContent);
        setMessages(prev => prev.map(msg => 
          msg.id === aiMessageId 
            ? { ...msg, content: cleanedContent }
            : msg
        ));
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Update the AI message with error
      setMessages(prev => prev.map(msg => 
        msg.id === aiMessageId 
          ? { ...msg, content: 'Sorry, I encountered an error while processing your message. Please check if the API is running and try again.' }
          : msg
      ));
    } finally {
      setIsStreaming(false);
      // Don't clear thinking content - keep it for the dropdown
      setCurrentAiMessageId('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const exportChat = () => {
    const chatText = messages.map(msg => 
      `${msg.role.toUpperCase()} (${msg.timestamp.toLocaleString()}): ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job-application-chat-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Add clear chat handler
  const handleClearChat = async () => {
    if (!sessionId) return; // Don't clear if no session ID yet
    
    if (window.confirm('Are you sure you want to clear this chat? This will erase the conversation.')) {
      try {
        // Call backend to clear session using the secure API client
        const response = await apiClient.post<ClearSessionResponse>('/chat/clear', {
          session_id: sessionId, // Use the generated session ID
        });

        if (!response.success) {
          throw new Error(response.error || 'Failed to clear chat');
        }

        const result = response.data;
        
                  if (result && result.success) {
            // Clear frontend state
            setMessages([]);
            setThinkingContent('');
            setShowThinking(false);
            setCurrentAiMessageId('');
            
            // Generate a new session ID for the next conversation
            const newSessionId = generateSessionId();
            setSessionId(newSessionId);
          
          console.log('Chat cleared successfully:', result.message);
          console.log('Generated new session ID:', newSessionId);
          
          // Show success feedback if messages were actually deleted
          if (result.messages_deleted > 0) {
            console.log(`Backend cleared ${result.messages_deleted} messages`);
          }
        } else {
          console.error('Failed to clear chat:', result?.message || 'Unknown error');
          // Still clear frontend as fallback and generate new session ID
          setMessages([]);
          setThinkingContent('');
          setShowThinking(false);
          setCurrentAiMessageId('');
          const newSessionId = generateSessionId();
          setSessionId(newSessionId);
          console.log('Generated new session ID after partial clear:', newSessionId);
          alert('Warning: Frontend cleared but there may have been an issue clearing the backend session.');
        }
      } catch (error) {
        console.error('Error clearing chat:', error);
        // Still clear frontend even if backend fails and generate new session ID
        setMessages([]);
        setThinkingContent('');
        setShowThinking(false);
        setThinkingExpanded(true);
        setCurrentAiMessageId('');
        const newSessionId = generateSessionId();
        setSessionId(newSessionId);
        console.log('Generated new session ID after error:', newSessionId);
        alert('Chat cleared from interface, but there may have been an issue clearing the backend session.');
      }
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-220px)] min-h-[600px] min-h-0 glass border-glass-border rounded-2xl overflow-hidden">
      {/* Sleek Header */}
      <div className="flex justify-between items-center p-2 bg-gradient-to-r from-blue-600/10 to-purple-600/10 border-b border-glass-border">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 gradient-primary rounded-md flex items-center justify-center shadow-glow">
            <MessageSquare className="h-3.5 w-3.5 text-white" />
          </div>
          <h2 className="text-base font-semibold text-gradient">AI Assistant</h2>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={exportChat}
            variant="outline"
            size="sm"
            disabled={messages.length === 0}
            className="glass border-glass-border hover:bg-white/20 h-7 px-2 text-[11px]"
            title="Export chat"
          >
            <Download className="h-3 w-3 mr-1" />
            Export
          </Button>
          <Button
            onClick={handleClearChat}
            variant="destructive"
            size="sm"
            disabled={messages.length === 0}
            className={`transition-all duration-200 h-7 px-2 text-[11px] ${
              messages.length === 0
                ? 'opacity-50 cursor-not-allowed bg-gray-100 text-gray-400 border-gray-200 hover:bg-gray-100'
                : 'bg-red-500 hover:bg-red-600 text-white border-red-500 hover:border-red-600 shadow-sm hover:shadow-md'
            }`}
            title={messages.length === 0 ? "No messages to clear" : "Clear chat conversation"}
          >
            <Trash2 className="h-3 w-3 mr-1" />
            Clear
          </Button>
        </div>
      </div>

      {/* Messages Container - Much larger */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3 sm:p-4 space-y-4" ref={messagesContainerRef} onScroll={handleScroll}>
        {!hasApiKeys ? (
          <div className="text-center text-muted-foreground mt-12">
            <div className="w-16 h-16 bg-muted/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <Bot className="h-8 w-8" />
            </div>
            <p className="text-lg font-medium">API Keys Required</p>
            <p className="text-sm mt-2">Please configure your API keys to start chatting with the AI assistant.</p>
            <p className="text-xs mt-1 text-muted-foreground">Your keys are stored securely and locally on your device.</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="text-center text-muted-foreground mt-12">
            <div className="w-16 h-16 gradient-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Sparkles className="h-8 w-8 text-primary" />
            </div>
            <p className="text-lg font-medium">Start a conversation!</p>
            <p className="text-sm mt-2">Ask questions about job applications, interview prep, or document optimization.</p>
            <div className="mt-6 p-4 bg-muted/30 rounded-lg border border-glass-border">
              <p className="text-xs text-muted-foreground">
                💡 <strong>Try asking:</strong> "How can I improve my resume for a software engineering role?"
              </p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-4 max-w-[85%] ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-sm ${
                    message.role === 'user'
                      ? 'gradient-primary text-white'
                      : 'bg-green-500 text-white'
                  }`}
                >
                  {message.role === 'user' ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
                </div>
                <div
                  className={`p-4 rounded-2xl shadow-sm ${
                    message.role === 'user'
                      ? 'gradient-primary text-white'
                      : 'glass border-glass-border bg-white/50'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                  <p className={`text-xs mt-3 ${
                    message.role === 'user' ? 'text-white/70' : 'text-muted-foreground'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
        
        {/* Show thinking process when available */}
        {showThinking && thinkingContent && (
          <div className="flex gap-4 justify-start">
            <div className="flex gap-4 max-w-[85%]">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-purple-500 text-white flex items-center justify-center shadow-sm">
                <Bot className="h-5 w-5" />
              </div>
              <div className="rounded-2xl bg-purple-50 border border-purple-200 shadow-sm">
                <div 
                  className="flex items-center gap-2 p-4 cursor-pointer hover:bg-purple-100 rounded-t-2xl"
                  onClick={() => setThinkingExpanded(!thinkingExpanded)}
                >
                  <div className="flex gap-1">
                    {thinkingExpanded && isStreaming ? (
                      <>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                      </>
                    ) : (
                      <ChevronDown className={`h-4 w-4 text-purple-600 transition-transform ${thinkingExpanded ? 'rotate-180' : ''}`} />
                    )}
                  </div>
                  <span className="text-sm font-medium text-purple-700">
                    {thinkingExpanded && isStreaming ? 'AI is thinking through this...' : 'AI reasoning process'}
                  </span>
                  {!thinkingExpanded && (
                    <span className="text-xs text-purple-500 ml-auto">Click to expand</span>
                  )}
                </div>
                {thinkingExpanded && (
                  <div className="px-4 pb-4 border-t border-purple-200">
                    <div 
                      ref={thinkingContentRef}
                      className="text-sm text-purple-800 whitespace-pre-wrap max-h-64 overflow-y-auto mt-2 scrollbar-thin scrollbar-thumb-purple-300 scrollbar-track-purple-100"
                    >
                      {thinkingContent}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {isStreaming && !showThinking && (
          <div className="flex gap-4 justify-start">
            <div className="flex gap-4 max-w-[85%]">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-green-500 text-white flex items-center justify-center shadow-sm">
                <Bot className="h-5 w-5" />
              </div>
              <div className="p-4 rounded-2xl glass border-glass-border bg-white/50 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-muted-foreground">AI is responding...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to bottom button */}
      {showScrollButton && (
        <Button
          onClick={scrollToBottom}
          className="fixed bottom-24 right-6 z-10 rounded-full w-12 h-12 p-0 shadow-lg gradient-primary hover:shadow-xl transition-all"
          size="sm"
        >
          <ChevronDown className="h-5 w-5 text-white" />
        </Button>
      )}

      {/* Provider/Model Selection */}
       {hasApiKeys && availableProviders.length > 0 && (
        <div className="px-3 py-2 border-t border-glass-border bg-white/20 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <Settings className="h-3.5 w-3.5 text-muted-foreground" />
            <div className="flex items-center gap-2 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">Provider:</span>
                <Select 
                  value={selectedProvider} 
                  onValueChange={(value) => {
                    setSelectedProvider(value);
                    const provider = availableProviders.find(p => p.key === value);
                    if (provider) {
                      setSelectedModel(provider.models[0]);
                    }
                  }}
                >
                  <SelectTrigger className="w-28 h-7 text-[11px]">
                    <SelectValue placeholder="Provider" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableProviders.map(provider => (
                      <SelectItem key={provider.key} value={provider.key}>
                        {provider.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">Model:</span>
                <Select 
                  value={selectedModel} 
                  onValueChange={setSelectedModel}
                >
                  <SelectTrigger className="w-44 h-7 text-[11px]">
                    <SelectValue placeholder="Model" />
                  </SelectTrigger>
                  <SelectContent>
                    {getAvailableModels().map(model => (
                      <SelectItem key={model} value={model}>
                        <span className="truncate">{model}</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              {/* Reasoning Controls - different UI for different model types */}
              {supportsReasoningEffort(selectedModel) && (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-muted-foreground">Reasoning:</span>



                  {/* Dropdown for OpenAI models (levels) */}
                  {supportsReasoningLevels(selectedModel) && (
                    <Select
                      value={reasoningEffort}
                      onValueChange={setReasoningEffort}
                    >
                      <SelectTrigger className="w-28 h-7 text-[11px]">
                        <SelectValue placeholder="Effort" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="minimal">Minimal</SelectItem>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 border-t border-glass-border bg-white/30 backdrop-blur-sm">
        <div className="flex gap-3">
          <Textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={hasApiKeys ? "Ask your AI assistant about job applications, interview tips, or document review..." : "Configure API keys to enable chat..."}
            className="resize-none min-h-[50px] max-h-[100px] glass border-glass-border focus:ring-2 focus:ring-primary/20"
            rows={2}
            disabled={isLoading || !hasApiKeys || isStreaming}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading || !hasApiKeys || isStreaming || !sessionId}
            className="px-4 py-2 h-[50px] gradient-primary text-white shadow-glow hover:shadow-xl transition-all rounded-lg"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
