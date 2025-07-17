
import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Download, ChevronDown, Sparkles, MessageSquare, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { apiClient } from '@/lib/api';

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

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ onSendMessage, isLoading, hasApiKeys = true }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

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
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleScroll = () => {
    setShowScrollButton(!isNearBottom());
  };

  useEffect(() => {
    // Only auto-scroll if user is already near the bottom or if it's a new message
    if (isNearBottom() || messages.length === 0) {
      scrollToBottom();
    }
  }, [messages]);

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

    try {
      // Use secure API client for streaming
      const reader = await apiClient.stream('/chat/stream', {
        message: messageToSend,
        session_id: sessionId, // Use the generated session ID
        conversation_history: messages.slice(-10) // Send last 10 messages for context
      });

      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();
      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.chunk) {
                accumulatedContent += data.chunk;
                // Update the AI message with accumulated content
                setMessages(prev => prev.map(msg => 
                  msg.id === aiMessageId 
                    ? { ...msg, content: accumulatedContent }
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
              // Skip invalid JSON lines
              continue;
            }
          }
        }
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
          const newSessionId = generateSessionId();
          setSessionId(newSessionId);
          console.log('Generated new session ID after partial clear:', newSessionId);
          alert('Warning: Frontend cleared but there may have been an issue clearing the backend session.');
        }
      } catch (error) {
        console.error('Error clearing chat:', error);
        // Still clear frontend even if backend fails and generate new session ID
        setMessages([]);
        const newSessionId = generateSessionId();
        setSessionId(newSessionId);
        console.log('Generated new session ID after error:', newSessionId);
        alert('Chat cleared from interface, but there may have been an issue clearing the backend session.');
      }
    }
  };

  return (
    <div className="flex flex-col h-[600px] glass border-glass-border rounded-2xl overflow-hidden">
      {/* Sleek Header */}
      <div className="flex justify-between items-center p-6 bg-gradient-to-r from-blue-600/10 to-purple-600/10 border-b border-glass-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 gradient-primary rounded-xl flex items-center justify-center shadow-glow">
            <MessageSquare className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gradient">AI Assistant</h2>
            <p className="text-sm text-muted-foreground">Your career companion</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={exportChat}
            variant="outline"
            size="sm"
            disabled={messages.length === 0}
            className="glass border-glass-border hover:bg-white/20"
            title="Export chat"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button
            onClick={handleClearChat}
            variant="destructive"
            size="sm"
            disabled={messages.length === 0}
            className={`transition-all duration-200 ${
              messages.length === 0
                ? 'opacity-50 cursor-not-allowed bg-gray-100 text-gray-400 border-gray-200 hover:bg-gray-100'
                : 'bg-red-500 hover:bg-red-600 text-white border-red-500 hover:border-red-600 shadow-sm hover:shadow-md'
            }`}
            title={messages.length === 0 ? "No messages to clear" : "Clear chat conversation"}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Clear
          </Button>
        </div>
      </div>

      {/* Messages Container - Much larger */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6" ref={messagesContainerRef} onScroll={handleScroll}>
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
        
        {isStreaming && (
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
                  <span className="text-sm text-muted-foreground">AI is thinking...</span>
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

      {/* Input Area */}
      <div className="p-6 border-t border-glass-border bg-white/30 backdrop-blur-sm">
        <div className="flex gap-3">
          <Textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={hasApiKeys ? "Ask your AI assistant about job applications, interview tips, or document review..." : "Configure API keys to enable chat..."}
            className="resize-none min-h-[60px] max-h-[120px] glass border-glass-border focus:ring-2 focus:ring-primary/20"
            rows={2}
            disabled={isLoading || !hasApiKeys || isStreaming}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading || !hasApiKeys || isStreaming || !sessionId}
            className="px-6 py-3 h-auto gradient-primary text-white shadow-glow hover:shadow-xl transition-all rounded-xl"
          >
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
};
