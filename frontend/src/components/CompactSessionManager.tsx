import React, { useState } from 'react';
import { ChevronDown, Settings, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface SessionData {
  personalDocsCount: number;
  jobSpecificDocsCount: number;
  chatMessagesCount: number;
  lastActivity: Date;
  currentJobTitle?: string;
}

interface CompactSessionManagerProps {
  sessionData: SessionData;
  onClearPersonalDocs: () => Promise<void>;
  onClearJobDocs: () => Promise<void>;
  onShowApiKeys: () => void;
  isLoading: boolean;
  hasApiKeys: boolean;
}

export const CompactSessionManager: React.FC<CompactSessionManagerProps> = ({
  sessionData,
  onClearPersonalDocs,
  onClearJobDocs,
  onShowApiKeys,
  isLoading,
  hasApiKeys
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex items-center gap-3">
      {/* Stats Display */}
      <Card className="glass border-none">
        <CardContent className="p-3">
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-success"></div>
              <span className="text-muted-foreground">Personal:</span>
              <span className="font-medium">{sessionData.personalDocsCount}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-primary"></div>
              <span className="text-muted-foreground">Job:</span>
              <span className="font-medium">{sessionData.jobSpecificDocsCount}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-warning"></div>
              <span className="text-muted-foreground">Messages:</span>
              <span className="font-medium">{sessionData.chatMessagesCount}</span>
            </div>
          </div>
        </CardContent>
      </Card>



      {/* Dropdown Menu */}
      <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className="glass border-none hover:bg-white/20 transition-all"
          >
            <Settings className="h-4 w-4 mr-2" />
            <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56 glass border-glass-border">
          <DropdownMenuItem
            onClick={onShowApiKeys}
            className="flex items-center gap-2 hover:bg-white/15 focus:bg-white/15 transition-colors cursor-pointer"
          >
            <Settings className="h-4 w-4" />
            {hasApiKeys ? 'Update API Keys' : 'Setup API Keys'}
          </DropdownMenuItem>
          
          <DropdownMenuSeparator className="bg-white/10" />
          
          <DropdownMenuItem
            onClick={onClearPersonalDocs}
            disabled={sessionData.personalDocsCount === 0 || isLoading}
            className="flex items-center gap-2 text-warning hover:bg-warning/15 focus:bg-warning/15 transition-colors cursor-pointer data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed"
          >
            <BarChart3 className="h-4 w-4" />
            Clear Personal Documents ({sessionData.personalDocsCount})
          </DropdownMenuItem>
          
          <DropdownMenuItem
            onClick={onClearJobDocs}
            disabled={sessionData.jobSpecificDocsCount === 0 || isLoading}
            className="flex items-center gap-2 text-warning hover:bg-warning/15 focus:bg-warning/15 transition-colors cursor-pointer data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed"
          >
            <BarChart3 className="h-4 w-4" />
            Clear Job Documents ({sessionData.jobSpecificDocsCount})
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
};