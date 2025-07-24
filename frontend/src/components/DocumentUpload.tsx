
import React, { useState, useCallback, useEffect } from 'react';
import { Upload, File, Trash2, Calendar, FileText, Award, Briefcase, Loader2, ArrowRight, Image, Video, Music, Archive, Code, FileSpreadsheet, Presentation, FileText as FilePdf, FileText as FileWord, FileSpreadsheet as FileExcel, Presentation as FilePowerpoint, Clock, HardDrive, Tag } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface Document {
  id: string;
  filename: string;
  type: string;
  size: number;
  upload_date: string;
  category: 'personal' | 'job-specific';
  tags: string[];
}

interface DocumentUploadProps {
  onUpload: (file: File, category: 'personal' | 'job-specific') => Promise<void>;
  onDelete: (documentId: string) => Promise<void>;
  isUploading: boolean;
  existingDocuments?: Document[];
  onDocumentsLoaded?: (documents: Document[]) => void;
  onNext?: () => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUpload,
  onDelete,
  isUploading,
  existingDocuments = [],
  onDocumentsLoaded,
  onNext
}) => {
  const [documents, setDocuments] = useState<Document[]>(existingDocuments);
  const [dragActive, setDragActive] = useState<{ personal: boolean, jobSpecific: boolean }>({
    personal: false,
    jobSpecific: false
  });

  // Update documents when existingDocuments prop changes
  useEffect(() => {
    setDocuments(existingDocuments);
    if (onDocumentsLoaded) {
      onDocumentsLoaded(existingDocuments);
    }
  }, [existingDocuments, onDocumentsLoaded]);

  const handleDrag = useCallback((e: React.DragEvent, category: 'personal' | 'job-specific') => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(prev => ({ ...prev, [category === 'personal' ? 'personal' : 'jobSpecific']: true }));
    } else if (e.type === "dragleave") {
      setDragActive(prev => ({ ...prev, [category === 'personal' ? 'personal' : 'jobSpecific']: false }));
    }
  }, []);

  const handleFileUpload = useCallback(async (file: File, category: 'personal' | 'job-specific') => {
    try {
      await onUpload(file, category);
      // Parent component will handle refreshing the document list
    } catch (error) {
      console.error('Upload failed:', error);
    }
  }, [onUpload]);

  const handleDrop = useCallback(async (e: React.DragEvent, category: 'personal' | 'job-specific') => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive({ personal: false, jobSpecific: false });

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      await handleFileUpload(file, category);
    }
  }, [handleFileUpload]);

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>, category: 'personal' | 'job-specific') => {
    if (e.target.files && e.target.files[0]) {
      await handleFileUpload(e.target.files[0], category);
      e.target.value = ''; // Reset input
    }
  };

  const handleDelete = async (documentId: string) => {
    try {
      await onDelete(documentId);
      setDocuments(prev => prev.filter(doc => doc.id !== documentId));
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.toLowerCase().split('.').pop();
    switch (extension) {
      case 'pdf':
        return <FilePdf className="h-5 w-5 text-red-500" />;
      case 'doc':
      case 'docx':
        return <FileWord className="h-5 w-5 text-blue-600" />;
      case 'xls':
      case 'xlsx':
        return <FileExcel className="h-5 w-5 text-green-600" />;
      case 'ppt':
      case 'pptx':
        return <FilePowerpoint className="h-5 w-5 text-orange-500" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'svg':
        return <Image className="h-5 w-5 text-purple-500" />;
      case 'mp4':
      case 'avi':
      case 'mov':
        return <Video className="h-5 w-5 text-pink-500" />;
      case 'mp3':
      case 'wav':
      case 'flac':
        return <Music className="h-5 w-5 text-indigo-500" />;
      case 'zip':
      case 'rar':
      case '7z':
        return <Archive className="h-5 w-5 text-yellow-600" />;
      case 'txt':
      case 'md':
        return <FileText className="h-5 w-5 text-gray-600" />;
      case 'json':
      case 'xml':
      case 'csv':
        return <Code className="h-5 w-5 text-teal-500" />;
      default:
        return <File className="h-5 w-5 text-gray-500" />;
    }
  };

  const getFileTypeColor = (fileName: string) => {
    const extension = fileName.toLowerCase().split('.').pop();
    switch (extension) {
      case 'pdf':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'doc':
      case 'docx':
        return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'xls':
      case 'xlsx':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'ppt':
      case 'pptx':
        return 'bg-orange-100 text-orange-700 border-orange-200';
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'svg':
        return 'bg-purple-100 text-purple-700 border-purple-200';
      case 'mp4':
      case 'avi':
      case 'mov':
        return 'bg-pink-100 text-pink-700 border-pink-200';
      case 'mp3':
      case 'wav':
      case 'flac':
        return 'bg-indigo-100 text-indigo-700 border-indigo-200';
      case 'zip':
      case 'rar':
      case '7z':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'txt':
      case 'md':
        return 'bg-gray-100 text-gray-700 border-gray-200';
      case 'json':
      case 'xml':
      case 'csv':
        return 'bg-teal-100 text-teal-700 border-teal-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const personalDocs = documents.filter(doc => doc.category === 'personal');
  const jobSpecificDocs = documents.filter(doc => doc.category === 'job-specific');

  const UploadArea = ({ category, title, description, icon, gradientClass }: {
    category: 'personal' | 'job-specific',
    title: string,
    description: string,
    icon: React.ReactNode,
    gradientClass: string
  }) => (
    <Card className="h-full glass border-glass-border overflow-hidden">
      <CardHeader className={`${gradientClass} bg-opacity-10 border-b border-glass-border`}>
        <CardTitle className="flex items-center gap-3 text-xl font-bold">
          <div className="p-2 rounded-lg bg-white/20 backdrop-blur-sm">
            {icon}
          </div>
          <div>
            <div className="text-gradient">{title}</div>
            <div className="text-sm font-normal text-muted-foreground mt-1">{description}</div>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <div
          className={`
            border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer group relative overflow-hidden
            ${dragActive[category === 'personal' ? 'personal' : 'jobSpecific']
              ? 'border-primary bg-primary/10 shadow-glow scale-105'
              : 'border-border hover:border-primary hover:bg-primary/5 hover:scale-[1.02]'
            }
            ${isUploading ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-xl'}
          `}
          onDragEnter={(e) => handleDrag(e, category)}
          onDragLeave={(e) => handleDrag(e, category)}
          onDragOver={(e) => handleDrag(e, category)}
          onDrop={(e) => handleDrop(e, category)}
          onClick={() => !isUploading && document.getElementById(`file-input-${category}`)?.click()}
        >
          {/* Background gradient effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          
          <div className="relative z-10">
            <div className="w-20 h-20 mx-auto mb-6 gradient-primary rounded-2xl flex items-center justify-center shadow-glow group-hover:shadow-2xl group-hover:scale-110 transition-all duration-300">
              {isUploading ? (
                <Loader2 className="h-10 w-10 text-white animate-spin" />
              ) : (
                <Upload className="h-10 w-10 text-white" />
              )}
            </div>
            <h3 className="text-xl font-bold mb-3 text-gradient">
              {isUploading ? 'Processing...' : 'Drop files here'}
            </h3>
            <p className="text-muted-foreground mb-3 text-base">
              {isUploading ? 'Please wait while your file is being processed' : 'Click to browse or drag & drop files'}
            </p>
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <FileText className="h-3 w-3 text-red-500" />
                PDF
              </span>
              <span className="flex items-center gap-1">
                <FileText className="h-3 w-3 text-blue-600" />
                DOC
              </span>
              <span className="flex items-center gap-1">
                <FileText className="h-3 w-3 text-gray-500" />
                TXT
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {isUploading ? 'This may take a few moments' : 'Max 10MB per file'}
            </p>
          </div>
          
          <input
            id={`file-input-${category}`}
            type="file"
            className="hidden"
            accept=".pdf,.doc,.docx,.txt"
            onChange={(e) => handleFileInput(e, category)}
            disabled={isUploading}
          />
        </div>

        {/* Documents List */}
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gradient">
              Uploaded Documents
            </h4>
            <Badge variant="secondary" className="bg-white/20 text-muted-foreground">
              {(category === 'personal' ? personalDocs : jobSpecificDocs).length} files
            </Badge>
          </div>
          
          <div className="space-y-3">
            {(category === 'personal' ? personalDocs : jobSpecificDocs).map((doc) => (
              <div key={doc.id} className="group relative glass rounded-xl border-glass-border hover:bg-white/10 transition-all duration-200 hover:shadow-lg overflow-hidden">
                <div className="flex items-center p-4">
                  <div className="flex items-center gap-4 flex-1 min-w-0">
                    <div className="p-2 rounded-lg bg-white/10 backdrop-blur-sm">
                      {getFileIcon(doc.filename)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="text-sm font-semibold truncate text-gradient">{doc.filename}</p>
                        <span className={`text-xs px-2 py-0.5 border rounded-full ${getFileTypeColor(doc.filename)}`}>
                          {doc.filename.split('.').pop()?.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <HardDrive className="h-3 w-3" />
                          {formatFileSize(doc.size)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {new Date(doc.upload_date).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(doc.id)}
                    className="opacity-0 group-hover:opacity-100 text-destructive hover:text-destructive hover:bg-destructive/10 transition-all duration-200"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
            
            {(category === 'personal' ? personalDocs : jobSpecificDocs).length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <File className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">No documents uploaded yet</p>
                <p className="text-xs">Upload your first document to get started</p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <UploadArea
          category="personal"
          title="Personal Documents"
          description="CV, cover letters, certificates, portfolio"
          icon={<Award className="h-6 w-6 text-green-600" />}
          gradientClass="from-green-500/20 to-emerald-500/20"
        />
        
        <UploadArea
          category="job-specific"
          title="Job-Specific Documents"
          description="Job descriptions, company info, requirements"
          icon={<Briefcase className="h-6 w-6 text-blue-600" />}
          gradientClass="from-blue-500/20 to-indigo-500/20"
        />
      </div>

      {/* Next Button - Show when at least one document is uploaded */}
      {documents.length > 0 && onNext && (
        <div className="flex justify-end pt-6 border-t border-glass-border">
          <Button
            onClick={onNext}
            disabled={isUploading}
            className="gradient-primary text-white shadow-glow hover:shadow-xl transition-all px-8 py-3 text-lg"
          >
            Next: Start AI Chat
            <ArrowRight className="h-5 w-5 ml-2" />
          </Button>
        </div>
      )}
    </div>
  );
};
