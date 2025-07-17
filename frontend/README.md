# Job Application Helper - Frontend

A modern React frontend for the Job Application Helper, providing an intuitive interface for AI-powered job application assistance.

## 🏗️ Technology Stack

- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - High-quality, accessible UI components
- **Lucide React** - Beautiful, customizable icons

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend API running (see backend README)

### Development Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# The frontend will be available at http://localhost:8080
```

### Production Build
```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 🎯 Features

- **Document Upload**: Drag-and-drop interface for uploading resumes, cover letters, and job descriptions
- **AI Chat Interface**: Interactive chat with context-aware responses based on uploaded documents
- **Session Management**: Organize conversations by job application
- **API Status Monitoring**: Real-time backend connection status
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Accessibility**: Built with WCAG guidelines in mind

## 📁 Project Structure

```
src/
├── components/           # React components
│   ├── ui/              # Reusable UI components (shadcn/ui)
│   ├── ApiStatus.tsx    # Backend connection status
│   ├── ChatInterface.tsx # Main chat interface
│   ├── DocumentUpload.tsx # File upload component
│   └── SessionManager.tsx # Session organization
├── hooks/               # Custom React hooks
├── lib/                 # Utility functions
├── pages/               # Page components
└── main.tsx            # Application entry point
```

## 🔧 Development

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues

### Environment Configuration
The frontend connects to the backend API. Ensure the backend is running on the expected port (default: 8000).

### Code Quality
- ESLint configuration for consistent code style
- TypeScript for type safety
- Prettier integration for code formatting

## 🎨 UI Components

This project uses shadcn/ui components for a consistent, accessible design system:

- **Forms**: Input, textarea, select, checkbox, radio
- **Navigation**: Tabs, breadcrumbs, pagination
- **Feedback**: Alerts, toasts, progress indicators
- **Layout**: Cards, separators, aspect ratios
- **Overlays**: Dialogs, popovers, tooltips

## 📱 Responsive Design

The interface is fully responsive and optimized for:
- Desktop (1024px+)
- Tablet (768px - 1023px)
- Mobile (320px - 767px)

## 🔐 Security

- No sensitive data stored in frontend
- API keys handled by backend only
- Secure communication with backend API
- Input validation and sanitization

## 🤝 Contributing

1. Follow the established code style (ESLint + Prettier)
2. Use TypeScript for all new components
3. Add proper error handling and loading states
4. Test components across different screen sizes
5. Ensure accessibility compliance

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../LICENSE) file for details.
