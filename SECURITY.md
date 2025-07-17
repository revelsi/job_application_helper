# Security Policy

## Overview

This document outlines the security measures implemented in the Job Application Helper application to protect user data and ensure safe operation.

## 🔒 Security Features

### Data Protection
- **Local Storage Only**: All user documents and data are stored locally on your machine
- **No Cloud Upload**: Documents are never uploaded to external servers
- **Encrypted Credentials**: API keys are encrypted using Fernet encryption with proper file permissions
- **Secure File Handling**: All file operations include path traversal protection and validation

### API Security
- **Input Validation**: All user inputs are validated and sanitized before processing
- **Error Sanitization**: Error messages are sanitized to prevent information disclosure
- **Request Timeouts**: API requests include proper timeouts and retry mechanisms
- **Security Headers**: Frontend includes CSRF protection headers

### Privacy First Design
- **No Data Collection**: The application does not collect or transmit user data
- **Local Processing**: All AI processing happens through your configured API keys
- **Session Isolation**: Each session is isolated with no cross-session data sharing
- **Secure Deletion**: Document deletion removes both files and associated metadata

**Last Updated**: 9 July 2025  