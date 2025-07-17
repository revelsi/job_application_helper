# Job Application Helper - Frontend Dockerfile
# Optimized for React 18, TypeScript, and Vite

FROM node:18-alpine

# Set environment variables
ENV NODE_ENV=development
ENV VITE_API_BASE_URL=http://localhost:8000

# Install system dependencies
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -u 1001

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm ci && \
    npm cache clean --force

# Copy application code
COPY src/ ./src/
COPY index.html .
COPY *.config.* ./
COPY tsconfig*.json ./

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Build the app for production
RUN npm run build

# Switch to root to install serve globally
USER root
RUN npm install -g serve

# Switch back to appuser
USER appuser

# Start the production server
CMD ["serve", "-s", "dist", "-l", "8080"] 