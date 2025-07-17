# Job Application Helper API Reference

Base URL: `http://localhost:8000/`

---

## 1. Health Check

**GET /health/**

- **Description:** Simple health check to verify the API is running.
- **Response Example:**
  ```json
  {
    "status": "ok",
    "timestamp": "2024-07-08T12:00:00.000Z",
      "services": {
    "llm": true,
    "document_service": true,
    "storage": true
  }
  }
  ```

---

## 2. Chat Completion (LLM Interaction)

**POST /chat/complete**

- **Description:**
  Send a user message and (optionally) conversation history to get an LLM-generated response.
- **Request Body:**
  ```json
  {
    "message": "string",
    "session_id": "string", // (Optional)
    "conversation_history": [
      { "role": "user" | "assistant", "content": "string" }
    ]
  }
  ```
- **Response Example:**
  ```json
  {
    "response": "LLM-generated text",
    "session_id": "string",
    "success": true,
    "error": null,
    "metadata": {
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "document_context_used": true,
      "context_documents": 2,
      "retrieval_time": 0.12,
      "tokens_used": 651,
      "processing_time": 10.09,
      "request_type": "general",
      "content_type": "general_response"
    }
  }
  ```

---

## 3. Streaming Chat Completion

**POST /chat/stream**

- **Description:**
  Send a user message and get a streaming response from the LLM. Returns chunks of text as they are generated.
- **Request Body:** Same as `/chat/complete`
- **Response:** Server-Sent Events (SSE) stream with chunks of text
- **Content-Type:** `text/event-stream`
- **Stream Format:**
  ```
  data: {"chunk": "Hello", "session_id": "string"}
  data: {"chunk": " there", "session_id": "string"}
  data: {"done": true, "session_id": "string"}
  ```

---

## 4. API Key Management

### Get API Keys Status

**GET /api/keys/status**

- **Description:** Get the status of all configured API keys.
- **Response Example:**
  ```json
  {
      "providers": {
    "openai": { "has_env_key": true, "has_stored_key": false, "configured": true, "source": "env" }
  },
    "has_any_configured": true,
    "has_env_configured": true
  }
  ```

### Get Provider Info

**GET /api/keys/providers**

- **Description:** Get detailed information about all available providers.
- **Response Example:**
  ```json
  [
    { "provider": "openai", "configured": true, ... }
  ]
  ```

### Set API Key

**POST /api/keys/set**

- **Description:** Store an API key for a specific provider.
- **Request Body:**
  ```json
  {
    "provider": "openai",
    "api_key": "your_api_key_here"
  }
  ```
- **Response Example:**
  ```json
  {
    "success": true,
    "message": "API key stored successfully",
    "provider": "openai",
    "configured": true
  }
  ```

### Remove API Key

**DELETE /api/keys/{provider}**

- **Description:** Remove a stored API key for a specific provider.
- **Response Example:**
  ```json
  {
    "success": true,
    "message": "API key removed successfully",
    "provider": "openai",
    "configured": false
  }
  ```

### Test API Key

**POST /api/keys/test/{provider}**

- **Description:** Test if an API key is valid for a specific provider.
- **Response Example:**
  ```json
  {
    "success": true,
    "message": "API key for openai is valid and working",
    "provider": "openai",
    "configured": true
  }
  ```

---

## 5. Document Management

### Upload Document

**POST /documents/upload**

- **Description:** Upload a document (CV, job description, etc.) for processing and inclusion in the document context system.
- **Request:** `multipart/form-data` with fields:
  - `file`: The document file
  - `category`: `personal` or `job-specific`
- **Response Example:**
  ```json
  {
    "success": true,
    "document_id": "string",
    "message": "Document uploaded and processed successfully",
    "metadata": {
      "file_name": "resume.pdf",
      "document_type": "cv",
      "text_length": 12345
    }
  }
  ```

### List Documents

**GET /documents/list**

- **Description:** List all uploaded documents and their metadata.
- **Response Example:**
  ```json
  {
    "success": true,
    "documents": [
      {
        "id": "string",
        "filename": "resume.pdf",
        "type": "cv",
        "category": "personal",
        "upload_date": "2024-07-08T12:00:00Z",
        "size": 12345,
        "tags": []
      }
    ]
  }
  ```

### Get Document Status

**GET /documents/status**

- **Description:** Get document statistics and status.
- **Response Example:**
  ```json
  {
    "success": true,
    "stats": {
      "total_documents": 5,
      "by_type": { "cv": 2, "job_description": 3 }
    }
  }
  ```

### Delete Document

**DELETE /documents/{document_id}**

- **Description:** Delete a document by its ID.
- **Response Example:**
  ```json
  {
    "success": true,
    "message": "Document deleted"
  }
  ```

### Clear Documents

**POST /documents/clear**

- **Description:** Clear all documents of a specific type or all documents.
- **Request Body:**
  ```json
  {
    "document_type": "cv" | "job_description" | "cover_letter" | "all"
  }
  ```
- **Response Example:**
  ```json
  {
    "success": true,
    "message": "Documents cleared successfully",
    "cleared_count": 5
  }
  ```

---

## 6. API Documentation

**GET /docs**

- **Description:** Interactive Swagger UI for exploring and testing all endpoints.

---

## Notes for Frontend Design

- **Authentication:**
  If you add user accounts in the future, endpoints may require authentication headers.
- **Session Management:**
  Use `session_id` to maintain chat continuity for each user.
- **File Uploads:**
  Use `multipart/form-data` for document uploads.
- **Error Handling:**
  All endpoints return a `success` boolean and an `error` field for error messages.
- **LLM Metadata:**
  The `metadata` field in chat responses provides insight into which model/provider was used, token usage, and timing.
- **Streaming Responses:**
  The streaming endpoint provides real-time text generation for better user experience.
- **API Key Management:**
  Environment variables take precedence over stored keys. Stored keys are encrypted for security.

---

## For More Details

- **Interactive API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) 