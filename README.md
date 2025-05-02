# Chat Service

A FastAPI-based chat service that provides a flexible and powerful chat interface with support for tools and conversation history.

## Project API

The project API provides a chat endpoint that supports conversation history, system prompts, and tool usage.

### Chat Endpoint

`POST /api/chat`

Processes a chat message and maintains conversation history when an identifier is provided.

#### Request Body

```json
{
  "prompt": "What's the weather like?",
  "system_prompt": "You are a helpful assistant.",
  "ident": "user123",
  "use_tools": true,
  "max_iterations": 5,
  "clean": false
}
```

##### Fields

- `prompt` (required): The user's message prompt
- `system_prompt` (optional): System prompt to set the context for the conversation
- `ident` (optional): Identifier to maintain chat history. If provided, the service will maintain up to 5 messages of history
- `use_tools` (optional, default: true): Whether to enable tool usage in the chat
- `max_iterations` (optional, default: 5): Maximum number of iterations for the agent to use (1-10)
- `clean` (optional, default: false): If true, clears the chat history for the given ident

#### Response

```json
{
  "response": "The weather is sunny and 75°F.",
  "tool_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "location": "New York"
      },
      "result": "{\"temperature\": 75, \"condition\": \"sunny\"}"
    }
  ],
  "history": [
    {
      "role": "user",
      "content": "What's the weather like?"
    },
    {
      "role": "assistant",
      "content": "The weather is sunny and 75°F."
    }
  ]
}
```

##### Fields

- `response`: The chat response from the AI
- `tool_calls`: List of tools called during the conversation, including their names, arguments, and results
- `history`: The conversation history (only included when an ident is provided)

### Features

- **Conversation History**: Maintains up to 5 messages of history per ident
- **System Prompts**: Supports setting context through system prompts
- **Tool Usage**: Enables/disables tool usage in conversations
- **History Management**: Ability to clear history for a specific ident
- **Iteration Control**: Configurable maximum iterations for complex conversations

### Example Usage

```python
import requests

# Start a new conversation
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "What's the weather like?",
        "ident": "user123",
        "use_tools": True
    }
)

# Continue the conversation
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "What about tomorrow?",
        "ident": "user123"
    }
)

# Clear history and start fresh
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "Let's start over. What's the weather like?",
        "ident": "user123",
        "clean": True
    }
)
```

## Configuration

The service requires the following environment variables:

### Application Configuration

These variables are prefixed with `CHAT_SERVICE_`:

- `CHAT_SERVICE_NAME`: Name of your service (default: "Chat Service")
- `CHAT_SERVICE_VERSION`: Version of your service (default: package version)
- `CHAT_SERVICE_LOG_LEVEL`: Logging level (default: 20 for INFO)

### AI Service Configuration

- `OPENROUTER_TOKEN`: Your OpenRouter API token for AI model interactions
- `OPENROUTER_BASE_URL`: Base URL for OpenRouter API (default: "https://openrouter.ai/api/v1")
- `MCP_SERVER_URL`: URL of the Model Context Protocol server (optional)

These can be set in your environment or in a `.env` file in the project root.

## Development

### Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   ```bash
   # Application configuration
   export CHAT_SERVICE_NAME="Chat Service"
   export CHAT_SERVICE_LOG_LEVEL=20

   # AI service configuration
   export OPENROUTER_TOKEN=your_api_key
   export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   export MCP_SERVER_URL=your_mcp_server_url  # Optional
   ```

### Running the Service

```bash
uvicorn src.chat_service.main:app --reload
```

The service will be available at `http://localhost:8000`.
