"""Project-level API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any
import json
import logging

from mbxai.openrouter import OpenRouterClient, OpenRouterModel
from mbxai.mcp import MCPClient
from ..config import get_mcp_config, get_openrouter_api_config

# Set up logging
logger = logging.getLogger(__name__)

# Create a router for project-level endpoints
router = APIRouter(prefix="/api", tags=["api"])

# In-memory storage for chat history
chat_history: dict[str, list[dict[str, str]]] = {}


class ToolCall(BaseModel):
    """Model for tool calls."""

    name: str
    arguments: dict[str, Any]
    result: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model."""

    prompt: str = Field(..., description="The user's message prompt")
    system_prompt: Optional[str] = Field(
        None, description="Optional system prompt to set the context"
    )
    ident: Optional[str] = Field(
        None, description="Optional identifier to maintain chat history"
    )
    use_tools: bool = Field(
        True, description="Whether to enable tool usage in the chat"
    )
    max_iterations: int = Field(
        5, description="Maximum number of iterations for the agent to use", ge=1, le=10
    )
    clean: bool = Field(
        False, description="If true, clears the chat history for the given ident"
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    history: list[dict[str, str]] = Field(default_factory=list)


class Answer(BaseModel):
    """Structured chat response model with parsed content."""

    content: str
    context: str

class StructuredChatResponse(BaseModel):
    """Structured chat response model with parsed content."""

    answers: list[Answer]


class ChatResponseFormat(BaseModel):
    """Format for the chat response."""

    content: str
    tool_calls: Optional[list[ToolCall]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message and maintain history if ident is provided.

    Args:
        request: The chat request containing the prompt and optional parameters

    Returns:
        ChatResponse containing the response, tool calls, and message history
    """
    try:
        # Initialize OpenRouter client
        openrouter_config = get_openrouter_api_config()
        openrouter_client = OpenRouterClient(
            token=openrouter_config.api_key,
            base_url=openrouter_config.base_url,
            model=OpenRouterModel.GPT41
        )

        # Initialize MCP client with OpenRouter client
        client = MCPClient(openrouter_client)

        # Get MCP config and connect to server if configured
        mcp_config = get_mcp_config()
        if mcp_config.server_url:
            client.register_mcp_server(
                name="mcp-server",
                base_url=mcp_config.server_url
            )
            logger.info("MCP server registered")

        # Prepare messages for the chat
        messages = []

        # Handle history and cleaning
        if request.ident:
            if request.clean:
                # Clear history for the ident
                chat_history[request.ident] = []
            elif request.ident not in chat_history:
                chat_history[request.ident] = []
            else:
                # Add existing history
                messages.extend(chat_history[request.ident])

        # Add system prompt if provided
        if request.system_prompt:
            # Check if this system prompt is already in the history
            has_system_prompt = any(
                msg["role"] == "system" and msg["content"] == request.system_prompt
                for msg in messages
            )

            if not has_system_prompt:
                # Add system prompt to messages for this request
                messages.append({"role": "system", "content": request.system_prompt})

                # Add system prompt to history if ident is provided
                if request.ident:
                    chat_history[request.ident].append(
                        {"role": "system", "content": request.system_prompt}
                    )

        # Add the current user message
        messages.append({"role": "user", "content": request.prompt})

        # Process the chat using OpenRouter
        try:
            logger.info(f"Processing chat with messages: {messages}")
            response = client.chat(
                messages=messages,
                model=OpenRouterModel.GPT41
            )

            if not response or not response.choices:
                logger.error("Received empty response from client.chat")
                raise HTTPException(status_code=500, detail="Received empty response from chat service")

            message = response.choices[0].message.content
            if not message:
                logger.error("Response missing message")
                raise HTTPException(status_code=500, detail="Response missing message")

            # Update history if ident is provided
            if request.ident:
                # Add user message to history
                chat_history[request.ident].append(
                    {"role": "user", "content": request.prompt}
                )

                # Add assistant response to history
                if not message:
                    logger.error(f"Response missing content: {response}")
                    raise HTTPException(status_code=500, detail="Response missing content")
                
                chat_history[request.ident].append(
                    response.choices[0].message.dict()
                )

                # Keep only the last 5 messages
                chat_history[request.ident] = chat_history[request.ident][-5:]

                return ChatResponse(
                    response=message,
                    history=chat_history[request.ident],
                )
            else:
                if not message:
                    logger.error(f"Response missing content: {response}")
                    raise HTTPException(status_code=500, detail="Response missing content")
                return ChatResponse(response=message)

        except Exception as e:
            logger.error(f"Error processing chat: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


@router.post("/chat-struct", response_model=StructuredChatResponse)
async def chat(request: ChatRequest) -> StructuredChatResponse:
    """Process a chat message and maintain history if ident is provided.

    Args:
        request: The chat request containing the prompt and optional parameters

    Returns:
        ChatResponse containing the response, tool calls, and message history
    """
    try:
        # Initialize OpenRouter client
        openrouter_config = get_openrouter_api_config()
        openrouter_client = OpenRouterClient(
            token=openrouter_config.api_key,
            base_url=openrouter_config.base_url,
            model=OpenRouterModel.GPT41
        )

        # Initialize MCP client with OpenRouter client
        client = MCPClient(openrouter_client)

        # Get MCP config and connect to server if configured
        mcp_config = get_mcp_config()
        if mcp_config.server_url:
            client.register_mcp_server(
                name="mcp-server",
                base_url=mcp_config.server_url
            )
            logger.info("MCP server registered")

        # Prepare messages for the chat
        messages = []

        # Handle history and cleaning
        if request.ident:
            if request.clean:
                # Clear history for the ident
                chat_history[request.ident] = []
            elif request.ident not in chat_history:
                chat_history[request.ident] = []
            else:
                # Add existing history
                messages.extend(chat_history[request.ident])

        # Add system prompt if provided
        if request.system_prompt:
            # Check if this system prompt is already in the history
            has_system_prompt = any(
                msg["role"] == "system" and msg["content"] == request.system_prompt
                for msg in messages
            )

            if not has_system_prompt:
                # Add system prompt to messages for this request
                messages.append({"role": "system", "content": request.system_prompt})

                # Add system prompt to history if ident is provided
                if request.ident:
                    chat_history[request.ident].append(
                        {"role": "system", "content": request.system_prompt}
                    )

        # Add the current user message
        messages.append({"role": "user", "content": request.prompt})

        # Process the chat using OpenRouter
        try:
            logger.info(f"Processing chat with messages: {messages}")
            response = client.parse(
                messages=messages,
                response_format=StructuredChatResponse,
                model=OpenRouterModel.GPT41
            )
            logger.info(f"Structured response received: {response}")

            if not response or not response.choices:
                logger.error("Received empty response from client.chat")
                raise HTTPException(status_code=500, detail="Received empty response from chat service")

            message = response.choices[0].message.content
            if not message:
                logger.error("Response missing message")
                raise HTTPException(status_code=500, detail="Response missing message")

            # Update history if ident is provided
            if request.ident:
                # Add user message to history
                chat_history[request.ident].append(
                    {"role": "user", "content": request.prompt}
                )

                # Add assistant response to history
                if not message:
                    logger.error(f"Response missing content: {response}")
                    raise HTTPException(status_code=500, detail="Response missing content")
                
                chat_history[request.ident].append(
                    response.choices[0].message.dict()
                )

                # Keep only the last 5 messages
                chat_history[request.ident] = chat_history[request.ident][-5:]

                return ChatResponse(
                    response=message,
                    history=chat_history[request.ident],
                )
            else:
                if not message:
                    logger.error(f"Response missing content: {response}")
                    raise HTTPException(status_code=500, detail="Response missing content")
                return ChatResponse(response=message)

        except Exception as e:
            logger.error(f"Error processing chat: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")
