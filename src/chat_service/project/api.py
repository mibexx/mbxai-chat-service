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


class ChatMessage(BaseModel):
    """Model for a single chat message."""
    role: str = Field(..., description="The role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="The content of the message")
    type: str = Field("input_text", description="The type of the message (input_text, input_image, input_file)")
    file_data: Optional[str] = Field(None, description="The base64 encoded data of the file if the type is input_file")
    image_url: Optional[str] = Field(None, description="The base64 encoded data of the image (data:image/jpeg;base64,\{base64_image\}) or the url to image if the type is input_image")


class ChatRequest(BaseModel):
    """Chat request model."""

    messages: list[ChatMessage] = Field(..., description="List of chat messages")
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


def convert_to_openai_format(message: ChatMessage) -> dict[str, Any]:
    """Convert a ChatMessage to OpenAI format.
    
    Args:
        message: The ChatMessage to convert
        
    Returns:
        dict: Message in OpenAI format
    """
    if message.type == "input_text":
        return {
            "role": message.role,
            "content": message.content
        }
    elif message.type == "input_image":
        return {
            "role": message.role,
            "content": [
                {
                    "type": "text",
                    "text": message.content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": message.image_url
                    }
                }
            ]
        }
    elif message.type == "input_file":
        return {
            "role": message.role,
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "file",  # You might want to add filename to your model
                        "file_data": message.file_data
                    }
                },
                {
                    "type": "text",
                    "text": message.content
                }
            ]
        }
    else:
        raise ValueError(f"Unknown message type: {message.type}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message and maintain history if ident is provided.

    Args:
        request: The chat request containing the messages and optional parameters

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

        # Convert and add the new messages
        converted_messages = [convert_to_openai_format(msg) for msg in request.messages]
        messages.extend(converted_messages)

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
                # Add new messages to history
                chat_history[request.ident].extend([msg.dict() for msg in request.messages])

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