"""Project-level API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any, Union
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
    image_url: Optional[str] = Field(None, description="The base64 encoded data of the image (data:image/jpeg;base64,{base64_image}) or the url to image if the type is input_image")


class StructuredMessage(BaseModel):
    """Model for structured messages with different types."""
    type: str = Field(..., description="The type of message: text, image, file, or code")
    text: Optional[str] = Field(None, description="Text content for text messages or captions for image messages")
    image_url: Optional[str] = Field(None, description="URL or base64 data for image messages")
    file_name: Optional[str] = Field(None, description="Name of the file for file messages")
    file_size: Optional[int] = Field(None, description="Size of the file in bytes for file messages")
    file_data: Optional[str] = Field(None, description="Base64 encoded file data for file messages")
    language: Optional[str] = Field(None, description="Programming language for code messages")
    sender: Optional[str] = Field("user", description="Sender identifier")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


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


class StructuredChatRequest(BaseModel):
    """Structured chat request model."""
    messages: list[StructuredMessage] = Field(..., description="List of structured messages")
    use_tools: bool = Field(True, description="Whether to enable tool usage in the chat")
    max_iterations: int = Field(5, description="Maximum number of iterations for the agent to use", ge=1, le=10)


class StructuredChatResponse(BaseModel):
    """Structured chat response model."""
    messages: list[StructuredMessage] = Field(..., description="List of response messages")


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    history: list[dict[str, str]] = Field(default_factory=list)


class Answer(BaseModel):
    """Structured chat response model with parsed content."""

    content: str
    context: str


class ParsedChatResponse(BaseModel):
    """Parsed chat response model with analyzed content."""

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


def convert_structured_message_to_openai(message: StructuredMessage) -> dict[str, Any]:
    """Convert a StructuredMessage to OpenAI format.
    
    Args:
        message: The StructuredMessage to convert
        
    Returns:
        dict: Message in OpenAI format
    """
    if message.type == "text":
        return {
            "role": "user",
            "content": message.text or ""
        }
    elif message.type == "image":
        content = []
        if message.text:
            content.append({"type": "text", "text": message.text})
        if message.image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": message.image_url}
            })
        return {
            "role": "user",
            "content": content if content else message.text or ""
        }
    elif message.type == "file":
        content = []
        if message.text:
            content.append({"type": "text", "text": message.text})
        if message.file_data:
            content.append({
                "type": "file",
                "file": {
                    "filename": message.file_name or "file",
                    "file_data": message.file_data
                }
            })
        return {
            "role": "user",
            "content": content if content else f"File: {message.file_name or 'unknown'}"
        }
    elif message.type == "code":
        code_content = f"```{message.language or 'text'}\n{message.text or ''}\n```"
        return {
            "role": "user",
            "content": code_content
        }
    else:
        return {
            "role": "user",
            "content": message.text or ""
        }


def generate_structured_response(messages: list[StructuredMessage]) -> list[StructuredMessage]:
    """Generate appropriate responses based on input message types.
    
    Args:
        messages: List of input messages
        
    Returns:
        List of response messages
    """
    responses = []
    
    for message in messages:
        if message.type == "image":
            response_text = "Thanks for sharing this image"
            if message.text:
                response_text += f' with caption: "{message.text}"'
            responses.append(StructuredMessage(
                type="text",
                text=response_text,
                sender="assistant"
            ))
            
        elif message.type == "file":
            file_size_mb = message.file_size / (1024 * 1024) if message.file_size else 0
            size_text = f"({file_size_mb:.1f} MB)" if file_size_mb > 0 else ""
            responses.append(StructuredMessage(
                type="text",
                text=f"I received your file: {message.file_name or 'unknown'} {size_text}",
                sender="assistant"
            ))
            
        elif message.type == "code":
            lang = message.language or "code"
            responses.append(StructuredMessage(
                type="text",
                text=f"Thanks for sharing this {lang} snippet! I'll take a look at it.",
                sender="assistant"
            ))
            
            # Special handling for JavaScript code
            if message.language in ["javascript", "js"]:
                responses.append(StructuredMessage(
                    type="text",
                    text="I noticed you're using JavaScript. Here's an example of an arrow function version:",
                    sender="assistant"
                ))
                responses.append(StructuredMessage(
                    type="code",
                    text="""// Arrow function version
const factorial = n => n <= 1 ? 1 : n * factorial(n - 1);

// Calculate factorial of 5
const result = factorial(5);
console.log(`Factorial of 5 is ${result}`);""",
                    language="javascript",
                    sender="assistant"
                ))
                
        else:  # text or unknown type
            responses.append(StructuredMessage(
                type="text",
                text=f"This is a response to your message: {message.text or ''}",
                sender="assistant"
            ))
    
    return responses


@router.post("/structured-chat", response_model=StructuredChatResponse)
async def structured_chat(request: StructuredChatRequest) -> StructuredChatResponse:
    """Process multiple structured messages without maintaining history.

    Args:
        request: The structured chat request containing multiple messages

    Returns:
        StructuredChatResponse containing multiple response messages
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

        # Convert structured messages to OpenAI format
        openai_messages = [convert_structured_message_to_openai(msg) for msg in request.messages]

        # Add a system message to provide context about the structured response
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that responds to various types of messages including text, images, files, and code. Provide appropriate responses based on the message type and content."
        }
        openai_messages.insert(0, system_message)

        try:
            logger.info(f"Processing structured chat with {len(request.messages)} messages")
            
            # Process the chat using OpenRouter
            response = client.chat(
                messages=openai_messages,
                model=OpenRouterModel.GPT41
            )

            if not response or not response.choices:
                logger.error("Received empty response from client.chat")
                # Fallback to structured response generation
                response_messages = generate_structured_response(request.messages)
                return StructuredChatResponse(messages=response_messages)

            ai_response = response.choices[0].message.content
            if not ai_response:
                logger.error("Response missing message content")
                # Fallback to structured response generation
                response_messages = generate_structured_response(request.messages)
                return StructuredChatResponse(messages=response_messages)

            # Create structured response from AI response
            response_messages = [StructuredMessage(
                type="text",
                text=ai_response,
                sender="assistant"
            )]

            return StructuredChatResponse(messages=response_messages)

        except Exception as e:
            logger.error(f"Error processing structured chat: {str(e)}", exc_info=True)
            # Fallback to structured response generation
            response_messages = generate_structured_response(request.messages)
            return StructuredChatResponse(messages=response_messages)

    except Exception as e:
        logger.error(f"Failed to process structured chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process structured chat: {str(e)}")