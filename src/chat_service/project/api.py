"""Project-level API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any
import json

from ..clients.openrouter import OpenRouterApiClient, OpenRouterModel

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
    tool_calls: list[ToolCall] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)


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
        client = OpenRouterApiClient(model=OpenRouterModel.GPT_41)

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
        response = await client.agent(
            messages=messages,
            max_iterations=request.max_iterations,
            call_tools=request.use_tools,
        )

        # Extract tool calls from the response
        tool_calls = []
        if "tool_calls" in response:
            for tool_call in response["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call["function"]["arguments"]),
                        result=tool_call.get("result"),
                    )
                )

        # Update history if ident is provided
        if request.ident:
            # Add user message to history
            chat_history[request.ident].append(
                {"role": "user", "content": request.prompt}
            )

            # Add assistant response to history
            chat_history[request.ident].append(
                {"role": "assistant", "content": response["content"]}
            )

            # Keep only the last 5 messages
            chat_history[request.ident] = chat_history[request.ident][-5:]

            return ChatResponse(
                response=response["content"],
                tool_calls=tool_calls,
                history=chat_history[request.ident],
            )
        else:
            return ChatResponse(response=response["content"], tool_calls=tool_calls)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")
