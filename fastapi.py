from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
import time
import uuid

app = FastAPI()

# --- Authentication Dependency (Simple API Key) ---
# In a real application, this would be more robust
API_KEYS = {
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx": "user1",
    "sk-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy": "user2",
}

async def get_api_key(request: Request):
    api_key = request.headers.get("Authorization")
    if not api_key:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    if not api_key.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must start with 'Bearer '")
    api_key = api_key.split("Bearer ")[1]
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[api_key]

# --- Data Models (Pydantic) ---
# These models should align with the OpenAI API schema

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None  # Optional name for the author of the message.

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None  # A unique identifier representing your end-user

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[dict] = None  # Add usage if you track tokens

# --- Error Handling ---
#  Define custom exception for specific error scenarios
class ModelNotFoundError(HTTPException):
    def __init__(self, model_name: str):
        super().__init__(status_code=400, detail=f"Model '{model_name}' not found")

# --- Mock LLM ---
#  Replace this with your actual LLM integration (e.g., calling another API,
#  loading a local model, etc.)
def mock_llm_response(messages: List[ChatMessage], model_name: str):
    """
    Generates a mock response based on the input messages.  This is where you'd
    integrate with your actual language model.

    Args:
        messages: The list of messages in the chat.
        model_name: The name of the model to use (for informational purposes).

    Returns:
        A mock ChatCompletionResponse.
    """
    if model_name not in ["gpt-3.5-turbo", "gpt-4"]:
        raise ModelNotFoundError(model_name)

    # Basic logic: Echo the user's last message as the assistant, with some modifications
    last_user_message = None
    for message in reversed(messages):
        if message.role == "user":
            last_user_message = message.content
            break

    if last_user_message is None:
        response_text = "I am here to help."
    else:
        response_text = f"Echoing: {last_user_message}"

    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}, # Mock usage
    )

# --- API Endpoints ---

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key)  # Require API key for this endpoint
) -> ChatCompletionResponse:
    """
    Creates a new chat completion for the provided messages.  This endpoint
    attempts to mirror the OpenAI API's /v1/chat/completions endpoint.
    """
    try:
        return mock_llm_response(request.messages, request.model)
    except ModelNotFoundError as e:
        raise e  # Re-raise the custom exception
    except Exception as e:
        # Log the error for debugging
        print(f"Error in /v1/chat/completions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Optional:  Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


curl -X POST \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'


