# Complete Python MCP Server with Tool Definitions
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional, Union
import httpx
import re
import os
import json
import math
import uvicorn
from datetime import datetime

# Create FastAPI app for the MCP server
app = FastAPI(title="MCP Server", description="Model Context Protocol server for FastAPI user management")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

# Models for user data
class UserCreate(BaseModel):
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    # Add any other fields your API expects

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, description="User's full name")
    email: Optional[str] = Field(None, description="User's email address")
    # Add any other fields your API allows updating

# Models for MCP requests and responses
class ContextRequest(BaseModel):
    query: str = Field(..., description="Natural language query about users")
    max_tokens: Optional[int] = Field(1000, description="Maximum number of tokens to return")

class ContextMatch(BaseModel):
    type: str
    content: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None

class ContextResponse(BaseModel):
    matches: List[ContextMatch]
    total_matches: int

class CitationRequest(BaseModel):
    citation_id: str = Field(..., description="ID of the citation to retrieve")

class CitationResponse(BaseModel):
    type: str
    content: Any
    metadata: Dict[str, Any]

class ActionRequest(BaseModel):
    action: str = Field(..., description="Action to perform")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the action")

class ActionResponse(BaseModel):
    result: str
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]

class ToolsResponse(BaseModel):
    tools: List[ToolDefinition]

# Helper function to call FastAPI endpoints
async def call_fastapi(method: str, endpoint: str, data: Dict[str, Any] = None):
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    async with httpx.AsyncClient() as client:
        try:
            if method.lower() == "get":
                response = await client.get(url)
            elif method.lower() == "post":
                response = await client.post(url, json=data)
            elif method.lower() == "put":
                response = await client.put(url, json=data)
            elif method.lower() == "delete":
                response = await client.delete(url)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}
            
            response.raise_for_status()
            return {"success": True, "data": response.json(), "status": response.status_code}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP error: {e}",
                "status": e.response.status_code,
                "data": e.response.text
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# MCP Tools Endpoint - Provides tool definitions for LLMs
@app.get("/tools", response_model=ToolsResponse)
async def get_tools():
    tools = [
        ToolDefinition(
            name="create_user",
            description="Creates a new user in the system",
            parameters={
                "type": "object",
                "properties": {
                    "user_data": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "User's full name"},
                            "email": {"type": "string", "description": "User's email address"}
                        },
                        "required": ["name", "email"]
                    }
                },
                "required": ["user_data"]
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "data": {"type": "object", "description": "Created user data"},
                    "message": {"type": "string", "description": "Success message"}
                }
            }
        ),
        ToolDefinition(
            name="get_all_users",
            description="Retrieves a list of all users",
            parameters={
                "type": "object",
                "properties": {}
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "data": {"type": "array", "items": {"type": "object"}, "description": "List of user objects"},
                    "message": {"type": "string", "description": "Success message"}
                }
            }
        ),
        ToolDefinition(
            name="get_user",
            description="Retrieves information about a specific user",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID of the user to retrieve"}
                },
                "required": ["user_id"]
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "data": {"type": "object", "description": "User data"},
                    "message": {"type": "string", "description": "Success message"}
                }
            }
        ),
        ToolDefinition(
            name="update_user",
            description="Updates information for an existing user",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID of the user to update"},
                    "user_data": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "User's full name"},
                            "email": {"type": "string", "description": "User's email address"}
                        }
                    }
                },
                "required": ["user_id", "user_data"]
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "data": {"type": "object", "description": "Updated user data"},
                    "message": {"type": "string", "description": "Success message"}
                }
            }
        ),
        ToolDefinition(
            name="delete_user",
            description="Deletes a user from the system",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID of the user to delete"}
                },
                "required": ["user_id"]
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "data": {"type": "null"},
                    "message": {"type": "string", "description": "Success message"}
                }
            }
        )
    ]
    
    return ToolsResponse(tools=tools)

# MCP Context Endpoint
@app.post("/api/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    query = request.query.lower()
    max_tokens = request.max_tokens
    results = []
    
    # Process natural language queries about users
    if "user" in query or "users" in query:
        # Get all users
        if "all users" in query or "users list" in query or "list all users" in query:
            users_response = await call_fastapi("get", "/")
            if users_response["success"]:
                results.append(
                    ContextMatch(
                        type="users_list",
                        content=json.dumps(users_response["data"]),
                        relevance_score=0.95
                    )
                )
        
        # Get specific user by ID
        user_id_match = re.search(r"user (?:id|ID)?\s*[:#]?\s*(\w+)", query) or re.search(r"user\s+(\w+)", query)
        if user_id_match:
            user_id = user_id_match.group(1)
            user_response = await call_fastapi("get", f"/{user_id}")
            if user_response["success"]:
                results.append(
                    ContextMatch(
                        type="specific_user",
                        content=json.dumps(user_response["data"]),
                        relevance_score=0.98,
                        metadata={"user_id": user_id}
                    )
                )
        
        # Extract user creation intent
        if "add user" in query or "create user" in query or "new user" in query:
            # Extract potential name
            name_match = re.search(r"(?:named|name|called)\s+([A-Za-z\s]+)(?:\s|$|with)", query)
            if not name_match:
                name_match = re.search(r"(?:add|create|new) user\s+([A-Za-z\s]+)(?:\s|$|with)", query)
            
            # Extract potential email
            email_match = re.search(r"email\s+(\S+@\S+\.\S+)", query)
            
            if name_match or email_match:
                context_data = {}
                if name_match:
                    context_data["name"] = name_match.group(1).strip()
                if email_match:
                    context_data["email"] = email_match.group(1)
                
                results.append(
                    ContextMatch(
                        type="user_creation_intent",
                        content=json.dumps(context_data),
                        relevance_score=0.97,
                        metadata={"intent": "create_user", "extracted_data": context_data}
                    )
                )
    
    # Sort by relevance
    results.sort(key=lambda x: x.relevance_score, reverse=True)
    
    # Apply token limit
    total_tokens = 0
    filtered_results = []
    
    for result in results:
        # Simple token estimation (characters / 4)
        estimated_tokens = math.ceil(len(result.content) / 4)
        if total_tokens + estimated_tokens <= max_tokens:
            filtered_results.append(result)
            total_tokens += estimated_tokens
        else:
            break
    
    return ContextResponse(
        matches=filtered_results,
        total_matches=len(results)
    )

# MCP Citation Endpoint
@app.post("/api/citation", response_model=CitationResponse)
async def get_citation(request: CitationRequest):
    citation_id = request.citation_id
    
    if not citation_id:
        raise HTTPException(status_code=400, detail="Citation ID is required")
    
    user_response = await call_fastapi("get", f"/{citation_id}")
    
    if user_response["success"]:
        return CitationResponse(
            type="user",
            content=user_response["data"],
            metadata={
                "source": "User Database",
                "user_id": citation_id,
                "retrieved_at": datetime.now().isoformat()
            }
        )
    else:
        raise HTTPException(
            status_code=user_response.get("status", 404),
            detail=f"User not found: {user_response.get('error', 'Unknown error')}"
        )

# MCP Action Endpoint
@app.post("/api/action", response_model=ActionResponse)
async def perform_action(request: ActionRequest):
    action = request.action
    parameters = request.parameters or {}
    
    if action == "create_user":
        if not parameters.get("user_data"):
            raise HTTPException(status_code=400, detail="User data is required")
        
        create_response = await call_fastapi("post", "/", parameters["user_data"])
        
        if create_response["success"]:
            return ActionResponse(
                result="success",
                data=create_response["data"],
                message="User created successfully"
            )
        else:
            raise HTTPException(
                status_code=create_response.get("status", 400),
                detail=f"Failed to create user: {create_response.get('error', 'Unknown error')}"
            )
    
    elif action == "get_all_users":
        users_response = await call_fastapi("get", "/")
        
        if users_response["success"]:
            return ActionResponse(
                result="success",
                data=users_response["data"],
                message="Retrieved all users successfully"
            )
        else:
            raise HTTPException(
                status_code=users_response.get("status", 400),
                detail=f"Failed to retrieve users: {users_response.get('error', 'Unknown error')}"
            )
    
    elif action == "get_user":
        if not parameters.get("user_id"):
            raise HTTPException(status_code=400, detail="User ID is required")
        
        user_id = parameters["user_id"]
        user_response = await call_fastapi("get", f"/{user_id}")
        
        if user_response["success"]:
            return ActionResponse(
                result="success",
                data=user_response["data"],
                message=f"Retrieved user {user_id} successfully"
            )
        else:
            raise HTTPException(
                status_code=user_response.get("status", 404),
                detail=f"User {user_id} not found: {user_response.get('error', 'Unknown error')}"
            )
    
    elif action == "update_user":
        if not parameters.get("user_id"):
            raise HTTPException(status_code=400, detail="User ID is required")
        if not parameters.get("user_data"):
            raise HTTPException(status_code=400, detail="User data is required for update")
        
        user_id = parameters["user_id"]
        update_response = await call_fastapi("put", f"/{user_id}", parameters["user_data"])
        
        if update_response["success"]:
            return ActionResponse(
                result="success",
                data=update_response["data"],
                message=f"Updated user {user_id} successfully"
            )
        else:
            raise HTTPException(
                status_code=update_response.get("status", 404),
                detail=f"Failed to update user {user_id}: {update_response.get('error', 'Unknown error')}"
            )
    
    elif action == "delete_user":
        if not parameters.get("user_id"):
            raise HTTPException(status_code=400, detail="User ID is required")
        
        user_id = parameters["user_id"]
        delete_response = await call_fastapi("delete", f"/{user_id}")
        
        if delete_response["success"]:
            return ActionResponse(
                result="success",
                message=f"Deleted user {user_id} successfully"
            )
        else:
            raise HTTPException(
                status_code=delete_response.get("status", 404),
                detail=f"Failed to delete user {user_id}: {delete_response.get('error', 'Unknown error')}"
            )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "fastapi_connection": FASTAPI_BASE_URL}

# Entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 3001))
    print(f"Starting MCP Server on port {port}")
    print(f"Connected to FastAPI at {FASTAPI_BASE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Example natural language queries:

1. "Add user John Doe with email john@gmail.com"
   - MCP will extract the name and email
   - LLM can use this to call the create_user action

2. "Show me all users"
   - MCP will understand this as a request for all users
   - LLM can use get_all_users action

3. "Get information about user 123"
   - MCP will extract the user ID
   - LLM can use get_user action

4. "Update user 456 with new email jane@example.com"
   - MCP will extract the user ID and email
   - LLM can use update_user action

5. "Delete user 789"
   - MCP will extract the user ID
   - LLM can use delete_user action
"""
