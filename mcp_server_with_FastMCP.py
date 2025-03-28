# FastMCP Server Implementation
import os
import re
import json
import httpx
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from pydantic import BaseModel, Field
from datetime import datetime

# Import FastMCP components
from fastmcp import FastMCP, Tool, ToolParameter, ContextProvider, ActionHandler
from fastmcp.models import ContextMatch, ContextRequest, CitationRequest, ActionRequest

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

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    # Add other fields that your API returns

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

# Context provider implementation for user queries
class UserContextProvider(ContextProvider):
    async def get_context(self, request: ContextRequest) -> List[ContextMatch]:
        query = request.query.lower()
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
                    
            # Extract update user intent
            if "update user" in query or "change user" in query or "modify user" in query:
                # Extract user ID
                user_id_match = re.search(r"(?:update|change|modify) user (?:id )?(\w+)", query)
                
                # Extract email update
                email_match = re.search(r"(?:with |to |email )(\S+@\S+\.\S+)", query)
                
                # Extract name update
                name_match = re.search(r"(?:with |to |name )['\"](.*?)['\"]", query)
                if not name_match:
                    name_match = re.search(r"(?:with |to |name )([A-Za-z\s]+)(?:\s|$)", query)
                
                if user_id_match and (email_match or name_match):
                    user_id = user_id_match.group(1)
                    update_data = {"user_id": user_id}
                    
                    if email_match:
                        update_data["email"] = email_match.group(1)
                    if name_match:
                        update_data["name"] = name_match.group(1).strip()
                    
                    results.append(
                        ContextMatch(
                            type="user_update_intent",
                            content=json.dumps(update_data),
                            relevance_score=0.96,
                            metadata={"intent": "update_user", "extracted_data": update_data}
                        )
                    )
            
            # Extract delete user intent
            if "delete user" in query or "remove user" in query:
                user_id_match = re.search(r"(?:delete|remove) user (?:id )?(\w+)", query)
                
                if user_id_match:
                    user_id = user_id_match.group(1)
                    
                    results.append(
                        ContextMatch(
                            type="user_deletion_intent",
                            content=json.dumps({"user_id": user_id}),
                            relevance_score=0.95,
                            metadata={"intent": "delete_user", "user_id": user_id}
                        )
                    )
                    
        return results

# Citation handler
async def get_citation(request: CitationRequest):
    citation_id = request.citation_id
    
    if not citation_id:
        raise ValueError("Citation ID is required")
    
    user_response = await call_fastapi("get", f"/{citation_id}")
    
    if user_response["success"]:
        return {
            "type": "user",
            "content": user_response["data"],
            "metadata": {
                "source": "User Database",
                "user_id": citation_id,
                "retrieved_at": datetime.now().isoformat()
            }
        }
    else:
        raise ValueError(f"User not found: {user_response.get('error', 'Unknown error')}")

# Action handlers
class CreateUserAction(ActionHandler):
    async def execute(self, request: ActionRequest):
        if not request.parameters or "user_data" not in request.parameters:
            raise ValueError("User data is required")
        
        user_data = request.parameters["user_data"]
        create_response = await call_fastapi("post", "/", user_data)
        
        if create_response["success"]:
            return {
                "result": "success",
                "data": create_response["data"],
                "message": "User created successfully"
            }
        else:
            raise ValueError(f"Failed to create user: {create_response.get('error', 'Unknown error')}")

class GetAllUsersAction(ActionHandler):
    async def execute(self, request: ActionRequest):
        users_response = await call_fastapi("get", "/")
        
        if users_response["success"]:
            return {
                "result": "success",
                "data": users_response["data"],
                "message": "Retrieved all users successfully"
            }
        else:
            raise ValueError(f"Failed to retrieve users: {users_response.get('error', 'Unknown error')}")

class GetUserAction(ActionHandler):
    async def execute(self, request: ActionRequest):
        if not request.parameters or "user_id" not in request.parameters:
            raise ValueError("User ID is required")
        
        user_id = request.parameters["user_id"]
        user_response = await call_fastapi("get", f"/{user_id}")
        
        if user_response["success"]:
            return {
                "result": "success",
                "data": user_response["data"],
                "message": f"Retrieved user {user_id} successfully"
            }
        else:
            raise ValueError(f"User {user_id} not found: {user_response.get('error', 'Unknown error')}")

class UpdateUserAction(ActionHandler):
    async def execute(self, request: ActionRequest):
        if not request.parameters or "user_id" not in request.parameters:
            raise ValueError("User ID is required")
        if "user_data" not in request.parameters:
            raise ValueError("User data is required for update")
        
        user_id = request.parameters["user_id"]
        user_data = request.parameters["user_data"]
        update_response = await call_fastapi("put", f"/{user_id}", user_data)
        
        if update_response["success"]:
            return {
                "result": "success",
                "data": update_response["data"],
                "message": f"Updated user {user_id} successfully"
            }
        else:
            raise ValueError(f"Failed to update user {user_id}: {update_response.get('error', 'Unknown error')}")

class DeleteUserAction(ActionHandler):
    async def execute(self, request: ActionRequest):
        if not request.parameters or "user_id" not in request.parameters:
            raise ValueError("User ID is required")
        
        user_id = request.parameters["user_id"]
        delete_response = await call_fastapi("delete", f"/{user_id}")
        
        if delete_response["success"]:
            return {
                "result": "success",
                "message": f"Deleted user {user_id} successfully"
            }
        else:
            raise ValueError(f"Failed to delete user {user_id}: {delete_response.get('error', 'Unknown error')}")

# Define tools
create_user_tool = Tool(
    name="create_user",
    description="Creates a new user in the system",
    parameters=[
        ToolParameter(
            name="user_data",
            description="User data including name and email",
            required=True,
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's full name"},
                    "email": {"type": "string", "description": "User's email address"}
                },
                "required": ["name", "email"]
            }
        )
    ],
    handler=CreateUserAction()
)

get_all_users_tool = Tool(
    name="get_all_users",
    description="Retrieves a list of all users",
    parameters=[],
    handler=GetAllUsersAction()
)

get_user_tool = Tool(
    name="get_user",
    description="Retrieves information about a specific user",
    parameters=[
        ToolParameter(
            name="user_id",
            description="ID of the user to retrieve",
            required=True,
            schema={"type": "string"}
        )
    ],
    handler=GetUserAction()
)

update_user_tool = Tool(
    name="update_user",
    description="Updates information for an existing user",
    parameters=[
        ToolParameter(
            name="user_id",
            description="ID of the user to update",
            required=True,
            schema={"type": "string"}
        ),
        ToolParameter(
            name="user_data",
            description="User data to update",
            required=True,
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's full name"},
                    "email": {"type": "string", "description": "User's email address"}
                }
            }
        )
    ],
    handler=UpdateUserAction()
)

delete_user_tool = Tool(
    name="delete_user",
    description="Deletes a user from the system",
    parameters=[
        ToolParameter(
            name="user_id",
            description="ID of the user to delete",
            required=True,
            schema={"type": "string"}
        )
    ],
    handler=DeleteUserAction()
)

# Initialize FastMCP
app = FastMCP(
    title="User Management MCP",
    description="MCP server for user management operations",
    context_providers=[UserContextProvider()],
    citation_handler=get_citation,
    tools=[
        create_user_tool,
        get_all_users_tool,
        get_user_tool,
        update_user_tool,
        delete_user_tool
    ]
)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "fastapi_connection": FASTAPI_BASE_URL}

# Entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3001))
    print(f"Starting FastMCP Server on port {port}")
    print(f"Connected to FastAPI at {FASTAPI_BASE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Example natural language queries:

1. "Add user John Doe with email john@gmail.com"
2. "Show me all users"
3. "Get information about user 123"
4. "Update user 456 with new email jane@example.com"
5. "Delete user 789"

This FastMCP server provides the MCP protocol interface for an LLM to process
these natural language queries and perform the corresponding actions on your
FastAPI user management application.
"""
