# LangChain and LangGraph Integration with MCP Server
import os
import json
import requests
from typing import Dict, List, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor

# Configure API keys and endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3001")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY, temperature=0)

# 1. First, let's define our MCP tool functions using LangChain's tool decorator

@tool
def get_context(query: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """
    Get relevant context from the MCP server based on a natural language query.
    
    Args:
        query: The natural language query to process
        max_tokens: Maximum number of tokens to return
        
    Returns:
        Dictionary containing context matches
    """
    response = requests.post(
        f"{MCP_SERVER_URL}/api/context",
        json={"query": query, "max_tokens": max_tokens}
    )
    response.raise_for_status()
    return response.json()

@tool
def get_available_tools() -> Dict[str, Any]:
    """
    Get available tools from the MCP server.
    
    Returns:
        Dictionary of available tools and their definitions
    """
    response = requests.get(f"{MCP_SERVER_URL}/tools")
    response.raise_for_status()
    return response.json()

@tool
def execute_action(action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an action on the MCP server.
    
    Args:
        action: The name of the action to execute
        parameters: Parameters required for the action
        
    Returns:
        Result of the action
    """
    response = requests.post(
        f"{MCP_SERVER_URL}/api/action",
        json={"action": action, "parameters": parameters}
    )
    response.raise_for_status()
    return response.json()

@tool
def get_citation(citation_id: str) -> Dict[str, Any]:
    """
    Get citation information from the MCP server.
    
    Args:
        citation_id: ID of the citation to retrieve
        
    Returns:
        Citation information
    """
    response = requests.post(
        f"{MCP_SERVER_URL}/api/citation",
        json={"citation_id": citation_id}
    )
    response.raise_for_status()
    return response.json()

# 2. Define LangGraph nodes

# Initial understanding and tool retrieval
def retrieve_context_and_tools(state):
    """Retrieve context and available tools based on the user query"""
    user_input = state["messages"][-1].content
    
    # Get available tools
    tools_response = get_available_tools()
    
    # Get context based on the query
    context_response = get_context(user_input)
    
    return {
        "messages": state["messages"],
        "tools": tools_response,
        "context": context_response,
        "execution_needed": len(context_response.get("matches", [])) > 0
    }

# Define the AI planning node - decides which action to take
def plan_action(state):
    """LLM decides which action to take based on context and tools"""
    user_input = state["messages"][-1].content
    context = state["context"]
    tools = state["tools"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that processes user requests about users in a system.
        Based on the context and available tools, determine the best action to take.
        
        Available tools: {tools}
        
        Context from the query: {context}
        
        You need to decide:
        1. Which tool to use (if any)
        2. What parameters to pass to the tool
        
        Respond with a JSON object with the following structure:
        {{"action": "tool_name", "parameters": {{"param1": "value1", ...}}}}
        
        If no action is needed, respond with:
        {{"action": null, "parameters": null}}
        """),
        ("human", "{query}")
    ])
    
    # Format the tools in a readable format
    formatted_tools = json.dumps(tools, indent=2)
    formatted_context = json.dumps(context, indent=2)
    
    # Call the LLM to decide on an action
    response = llm.invoke(
        prompt.format(
            tools=formatted_tools,
            context=formatted_context,
            query=user_input
        )
    )
    
    # Parse the response to extract the action plan
    try:
        # Try to parse as JSON
        action_plan = json.loads(response.content)
    except json.JSONDecodeError:
        # If not valid JSON, try to extract JSON from the text
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            try:
                action_plan = json.loads(json_match.group(0))
            except:
                action_plan = {"action": None, "parameters": None}
        else:
            action_plan = {"action": None, "parameters": None}
    
    return {
        "messages": state["messages"],
        "tools": state["tools"],
        "context": state["context"],
        "action_plan": action_plan,
        "execution_needed": action_plan.get("action") is not None
    }

# Execute the planned action
def execute_planned_action(state):
    """Execute the action planned by the LLM"""
    action_plan = state["action_plan"]
    
    if not action_plan or not action_plan.get("action"):
        return {
            "messages": state["messages"],
            "action_result": {"result": "no_action", "message": "No action was needed"}
        }
    
    # Execute the action using our MCP tool
    action_result = execute_action(
        action=action_plan["action"],
        parameters=action_plan["parameters"]
    )
    
    return {
        "messages": state["messages"],
        "tools": state["tools"],
        "context": state["context"],
        "action_plan": action_plan,
        "action_result": action_result
    }

# Generate the final response
def generate_response(state):
    """Generate a final response to the user based on action results"""
    user_input = state["messages"][-1].content
    action_result = state.get("action_result", {})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that manages user information.
        Respond to the user in a natural, conversational way.
        
        The user asked: {query}
        
        The result of processing their request was: {result}
        
        Don't mention the specific technical details of how you processed the request,
        just provide a friendly, human response that addresses their needs.
        """),
        ("human", "{query}")
    ])
    
    # Format the result
    formatted_result = json.dumps(action_result, indent=2)
    
    # Call the LLM to generate a human-friendly response
    response = llm.invoke(
        prompt.format(
            query=user_input,
            result=formatted_result
        )
    )
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "action_result": action_result
    }

# 3. Build the LangGraph

# Define the graph
builder = lg.GraphBuilder()
builder.add_node("retrieve_context", retrieve_context_and_tools)
builder.add_node("plan_action", plan_action)
builder.add_node("execute_action", execute_planned_action)
builder.add_node("generate_response", generate_response)

# Define the edges
builder.add_edge("retrieve_context", "plan_action")
builder.add_conditional_edges(
    "plan_action",
    lambda state: state.get("execution_needed", False),
    {
        True: "execute_action",
        False: "generate_response"
    }
)
builder.add_edge("execute_action", "generate_response")

# Set the entry point
builder.set_entry_point("retrieve_context")

# Compile the graph
graph = builder.compile()

# 4. Function to process a user request through the graph
def process_user_request(user_input: str) -> str:
    """
    Process a natural language user request through the LangGraph.
    
    Args:
        user_input: Natural language user request like "add user John Doe with email john@gmail.com"
        
    Returns:
        Response from the assistant
    """
    # Initialize the state with the user's message
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # Execute the graph
    final_state = graph.invoke(initial_state)
    
    # Return the last message from the assistant
    return final_state["messages"][-1].content

# Example usage
if __name__ == "__main__":
    # Example 1: Create a user
    user_query = "Add user John Doe with email john@gmail.com"
    response = process_user_request(user_query)
    print(f"User: {user_query}")
    print(f"Assistant: {response}")
    print()
    
    # Example 2: Get all users
    user_query = "Show me all the users in the system"
    response = process_user_request(user_query)
    print(f"User: {user_query}")
    print(f"Assistant: {response}")
    print()
    
    # Example 3: Update a user
    user_query = "Update user 123 with new email john.doe@newemail.com"
    response = process_user_request(user_query)
    print(f"User: {user_query}")
    print(f"Assistant: {response}")
