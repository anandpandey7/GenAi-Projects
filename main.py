from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Define LLM
llm = ChatOpenAI(temperature=0.7, api_key=openai_api_key)

# ✅ State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Build graph
graph_builder = StateGraph(State)

# Tools
tool = TavilySearchResults(max_results=2, api_key=tavily_api_key)  # ✅ fix param name
tools = [tool]

llm_with_tools = llm.bind_tools(tools)

# Chatbot node
def Chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph_builder.add_node("Chatbot", Chatbot)

# ✅ ToolNode auto-creates nodes for each tool (no need to hardcode "TavilySearch")
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)  # <-- name must be "tools"

# Edges
graph_builder.add_conditional_edges("Chatbot", tools_condition)  # directs to "tools"
graph_builder.add_edge("tools", "Chatbot")  # back to chatbot
graph_builder.add_edge(START, "Chatbot")

# Compile
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
