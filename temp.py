from typing import Annotated, List
import os
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.constants import START, END


# Streamlit for UI
import streamlit as st

# ---------------------------
# Load keys
# ---------------------------
load_dotenv()

openai_api_key = os.getenv("openai_api_key")

# ---------------------------
# Setup LLM
# ---------------------------
llm = ChatOpenAI(temperature=0.7, api_key=openai_api_key)

# ---------------------------
# Define State
# ---------------------------
class State(TypedDict):
    messages: Annotated[list, BaseMessage]

graph_builder = StateGraph(State)
checkpointer = InMemorySaver()

# ---------------------------
# Define Tools
# ---------------------------
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

# Chatbot Node
def Chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph_builder.add_node("Chatbot", Chatbot)

# Tool Node
tool_node = ToolNode(tool)
graph_builder.add_node("TavilySearch", tool_node)

# Conditional edges (decide when to call tools)
graph_builder.add_conditional_edges("Chatbot", tools_condition)

# Loop back tool output to chatbot
graph_builder.add_edge("TavilySearch", "Chatbot")

# Start → Chatbot
graph_builder.add_edge(START, "Chatbot")

# Compile graph
graph = graph_builder.compile(checkpointer=checkpointer)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="LangGraph AI Agent", layout="centered")
st.title("🤖 LangGraph AI Agent with Tavily Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"]=="user" else "assistant"):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run graph
    response = graph.invoke({"messages": st.session_state.messages})
    bot_reply = response["messages"][-1].content

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
