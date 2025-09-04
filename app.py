import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from main import graph  # Assuming your LangGraph code is saved as main.py

st.set_page_config(page_title="AI Assistant", page_icon="🤖")

st.title("🤖 LangGraph Chatbot with Tavily Search")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Add human message
    human_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(human_msg)

    st.chat_message("user").markdown(prompt)

    # Invoke graph
    result = graph.invoke(
    {"messages": st.session_state.messages},
    config={"configurable": {"thread_id": "default"}}
)


    ai_msg = result["messages"][-1]  # latest AI message
    st.session_state.messages.append(ai_msg)

    st.chat_message("assistant").markdown(ai_msg.content)
