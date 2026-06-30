import uuid
from typing import Annotated, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state

class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: dict
    metadata: dict

def chat_node(state: ChatState):
    messages = state["messages"]
    
    # We use a system prompt that includes the context of the hedge fund data
    context = ""
    if "analyst_signals" in state["data"]:
        context += "Analyst Signals: " + str(state["data"]["analyst_signals"]) + "\n"
    if "portfolio" in state["data"]:
        context += "Portfolio: " + str(state["data"]["portfolio"]) + "\n"
    if "decisions" in state["data"]:
        context += "Decisions: " + str(state["data"]["decisions"]) + "\n"

    system_message = SystemMessage(
        content=(
            "You are the AI Hedge Fund Manager. You have access to various analyst signals, "
            "portfolio data, and trading decisions. Your job is to answer the user's follow-up "
            "questions based on this context. Maintain a helpful and professional tone.\n\n"
            f"Context:\n{context}"
        )
    )

    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    provider = state["metadata"].get("model_provider", "OpenAI")
    model_name = state["metadata"].get("model_name", "gpt-4o")
    
    if provider == "OpenAI":
        llm = ChatOpenAI(model=model_name)
    elif provider == "Anthropic":
        llm = ChatAnthropic(model=model_name)
    elif provider == "Google":
        llm = ChatGoogleGenerativeAI(model=model_name)
    else:
        llm = ChatOpenAI(model=model_name) # default
        
    response = llm.invoke([system_message] + list(messages))
    return {"messages": [response]}

def create_chat_workflow():
    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

chat_agent = create_chat_workflow()
