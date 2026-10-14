from typing import TypedDict, NotRequired

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load environment variables
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(
    model="google/gemma-4-31b-it",
    nvidia_api_key=NVIDIA_API_KEY,
    timeout=120,
)


# Define the graph state
class State(TypedDict):
    topic: str
    info: str
    response: NotRequired[AIMessage]


# Choose a topic
def choose_topic(state: State):
    return {"topic": "front end development"}


# Define the research tool
@tool
def get_topic_info(topic: str) -> str:
    """Get information about a given topic."""

    if topic.lower() == "ai agents":
        return (
            "AI Agents are autonomous entities that perceive "
            "their environment and take actions to achieve "
            "specific goals."
        )

    elif topic.lower() == "machine learning":
        return (
            "Machine Learning is a subset of AI that focuses "
            "on algorithms that learn from data and make "
            "predictions."
        )

    elif topic.lower() == "natural language processing":
        return (
            "Natural Language Processing (NLP) focuses on "
            "interactions between computers and humans "
            "through natural language."
        )

    else:
        return "Topic information not available."


# Bind the tool to the LLM
llm_with_tools = llm.bind_tools([get_topic_info])


# Ask the LLM whether to use the tool
def llm_node(state: State):
    prompt = (
        f"Find information about {state['topic']} "
        "using your available tool. "
        "If the tool is appropriate, call it."
    )

    response = llm_with_tools.invoke(prompt)

    return {"response": response}


# Decide where the graph goes next
def route_after_llm(state: State):
    if state["response"].tool_calls:
        return "research"
    else:
        return "tool_not_used"


def no_tool_final_answer_node(state: State):
    response = state["response"]

    messages = [
        HumanMessage(
            content=(
                f"Answer the user's request about {state['topic']} "
                "using your existing knowledge. "
                "The research tool was not used."
            )
        ),
        response,
    ]

    final_response = llm.invoke(messages)

    return {"response": final_response}


# Execute the tool call requested by the LLM
def research_node(state: State):
    response = state["response"]
    tool_call = response.tool_calls[0]

    if tool_call["name"] != get_topic_info.name:
        raise ValueError(f"Unexpected tool: {tool_call['name']}")

    result = get_topic_info.invoke(tool_call["args"])

    return {"info": result}


# Generate a final answer using the tool result
def final_answer_node(state: State):
    response = state["response"]
    tool_call = response.tool_calls[0]

    messages = [
        HumanMessage(
            content=(
                f"Find information about {state['topic']} " "using your available tool."
            )
        ),
        response,
        ToolMessage(
            content=state["info"],
            tool_call_id=tool_call["id"],
        ),
    ]

    final_response = llm_with_tools.invoke(messages)

    return {"response": final_response}


# Build the graph
builder = StateGraph(State)

builder.add_node("choose_topic", choose_topic)
builder.add_node("llm_node", llm_node)
builder.add_node("research", research_node)

builder.add_node(
    "no_tool_final_answer",
    no_tool_final_answer_node,
)
builder.add_node("final_answer", final_answer_node)


builder.add_edge(START, "choose_topic")
builder.add_edge("choose_topic", "llm_node")

builder.add_conditional_edges(
    "llm_node",
    route_after_llm,
    {
        "research": "research",
        "tool_not_used": "no_tool_final_answer",
    },
)

builder.add_edge("research", "final_answer")
builder.add_edge("final_answer", END)
builder.add_edge("no_tool_final_answer", END)

graph = builder.compile()


# Run the graph
results = graph.invoke(
    {
        "topic": "",
        "info": "",
    }
)

print("Topic:", results["topic"])
print("Research:", results["info"])
print("Final answer:", results["response"].content)
