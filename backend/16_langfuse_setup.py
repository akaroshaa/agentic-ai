
from langchain_core.tools import tool
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
import os
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
import json
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        azure_deployment="sample-gpt-4o-deployment",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=0.5,
        top_p=1.0,
        max_tokens=4096,
    )

@tool
def fake_weather_tool(location: str) -> str:
    """A fake weather tool that returns a hardcoded weather report."""
    return f"The weather in {location} is sunny with a high of 25°C."

@tool
def add_two_numbers(a: int, b: int) -> int:
    """A simple tool that adds two numbers."""
    return a + b

tools = [fake_weather_tool, add_two_numbers]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    """A node that uses the LLM with tools to process messages."""

    print("=========  state['messages'] inside 'agent_node'  ================\n")
    for message in state["messages"]:
        print(type(message).__name__," ---> ",message)
        print()
    print("===============================================\n")

    result = llm_with_tools.invoke(state["messages"])
    
    # print(json.dumps(result.tool_calls, indent=2))

    print("-------------------------------")
    print("result = ", result.content)
    print("-------------------------------\n")

    return {"messages": [result]}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Determine if tools are required based on the messages."""
    # For simplicity, let's say if the last message contains "weather" or "add", we need tools
    messages = state.get("messages", [])

    print("=========  state['messages'] inside 'should_continue'  ================\n")
    for message in state["messages"]:
        print(type(message).__name__," ---> ",message)
        print()
    print("===============================================\n\n\n\n")

    if not messages:
        return "__end__"
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"

graph = StateGraph(MessagesState)
graph.add_node("Agent Node", agent_node)
graph.add_node("Tools Node", ToolNode(tools))

graph.add_edge(START, "Agent Node")
graph.add_conditional_edges(
                            "Agent Node", 
                            should_continue, 
                            {
                                "tools": "Tools Node",
                                "__end__": END
                            })
graph.add_edge("Tools Node", "Agent Node")

workflow = graph.compile()

# Save the visual graph to a PNG file
with open("16_graph.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

config = {
    "callbacks": [langfuse_handler],
    "metadata": {
        "user_id": "user_demo_001",
        "session_id": "session_001",
        "environment": "development",
    }
}

result = workflow.invoke(
    {
        "messages": [
            HumanMessage(
                # content="What is 41 + 9? Then tell me the weather in Paris (use tools)."
                # content="What is the capital of India? (use tools)"
                content="What is 41 + 9? Then tell me the weather in Paris."
                # content="What is the capital of India?"
            )
        ]
    }, config=config
)

print("\nFinal AI Response: ",result["messages"][-1].content)

print("\nLangfuse logs flushed successfully...")