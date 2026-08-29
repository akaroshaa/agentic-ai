from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, END, StateGraph
import os
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        azure_deployment="sample-gpt-4o-deployment",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=0.5,
        top_p=1.0,
        max_tokens=4096,
    )

class GreetState(TypedDict):
    topic: str
    reply: str

def greet_node(state: GreetState) -> dict:
    """A node is just a function: input state → return partial state updates."""
    msg = llm.invoke(
        [
            SystemMessage(
                content="You are a friendly tutor. Reply in 2 short sentences."
            ),
            HumanMessage(content=f"Explain briefly: {state['topic']}"),
        ]
    )
    return {"reply": msg.content}

graph = StateGraph(GreetState)
graph.add_node(node = "greet the user", action = greet_node)
graph.add_edge(START, "greet the user")
graph.add_edge("greet the user", END)

workflow = graph.compile()

# Save the visual graph to a PNG file
with open("11_graph.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

result = workflow.invoke({
    "topic": "what is a LangGraph node",
    "reply": ""
    })
print("Graph State --->  ", result["reply"])
