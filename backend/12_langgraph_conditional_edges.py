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

class RouterState(TypedDict):
    question: str
    answer: str
    route: str

def classify_node(state: RouterState) -> dict:
    text = state["question"].lower()
    if any(w in text for w in ("+", "plus", "times", "multiply", "what is", "calculate")):
        return {"route": "math"}
    return {"route": "chitchat"}

def math_bot(state: RouterState) -> dict:
    result = llm.invoke(
        [HumanMessage(content=f"Solve or explain briefly (show reasoning): {state['question']}")]
    )
    return {"answer": result.content}

def chitchat_bot(state: RouterState) -> dict:
    result = llm.invoke(
        [HumanMessage(content=f"Reply helpfully and briefly: {state['question']}")]
    )
    return {"answer": result.content}

def pick_route(state: RouterState) -> Literal["math", "chitchat"]:
    return state["route"]  # must match keys in the mapping below


graph = StateGraph(RouterState)
graph.add_node("Classify Nodes", classify_node)
graph.add_node("Math Operation Node", math_bot)
graph.add_node("Chit-Chat Node", chitchat_bot)

graph.add_edge(START, "Classify Nodes")
graph.add_conditional_edges(
    "Classify Nodes",
    pick_route,
    {
        "math": "Math Operation Node", 
        "chitchat": "Chit-Chat Node"
    })
graph.add_edge("Math Operation Node", END)
graph.add_edge("Chit-Chat Node", END)

workflow = graph.compile()

# Save the visual graph to a PNG file
with open("12_graph.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

math_result = workflow.invoke({
    "question": "What is 12 * 7?",
    "route": "", "answer": ""
    })
print("Math Result --->  ", math_result["answer"])
print("-"*70)
chitchat_result = workflow.invoke({
    "question": "Say hi in one sentence",
    "route": "", "answer": ""
    })
print("Chit-Chat Result --->  ", chitchat_result["answer"])
