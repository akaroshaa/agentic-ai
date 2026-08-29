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
    query: str
    priority: int
    resolution: str
    route: str

def classify_node(state: RouterState) -> dict:
    priority = state["priority"]
    if priority in (1,2):
        return {"route": "primary"}
    return {"route": "secondary"}

def urgent_query_bot(state: RouterState) -> dict:
    result = llm.invoke(
        [HumanMessage(content=f"Give quick resolution of the query as it is very urgent: {state['query']}")]
    )
    return {"resolution": result.content}

def non_urgent_query_bot(state: RouterState) -> dict:
    result = llm.invoke(
        [HumanMessage(content=f"Give resolution of the query but it is not urgent: {state['query']}")]
    )
    return {"resolution": result.content}

def pick_route(state: RouterState) -> Literal["primary", "secondary"]:
    return state["route"]  # must match keys in the mapping below


graph = StateGraph(RouterState)
graph.add_node("Classify Nodes", classify_node)
graph.add_node("Urgent Query Node", urgent_query_bot)
graph.add_node("Non-Urgent Query Node", non_urgent_query_bot)

graph.add_edge(START, "Classify Nodes")
graph.add_conditional_edges(
    "Classify Nodes",
    pick_route,
    {
        "primary": "Urgent Query Node", 
        "secondary": "Non-Urgent Query Node"
    })
graph.add_edge("Urgent Query Node", END)
graph.add_edge("Non-Urgent Query Node", END)

workflow = graph.compile()

# Save the visual graph to a PNG file
with open("13_graph.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

urgent_query_result = workflow.invoke({
     "query": "My phone is not working, I need it fixed immediately",
     "priority": 1,
     "resolution": "",
     "route": ""
     })
print("Urgent Query Result --->  ", urgent_query_result["resolution"])
print("-"*70)
non_urgent_query_result = workflow.invoke({
     "query": "Which iphone model is best for photography?",
     "priority": 5,
     "resolution": "",
     "route": ""
    })
print("Non-Urgent Result --->  ", non_urgent_query_result["resolution"])
