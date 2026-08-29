from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
import os
import asyncio
from dotenv import load_dotenv
import sys

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

mcp_client = MultiServerMCPClient({
                "hotels": {

                        # # ======  if MCP server is on the same machine only ======
                        # "transport": "stdio",
                        # "command": sys.executable,
                        # "args": [os.path.abspath(
                        #         os.path.join(os.path.dirname(__file__), "9_MCP_server_setup.py")
                        #         )],


                        # ======  if MCP server is on a remote machine ======
                        "transport": "streamable-http",
                        "url": os.environ.get("MCP_URL")
                    }
                })


async def main():
    # connect to the remote MCP server and load its tools
    tools = await mcp_client.get_tools()
    print("==========  Loaded MCP tools ===========\n\n")
    for tool in tools:
        print(f"Tool name: {tool.name}\n\nDescription: {tool.description}")
        print("="*50)
        print("\n")
    hotel_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant."
    )

    question = """
                which hotel in da nang has the best reviews and why?
                give the average rating for each.
                """
    result = await hotel_agent.ainvoke({
                    "messages": [{"role": "user", "content": question}]
    })
    print("Agent response:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())