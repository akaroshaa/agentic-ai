from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

load_dotenv()  # Load environment variables from .env file

def call_azure_openai_llm(user_input):

    llm = AzureChatOpenAI(
        azure_endpoint="https://crook-mt496dmp-eastus2.openai.azure.com/",
        azure_deployment="sample-gpt-4o-deployment",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=0.5,
        top_p=1.0,
        max_tokens=4096,
    )

    # Prepare messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_input),
    ]

    # Invoke the model
    response = llm.invoke(messages)

    return response.content