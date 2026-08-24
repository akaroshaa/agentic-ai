import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# RERANKER 1: FlashRank
# ============================================================

from flashrank import Ranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# ============================================================

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

# Disable the faulty deallocator
AzureSearch.__del__ = lambda self: None

load_dotenv()

# pdf_path = "C:/Users/HP/Downloads/Comm626.pdf" 

# # 1. Load the PDF and create Document objects
# reader = pypdf.PdfReader(pdf_path)
# print(f"Loaded PDF from: {pdf_path}")
# docs = [
#         Document(
#             page_content=page.extract_text() or "",
#             metadata={"source": pdf_path, "page": i},
#         )
#         for i, page in enumerate(reader.pages)
#     ]
# print(f"Document has {len(docs)} pages")

# # 2. Splitting the document into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(docs)
# print(f"Number of text splits: {len(all_splits)}")


# 3. Setup Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="sample-text-embedding-3-small",
    api_version="2024-12-01-preview",
)

# 4. Initialize Azure AI Search
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
    index_name="sample-index",
    embedding_function=embeddings.embed_query,
)

# 5. Index chunks
# ids = vector_store.add_documents(documents=all_splits)
# print(f"Indexed {len(ids)} document chunks to Azure AI Search.")


# 6. Query the Azure AI Search index
query = "What is the key takeaway from the document?"

base_retriever = vector_store.as_retriever(
    search_type="similarity", k=6
    )

base_compressor = FlashrankRerank(top_n=3)

flashrank_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=base_compressor
)

# reranked_results = flashrank_retriever.invoke(query)
# for i, doc in enumerate(reranked_results):
#     score = doc.metadata.get("relevance_score", "N/A")
#     print(f"  [{i+1}] Relevance Score: {score}")
#     print(f"      \"{doc.page_content[:150]}...\"")
#     print()

llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        azure_deployment="sample-gpt-4o-deployment",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=0.5,
        top_p=1.0,
        max_tokens=4096,
    )

template = """Answer the question based ONLY on the following context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": flashrank_retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(query)
print(f"\n--- Final Answer ---\n{response}")