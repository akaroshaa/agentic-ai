import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

# Disable the faulty deallocator
AzureSearch.__del__ = lambda self: None

load_dotenv()

pdf_path = "C:/Users/HP/Downloads/Comm626.pdf"  # Replace with the actual path to your PDF file

# 1. Load the PDF and create Document objects
reader = pypdf.PdfReader(pdf_path)
print(f"Loaded PDF from: {pdf_path}")
docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]
print(f"Document has {len(docs)} pages")

# 2. Splitting the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"Number of text splits: {len(all_splits)}")


# 3. Setup Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://crook-mt496dmp-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="sample-text-embedding-3-small",
    api_version="2024-12-01-preview",
)

# 4. Initialize Azure AI Search
vector_store = AzureSearch(
    azure_search_endpoint="https://sample-ai-search-drop.search.windows.net",
    azure_search_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
    index_name="sample-index",
    embedding_function=embeddings.embed_query,
)

# 5. Index chunks
ids = vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(ids)} document chunks to Azure AI Search.")


# 6. Query the Azure AI Search index
query = "What is the key takeaway from the document?"
retriever = vector_store.as_retriever(
    search_type="similarity", k=3
    )
docs = retriever.invoke(query)

for idx, doc in enumerate(docs, start=1):
    print(f"\n--- Result {idx} (Page: {doc.metadata.get('page')}) ---")
    print(doc.page_content[:300])


# # 7. Query the Azure AI Search index (with Similarity Scores)
# threshold = 0.2
# filtered_docs = [
#     (doc, score) for doc, score in vector_store.similarity_search_with_score(query=query, k=5)
#     if score < threshold
# ]
# for idx, (doc, raw_score) in enumerate(filtered_docs, start=1):
#     print(f"\n--- Result {idx} | Raw Score: {raw_score:.4f} ---")
#     print(f"Page: {doc.metadata.get('page')}")
#     print(f"Content: {doc.page_content[:250]}...")