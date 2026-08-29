import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# Hybrid Search (Keyword + Semantic) + Flashrank
# ============================================================

from flashrank import Ranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# ============================================================

from langchain_neo4j import Neo4jGraph
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


kg = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)
kg.refresh_schema()
print("Connected to Neo4j Knowledge Graph. Schema refreshed.")

query1 = """
        CREATE (p1:Person {name: 'Alice Johnson', age: 28, city: 'New York'})
        CREATE (p2:Person {name: 'Bob Smith', age: 35, city: 'San Francisco'})
        CREATE (p3:Person {name: 'Carol White', age: 42, city: 'Boston'})
        CREATE (p4:Person {name: 'David Brown', age: 31, city: 'Seattle'})
        RETURN 'Created 4 Person nodes' AS result
        """

query2 = """CREATE (c1:Company:TechCompany {
            name: 'TechCorp', 
            founded: 2010, 
            industry: 'Software',
            employees: 500
        })
        CREATE (c2:Company:TechCompany {
            name: 'DataSolutions', 
            founded: 2015, 
            industry: 'Data Analytics',
            employees: 150
        })
        RETURN 'Created 2 Company nodes with multiple labels' AS result
        """

query3 = """
        MATCH (p:Person {name: 'Alice Johnson'}), (c:Company {name: 'TechCorp'})
        CREATE (p)-[r:WORKS_AT {since: 2020, position: 'Software Engineer'}]->(c)
        RETURN p.name + ' works at ' + c.name AS result
        """

result1 = kg.query(query1)
print(result1)

# result2 = kg.query(query2)
# print(result2)

# result3 = kg.query(query3)
# print(result3)



# # 3. Setup Embeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_deployment="sample-text-embedding-3-small",
#     api_version="2024-12-01-preview",
# )

# # 4. Initialize Azure AI Search
# vector_store = AzureSearch(
#     azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
#     azure_search_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
#     index_name="sample-index",
#     embedding_function=embeddings.embed_query,
# )

# # 5. Index chunks
# # ids = vector_store.add_documents(documents=all_splits)
# # print(f"Indexed {len(ids)} document chunks to Azure AI Search.")


# # 6. Query the Azure AI Search index
# query = "What is the key takeaway from the document?"



# # =====================================================================
# #                           IF NOT USING AZURE 
# # =====================================================================


# # # ---  BUILD HYBRID RETRIEVER (BM25 + Vector Search) ---

# # # pip install rank-bm25

# # # from langchain_community.retrievers import BM25Retriever
# # # from langchain_classic.retrievers import EnsembleRetriever

# # # A. Dense Vector Retriever
# # vector_retriever = vector_store.as_retriever(search_type="similarity", k=6)

# # # B. Sparse BM25 Keyword Retriever (Built from document chunks)
# # bm25_retriever = BM25Retriever.from_documents(all_splits)
# # bm25_retriever.k = 6

# # # C. Combine both with Ensemble (50% BM25, 50% Vector)
# # hybrid_ensemble_retriever = EnsembleRetriever(
# #     retrievers=[bm25_retriever, vector_retriever],
# #     weights=[0.5, 0.5]
# # )

# # base_compressor = FlashrankRerank(top_n=3)

# # flashrank_retriever = ContextualCompressionRetriever(
# #     base_retriever=hybrid_ensemble_retriever,
# #     base_compressor=base_compressor
# # )


# base_retriever = vector_store.as_retriever(
#     search_type="hybrid", k=10
#     )

# base_compressor = FlashrankRerank(top_n=3)

# flashrank_retriever = ContextualCompressionRetriever(
#     base_retriever=base_retriever,
#     base_compressor=base_compressor
# )

# # reranked_results = flashrank_retriever.invoke(query)
# # for i, doc in enumerate(reranked_results):
# #     score = doc.metadata.get("relevance_score", "N/A")
# #     print(f"  [{i+1}] Relevance Score: {score}")
# #     print(f"      \"{doc.page_content[:150]}...\"")
# #     print()

# llm = AzureChatOpenAI(
#         azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
#         azure_deployment="sample-gpt-4o-deployment",
#         api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#         api_version="2024-12-01-preview",
#         temperature=0.5,
#         top_p=1.0,
#         max_tokens=4096,
#     )

# template = """Answer the question based ONLY on the following context. 
# If the context doesn't contain the answer, say "I don't have enough information to answer this."

# Context:
# {context}

# Question: {question}

# Answer:"""

# prompt = ChatPromptTemplate.from_template(template)

# rag_chain = (
#     {"context": flashrank_retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# response = rag_chain.invoke(query)
# print(f"\n--- Final Answer ---\n{response}")