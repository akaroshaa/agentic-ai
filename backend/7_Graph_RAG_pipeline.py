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

from langchain_experimental.graph_transformers import LLMGraphTransformer

# Disable the faulty deallocator
AzureSearch.__del__ = lambda self: None

load_dotenv()


raw_documents = [
    Document(
        page_content=(
            "POLICY PACK: MedFridge Ltd - Cold-chain oncology injectables (EU GDP). "
            "SKUs: MF-Insulin-Pro (2-8 deg C), MF-MAB-Cool (-25 to -15 deg C frozen). "
            "Batch B-4421 (MF-Insulin-Pro) was released from Plant Dresden on 2025-11-02 "
            "with a stability budget of 48 aggregate excursion-minutes above 8 deg C before "
            "lot disposition must be escalated to QP (Qualified Person). Courier Apex "
            "Cold carried the pallet to distributor NordicCare. NordicCare's receiving "
            "SOP states: if logger shows any single contiguous excursion >20 minutes "
            "above 8 deg C, the pallet is QUARANTINED and cannot be merged with sellable "
            "inventory until QA completes investigation. However, NordicCare may "
            "PRE-RELEASE to hospital back-order if (a) the excursion is at most 25 minutes, "
            "(b) MedFridge Medical Affairs emails written exception referencing batch "
            "stability report SR-B4421-REV-C, and (c) the hospital accepts residual "
            "risk form RF-77. Batch B-4408 is unrelated (different stability report). "
            "Regulatory: EMA variation VAR-1189 caps combined truck + warehouse hand-off "
            "delay at 36 hours for MF-Insulin-Pro; Apex logged 31 hours end-to-end for "
            "B-4421. If quarantine is triggered, NordicCare must notify EudraVigilance "
            "only when product has already left quarantine to a patient - not while "
            "held on-site. Finance rule: revenue for B-4421 cannot be recognized at "
            "MedFridge until NordicCare posts 'Available for sale' in ERP ledger code "
            "NCF-AS-01."
        ),
        metadata={
            "title": "MedFridge cold-chain and NordicCare release rules",
            "source": "business:medfridge-policy-pack",
            "domain": "pharma_logistics",
        },
    ),
    Document(
        page_content=(
            "COMMERCIAL MEMO: HelioStack SaaS - Enterprise Agreement EA-2024-771 with "
            "ACME Corp (manufacturing vertical). Contract term 2024-07-01 to 2027-06-30. "
            "Committed ARR: $1.2M for 2,400 named seats of module 'HelioMES'; unit list "
            "price $600/seat/year before tiered discount. Clause 14(b): at each renewal "
            "anniversary, customer may downgrade up to 20% of committed seats without "
            "termination fee; downgraded seats convert to month-to-month list price for "
            "remainder of term unless re-committed. Clause 14(c): if downgrade exceeds "
            "20%, HelioStack may terminate for convenience with 90-day notice and forfeit "
            "only unbilled future periods (no clawback of cash already collected). "
            "Revenue policy (ASC 606 memo FIN-HS-09): for multi-year prepay deals, "
            "HelioStack recognizes revenue straight-line over the service period; "
            "if seats are removed mid-period, deferred revenue is reduced prospectively "
            "from the downgrade effective date - never retroactively restating prior "
            "quarters. Customer success owns 'adoption score' gates: if adoption score "
            "<40 at day-180, auto-renew uplift of 7% is waived. ACME's plant in "
            "Gdansk shares one tenant with subsidiary ACME Baltics; Baltics seats are "
            "counted inside the same 2,400 pool (not additive). Competitor OrionMES is "
            "excluded from data residency routing per Annex D."
        ),
        metadata={
            "title": "HelioStack ACME enterprise agreement and revenue rules",
            "source": "business:heliostack-ea-771",
            "domain": "b2b_saas_contracts",
        },
    ),
    Document(
        page_content=(
            "TRADE OPS BRIEF: FobCo (Mumbai exporter) sold specialty dyes to RhineChem "
            "AG under contract CTR-RH-303. Payment: irrevocable letter of credit LCI-9001 "
            "issued by Deutsche Handelsbank (confirming bank added). Required documents "
            "under UCP 600 field 46A: commercial invoice, packing list, certificate of "
            "origin, full set onboard bill of lading showing shipment from Nhava Sheva "
            "to Hamburg, and phytosanitary certificate. Contract Incoterms 2020: CIF "
            "Hamburg - seller arranges carriage and insurance to named port; risk "
            "transfers when goods pass ship's rail at origin port per Incoterms rules. "
            "The B/L received by the negotiating bank shows 'FOB Nhava Sheva' and "
            "freight 'collect'. Discrepancy playbook: if Incoterms on B/L contradict "
            "contract, RhineChem's treasury may ACCEPT waiver if RhineChem signs "
            "discrepancy indemnity DI-IND-01 before presentation; otherwise documents "
            "must be re-issued. Force majeure addendum FM-22: port worker strikes at "
            "Hamburg do NOT excuse FobCo from presenting conforming documents - they only "
            "extend delivery tolerance by 10 calendar days for physical arrival, not for "
            "documentary compliance. Sanctions screen: RhineChem is Tier-1 cleared; "
            "transhipment via sanctioned corridor SC-List-B is forbidden even if cheaper."
        ),
        metadata={
            "title": "FobCo RhineChem LC and Incoterms discrepancy handling",
            "source": "business:fobco-trade-ctr-rh-303",
            "domain": "trade_finance",
        },
    ),
]
print(f"Loaded {len(raw_documents)} source document(s).")


# 2. Splitting the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(raw_documents)
print(f"Number of text splits: {len(all_splits)}")

print("Sample chunk metadata:", [d.metadata.get("title") for d in all_splits[:3]])

kg = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)
kg.refresh_schema()
print("Connected to Neo4j Knowledge Graph. Schema refreshed.")


# # 3. Setup Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="sample-text-embedding-3-small",
    api_version="2024-12-01-preview",
)

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

llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        azure_deployment="sample-gpt-4o-deployment",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=0.5,
        top_p=1.0,
        max_tokens=4096,
    )

llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(all_splits)

# Inspect what was extracted (nodes + relationships per chunk).
for gd in graph_documents:
    print("Nodes:", [n.id for n in gd.nodes])
    print("Rels :", [(r.source.id, r.type, r.target.id) for r in gd.relationships])
    print("-" * 60)

kg.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)
print("Graph written to Neo4j.")



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