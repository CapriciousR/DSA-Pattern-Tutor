import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, ".chroma")
STORE_PATH = os.path.join(BASE_DIR, ".store.pkl")

def build_pdr_hybrid_retriever():
    print("💾 Loading Embeddings and ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("💾 Loading Pickled Parent Document Store...")
    try:
        with open(STORE_PATH, "rb") as f:
            store = pickle.load(f)
    except FileNotFoundError:
        print("❌ ERROR: .store.pkl not found. Run ingest.py first!")
        return None

    # Re-declare the splitters just to satisfy Pydantic's strict type validation
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 1. Rebuild the PDR Engine
    pdr = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter, 
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3} 
    )

    # 2. Build the BM25 Keyword Engine purely on the Parent Documents
    print("⚡ Building BM25 Index on Parent Documents...")
    parent_docs = []
    for key in store.yield_keys():
        parent_docs.extend(store.mget([key]))

    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 3

    # 3. Fuse the Engines (Phase 3 Architecture)
    print("🧬 Fusing into Hybrid Ensemble Engine (25% BM25 / 75% PDR)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pdr],
        weights=[0.25, 0.75]
    )

    return ensemble_retriever