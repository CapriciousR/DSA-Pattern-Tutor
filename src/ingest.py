import os
import shutil
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, ".chroma")
STORE_PATH = os.path.join(BASE_DIR, ".store.pkl")
REPO_DIR = os.path.join(BASE_DIR, "data", "solutions", "Python")

def ingest_pdr():
    print("🧹 Nuking old vector database and document store...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)

    # 1. Setup our splitters
    # Parent chunks: Large enough to capture the full algorithmic context
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # Child chunks: Small enough to capture tight, specific semantic meaning
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 2. Setup Vector Store & In-Memory Doc Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    store = InMemoryStore()

    # 3. Initialize the Parent Document Retriever Engine
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    documents = []
    print("📂 Scanning repository...")

    for root, _, files in os.walk(REPO_DIR):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file
                    documents.extend(docs)
                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

    if not documents:
        print(f"❌ ERROR: No Python files found in {REPO_DIR}. Check your path!")
        return

    print(f"🔪 Splitting and indexing {len(documents)} parent files in batches...")
    
    # We send 200 parent files at a time to stay safely under the 5461 child chunk limit
    BATCH_SIZE = 200 
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        print(f"  -> Processing batch {i // BATCH_SIZE + 1} of {(len(documents) // BATCH_SIZE) + 1}...")
        
        # Add this specific batch to the retriever
        retriever.add_documents(batch, ids=None)

    print("💾 Pickling the parent document store...")
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)

    print("🚀 PDR Ingestion complete! The database is locked and loaded.")

if __name__ == "__main__":
    ingest_pdr()