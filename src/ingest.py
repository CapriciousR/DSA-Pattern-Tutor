import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, ".chroma")
REPO_DIR = os.path.join(BASE_DIR, "data", "solutions", "Python") # Update this if your repo path is different

def ingest_with_metadata():
    print("🧹 Nuking old vector database...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    documents = []
    print("📂 Scanning repository and tagging metadata...")

    for root, _, files in os.walk(REPO_DIR):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                
                # 1. Dynamically generate the category from the parent folder name
                # e.g., "TheAlgorithms/graphs/dijkstra.py" -> "Graphs"
                raw_category = os.path.basename(root)
                clean_category = raw_category.replace("_", " ").title()
                
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    # 2. INJECT THE CRITICAL METADATA HERE
                    for doc in docs:
                        doc.metadata["category"] = clean_category
                        doc.metadata["source"] = file
                        
                    documents.extend(docs)
                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

    print(f"✅ Loaded {len(documents)} tagged files.")

    # 3. Split the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    print(f"🔪 Splitting into {len(chunks)} chunks...")

    # 4. Embed and store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("💾 Embedding and saving to ChromaDB...")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("🚀 Ingestion complete! The database is now fully tagged.")

if __name__ == "__main__":
    ingest_with_metadata()