import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define absolute paths so the script runs from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "solutions", "Python")
DB_DIR = os.path.join(BASE_DIR, ".chroma")

def ingest_code():
    print(f"Scanning for Python files in: {DATA_DIR}")
    
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.py", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    
    if not documents:
        print("No Python files found! Drop a script in data/solutions/ and run again.")
        return

    print(f"Loaded {len(documents)} files. Applying syntax-aware chunking...")
    
    # AST-aware chunking for Python
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    chunks = python_splitter.split_documents(documents)
    print(f"Created {len(chunks)} logical chunks. Generating embeddings...")
    
    # Fast, local embedding model (runs offline)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store vectors locally in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print("✅ Ingestion complete! Vector database is staged and ready.")

if __name__ == "__main__":
    ingest_code()