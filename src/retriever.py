import os
import re
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, ".chroma")
REPO_DIR = os.path.join(BASE_DIR, "data", "solutions", "Python") # Path to your local codebase

class HybridRouterRetriever:
    def __init__(self, repo_path=REPO_DIR):
        self.repo_path = repo_path
        print("⚡ Extracting filesystem layout to build Fast Lane lookup...")
        self.algo_router = self._auto_generate_keywords()
        
        print("💾 Initializing Vector Store & Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        
        # Build the LLM Router (Smart Lane Safety Net)
        self.llm = OllamaLLM(model="llama3.2:3b", temperature=0.0)
        router_prompt = PromptTemplate.from_template("""
You are a strict, highly efficient text classifier for data structures and algorithms. 
Classify the query into exactly ONE of the following known categories:
{categories}

Output ONLY the exact category name. No explanations, no markdown, no problem numbers.

Example 1:
Query: "Find the maximum profit from buying and selling stocks over given days."
Category: Dynamic Programming

Example 2:
Query: "Write a function to traverse a hierarchy and print leaf nodes."
Category: Trees

Example 3:
Query: {query}
Category:""")
        self.router_chain = router_prompt | self.llm

    def _auto_generate_keywords(self):
        """Scans the codebase folders to dynamically map algorithm filenames to their categories."""
        algo_router = {}
        if not os.path.exists(self.repo_path):
            # Fallback mock map for testing if the repo directory isn't fully pulled yet
            return {
                "dijkstra": "Graphs", "floyd warshall": "Graphs", "bellman ford": "Graphs",
                "combination sum": "Backtracking", "subsets": "Backtracking",
                "knapsack": "Dynamic Programming", "longest common subsequence": "Dynamic Programming"
            }
            
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    raw_name = file.replace(".py", "")
                    # Parent folder name becomes the clean Category (e.g., "dynamic_programming" -> "Dynamic Programming")
                    category = os.path.basename(root).replace("_", " ").title()
                    clean_phrase = raw_name.replace("_", " ")
                    
                    algo_router[clean_phrase] = category
                    if "cache" in clean_phrase:
                        algo_router[clean_phrase.replace(" cache", "")] = category
        return algo_router

    def route_query(self, query: str) -> str:
        """Determines the target metadata category using the Hybrid Router strategy."""
        query_lower = query.lower()
        
        # --- LANE 1: DYNAMIC FAST LANE (0.01ms) ---
        # Sort by length descending to catch multi-word phrases first ("binary search tree" before "binary search")
        sorted_keywords = sorted(self.algo_router.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, query_lower):
                print(f"🎯 [Fast Lane Match]: Found explicit keyword '{keyword}'. Routing to -> {self.algo_router[keyword]}")
                return self.algo_router[keyword]
                
        # --- LANE 2: SMART LANE LLM FALLBACK (3-4s) ---
        unique_categories = list(set(self.algo_router.values())) if self.algo_router else ["Graphs", "Backtracking", "Dynamic Programming"]
        print("🤔 [Smart Lane Triggered]: No explicit keyword matches. Asking LLM to classify...")
        
        llm_output = self.router_chain.invoke({
            "query": query,
            "categories": ", ".join(unique_categories)
        }).strip()
        
        print(f"🤖 [LLM Routed]: Classified query as -> {llm_output}")
        return llm_output

    def invoke(self, query: str, k=3):
        """Routes the query, applies a hard metadata filter, and executes a Hybrid Search."""
        category = self.route_query(query)
        
        # 1. Apply hard metadata tracking filter to the Vector Search
        chroma_retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"category": category} # Hard filter injection
            }
        )
        
        # 2. Extract docs isolated to this category to build a clean BM25 engine on the fly
        category_docs_data = self.vectorstore.get(
            where={"category": category}, 
            include=["metadatas", "documents"]
        )
        
        # Fallback if metadata tag yields 0 files (database mismatch)
        if not category_docs_data["documents"]:
            print(f"⚠️ Metadata category '{category}' not found in active index. Falling back to global search.")
            chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            all_docs_data = self.vectorstore.get(include=["metadatas", "documents"])
            docs = [Document(page_content=d, metadata=m) for d, m in zip(all_docs_data["documents"], all_docs_data["metadatas"])]
        else:
            docs = [Document(page_content=d, metadata=m) for d, m in zip(category_docs_data["documents"], category_docs_data["metadatas"])]
            
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        
        # 3. Fuse the restricted engines
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.0, 1.0]
        )
        
        return ensemble_retriever.invoke(query)