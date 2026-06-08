import streamlit as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


# 1. Setup paths (Must match ingest.py exactly)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

st.title("🧠 DSA Pattern Tutor")
st.caption("Lightweight Local RAG Pipeline optimized for 4GB VRAM")

# 2. Load the exact same embedding model used during ingestion
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the existing database from disk
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Configure retriever (Fetch top 3 matching code snippets)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Connect to your local Ollama model
    # CHANGE "llama3.2" TO THE EXACT NAME IN YOUR SCREENSHOT IF IT'S DIFFERENT
    llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
    
    return retriever, llm

retriever, llm = load_rag_system()

# 3. Create a strict system prompt for interview prep
prompt_template = """
You are an expert technical interview coach. Your job is to help the user understand Data Structures and Algorithms (DSA).
You are given a set of reference code implementations from past solutions (Context) and a user query.

Guidelines:
1. Use the Context to provide a conceptual explanation, algorithmic breakdown, or step-by-step hint.
2. DO NOT give them the direct, complete code solution. Guide them so they can write it themselves.
3. If the Context doesn't contain relevant information, use your core knowledge to guide them conceptually.

Context:
{context}

User Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 4. Build the LangChain RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_query := st.chat_input("Ask about a pattern (e.g., How do I implement a sliding window?)"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing local code vector space..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})