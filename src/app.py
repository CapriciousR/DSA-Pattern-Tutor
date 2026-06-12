import streamlit as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import sys
from retriever import build_pdr_hybrid_retriever

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 1. Setup paths (Must match ingest.py exactly)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

st.title("🧠 DSA Pattern Tutor")
st.caption("Lightweight Local RAG Pipeline optimized for 4GB VRAM")

# 2. Load the exact same embedding model used during ingestion
@st.cache_resource
def load_rag_system():
    retriever = build_pdr_hybrid_retriever()
    llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)
    return retriever, llm

retriever, llm = load_rag_system()

# 3. Create a strict system prompt for interview prep
prompt_template = """
You are an encouraging technical mentor.
Your goal is to give a gentle, high-level nudge to point the user in the right direction. 
Do NOT explain the exact mathematical logic, and NEVER write Python code.

SECRET REFERENCE SOLUTION:
{context}

USER QUERY: {question}

Provide a casual, 1-2 sentence suggestion. Mention the optimal algorithmic pattern and the target time complexity, but leave the actual implementation up to the user.

Example Output:
"Try using a two-pointer approach from both ends of the array. Think about how you can use that to track the maximums and reach an O(n) solution without needing any nested loops!"

Your Suggestion:"""

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