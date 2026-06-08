# 🧠 DSA Pattern Tutor

A hardware-optimized, local Retrieval-Augmented Generation (RAG) pipeline designed to act as an intelligent tutor for Data Structures and Algorithms. 

Rather than just dumping raw Python solutions, this system is engineered to provide strategic, conceptual hints (like sliding window logic or backtracking templates) to aid in interview preparation.

## 🏗️ Architecture & Stack
- **Frontend UI:** Streamlit
- **Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Local Inference:** Llama 3.2 (3B) via Ollama
- **Retrieval Strategy:** Dual-Tier Storage (Vector Math + Raw Text Payloads)

## 🚀 Getting Started

1. Clone the repository and set up the environment
```bash
git clone https://github.com/CapriciousR/DSA-Pattern-Tutor
cd DSA-Pattern-Tutor
pip install -r requirements.txt
```

2. Ingest the Data
Note: Ensure your local .docstore and .chroma directories are clean before running the ingestion pipeline.
```bash
python src/ingest.py
```

3. Launch the Application
```bash
streamlit run src/app.py
```

🗺️ Future Roadmap
Check out ROADMAP.md for the current evaluation and feature rollout plan.
