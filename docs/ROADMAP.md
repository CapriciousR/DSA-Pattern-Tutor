# 🗺️ DSA Pattern Tutor - Engineering Roadmap

### Phase 1: Establish the Evaluation Baseline
* [ ] **Create the Benchmark:** Write a test script (`scripts/evaluate_retrieval.py`) with 25 hardcoded DSA questions.
* [ ] **Measure Accuracy:** Calculate Top-1, Top-3, and Top-5 retrieval accuracy percentages.
* [ ] **Raw Inspection:** Log the exact text `retriever.invoke(query)` returns to inspect chunk quality directly.

### Phase 2: Advanced Retrieval & Context Engineering
* [ ] **Parent-Document Retrieval:** Separate math vectors (`.chroma`) from heavy code payloads (`.docstore`) to prevent algorithm fragmentation.
* [ ] **Hybrid Search Integration:** Layer **BM25** (lexical keyword search) over ChromaDB to handle exact algorithm keywords alongside semantic meaning.
* [ ] **Metadata Tagging:** Tag chunks during ingestion with category filters (`{"category": "Dynamic Programming"}`).

### Phase 3: Model & Hardware Optimization
* [ ] **A/B Model Testing:** Run evaluation benchmarks using `Llama-3.2:3B`, `Qwen2.5:3B`, and `Gemma-2:2B` to see which performs best on local constraints.
* [ ] **Prompt Engineering Guardrails:** Lock down the system prompt to guarantee the model provides strategic architectural hints rather than direct solution dumps.

### Phase 4: Production Packaging & Deployment
* [x] **CI/CD Pipeline:** Configure GitHub Actions to automatically run the evaluation framework on every push.
* [ ] **Containerization:** Package the backend, databases, and Streamlit UI into a single `Dockerfile`.