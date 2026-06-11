# 📊 RAG Architecture Evaluation & Benchmarks

**Objective:** Build an intelligent Retrieval-Augmented Generation (RAG) tutor for Data Structures and Algorithms. 
**Dataset:** `TheAlgorithms/Python` (100+ raw algorithm scripts).
**Testing Methodology:** 22 highly specific LeetCode-style problem descriptions mapped to exact python scripts.

---

## Phase 1: Baseline Character Splitting
* **Splitter:** LangChain `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap).
* **Search Engine:** ChromaDB Dense Vectors (`all-MiniLM-L6-v2`).
* **Metrics:**
  * Top-1 Accuracy: 31.8%
  * Top-3 Accuracy: 50.0%
* **Conclusion:** Small chunks are decent at catching semantic keywords, but they physically shred the Python logic in half. If the LLM receives chopped code, it hallucinates. The 50% failure rate is unacceptable for a tutor.

## Phase 2: Structural (AST) Chunking
* **Splitter:** Custom Abstract Syntax Tree (AST) parser extracting whole functions/classes.
* **Search Engine:** ChromaDB Dense Vectors (`all-MiniLM-L6-v2`).
* **Metrics:**
  * Top-1 Accuracy: 22.7%
  * Top-3 Accuracy: ~50.0%
* **Conclusion:** Completely solves the LLM context problem by keeping algorithms unbroken. However, the massive chunk sizes cause "Vector Dilution." The dense embeddings get confused by all the raw Python boilerplate (`def`, `while`, `self`), causing severe misses on highly similar algorithmic patterns (e.g., Combination Sum I vs II).

## Phase 3: Hybrid Search (BM25 + Chroma)
* **Architecture:** Ensemble Retriever using Reciprocal Rank Fusion (RRF).
* **Weights:** `[0.0, 1.0]` (0% BM25 / 100% Chroma)
* **Metrics:** 
  * Top-1 Accuracy: 36.4%
  * Top-3 Accuracy: 59.1%
* **Observation:** Running pure Chroma on the expanded 22-case dataset with multi-answer safety nets eliminated false negatives. However, vector dilution from the large AST chunks still bottlenecked the system at a ~60% maximum retrieval ceiling.

### Test 3B: The 50/50 Split
* **Weights:** `[0.5, 0.5]` (50% BM25 / 50% Chroma)
* **Metrics:**
  * Top-1 Accuracy: 40.9%
  * Top-3 Accuracy: 45.5%
* **Observation:** BM25 successfully acted as a sniper for highly constrained queries (like "missing number"), immediately boosting Top-1 accuracy. However, because both engines had equal voting power, Chroma's noisy "diluted" results clashed with BM25, and the RRF algorithm penalized the Top-3 average heavily.

### Test 3C: The Tuned Configuration (Winner)
* **Weights:** `[0.25, 0.75]` (25% BM25 / 75% Chroma)
* **Metrics:**
  * Top-1 Accuracy: 45.5%
  * Top-3 Accuracy: 54.5%
* **Conclusion:** This configuration yielded the best balance. Allowing Chroma to cast a wide semantic net (75% weight) preserved the Top-3 safety net, while giving BM25 just enough voting power (25%) to push exact keyword matches into the absolute #1 spot.

## Phase 4: Two-Stage Pipeline (Cross-Encoder Re-ranking)
* **Architecture:** Phase 3 Hybrid Retriever + `ms-marco-MiniLM-L-6-v2` Cross-Encoder.
* **Test 4A (Top-3 Retrieval):**
  * Top-1 Accuracy: 27.3%
  * Top-3 Accuracy: 59.1%
* **Test 4B (Expanded Top-5 Retrieval):**
  * Top-1 Accuracy: 27.3%
  * Top-3 (Top-5) Accuracy: 68.2%
* **Conclusion:** **Failed Experiment.** The standard MS-MARCO cross-encoder is trained on English web-search queries, not Python AST syntax. It fundamentally misunderstood the algorithmic boilerplate, aggressively penalizing correct matches and tanking Top-1 accuracy down to 27%. While widening the net to Top-5 improved categorical grouping (68.2%), the compute overhead of running sequential transformer inferences locally caused massive latency spikes. 
* **Final Decision:** Roll back to Phase 3 (Tuned Hybrid Search) as the production baseline.

## Phase 5: Hypothetical Document Embeddings (HyDE)
* **Architecture:** Local Llama 3.2:3b generated hypothetical python snippets $\rightarrow$ Chroma Dense Vector Search + 25/75 BM25 Hybrid Fusion.
* **Metrics:**
  * Top-1 Accuracy: 40.9%
  * Top-3 Accuracy: 54.5%
* **Conclusion:** **Failed Experiment.** While HyDE is an industry standard for natural language, it failed on rigid algorithmic code. The structural style of the LLM-generated code (e.g., flat functions) did not perfectly match the specific architectural style of `TheAlgorithms/Python` repository (e.g., class-based implementations). Furthermore, the massive compute overhead of running local LLM inference before every search introduced unacceptable latency for a 0% improvement.

## Phase 6: Algorithmic Metadata Routing
* **Architecture:** O(1) Dictionary Fast Lane + Llama 3.2 Smart Lane -> Hard Metadata Filter -> 25/75 BM25/Chroma Hybrid Search.
* **Metrics:**
  * Top-1 Accuracy: 27.3%
  * Top-3 Accuracy: 45.5%
* **Conclusion:** **Failed Experiment.** Hard metadata filtering caused "Intra-Category Collapse". By restricting the search space to a single folder, BM25 lost its term-frequency sorting power (every file had the same keywords), and dense vectors struggled to differentiate highly homogenous code files. 
* **ULTIMATE DECISION:** The architecture is officially rolled back to **Phase 3 (Global Hybrid Search)**. It balances global keyword rarity with semantic structural matching flawlessly and remains the fastest, most resilient engine for this specific codebase.