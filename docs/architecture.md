graph TD
    %% Styling
    classDef user fill:#2d3436,stroke:#b2bec3,stroke-width:2px,color:#fff
    classDef engine fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    classDef db fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    classDef logic fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff

    Q["👤 User Query"]:::user --> Split["Parallel Search Strategy"]:::logic

    %% BM25 Pathway
    Split -->|25% Weight| BM25["BM25 Keyword Engine"]:::engine
    BM25 -.->|Exact Match| Store[("In-Memory Store<br/>(Parent AST Documents)")]:::db

    %% Chroma Pathway
    Split -->|75% Weight| Chroma["ChromaDB Vector Engine"]:::engine
    Chroma -.->|Semantic Match| Vec[("Dense Vector DB<br/>(400-char Child Chunks)")]:::db
    
    %% Parent Document Retriever (PDR) mapping
    Vec -->|parent_id lookup| Store

    %% Retrieval Fusion
    Store --> Fusion["Reciprocal Rank Fusion<br/>(Ensemble Retriever)"]:::logic
    BM25 --> Fusion

    %% LLM Generation
    Fusion -->|Top 3 Full Scripts| Prompt["Prompt Formatting<br/>(Strict Socratic Template)"]:::logic
    Prompt --> LLM["Llama 3.2 (3B)<br/>Local Inference"]:::engine
    LLM --> Final["🎯 Socratic Hint & Complexity"]:::user