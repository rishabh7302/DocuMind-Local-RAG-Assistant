# 📄 PDF Document Q&A — RAG-based Application

A Retrieval-Augmented Generation (RAG) application that lets users upload one or more PDF documents and ask natural-language questions about their content. The system retrieves the most relevant chunks from the document(s) and generates a grounded, context-based answer using a local LLM — with source citations, so answers can be verified against the original document.

---

## 🎯 What This Project Does

Instead of relying on an LLM's general knowledge (which can hallucinate facts), this app:

1. Lets a user upload PDF(s) — single or multiple, at runtime.
2. Splits the document text into small, overlapping chunks.
3. Converts each chunk into a vector embedding (numerical representation of meaning).
4. Stores these vectors in a FAISS vector database for fast similarity search.
5. When a user asks a question, retrieves the top-k most relevant chunks.
6. Passes those chunks as context to a local LLM (via Ollama), which generates an answer **strictly grounded in the retrieved content**.
7. Displays the answer along with the source chunks/pages it came from, so the user can verify it.

This is a fully **local, private, and free** pipeline — no data leaves the machine, and no paid API keys are required.

---

## 🧩 Why This Project Is Meaningful (for a portfolio)

- Demonstrates understanding of the full RAG pipeline: chunking, embeddings, vector search, and grounded generation — not just "calling an LLM API."
- Includes citation/source display, which addresses one of RAG's biggest real-world problems: hallucination and unverifiable answers.
- Uses a fully local stack (Ollama for both LLM and embeddings), showing awareness of cost, privacy, and offline-capable AI system design.
- Supports multiple document uploads at runtime with session-based reset — a practical, product-like feature rather than a static notebook demo.

---

## 🛠️ Tech Stack

| Component | Tool Used | Purpose |
|---|---|---|
| Orchestration | LangChain | Connects loading, splitting, embedding, retrieval, and generation |
| LLM (answer generation) | Ollama — `qwen3-vl:2b` | Generates the final natural-language answer |
| Embeddings | Ollama — `nomic-embed-text` | Converts text chunks into vectors |
| Vector Store | FAISS | Stores embeddings and performs similarity search |
| PDF Reading | pypdf / PyPDFLoader | Extracts text from uploaded PDF files |
| UI | Streamlit | Web interface for upload, questions, and answers |

---

## 🔄 Pipeline Flow

```
PDF Upload (single or multiple)
        │
        ▼
Text Extraction (PyPDFLoader)
        │
        ▼
Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
Embedding (nomic-embed-text via Ollama)
        │
        ▼
Vector Store (FAISS) — held in session memory
        │
        ▼
User Question
        │
        ▼
Similarity Search (top-k relevant chunks)
        │
        ▼
LLM Answer Generation (qwen3-vl:2b) — grounded in retrieved context
        │
        ▼
Answer + Source Citations displayed in UI
```

---

## ✨ Key Features

- Upload a single PDF or multiple PDFs at once.
- Manual "Process PDFs" step, so processing only happens when the user is ready.
- Fresh state on every page reload — no leftover data from a previous session.
- Answers are grounded strictly in the uploaded document content (explicit anti-hallucination instruction in the prompt).
- Expandable "Sources" section showing which chunk/page/file an answer came from.
- Fully local and free: no external API keys, no cloud costs.

---

## 📌 Status

This README describes the project's purpose and design. Setup and run instructions will be added once the full step list is finalized.
