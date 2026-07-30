# 🕷️ Dr. Lowkey: Evidence-Based Medical RAG Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-fd4258)
![Gemini](https://img.shields.io/badge/Gemini-3.5_Flash_Lite-4285F4)

**Dr. Lowkey** is an advanced, safety-first Retrieval-Augmented Generation (RAG) system built to answer health-related queries using strictly verified medical guidelines (WHO, NHS, CDC, ICMR). It combines hybrid vector search, cross-encoder reranking, and multi-layered safety guardrails to deliver fast, accurate, and hallucination-free medical information via a custom-built, streaming web UI.

---

## ✨ Key Features

*   **🔍 Hybrid Search Engine:** Combines Dense embeddings (`BAAI/bge-small-en-v1.5`) for semantic understanding with Sparse BM25 embeddings (`Qdrant/bm25`) for exact medical terminology matching.
*   **⚖️ Cross-Encoder Reranking:** Uses `FlashRank` (`ms-marco-MiniLM-L-12-v2`) to rerank the top 10 search results down to the absolute top 3, maximizing LLM context precision.
*   **🛡️ Multi-Tier Safety Guardrails:**
    *   **Emergency Triage:** Detects critical symptoms (e.g., "chest pain", "stroke") and intercepts the query to advise calling local emergency services (108).
    *   **Intent Matcher:** Detects self-harm, violence, or dangerous drug practices, bypassing the RAG pipeline to deliver a hardcoded, caring safety response.
*   **🗣️ Conversational Bypass:** Uses Regex intent matching to gracefully handle casual greetings and identity questions without triggering "insufficient medical evidence" errors.
*   **⚡ Real-Time Streaming UI:** A sleek, medical-blue "spider-web" themed frontend that utilizes Server-Sent Events (SSE) to stream tokens directly from the Gemini API for a seamless user experience.

---

## 🛠️ Technology Stack

*   **Backend:** FastAPI, Uvicorn, Python
*   **Vector Database:** Qdrant (Local instance)
*   **Embeddings:** FastEmbed (Dense & Sparse)
*   **Reranker:** FlashRank
*   **LLM Generator:** Google Gemini API (`gemini-3.5-flash-lite`)
*   **Data Pipeline:** BeautifulSoup4, Markdownify, PyPDF, Pandas
*   **Frontend:** HTML5, CSS3 (Custom Variables & Keyframes), Vanilla JavaScript

---

## 📂 Project Structure

```text
.
├── data/
│   └── raw/                # Downloaded PDFs, scraped HTML, and MedQuAD datasets
    ├── process_data.py     # Data scraper and PDF extractor
├── qdrant_db/              # Local Qdrant vector storage
├── src/
│   ├── config.py           # Global paths, models, and thresholds
│   ├── generator.py        # Gemini API integration and strict prompt engineering
│   ├── ingestion.py        # Chunking logic and hybrid embedding generation
│   ├── retrieval.py        # Hybrid Reciprocal Rank Fusion (RRF) search logic
│   └── triage_guard.py     # Emergency clinical intent matching
├── static/
│   ├── index.html          # Frontend UI layout
│   ├── style.css           # Medical Blue & Spider theme styling
│   └── script.js           # API communication and SSE token rendering
├── app.py                  # FastAPI application and routing logic
└── README.md
