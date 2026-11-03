# SEC 10-K Financial RAG Engine

**Live Demo:** [Click here to launch the web app](https://financial-rag-engine-ktlbirflinzxinqgjqno3o.streamlit.app/)

A production-grade Retrieval-Augmented Generation (RAG) application designed to analyze complex SEC 10-K financial reports. This engine uses a **Two-Stage Retrieval Pipeline** to extract exact figures from dense financial tables and corporate narratives, ensuring zero hallucinations and high-fidelity synthesis.

Currently configured to analyze the **Alphabet Inc. (GOOG) 2024 Form 10-K**.

## 🧠 Architecture & Two-Stage Pipeline

To handle the complexity of regulatory filings where standard vector search often fails (especially with numerical tables and nuanced corporate jargon), this engine employs a two-stage retrieval process:

1. **Document Parsing & Chunking:** `PyMuPDF` reads the raw PDF, preserving table structures better than standard extractors. `LangChain` splits the text into 1200-character chunks with a 150-character overlap to prevent severing financial data tables.
2. **Stage 1 - High Recall (Bi-Encoder):** `FAISS` and `BGE-Small-EN-v1.5` instantly retrieve the top 10 most semantically relevant chunks across the 80+ page document.
3. **Stage 2 - High Precision (Cross-Encoder):** `MS-MARCO MiniLM-L-6-v2` re-evaluates the retrieved chunks against the exact user query, surfacing the top 3 absolute best facts.
4. **LLM Synthesis (Gemini):** Google's Gemini models synthesize the strictly grounded facts to answer the user's question, applying domain-specific guardrails.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit (Python)
* **Document Extraction:** PyMuPDF
* **Text Chunking:** LangChain (`RecursiveCharacterTextSplitter`)
* **Vector Database:** FAISS (CPU-optimized)
* **Embedding Model:** Sentence-Transformers (`BAAI/bge-small-en-v1.5`)
* **Reranking Model:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **LLM Integration:** Google GenAI SDK (`gemini-3.5-flash-lite` / `gemini-3.8-flash`)

---

## ✨ Key Features & Engineering Highlights

* **Zero-Hallucination Guardrails:** The LLM is strictly prompted to refuse answering if the exact figures or facts are not present in the retrieved context.
* **API Fault Tolerance:** Implemented a custom `try-except` routing logic. If the primary lightweight model (`gemini-3.5-flash-lite`) hits a `503 UNAVAILABLE` capacity limit, the engine automatically falls back to `gemini-3.8-flash` to ensure uninterrupted user experience.
* **Transparent "Behind the Scenes" Extraction:** The UI includes an expandable inspector that displays the exact, raw textual chunks the Cross-Encoder passed to the LLM, allowing for total auditability of the financial data.
* **Financial Table Preservation:** Optimized chunk sizing (1200 chars) specifically designed to keep SEC financial grid data intact during the vectorization process.

---

## 💻 Run it Locally

### 1. Clone the repository
```bash
git clone [https://github.com/YourUsername/financial-rag-engine.git](https://github.com/YourUsername/financial-rag-engine.git)
cd financial-rag-engine
```

### 2. Set up a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate #On Windows
source venv/bin/activate #On Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
Add your Google API key to a `.env` file: GEMINI_API_KEY="your_key"
```

### 5.Launch the Engine
```bash
 Launch the app: streamlit run app.py
