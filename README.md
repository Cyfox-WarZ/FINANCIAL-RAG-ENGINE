# SEC 10-K Financial RAG Engine

**Live Demo:** [Click here to launch the web app](https://financial-rag-engine-ktlbirflinzxinqgjqno3o.streamlit.app/)

## Overview
This project is a production-grade Retrieval-Augmented Generation (RAG) engine designed to query financial documents (like SEC 10-K filings) with high accuracy. It was built to demonstrate how to effectively eliminate LLM hallucinations when dealing with strict corporate financial data by utilizing advanced reranking algorithms and prompt engineering.

## Architecture: Two-Stage Retrieval Pipeline
Standard vector databases often fail on financial data because embedding models treat concepts like "CEO stepping down" and "laying off staff" as semantically identical (both map to "employment termination"). To solve this, this engine uses a two-stage pipeline:

1. **Stage 1 - High Recall (Bi-Encoder + FAISS):** The user's query is vectorized using `BAAI/bge-small-en-v1.5`. FAISS performs a fast $O(1)$ nearest-neighbor search to retrieve the top 3 broad candidate facts.
2. **Stage 2 - High Precision (Cross-Encoder):** A `ms-marco-MiniLM` Cross-Encoder scores the exact semantic relationship between the query and the 3 candidates simultaneously, forcefully rejecting irrelevant data and passing only the single most accurate fact to the LLM.
3. **Stage 3 - Generation (Gemini API):** Google's Gemini synthesizes the final answer using strict prompt engineering to interpret business terminology without hallucinating outside the provided context.

## Tech Stack
* **Frontend UI & Cloud Deployment:** Streamlit
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings & Reranking:** Sentence-Transformers (Hugging Face)
* **LLM:** Google GenAI SDK (Gemini)
* **Environment Management:** Python `python-dotenv`

## Key Engineering Features
* **Zero-Hallucination Guardrails:** The Cross-Encoder guarantees the LLM only receives strictly validated context.
* **Semantic Prompt Engineering:** The LLM is instructed to act as a financial analyst, allowing it to correctly map natural language queries (e.g., "fired") to corporate terminology (e.g., "headcount reduction").
* **Graceful Degradation:** Wrapped the external API calls in robust `try-except` blocks to handle Google server rate-limits (503 UNAVAILABLE) without crashing the application. 

## Run Locally
1. Clone the repository: `git clone https://github.com/Cyfox-WarZ/financial-rag-engine.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Google API key to a `.env` file: `GEMINI_API_KEY="your_key"`
4. Launch the app: `streamlit run app.py`
