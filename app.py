import os
import streamlit as st
import numpy as np
import faiss
import pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

# --- PAGE CONFIGURATION ---
# MUST be the first Streamlit command
st.set_page_config(page_title="SEC 10-K RAG Engine", page_icon="📈", layout="wide")

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- UI: SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Architecture")
    st.markdown("""
    **Two-Stage Pipeline:**
    1. **Extraction:** PyMuPDF
    2. **Chunking:** LangChain (1200 chars)
    3. **Vector DB:** FAISS
    4. **Recall:** BGE-Small (Bi-Encoder)
    5. **Precision:** MS-MARCO (Cross-Encoder)
    6. **Synthesis:** Google Gemini
    """)
    st.caption("Built to analyze Alphabet's Form 10-K with zero hallucinations.")

# --- UI: MAIN HEADER ---
st.title("📈 SEC 10-K Financial RAG Engine")
st.markdown("Ask natural language questions about Alphabet's corporate filings. The engine retrieves exact excerpts and synthesizes the data.")
st.divider()

# --- 1. LOAD MODELS & FAISS DATABASE ---
@st.cache_resource
def load_database():
    embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Read PDF text safely with PyMuPDF
    pdf_filename = "GOOG-10-K-2025.pdf"
    raw_text = ""
    
    if os.path.exists(pdf_filename):
        try:
            doc = pymupdf.open(pdf_filename)
            for page in doc:
                text = page.get_text()
                if text:
                    raw_text += text + "\n"
            doc.close()
        except Exception as e:
            st.warning(f"Could not read PDF completely: {e}")

    # Fallback to sample text if PDF is missing or unreadable
    if not raw_text.strip():
        raw_text = (
            "The company's Q3 revenue grew by 15% year-over-year.\n"
            "The CEO stepped down in November due to health reasons.\n"
            "We reduced headcount by 500 employees to cut operational costs.\n"
            "Our primary server farms are located in Virginia."
        )

    # Chunk text into larger overlapping segments to preserve financial tables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = text_splitter.split_text(raw_text)

    # Build FAISS vector database
    embeddings = embed_model.encode(documents, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embed_model, reranker, index, documents

embed_model, reranker, faiss_index, docs = load_database()

# --- 2. USER INTERFACE ---
user_query = st.text_input("Ask a question (e.g., 'What was Google Cloud revenue?'):")

# --- 3. THE RAG LOGIC ---
if st.button("Search", type="primary"):
    if user_query:
        with st.spinner("Executing Two-Stage Retrieval and Analyzing..."):
            # Stage 1: FAISS Search (retrieve top 10 candidates)
            query_vector = embed_model.encode([user_query], normalize_embeddings=True)
            k = min(10, len(docs))
            distances, indices = faiss_index.search(query_vector, k)
            candidate_docs = [docs[i] for i in indices[0]]

            # Stage 2: Cross-Encoder High-Precision Rerank
            pairs = [[user_query, doc] for doc in candidate_docs]
            scores = reranker.predict(pairs)
            
            # Select top 3 reranked chunks to prevent cutoffs in tables and metrics
            top_ranked_indices = np.argsort(scores)[::-1][:3]
            top_facts = [candidate_docs[i] for i in top_ranked_indices]
            combined_context = "\n---\n".join(top_facts)

            # Stage 3: LLM Synthesis with Domain-Specific Prompting
            prompt = f"""
            You are a financial AI assistant. Answer the user's question accurately using the provided financial report excerpts.
            Use your knowledge of business terminology to interpret the facts accurately (e.g., 'reducing headcount' is synonymous with firing or laying off staff).
            Synthesize information across multiple excerpts if necessary.

            Excerpts:
            {combined_context}

            Question: {user_query}
            """

            try:
                client = genai.Client(api_key=api_key)
                
                # Primary lightweight model with high-demand fallback
                try:
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )
                except Exception:
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )
                
                st.success("Analysis Complete!")
                st.write(f"**Gemini's Answer:** {response.text}")
                
                with st.expander("🔍 View Behind-the-Scenes (Extracted Excerpts)"):
                    st.write("The Cross-Encoder selected these exact chunks from the PDF:")
                    # st.info creates the blue background; the triple backticks preserve table spacing
                    st.info(f"```text\n{combined_context}\n```")
                
            except Exception as e:
                st.error("Google's Gemini API is currently unavailable. Please wait a moment and try again.")
                st.warning(f"Error Details: {e}")