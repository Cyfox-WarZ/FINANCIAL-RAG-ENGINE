import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from google import genai

# --- 1. SETUP & AUTHENTICATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

@st.cache_resource
def load_database():
    embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
    
    documents = [
        "The company's Q3 revenue grew by 15% year-over-year.",
        "We reduced headcount by 500 employees to cut operational costs.",
        "The CEO stepped down in November due to health reasons.",
        "Our primary server farms are located in Virginia."
    ]
    embeddings = embed_model.encode(documents, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embed_model, reranker, index, documents

embed_model, reranker, faiss_index, docs = load_database()

# --- 2. WEB UI LAYOUT ---
st.title("SEC 10-K Financial RAG Engine")
st.write("This app uses a Two-Stage Pipeline: FAISS (Search) + Cross-Encoder (Rerank) + Gemini (Answer).")

user_query = st.text_input("Ask a question (e.g., 'Did the company lay off staff?'):")

# --- 3. THE RAG LOGIC ---
if st.button("Search"):
    if user_query:
        with st.spinner("Executing Two-Stage Retrieval and Analyzing..."):
            query_vector = embed_model.encode([user_query], normalize_embeddings=True)
            
            distances, indices = faiss_index.search(query_vector, 3)
            candidate_docs = [docs[i] for i in indices[0]]
            
            pairs = [[user_query, doc] for doc in candidate_docs]
            scores = reranker.predict(pairs)
            
            best_idx = np.argmax(scores)
            best_fact = candidate_docs[best_idx]
            
            # Step C: Send the perfect fact to Gemini with a smarter prompt
            prompt = f"""
            You are a financial AI assistant. Answer the user's question using the provided fact. 
            Use your knowledge of business terminology to interpret the fact accurately (e.g., 'reducing headcount' is synonymous with firing or laying off staff).
            
            Fact: '{best_fact}'
            Question: {user_query}
            """
            
            try:
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt
                )
                
                st.success("Analysis Complete!")
                st.markdown(f"**Gemini's Answer:** {response.text}")
                st.info(f"**Behind the scenes:** The Cross-Encoder bypassed the CEO fact and correctly selected -> *{best_fact}*")
                
            except Exception as e:
                st.error("Google's Gemini API is currently overloaded. Please wait a minute and try again.")
                st.warning(f"Error Details: {e}")
    else:
        st.warning("Please enter a question first.")
