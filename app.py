from dotenv import load_dotenv
# Load environment variables (for GOOGLE_API_KEY) as early as possible
load_dotenv()

import streamlit as st
import os
import pandas as pd
from datetime import datetime
from data_loader import load_all_csvs, preprocess_documents
from vector_store import VectorStore
from rag_engine import RAGEngine

st.set_page_config(page_title="Pakistani News RAG System", layout="wide")

st.title("🇵🇰 Pakistani News RAG System")
st.markdown("""
Analyze recent news from **The News** and **The Express Tribune**. 
Compare standard LLM answers with RAG-enhanced answers using local document context.
""")

# --- Sidebar: Configuration & Stats ---
st.sidebar.header("Configuration")
google_api_key = st.sidebar.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize Vector Store
@st.cache_resource
def get_vector_store():
    vs = VectorStore()
    if not vs.load_index():
        st.sidebar.warning("Index not found. Building index from CSVs...")
        df = load_all_csvs()
        if not df.empty:
            docs = preprocess_documents(df)
            vs.build_index(docs)
            vs.save_index()
            st.sidebar.success(f"Built index with {len(docs)} documents.")
        else:
            st.sidebar.error("No data found in 'data/' folder.")
    return vs

vs = get_vector_store()
rag_engine = RAGEngine(vs)

st.sidebar.divider()
st.sidebar.subheader("Filters")
newspaper = st.sidebar.selectbox("Newspaper", ["All", "The News", "Tribune"])

# --- Main UI: Search ---
query = st.text_input("Enter your question about Pakistani news:", placeholder="e.g., What are the latest developments in the PSL?", key="query_input")

if query:
    if not google_api_key:
        st.error("Please provide a Google API Key in the sidebar.")
    else:
        with st.spinner("Generating answers..."):
            # Columns for comparison
            col1, col2 = st.columns(2)
            
            # --- RAG Answer ---
            with col1:
                st.subheader("🔍 With RAG")
                rag_answer, sources = rag_engine.generate_rag_answer(query, newspaper_filter=newspaper)
                st.write(rag_answer)
                
                if sources:
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**{i+1}. {doc['metadata']['title']}** ({doc['metadata']['newspaper']} - {doc['metadata']['date']})")
                            st.caption(doc['text'][:300] + "...")
                            st.divider()

            # --- Plain LLM Answer ---
            with col2:
                st.subheader("🤖 Plain LLM (No RAG)")
                plain_answer = rag_engine.generate_plain_answer(query)
                st.write(plain_answer)

# --- Footer ---
st.sidebar.divider()
if vs.index:
    st.sidebar.info(f"Vector Store Status: Ready ({vs.index.ntotal} docs)")
else:
    st.sidebar.error("Vector Store Status: Not initialized")
