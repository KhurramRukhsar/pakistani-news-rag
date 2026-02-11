import os
import google.generativeai as genai
from vector_store import VectorStore

MODEL_NAME = "gemini-flash-latest"

class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # Configure Gemini API if not already configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            
        self.model = genai.GenerativeModel(MODEL_NAME)

    def generate_rag_answer(self, query, newspaper_filter="All", date_filter=None):
        """
        Generates an answer using RAG (Retrieval-Augmented Generation).
        Returns: answer_text, source_documents (list of dicts)
        """
        # 1. Retrieve context
        print(f"Retrieving for query: {query}, filter: {newspaper_filter}")
        
        # If vector store is empty, try to load or build
        if self.vector_store.index is None:
             print("Index not loaded. Attempting to load...")
             if not self.vector_store.load_index():
                 return "Error: Vector store not initialized. Please build the index first.", []

        retrieved_docs = self.vector_store.search(
            query, 
            top_k=5, 
            newspaper_filter=newspaper_filter if newspaper_filter != "All" else None,
            date_filter=date_filter
        )
        
        if not retrieved_docs:
            return "No relevant documents found to answer your question.", []

        # 2. Construct Prompt
        context_str = "\n\n".join(
            [f"--- Document {i+1} ---\n{doc['text']}" for i, doc in enumerate(retrieved_docs)]
        )
        
        system_instruction = (
            "You are a helpful assistant for a RAG system analyzing Pakistani news.\n"
            "Answer the user's question based ONLY on the provided context.\n"
            "If the answer is not in the context, say 'I cannot answer this based on the available news articles.'\n"
            "Cite the newspaper if relevant."
        )
        
        prompt = f"{system_instruction}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        # 3. Generate Answer
        try:
            response = self.model.generate_content(prompt)
            return response.text, retrieved_docs
        except Exception as e:
            return f"Error generating answer: {e}", []

    def generate_plain_answer(self, query):
        """
        Generates an answer using the LLM's internal knowledge only (No RAG).
        """
        prompt = f"Answer the following question based on your general knowledge.\n\nQuestion: {query}\n\nAnswer:"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"

if __name__ == "__main__":
    # Test
    if not GOOGLE_API_KEY:
        print("Skipping RAG Engine test: GOOGLE_API_KEY not set.")
    else:
        from vector_store import VectorStore
        vs = VectorStore()
        if vs.load_index():
            rag = RAGEngine(vs)
            ans, sources = rag.generate_rag_answer("Who won the PSL final?")
            print("RAG Answer:\n", ans)
            print("\nSources:", [s['metadata']['title'] for s in sources])
            
            plain_ans = rag.generate_plain_answer("Who won the PSL final?")
            print("\nPlain Answer:\n", plain_ans)
