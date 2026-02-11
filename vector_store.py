import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.documents = [] # List of dicts: {'text': ..., 'metadata': ...}

    def build_index(self, documents):
        """
        Builds a FAISS index from the given documents.
        """
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        print("Encoding documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Convert to float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} documents.")

    def save_index(self):
        """Saves the index and documents (metadata) to disk."""
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
            
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Index saved to {VECTOR_STORE_DIR}")

    def load_index(self):
        """Loads the index and documents from disk."""
        if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
            print("Index not found.")
            return False
            
        self.index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Index loaded with {self.index.ntotal} documents.")
        return True

    def search(self, query, top_k=5, newspaper_filter=None, date_filter=None):
        """
        Searches the index for the query.
        Returns top_k matching documents with metadata.
        """
        if self.index is None:
            raise ValueError("Index not loaded or built.")

        query_vector = self.model.encode([query]).astype('float32')
        
        # We retrieve more than top_k initially to allow for filtering
        fetch_k = top_k * 5 
        distances, indices = self.index.search(query_vector, fetch_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            doc = self.documents[idx]
            metadata = doc['metadata']
            
            # Apply filters
            if newspaper_filter and newspaper_filter != "All":
                if metadata['newspaper'] != newspaper_filter:
                    continue
            
            # simple date filtering (exact match for now, could be range)
            if date_filter:
                # Assuming date_filter format YYYYMMDD
                if metadata['date'] != date_filter:
                    continue
            
            results.append(doc)
            if len(results) >= top_k:
                break
                
        return results

if __name__ == "__main__":
    # Test
    from data_loader import load_all_csvs, preprocess_documents
    
    print("Initializing VectorStore...")
    vs = VectorStore()
    
    if not vs.load_index():
        print("Building new index...")
        df = load_all_csvs()
        docs = preprocess_documents(df)
        vs.build_index(docs)
        vs.save_index()
    
    results = vs.search("What happened with the Indus Water Treaty?")
    for res in results:
        print(f"\n--- {res['metadata']['title']} ({res['metadata']['newspaper']}) ---\n{res['text'][:100]}...")
