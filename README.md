# 🇵🇰 Pakistani News RAG System

A professional Retrieval-Augmented Generation (RAG) system designed to analyze and query news from major Pakistani publications (**The News** and **The Express Tribune**).

This system provides a side-by-side comparison between standard LLM responses and RAG-enhanced responses, highlighting the importance of local context in news analysis.

## 🚀 Features

- **Side-by-Side Comparison**: Compare "With RAG" vs "Plain LLM" answers to see how local data improves accuracy.
- **Source Transparency**: View exact snippets from the news articles used to generate the RAG answer.
- **Filtering**: Filter queries by specific newspaper sources.
- **Local Vector Store**: Uses FAISS for efficient similarity search over local news data.
- **Modern UI**: Built with Streamlit for a clean, interactive experience.

## 🛠️ Technology Stack

- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **UI Framework**: Streamlit
- **Data Handling**: Pandas

## 📋 Prerequisites

- Python 3.8+
- A Google API Key (Get it at [Google AI Studio](https://aistudio.google.com/app/apikey))

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KhurramRukhsar/pakistani-news-rag.git
   cd pakistani-news-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   Create a `.env` file in the root directory and add your key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Prepare Data**:
   Ensure your news CSV files are in the `data/` folder. The system will automatically build the vector index on the first run.

## 🏃 How to Run

Start the Streamlit application:

```bash
streamlit run app.py
```
<img width="1485" height="977" alt="image" src="https://github.com/user-attachments/assets/45b11dfc-b7c5-4535-9e8d-98a6ab68cb55" />


## 📂 Project Structure

- `app.py`: The Streamlit frontend and main entry point.
- `rag_engine.py`: Logic for retrieval and generation using Gemini.
- `vector_store.py`: FAISS index management and document encoding.
- `data_loader.py`: Utilities for loading and preprocessing news CSVs.
- `.gitignore`: Ensures sensitive keys and bulky local data stay private.
