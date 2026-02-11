import os
import pandas as pd
import glob
import re

DATA_DIR = "data"

def load_all_csvs(data_dir=DATA_DIR):
    """
    Loads all CSV files from the data directory.
    Returns a unified DataFrame with 'newspaper' and 'date' metadata.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []

    for filename in all_files:
        basename = os.path.basename(filename)
        
        # Extract newspaper and date from filename
        # Pattern: {newspaper}_{date}.csv
        # date format: YYYYMMDD
        
        newspaper = ""
        date_str = ""
        
        if basename.startswith("the_news_"):
            newspaper = "The News"
            match = re.search(r"the_news_(\d{8})\.csv", basename)
            if match:
                date_str = match.group(1)
        elif basename.startswith("tribune_"):
            newspaper = "Tribune"
            match = re.search(r"tribune_(\d{8})\.csv", basename)
            if match:
                date_str = match.group(1)
        
        try:
            df = pd.read_csv(filename)
            df['newspaper'] = newspaper
            df['date'] = date_str
            df['source_file'] = basename
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

def preprocess_documents(df):
    """
    Preprocesses the DataFrame into a list of document dictionaries.
    Each document has 'text' (title + content) and 'metadata'.
    """
    documents = []
    
    # Ensure required columns exist (handling potential missing columns in some files)
    required_cols = ['title', 'content']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Replace NaN with empty string
    df = df.fillna("")

    for _, row in df.iterrows():
        title = str(row['title']).strip()
        content = str(row['content']).strip()
        
        # Skip if title or content is missing or indicates error
        if not title or not content or "Title not found" in title:
            continue

        # Basic cleaning - remove copyright footer if present (common in scraped data)
        # This is a heuristic; adjust if needed based on actual data inspection
        copyright_pattern = r"Copyright © \d{4}\. The News International.*"
        content = re.sub(copyright_pattern, "", content, flags=re.IGNORECASE).strip()

        # Combine Title and Content for the embedding text, as requested
        # We'll use a format that clearly separates them for the model
        combined_text = f"Title: {title}\nContent: {content}"
        
        metadata = {
            "newspaper": row['newspaper'],
            "date": row['date'],
            "title": title,
            "link": row.get('link', ''), # Tribune might not have link, default to empty
            "sentiment": row.get('title_sentiment', 'UNKNOWN'),
            "source_file": row['source_file']
        }

        documents.append({
            "text": combined_text,
            "metadata": metadata
        })

    return documents

if __name__ == "__main__":
    # Test run
    print("Loading data...")
    df = load_all_csvs()
    print(f"Loaded {len(df)} rows.")
    
    docs = preprocess_documents(df)
    print(f"Processed {len(docs)} documents.")
    if docs:
        print("Sample document:")
        print(docs[0])
