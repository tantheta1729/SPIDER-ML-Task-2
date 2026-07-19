import os
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup our structural text-cutter tool
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # A bit bigger for rich context
    chunk_overlap=150,
    length_function=len
)

all_processed_chunks = []

# --- PART 1: INGEST THE STRUCTURED MEDQUAD DATA ---
def ingest_medquad(csv_path):
    print(f"📖 Loading structured MedQuAD data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found! Skipping MedQuAD.")
        return
        
    # Read the dataset using Pandas
    df = pd.read_csv(csv_path) 
    
    # Loop through every row in the dataset
    for idx, row in df.iterrows():
        question = row.get('Question', '')
        answer = row.get('Answer', '')
        
        # Merge the question and answer together so context isn't lost
        combined_text = f"Question: {question}\nAnswer: {answer}"
        
        # Split it into chunks if it's very long
        chunks = text_splitter.split_text(combined_text)
        
        for chunk in chunks:
            all_processed_chunks.append({
                "text": chunk,
                "source": "MedQuAD Dataset" # Explicit tag for citations later!
            })

# --- PART 2: INGEST UNSTRUCTURED MANUALS (PDFs) ---
def ingest_pdf_guidelines(pdf_path, source_name):
    print(f"📄 Scraping text from PDF manual: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"⚠️ {pdf_path} not found! Skipping.")
        return
        
    reader = PdfReader(pdf_path)
    full_text = ""
    
    # Extract raw text page by page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    # Chop up the massive document text
    chunks = text_splitter.split_text(full_text)
    
    for chunk in chunks:
        all_processed_chunks.append({
            "text": chunk,
            "source": source_name # Tag it (e.g., "CDC Heart Guidelines")
        })

# --- EXECUTE THE PIPELINE ---
if __name__ == "__main__":
    # 1. Process MedQuAD rows
    ingest_medquad("data/medquad.csv")
    
    # 2. Process an example external rule manual
    ingest_pdf_guidelines("data/cdc_heart_guide.pdf", "CDC Heart Recommendations")
    
    print("\n--- INGESTION COMPLETE ---")
    print(f"Total baby-sized chunks ready for database: {len(all_processed_chunks)}")
    
    # Sneak peek at the first document structure
    if all_processed_chunks:
        print("\nSample Chunk Structure:")
        print(f"SOURCE TAG: {all_processed_chunks[0]['source']}")
        print(f"TEXT CONTENT:\n{all_processed_chunks[0]['text'][:300]}...")