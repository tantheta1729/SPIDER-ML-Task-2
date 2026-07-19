import os
import pandas as pd
import chromadb
from PyPDF2 import PdfReader

# 1. Initialize the Database
client = chromadb.PersistentClient(path="./medical_db")

# Clear out the old database to avoid duplicate overlapping data
try:
    client.delete_collection("healthcare_docs")
    print("🗑️ Cleared old vector database...")
except Exception:
    pass

collection = client.create_collection("healthcare_docs")
print("✅ Created fresh multi-format collection...")

# 2. Universal Chunking Function
def chunk_text(text, source_name, words_per_chunk=250):
    """Breaks large strings of text into smaller, digestible pieces."""
    chunks = []
    words = str(text).split()
    for i in range(0, len(words), words_per_chunk):
        chunk_str = " ".join(words[i:i + words_per_chunk])
        if chunk_str.strip():  # Ignore empty chunks
            chunks.append({"text": chunk_str, "source": source_name})
    return chunks

# 3. Multi-Format Extraction Engine
def process_all_files(directory_path):
    all_chunks = []
    
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        # --- Handle PDF Files ---
        if filename.endswith(".pdf"):
            print(f"📄 Processing PDF: {filename}")
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + " "
                all_chunks.extend(chunk_text(full_text, filename))
            except Exception as e:
                print(f"⚠️ Failed to read {filename}: {e}")

        # --- Handle CSV Files ---
        elif filename.endswith(".csv"):
            print(f"📊 Processing CSV: {filename}")
            try:
                df = pd.read_csv(filepath)
                # Convert every row into a single string of text
                for index, row in df.iterrows():
                    # Drop empty columns for this row and join with spaces
                    row_text = " ".join([str(val) for val in row.dropna().values])
                    all_chunks.extend(chunk_text(row_text, filename))
            except Exception as e:
                print(f"⚠️ Failed to read {filename}: {e}")

        # --- Handle Text Files ---
        elif filename.endswith(".txt"):
            print(f"📝 Processing TXT: {filename}")
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()
                all_chunks.extend(chunk_text(full_text, filename))
            except Exception as e:
                print(f"⚠️ Failed to read {filename}: {e}")
                
        else:
            print(f"⏭️ Skipping unsupported file type: {filename}")
            
    return all_chunks

# 4. Run the Engine
data_folder = "./data"
print(f"🔍 Scanning '{data_folder}' for documents...\n")
document_chunks = process_all_files(data_folder)

# 5. Load Vectors into ChromaDB
if not document_chunks:
    print("\n❌ No compatible data found in the folder!")
else:
    print(f"\n⚙️ Vectorizing {len(document_chunks)} total text chunks. This might take a moment...")
    
    documents = [chunk["text"] for chunk in document_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in document_chunks]
    ids = [f"chunk_{i}" for i in range(len(document_chunks))]

    # Find the maximum batch size your computer allows (usually around 5461)
    # We will use 5000 just to be safe!
    batch_size = 5000
    
    # Loop through the massive lists and send them in smaller batches
    for i in range(0, len(documents), batch_size):
        print(f"📦 Inserting batch {i} to {i + batch_size}...")
        collection.add(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        
    print("🚀 SUCCESS! All data formats have been compiled and saved to the database.")