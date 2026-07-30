import os
import glob
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from src.config import QDRANT_PATH, COLLECTION_NAME, DENSE_MODEL_NAME, SPARSE_MODEL_NAME, DATA_DIR
class HybridIngestor:
    def __init__(self):
        print("Initializing Qdrant client and embedding models...")
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def setup_collection(self):
        """Creates Qdrant collection configured for hybrid dense + sparse vectors."""
        if self.client.collection_exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists. Recreating it...")
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,  # bge-small-en-v1.5 vector dimension
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
        print(f"Collection '{COLLECTION_NAME}' setup complete.\n")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Splits Markdown documents into chunks preserving paragraph structure."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current_chunk) + len(p) + 2 <= chunk_size:
                current_chunk += ("\n\n" + p) if current_chunk else p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(p) > chunk_size:
                    for i in range(0, len(p), chunk_size - overlap):
                        chunks.append(p[i:i + chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def load_documents(self):
        documents = []
        payloads = []
        raw_dir = DATA_DIR / "raw"

        # 1. Load MedQuAD CSV
        csv_file = raw_dir / "medquad.csv"
        if csv_file.exists():
            print(f"Loading CSV: {csv_file.name}...")
            df = pd.read_csv(csv_file).dropna(subset=["question", "answer"])
            for _, row in df.iterrows():
                text = f"Question: {row['question']}\nAnswer: {row['answer']}"
                documents.append(text)
                payloads.append({
                    "source": "MedQuAD",
                    "topic": str(row.get("focus_area", "General")),
                    "content": text
                })
            print(f"Loaded {len(df)} MedQuAD pairs.")
        else:
            print(f"⚠️ Warning: {csv_file} not found! Skipping MedQuAD CSV.")

        # 2. Load all Markdown files
        md_files = glob.glob(str(raw_dir / "*.md"))
        print(f"Found {len(md_files)} Markdown guideline files.")

        for md_path in md_files:
            file_name = Path(md_path).stem
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self.chunk_text(content)
            for chunk in chunks:
                documents.append(chunk)
                payloads.append({
                    "source": file_name,
                    "topic": file_name.replace("-", " ").title(),
                    "content": chunk
                })

        print(f"\nTotal document chunks prepared for indexing: {len(documents)}\n")
        return documents, payloads

    def ingest_all(self):
        documents, payloads = self.load_documents()

        if not documents:
            print("❌ No documents found to ingest! Check your data/raw folder.")
            return

        print("Generating hybrid embeddings (Dense + Sparse BM25)...")
        dense_vectors = list(self.dense_model.embed(documents))
        sparse_vectors = list(self.sparse_model.embed(documents))

        points = []
        for idx in range(len(documents)):
            points.append(
                models.PointStruct(
                    id=idx,
                    vector={
                        "dense": dense_vectors[idx].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vectors[idx].indices.tolist(),
                            values=sparse_vectors[idx].values.tolist()
                        )
                    },
                    payload=payloads[idx]
                )
            )

        # Inserting batches into Qdrant
        batch_size = 200
        print("Upserting vectors into local Qdrant database...")
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + batch_size]
            )
            print(f"Indexed batch {i} to {min(i + batch_size, len(points))}")

        print("\n✅ Ingestion Complete! All documents & guidelines indexed into Qdrant.")

if __name__ == "__main__":
    ingestor = HybridIngestor()
    ingestor.setup_collection()
    ingestor.ingest_all()