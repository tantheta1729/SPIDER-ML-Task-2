import os
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from flashrank import Ranker, RerankRequest
from src.config import QDRANT_PATH, COLLECTION_NAME, DENSE_MODEL_NAME, SPARSE_MODEL_NAME, RERANK_MODEL_NAME

class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retrieval Engine...")
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        self.reranker = Ranker(model_name=RERANK_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = 10, rerank_k: int = 3):
        # 1. Generate Query Embeddings
        dense_query = list(self.dense_model.embed([query]))[0]
        sparse_query = list(self.sparse_model.embed([query]))[0]

        # 2. Hybrid RRF Search in Qdrant
        search_results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using="dense",
                    limit=top_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist()
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        raw_docs = []
        for pt in search_results.points:
            raw_docs.append({
                "id": pt.id,
                "text": pt.payload["content"],
                "meta": pt.payload
            })

        if not raw_docs:
            return []

        # 3. FlashRank Cross-Encoder Reranking
        rerank_req = RerankRequest(query=query, passages=raw_docs)
        reranked_results = self.reranker.rerank(rerank_req)

        # 4. Format and Return Top Reranked Evidence
        final_docs = []
        for item in reranked_results[:rerank_k]:
            final_docs.append({
                "content": item["text"],
                "score": round(float(item["score"]), 4),
                "source": item["meta"].get("source", "Unknown"),
                "topic": item["meta"].get("topic", "General")
            })

        return final_docs

    def close(self):
        """Explicitly closes the Qdrant connection before Python shuts down."""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()


if __name__ == "__main__":
    retriever = HybridRetriever()
    try:
        test_query = "What are the symptoms and home remedies for common cold?"
        print(f"\n🔍 Testing Search Query: '{test_query}'\n")
        results = retriever.retrieve(test_query)

        for i, res in enumerate(results, 1):
            print(f"--- Result {i} [Relevance Score: {res['score']}] ---")
            print(f"Source: {res['source']} ({res['topic']})")
            print(f"Content snippet: {res['content'][:200]}...\n")
    finally:
        retriever.close()