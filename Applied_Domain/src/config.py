import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QDRANT_PATH = BASE_DIR / "qdrant_db"

# Model Selection
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
GEN_MODEL_NAME = "gemini-3.5-flash-lite" 

# Safety & Search Thresholds
COLLECTION_NAME = "medical_knowledge"
SIMILARITY_TOP_K = 10
RERANK_TOP_K = 3

# Red Flagging Medical Triage Terms
EMERGENCY_KEYWORDS = [
    "sharp chest pain", "difficulty breathing", "shortness of breath", 
    "sudden numbness", "slurred speech", "severe bleeding", 
    "suicide", "overdose", "unconscious", "heart attack", "stroke"
]