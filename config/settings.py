import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Cache (unchanged from original) ---
    CACHE_DIR = "document_cache"
    CACHE_EXPIRE_DAYS = 7

    # --- Retrieval (unchanged from original) ---
    VECTOR_SEARCH_K = 3
    HYBRID_RETRIEVER_WEIGHTS = [0.3, 0.7]  # [BM25_weight, Vector_weight]

    # --- Groq LLM config ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Fast + cheap model for relevance checking & first-pass research
    GROQ_MODEL_FAST = "openai/gpt-oss-20b"
    # Stronger model for verification (accuracy matters most here)
    GROQ_MODEL_STRONG = "openai/gpt-oss-120b"

    # --- Hugging Face embeddings config ---
    HF_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

    # --- Pinecone config ---
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docchat-index")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_DIMENSION = 768  # matches BAAI/bge-base-en-v1.5 output dim


settings = Settings()