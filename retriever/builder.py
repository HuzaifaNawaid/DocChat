from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever   # moved here in langchain v1.0+
from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
import logging
import uuid

logger = logging.getLogger(__name__)


class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with HuggingFace embeddings + Pinecone client."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL
        )

        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create the Pinecone index once if it doesn't already exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index '{settings.PINECONE_INDEX_NAME}'...")
            self.pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )
            logger.info("Pinecone index created.")

    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 (local) and Pinecone vector search."""
        try:
            # Use a fresh namespace per document set so different uploads
            # don't mix results together inside the same Pinecone index.
            namespace = str(uuid.uuid4())

            # Create Pinecone vector store from the document chunks
            vector_store = PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                index_name=settings.PINECONE_INDEX_NAME,
                namespace=namespace,
            )
            logger.info("Pinecone vector store created successfully.")

            # Create BM25 retriever (local, in-memory — unaffected by vector DB choice)
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = settings.VECTOR_SEARCH_K
            logger.info("BM25 retriever created successfully.")

            # Create vector-based retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created successfully.")

            # Combine retrievers into a hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise