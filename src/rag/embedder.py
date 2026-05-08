"""Embeds code examples and queries using sentence-transformers."""

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EMBEDDING_DIM = 384


class CodeEmbedder:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        logger.info("CodeEmbedder initialised")

    def load(self):
        if not self.model:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded")
            except Exception as e:
                logger.warning("Could not load embedding model: %s - using fallback", e)
                self.model = None

    def embed(self, text: str) -> list[float]:
        self.load()
        if not self.model:
            return [0.0] * _EMBEDDING_DIM
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.load()
        if not self.model:
            return [[0.0] * _EMBEDDING_DIM for _ in texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


code_embedder = CodeEmbedder()
