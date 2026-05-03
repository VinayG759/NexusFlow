"""Embeds code examples and queries using sentence-transformers."""

from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeEmbedder:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        logger.info("CodeEmbedder initialised")

    def load(self):
        if not self.model:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")

    def embed(self, text: str) -> list[float]:
        self.load()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.load()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


code_embedder = CodeEmbedder()
