"""Vector store using ChromaDB for storing and retrieving code examples."""

import os
from src.rag.embedder import code_embedder
from src.rag.knowledge_base import PERFECT_EXAMPLES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self.db_path = os.path.join(os.path.dirname(__file__), "../../chroma_db")
        logger.info("VectorStore initialised")

    def initialize(self):
        """Initialize ChromaDB and populate with knowledge base."""
        try:
            import chromadb
            logger.info("Initialising ChromaDB at %s", self.db_path)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name="nexusflow_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            count = self.collection.count()
            if count < len(PERFECT_EXAMPLES):
                logger.info("Upserting %d examples into knowledge base", len(PERFECT_EXAMPLES))
                self._populate()
            logger.info("ChromaDB initialized with %d examples", self.collection.count())
        except Exception as e:
            logger.warning("ChromaDB unavailable: %s - RAG will use templates only", e)
            self.client = None
            self.collection = None

    def _populate(self):
        """Add all perfect examples to the vector store."""
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for example in PERFECT_EXAMPLES:
            text = f"{example['description']} {' '.join(example['tags'])} {example['code']}"
            embedding = code_embedder.embed(text)

            ids.append(example['id'])
            documents.append(text[:1000])
            embeddings.append(embedding)
            metadatas.append({
                "category": example['category'],
                "subcategory": example['subcategory'],
                "file_path": example['file_path'],
                "description": example['description'],
                "tags": ",".join(example['tags']),
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Knowledge base populated with %d examples", len(PERFECT_EXAMPLES))

    def retrieve(self, query: str, n_results: int = 5, category: str = None) -> list[dict]:
        """Retrieve most relevant examples for a query."""
        if not self.collection:
            self.initialize()
        if not self.collection:
            return []

        query_embedding = code_embedder.embed(query)

        where = {"category": category} if category else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i, doc_id in enumerate(results['ids'][0]):
            example = next((e for e in PERFECT_EXAMPLES if e['id'] == doc_id), None)
            if example:
                retrieved.append({
                    "id": doc_id,
                    "description": results['metadatas'][0][i]['description'],
                    "file_path": results['metadatas'][0][i]['file_path'],
                    "code": example['code'],
                    "similarity": 1 - results['distances'][0][i],
                    "category": results['metadatas'][0][i]['category'],
                })

        logger.info("Retrieved %d examples for query: %s", len(retrieved), query[:50])
        return retrieved

    def add_example(self, example: dict):
        """Add a new example to the knowledge base at runtime."""
        if not self.collection:
            self.initialize()
        if not self.collection:
            return

        text = f"{example['description']} {example.get('code', '')}"
        embedding = code_embedder.embed(text)

        self.collection.upsert(
            ids=[example['id']],
            documents=[text[:1000]],
            embeddings=[embedding],
            metadatas=[{
                "category": example.get('category', 'general'),
                "subcategory": example.get('subcategory', 'general'),
                "file_path": example.get('file_path', ''),
                "description": example['description'],
                "tags": ",".join(example.get('tags', [])),
            }]
        )


vector_store = VectorStore()
