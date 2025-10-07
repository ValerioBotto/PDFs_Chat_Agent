import logging
import torch
from typing import List
from langchain_core.documents import Document as LangchainDocument
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from .graph_db import GraphDB #importo la mia classe GraphDB

logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Indexer:
    def __init__(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Stiamo usando: {device}")

            #specifico il dispositivo 'cpu' per assicurarmi che venga caricato lì dato che non ho una gpu dedicata e a volte provava ad accedervi
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
            self.embedding_dimensions = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully on CPU with {self.embedding_dimensions} dimensions.")
        except Exception as e:
            logger.error(f"error loading embedding model: {e}")
            raise

    def generate_embeddings(self, text: str) -> List[float]:
        #genera l'embedding per un dato testo
        return self.embedding_model.encode(text).tolist()

    def index_chunks_to_neo4j(self, filename: str, chunks: List[LangchainDocument]):
        """
        processa una lista di chunk, genera gli embedding e li salva in neo4j.
        """
        if not chunks:
            logger.warning("no chunks provided for indexing.")
            return

        graph_db = None
        try:
            graph_db = GraphDB() #inizializzo cosi la connessione a neo4j
            
            #creo l'indice vettoriale per i chunk se non esiste già
            #uso il label 'Chunk' e la proprietà 'embedding'
            graph_db.create_vector_index(
                index_name="chunk_embeddings_index",
                node_label="Chunk",
                property_name="embedding",
                vector_dimensions=self.embedding_dimensions
            )

            for i, chunk in enumerate(chunks):
                content = chunk.page_content
                metadata = chunk.metadata
                chunk_id = metadata.get("chunk_id", f"{filename}_chunk_{i}") # usa chunk_id dai metadati o genera uno
                
                #genero l'embedding
                embedding = self.generate_embeddings(content)
                
                #salva il chunk e il suo embedding in neo4j, collegandolo al documento
                graph_db.add_chunk_to_neo4j(filename, chunk_id, content, embedding, metadata)
                logger.debug(f"indexed chunk '{chunk_id}' for file '{filename}'.")
            
            logger.info(f"successfully indexed {len(chunks)} chunks for document '{filename}' into neo4j.")

        except Exception as e:
            logger.error(f"error during neo4j indexing for file '{filename}': {e}")
            raise
        finally:
            if graph_db:
                graph_db.close()
