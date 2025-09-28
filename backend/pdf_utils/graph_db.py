# backend/pdf_utils/graph_db.py

from neo4j import GraphDatabase, exceptions # importa anche exceptions
import logging
import os
import hashlib
import datetime
from typing import List, Dict, Any

# setup logging per debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphDB:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=None):
        if not password:
            password = os.getenv("NEO4J_PASSWORD", "logogramma") 
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.create_indexes() # Chiamata a create_indexes() qui
            logger.info("connessione a neo4j stabilita con successo")
        except Exception as e:
            logger.error(f"errore durante la connessione a neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("connessione a neo4j chiusa")
    
    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result

    def create_indexes(self):
        # crea indici e vincoli per migliorare le performance
        
        # Lista di query per vincoli/indici che potrebbero causare errori se ricreati
        index_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:PERSONA) REQUIRE e.name IS UNIQUE", # per esempio, per un tipo di entità specifico
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.section)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
        ]

        with self.driver.session() as session:
            for query in index_queries:
                try:
                    session.run(query)
                    logger.debug(f"query '{query}' executed successfully.")
                except exceptions.ClientError as e:
                    # cattura solo l'errore specifico di indice/vincolo già esistente
                    if "IndexAlreadyExists" in e.message or "ConstraintAlreadyExists" in e.message:
                        logger.info(f"skipping: {e.message} (likely already exists).")
                    else:
                        logger.error(f"error executing index query '{query}': {e}")
                        raise # rilancia altri tipi di errori
                except Exception as e:
                    logger.error(f"unexpected error executing index query '{query}': {e}")
                    raise
        
        logger.info("indici e vincoli neo4j verificati/creati.")
    
    # ... il resto del file rimane invariato ...
    def create_user(self, user_id):
        query = """
        MERGE (u:User {id: $user_id})
        ON CREATE SET u.created_at = datetime(), u.last_activity = datetime()
        ON MATCH SET u.last_activity = datetime()
        RETURN u
        """
        return self.run_query(query, {"user_id": user_id})
    
    def create_document(self, filename: str, title: str = None, file_size: int = None, sections_count: int = None):
        query = """
        MERGE (d:Document {filename: $filename})
        ON CREATE SET 
            d.created_at = datetime(), 
            d.title = COALESCE($title, $filename)
        ON MATCH SET d.last_updated = datetime()
        RETURN d
        """
        return self.run_query(query, {"filename": filename, "title": title})
    
    def link_user_to_document(self, user_id: str, filename: str):
        query = """
        MATCH (u:User {id: $user_id})
        MATCH (d:Document {filename: $filename})
        MERGE (u)-[r:ACCESSED]->(d)
        ON CREATE SET r.first_access = datetime(), r.last_access = datetime()
        ON MATCH SET r.last_access = datetime()
        RETURN u, d, r
        """
        return self.run_query(query, {"user_id": user_id, "filename": filename})
        
    def add_chunk_to_neo4j(self, filename: str, chunk_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]):
        query = """
        MATCH (d:Document {filename: $filename})
        MERGE (c:Chunk {chunk_id: $chunk_id})
        ON CREATE SET 
            c.content = $content, 
            c.embedding = $embedding, 
            c.section = $section, 
            c.source = $filename,
            c.created_at = datetime()
        ON MATCH SET 
            c.content = $content, 
            c.embedding = $embedding, 
            c.section = $section, 
            c.source = $filename,
            c.last_updated = datetime()
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c
        """
        parameters = {
            "filename": filename,
            "chunk_id": chunk_id,
            "content": content,
            "embedding": embedding,
            "section": metadata.get("section", "unspecified"),
            "source": filename
        }
        return self.run_query(query, parameters)
        
    def create_vector_index(self, index_name: str, node_label: str, property_name: str, vector_dimensions: int):
        """
        Creates a vector index for similarity search using the vector-2.0 provider
        """
        
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS 
        FOR (n:{node_label})
        ON (n.{property_name})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            self.run_query(query)
            logger.info(f"Vector index '{index_name}' created successfully")
        except Exception as e:
            logger.error(f"Error creating vector index '{index_name}': {e}")
            logger.error(f"Query that generated the error: \n{query}")
            raise
    
    def add_topic_to_neo4j(self, topic_name: str, filename: str = None):
        """
        crea un nodo Topic e lo collega a un Documento (se filename fornito).
        """
        query = """
        MERGE (t:Topic {name: $topic_name})
        ON CREATE SET t.created_at = datetime()
        ON MATCH SET t.last_updated = datetime()
        """
        params = {"topic_name": topic_name}

        if filename:
            query += """
            WITH t
            MATCH (d:Document {filename: $filename})
            MERGE (d)-[:HAS_TOPIC]->(t)
            """
            params["filename"] = filename
        
        return self.run_query(query, params)

    def add_entity_to_neo4j(self, entity_type: str, entity_name: str, filename: str = None, chunk_id: str = None):
        """
        crea un nodo Entity con una proprietà 'type' e lo collega a un Documento/Chunk.
        """
        # Creiamo un nodo generico :Entity e usiamo la proprietà 'type' per il tipo specifico
        query = f"""
        MERGE (e:Entity {{name: $entity_name, type: $entity_type_prop}})
        ON CREATE SET e.created_at = datetime()
        ON MATCH SET e.last_updated = datetime()
        """
        # Usiamo entity_type_prop per evitare conflitti con la variabile entity_type
        params = {"entity_name": entity_name, "entity_type_prop": entity_type} 

        if filename:
            query += """
            WITH e
            MATCH (d:Document {filename: $filename})
            MERGE (d)-[:HAS_ENTITY]->(e)
            """
            params["filename"] = filename
        
        if chunk_id: # Colleghiamo solo se il chunk esiste e ha senso
            # Se un'entità è estratta dal testo completo, il collegamento ai chunk potrebbe essere eccessivo
            # o gestire solo l'entità primaria. Valutare se questo collegamento è sempre desiderabile.
            query += """
            WITH e
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:CONTAINS_ENTITY]->(e)
            """
            params["chunk_id"] = chunk_id

        return self.run_query(query, params)
        
    def link_topics_to_entity(self, entity_name: str, topic_names: List[str]):
        """
        Collega un'entità a uno o più topic a cui è correlata.
        """
        if not topic_names:
            return

        query = """
        MATCH (e:ENTITY {name: $entity_name})
        """
        for i, topic in enumerate(topic_names):
            query += f"""
            MERGE (t{i}:Topic {{name: $topic_{i}}})
            MERGE (e)-[:RELATED_TO]->(t{i})
            """
            params = {"entity_name": entity_name}
            params.update({f"topic_{i}": t for i, t in enumerate(topic_names)})
            
        return self.run_query(query, params)
    
    def query_vector_index(self, index_name: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        esegue una ricerca di similarità vettoriale sull'indice neo4j.
        restituisce una lista di dizionari contenenti il contenuto del nodo
        e il punteggio di similarità.
        """
        #la procedura db.index.vector.queryNodes è il modo per interrogare indici vettoriali
        #in Neo4j.
        query = f"""
        CALL db.index.vector.queryNodes('{index_name}', {k}, $query_embedding)
        YIELD node, score
        RETURN node.content AS node_content, score, node.chunk_id AS chunk_id, node.section AS section
        """
        parameters = {"query_embedding": query_embedding}
        
        results = []
        try:
            with self.driver.session() as session:
                for record in session.run(query, parameters):
                    results.append({
                        "node_content": record["node_content"],
                        "score": record["score"],
                        "chunk_id": record["chunk_id"],
                        "section": record["section"]
                    })
            logger.debug(f"ricerca vettoriale su '{index_name}' ha trovato {len(results)} risultati.")
            return results
        except Exception as e:
            logger.error(f"errore durante la query dell'indice vettoriale '{index_name}': {e}", exc_info=True)
            raise    