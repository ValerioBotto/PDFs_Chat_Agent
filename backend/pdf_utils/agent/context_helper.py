"""
Context Helper per ricerca cross-document intelligente.
Permette all'agente di cercare informazioni in documenti correlati quando il documento corrente non contiene abbastanza informazioni.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ContextHelper:
    """
    Helper per arricchire il contesto dell'agente con informazioni da documenti correlati.
    Usa una strategia ibrida: topic matching + vector similarity search.
    """
    
    def __init__(self, graph_db, indexer):
        """
        Inizializza il ContextHelper.
        
        Args:
            graph_db: Istanza di GraphDB per query Cypher
            indexer: Istanza di Indexer per embeddings e vector search
        """
        self.graph_db = graph_db
        self.indexer = indexer
        logger.info("ContextHelper inizializzato")
    
    def find_documents_by_topics(self, current_filename: str, max_docs: int = 5) -> List[str]:
        """
        Trova documenti che condividono almeno un topic con il documento corrente.
        
        Args:
            current_filename: Nome del documento corrente
            max_docs: Numero massimo di documenti da ritornare
            
        Returns:
            Lista di filename di documenti correlati (esclude il documento corrente)
        """
        try:
            #query: trova documenti con topic in comune
            query = """
            MATCH (current:Document {filename: $current_filename})-[:HAS_TOPIC]->(topic:Topic)
            MATCH (other:Document)-[:HAS_TOPIC]->(topic)
            WHERE other.filename <> $current_filename
            WITH other, COUNT(DISTINCT topic) as shared_topics
            ORDER BY shared_topics DESC
            LIMIT $max_docs
            RETURN other.filename as filename, shared_topics
            """
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, {
                    "current_filename": current_filename,
                    "max_docs": max_docs
                })
                
                related_docs = []
                for record in result:
                    filename = record.get("filename")
                    shared_topics = record.get("shared_topics", 0)
                    if filename:
                        related_docs.append(filename)
                        logger.debug(f"Documento correlato trovato: {filename} (topic condivisi: {shared_topics})")
                
                logger.info(f"Trovati {len(related_docs)} documenti correlati per topic a '{current_filename}'")
                return related_docs
                
        except Exception as e:
            logger.exception(f"Errore nella ricerca documenti per topic: {e}")
            return []
    
    def search_chunks_in_documents(
        self, 
        query: str, 
        target_filenames: List[str], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Esegue vector search limitata a specifici documenti.
        
        Args:
            query: Testo della domanda da cercare
            target_filenames: Lista di filename dove cercare
            k: Numero di chunk da ritornare per documento
            
        Returns:
            Lista di chunk con metadata (filename, chunk_id, content, score)
        """
        if not target_filenames:
            logger.warning("Nessun documento target specificato per la ricerca")
            return []
        
        try:
            #genera embedding della query
            query_embedding = self.indexer.generate_embeddings(query)
            
            all_chunks = []
            
            #esegui ricerca su ogni documento target
            for filename in target_filenames:
                try:
                    results = self.graph_db.query_vector_index(
                        "chunk_embeddings_index",
                        query_embedding,
                        k=k,
                        filename=filename
                    )
                    
                    for result in results:
                        chunk_data = {
                            "filename": result.get("filename", filename),
                            "chunk_id": result.get("chunk_id", "unknown"),
                            "content": result.get("node_content", ""),
                            "score": result.get("score", 0.0)
                        }
                        all_chunks.append(chunk_data)
                        
                    logger.debug(f"Trovati {len(results)} chunk in '{filename}'")
                    
                except TypeError:
                    #fallback se query_vector_index non supporta il parametro filename
                    logger.warning(f"Impossibile filtrare per filename '{filename}', uso ricerca globale")
                    continue
                except Exception as e:
                    logger.error(f"Errore nella ricerca chunk in '{filename}': {e}")
                    continue
            
            #ordina per score decrescente
            all_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            logger.info(f"Totale chunk trovati da documenti correlati: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.exception(f"Errore nella ricerca vettoriale cross-document: {e}")
            return []
    
    def enrich_context_with_related_docs(
        self, 
        current_filename: str, 
        question: str, 
        max_external_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        METODO PRINCIPALE: Arricchisce il contesto con informazioni da documenti correlati.
        Strategia ibrida:
        1. Trova documenti con topic simili (phase 1: topic filtering)
        2. Esegue vector search su questi documenti (phase 2: similarity search)
        3. Ritorna i chunk più rilevanti con metadata chiare
        
        Args:
            current_filename: Nome del documento corrente
            question: Domanda dell'utente
            max_external_chunks: Numero massimo di chunk esterni da includere
            
        Returns:
            Dizionario con:
            - related_documents: lista di filename usati
            - external_chunks: lista di chunk trovati con metadata
            - summary: stringa formattata per il planner
        """
        logger.info(f"Arricchimento contesto per documento '{current_filename}'")
        
        #fase 1: trova documenti correlati per topic
        related_docs = self.find_documents_by_topics(current_filename, max_docs=5)
        
        if not related_docs:
            logger.info("Nessun documento correlato trovato, ritorno contesto vuoto")
            return {
                "related_documents": [],
                "external_chunks": [],
                "summary": "Nessun documento correlato trovato nel knowledge graph."
            }
        
        #fase 2: vector search sui documenti correlati
        external_chunks = self.search_chunks_in_documents(
            query=question,
            target_filenames=related_docs,
            k=max_external_chunks
        )
        
        #limita al numero richiesto
        external_chunks = external_chunks[:max_external_chunks]
        
        if not external_chunks:
            logger.info("Nessun chunk rilevante trovato nei documenti correlati")
            return {
                "related_documents": related_docs,
                "external_chunks": [],
                "summary": f"Trovati {len(related_docs)} documenti correlati ma nessun chunk rilevante."
            }
        
        #formatta risultati per il planner
        summary_parts = []
        summary_parts.append(f"=== INFORMAZIONI DA DOCUMENTI CORRELATI ===\n")
        summary_parts.append(f"Documenti consultati: {', '.join(related_docs)}\n")
        summary_parts.append(f"Chunk rilevanti trovati: {len(external_chunks)}\n\n")
        
        for idx, chunk in enumerate(external_chunks, start=1):
            source_doc = chunk.get("filename", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            content = chunk.get("content", "")[:500]  #limita lunghezza
            score = chunk.get("score", 0.0)
            
            summary_parts.append(f"[{idx}] Fonte: {source_doc} (chunk: {chunk_id}, score: {score:.3f})\n")
            summary_parts.append(f"    {content}\n\n")
        
        summary = "".join(summary_parts)
        
        result = {
            "related_documents": related_docs,
            "external_chunks": external_chunks,
            "summary": summary
        }
        
        logger.info(f"Contesto arricchito con {len(external_chunks)} chunk da {len(related_docs)} documenti")
        return result
    
    def format_answer_with_sources(
        self, 
        answer: str, 
        used_documents: List[str], 
        current_filename: str
    ) -> str:
        """
        Formatta la risposta finale indicando esplicitamente le fonti utilizzate.
        
        Args:
            answer: Risposta generata dall'agente
            used_documents: Lista di documenti esterni utilizzati
            current_filename: Nome del documento corrente
            
        Returns:
            Risposta formattata con indicazione delle fonti
        """
        if not used_documents:
            return answer
        
        #aggiungi nota sulle fonti alla fine della risposta
        sources_note = "\n\n---\n📚 **Fonti consultate:**\n"
        sources_note += f"- Documento principale: `{current_filename}`\n"
        
        for doc in used_documents:
            sources_note += f"- Documento correlato: `{doc}`\n"
        
        return answer + sources_note
    
    def log_cross_document_usage(
        self,
        answer_id: str,
        external_chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Crea relazioni ENRICHED_BY tra Answer e documenti esterni usati.
        Include metadata sui chunk specifici usati (chunk_ids, scores).
        
        Args:
            answer_id: ID hash della risposta (Answer.id)
            external_chunks: Lista di chunk esterni da enrich_context_with_related_docs
        """
        if not external_chunks:
            logger.debug("Nessun chunk esterno da loggare")
            return
        
        #raggruppa chunk per documento
        docs_data = {}
        for chunk in external_chunks:
            filename = chunk.get("filename")
            chunk_id = chunk.get("chunk_id")
            score = chunk.get("score", 0.0)
            
            if not filename:
                continue
            
            if filename not in docs_data:
                docs_data[filename] = {
                    "chunk_ids": [],
                    "chunk_scores": []
                }
            
            docs_data[filename]["chunk_ids"].append(chunk_id)
            docs_data[filename]["chunk_scores"].append(score)
        
        #crea relazioni ENRICHED_BY per ogni documento esterno
        for filename, data in docs_data.items():
            query = """
            MATCH (a:Answer {id: $answer_id})
            MATCH (d:Document {filename: $filename})
            MERGE (a)-[r:ENRICHED_BY]->(d)
            SET r.chunk_ids = $chunk_ids,
                r.chunk_scores = $chunk_scores,
                r.chunk_count = $chunk_count,
                r.used_at = timestamp()
            """
            
            try:
                with self.graph_db.driver.session() as session:
                    session.run(query, {
                        "answer_id": answer_id,
                        "filename": filename,
                        "chunk_ids": data["chunk_ids"],
                        "chunk_scores": data["chunk_scores"],
                        "chunk_count": len(data["chunk_ids"])
                    })
                logger.info(f"Creata relazione ENRICHED_BY: Answer({answer_id}) -> {filename} ({len(data['chunk_ids'])} chunks)")
            except Exception as e:
                logger.error(f"Errore nella creazione ENRICHED_BY per {filename}: {e}")
    
    def log_primary_chunks_usage(
        self,
        answer_id: str,
        primary_chunks: List[Dict],
        current_filename: str
    ) -> None:
        """
        Crea relazione (Answer)-[:ENRICHED_BY]->(Document) per il documento corrente.
        
        Args:
            answer_id: ID hash della risposta
            primary_chunks: Lista di dict con chunk_id, filename, score
            current_filename: Nome del documento corrente
        """
        if not primary_chunks or not current_filename:
            logger.debug("Nessun chunk primario da loggare")
            return
        
        # Estrai chunk IDs e scores
        chunk_ids = [str(c.get("chunk_id", "?")) for c in primary_chunks]
        chunk_scores = [float(c.get("score", 0.0)) for c in primary_chunks]
        
        if not chunk_ids:
            logger.debug("Nessun chunk_id valido trovato")
            return
        
        #crea relazione ENRICHED_BY per il documento corrente
        query = """
        MATCH (a:Answer {id: $answer_id})
        MATCH (d:Document {filename: $filename})
        MERGE (a)-[r:ENRICHED_BY]->(d)
        SET r.chunk_ids = $chunk_ids,
            r.chunk_scores = $chunk_scores,
            r.chunk_count = $chunk_count,
            r.used_at = timestamp()
        """
        
        try:
            with self.graph_db.driver.session() as session:
                session.run(query, {
                    "answer_id": answer_id,
                    "filename": current_filename,
                    "chunk_ids": chunk_ids,
                    "chunk_scores": chunk_scores,
                    "chunk_count": len(chunk_ids)
                })
            logger.info(f"Creata relazione ENRICHED_BY (primary): Answer({answer_id}) -> {current_filename} ({len(chunk_ids)} chunks)")
        except Exception as e:
            logger.error(f"Errore nella creazione ENRICHED_BY per primary document {current_filename}: {e}")
