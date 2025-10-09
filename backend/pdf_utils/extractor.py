# backend/pdf_utils/extractor.py

import logging
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_together import ChatTogether
from dotenv import load_dotenv
import os
import json

logger = logging.getLogger(__name__)

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class Extractor:
    def __init__(self, llm_model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        if not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY non è impostata nell'ambiente.")
        
        self.llm = ChatTogether(
            model=llm_model_name,
            temperature=0.1, #temperatura bassa per risposte più consistenti e fattuali
            together_api_key=TOGETHER_API_KEY,
            max_tokens=512
        )
        logger.info(f"LLM initialized for extraction with model: {llm_model_name}")

        #prompt per estrarre topic
        self.topic_extraction_prompt = PromptTemplate(
            template="""
            Analizza il seguente testo e identifica un massimo di {max_topics} concetti principali (topic) che descrivono il contenuto. 
            Sii conciso e rispondi solo con una lista di parole chiave separate da virgole, senza spiegazioni o prefissi. 
            NON includere termini generici come "documento", "testo", "informazioni".
            NON aggiungere altro testo, come ad esempio "Certo! ecco a te una lista di topic...".
            Se non segui esattamente il formato richiesto, verrai penalizzato.
            
            Testo:
            ---
            {text}
            ---
            
            Topics (max {max_topics}, comma-separated):""",
            input_variables=["text", "max_topics"]
        )

        #definizione dei tipi di entità che vogliamo estrarre, scritti in inglese

        self.selected_entity_types = [
            "Person", "Organization", "Location", "GPE", "DateTime", "Product",
            "Event", "URL", "Email", "Currency", "Percentage", "Address",
            "CountryRegion", "Continent", "CulturalEvent", "NaturalEvent"
        ]
        
        #genera il prompt per l'estrazione di entità in base ai tipi selezionati
        entity_format_guide = "\n".join([f"- {ent_type}: [Lista di entità di questo tipo]" for ent_type in self.selected_entity_types])
        self.entity_extraction_prompt = PromptTemplate(
            template=f"""
            Estrai dal seguente testo entità dei seguenti tipi:
            {", ".join(self.selected_entity_types)}.
            
            Formato di risposta (un tipo per riga, entità separate da virgole. Se un tipo non è presente, ometti la sua riga):
            {entity_format_guide}
            
            NON includere altro testo o spiegazioni. Se non segui esattamente il formato richiesto, verrai penalizzato.
            
            Testo:
            ---
            {{text}}
            ---
            """,
            input_variables=["text"]
        )
        
        # cache in memoria {synonym -> canonical} popolata da Neo4j/LLM
        self.synonym_map: Dict[str, str] = {}
        logger.info("synonym map initialized (empty cache; dynamic via Neo4j/LLM).")

        # Prompt per indurre forma canonica + sinonimi (risposta SOLO JSON)
        self.synonym_induction_prompt = PromptTemplate(
            template=(
                """
                Dato un termine di topic e un estratto del documento, proponi una normalizzazione:
                - canonical: forma canonica breve (lemma/base) del termine in italiano quando possibile
                - synonyms: al massimo 4 sinonimi/varianti e traduzioni rilevanti (IT/EN), includendo il termine originale
                Rispondi SOLO con un oggetto JSON valido, senza backtick e senza testo extra:
                {{"canonical": "...", "synonyms": ["...", "..."]}}

                Termine: {term}
                Contesto (troncato): {context}
                """
            ).strip(),
            input_variables=["term", "context"],
        )

        # riferimenti opzionali a Neo4j e filename attivo
        self.graph_db = None
        self.active_filename = None

    def attach_graph(self, graph_db, active_filename: str | None = None):
        """Collega Neo4j e carica la mappa sinonimi iniziale.

        Strategia di preload:
        - Carica prima i sinonimi globali (tutti i documenti)
        - Se viene passato un filename attivo, carica anche i sinonimi specifici del documento
          sovrascrivendo eventuali collisioni (priorità al documento corrente).
        """
        self.graph_db = graph_db
        self.active_filename = active_filename
        try:
            # preload global synonyms
            global_loaded = graph_db.load_synonym_map(None)
            if isinstance(global_loaded, dict):
                self.synonym_map.update(global_loaded)
            # then document-scoped to override globals when needed
            doc_loaded = {}
            if active_filename:
                doc_loaded = graph_db.load_synonym_map(active_filename)
                if isinstance(doc_loaded, dict):
                    self.synonym_map.update(doc_loaded)
            logger.info(
                f"loaded synonyms from Neo4j -> global={len(global_loaded) if isinstance(global_loaded, dict) else 0}, "
                f"doc={len(doc_loaded) if isinstance(doc_loaded, dict) else 0}, total_cache={len(self.synonym_map)}"
            )
        except Exception:
            logger.exception("failed loading synonyms from Neo4j")

    def _resolve_canonical_via_graph(self, term: str) -> str | None:
        t = term.lower().strip()
        # prima dalla mappa locale
        if t in self.synonym_map:
            return self.synonym_map[t]
        # poi dal grafo
        if self.graph_db:
            try:
                canon = self.graph_db.get_canonical_topic(t)
                if canon:
                    canon_l = canon.lower().strip()
                    self.synonym_map[t] = canon_l
                    return canon_l
            except Exception:
                logger.exception("get_canonical_topic error")
        return None

    def _call_llm(self, prompt_template: PromptTemplate, **kwargs) -> str:
        """helper function to call LLM with a given prompt and parse response."""
        chain = prompt_template | self.llm
        response = chain.invoke(kwargs)
        # estraiamo il contenuto dal wrapper di langchain
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        return str(response)

    def extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """estrae topic da un testo usando l'LLM e li normalizza (dinamicamente via Neo4j quando disponibile)."""
        raw_topics_str = self._call_llm(self.topic_extraction_prompt, text=text, max_topics=max_topics)

        # pulisce e spacchetta i topic
        topics = [t.strip().lower() for t in raw_topics_str.split(',') if t.strip()]

        # filtra termini generici che l'LLM potrebbe comunque aver incluso
        generic_terms = {"documento", "testo", "informazioni", "contenuto", "dati", "elemento", "parte", "sezione", "pdf", "file", "articolo"}
        topics = [t for t in topics if t not in generic_terms]

        normalized: List[str] = []
        for t in topics:
            canon = self._resolve_canonical_via_graph(t)
            if not canon:
                # fallback alla mappa locale
                canon = self.synonym_map.get(t)
            if not canon:
                # nuova conoscenza: chiedi all'LLM e persisti
                canon = self._dynamic_synonym_update(t, context_text=text) or t
            normalized.append(canon)

        # rimuove duplicati e rispetta max_topics
        unique_norm = []
        for c in normalized:
            if c not in unique_norm:
                unique_norm.append(c)
        return unique_norm[:max_topics]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Estrae entità da un testo usando l'LLM."""
        raw_entities_str = self._call_llm(self.entity_extraction_prompt, text=text)
        
        #Inizializza il dizionario delle entità con le chiavi dei tipi selezionati
        entities = {ent_type: [] for ent_type in self.selected_entity_types}

        lines = raw_entities_str.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            #cerca una corrispondenza con i tipi di entità definite
            for ent_type in self.selected_entity_types:
                prefix = f"- {ent_type}:"
                if line.startswith(prefix):
                    parts = line.split(prefix, 1)
                    if len(parts) > 1:
                        # Estrai i valori, pulisci e converti in minuscolo
                        entity_list = [e.strip().lower() for e in parts[1].split(',') if e.strip()]
                        entities[ent_type].extend(entity_list)
                    break # Passa alla prossima riga dopo aver trovato un match
        
        # Rimuove i duplicati per ogni tipo e filtra i tipi di entità che alla fine risultano vuoti
        return {k: list(set(v)) for k, v in entities.items() if v}

    def _dynamic_synonym_update(self, new_topic: str, context_text: str = None):
        """
         aggiorna la mappa locale.
        """
        try:
            context = (context_text or "")[:2000]
            resp = self._call_llm(self.synonym_induction_prompt, term=new_topic, context=context)
            data = None
            try:
                data = json.loads(resp)
            except Exception:
                # tenta di estrarre l'ultimo oggetto JSON da testo rumoroso
                import re
                matches = re.findall(r"(\{[\s\S]*\})", resp)
                for m in reversed(matches):
                    try:
                        data = json.loads(m)
                        break
                    except Exception:
                        continue
            if not isinstance(data, dict):
                return None

            canonical = str(data.get("canonical", "")).lower().strip()
            syns = [s.lower().strip() for s in data.get("synonyms", []) if isinstance(s, str) and s.strip()]
            base = new_topic.lower().strip()
            if base and base not in syns:
                syns.append(base)
            if not canonical:
                return None

            # dedup, rimuovi canonical e limita a massimo 4
            seen = set()
            filtered = []
            for s in syns:
                if not s or s == canonical:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                filtered.append(s)
            syns = filtered[:4]

            if self.graph_db and syns:
                try:
                    self.graph_db.add_synonyms(canonical, syns)
                except Exception:
                    logger.exception("add_synonyms error")

            #aggiorna mappa in memoria
            for s in syns:
                self.synonym_map[s] = canonical

            return canonical
        except Exception:
            logger.exception("dynamic synonym update failed")
            return None
