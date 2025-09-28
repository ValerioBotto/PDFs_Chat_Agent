# backend/pdf_utils/extractor.py

import logging
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_together import ChatTogether
from dotenv import load_dotenv
import os
import re

logger = logging.getLogger(__name__)

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class Extractor:
    def __init__(self, llm_model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"): # Ho ripristinato il tuo modello originale
        if not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY non è impostata nell'ambiente.")
        
        self.llm = ChatTogether(
            model=llm_model_name,
            temperature=0.1, # temperatura bassa per risposte più consistenti e fattuali
            together_api_key=TOGETHER_API_KEY,
            max_tokens=512
        )
        logger.info(f"LLM initialized for extraction with model: {llm_model_name}")

        # prompt per estrarre topic
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

        # definizione dei tipi di entità che vogliamo estrarre, in inglese
        # ho selezionato un sottoinsieme più gestibile e rilevante per documenti tecnici inizialmente.
        # possiamo espanderlo in seguito.
        self.selected_entity_types = [
            "Person", "Organization", "Location", "GPE", "DateTime", "Product",
            "Event", "URL", "Email", "Currency", "Percentage", "Address",
            "CountryRegion", "Continent", "CulturalEvent", "NaturalEvent" # Aggiungo i nomi generici per copertura
        ]
        
        # genera il prompt per l'estrazione di entità in base ai tipi selezionati
        # la formattazione nel prompt deve essere esatta perché l'LLM la segua.
        entity_format_guide = "\n".join([f"- {ent_type}: [Lista di entità di questo tipo]" for ent_type in self.selected_entity_types])

        # La formattazione nel prompt deve essere esatta perché l'LLM la segua.
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
        
        # dizionario di sinonimi (placeholder dinamico)
        # in una fase successiva, potremmo caricare da neo4j o un altro storage
        self.synonym_map = {
            "automatizzazione": "automazione",
            "sicurezza informatica": "sicurezza",
            "codifica": "programmazione",
            "installazione software": "installazione",
            "manutenzione preventiva": "manutenzione",
            "elettronica digitale": "elettronica",
            "controllo qualità": "controllo"
        }
        logger.info("synonym map initialized (placeholder).")

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
        """estrae topic da un testo usando l'llm e li normalizza."""
        raw_topics_str = self._call_llm(self.topic_extraction_prompt, text=text, max_topics=max_topics)
        
        # pulisce e spacchetta i topic
        topics = [t.strip().lower() for t in raw_topics_str.split(',') if t.strip()]
        
        # filtra termini generici che l'llm potrebbe comunque aver incluso
        generic_terms = {"documento", "testo", "informazioni", "contenuto", "dati", "elemento", "parte", "sezione", "pdf", "file", "articolo"}
        topics = [t for t in topics if t not in generic_terms]
        
        # applica normalizzazione tramite sinonimi
        normalized_topics = [self.synonym_map.get(t, t) for t in topics]
        
        # rimuove duplicati dopo la normalizzazione
        return list(set(normalized_topics))

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Estrae entità da un testo usando l'LLM."""
        raw_entities_str = self._call_llm(self.entity_extraction_prompt, text=text)
        
        # Inizializza il dizionario delle entità con le chiavi dei tipi selezionati
        entities = {ent_type: [] for ent_type in self.selected_entity_types}

        lines = raw_entities_str.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Cerca una corrispondenza con i tipi di entità definite
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
        placeholder per una futura logica di aggiornamento dinamico dei sinonimi.
        potrebbe coinvolgere:
        1. query a neo4j per sinonimi esistenti.
        2. chiamata a llm per suggerire sinonimi basati sul contesto.
        3. persistenza dei nuovi sinonimi in neo4j.
        """
        logger.debug(f"simulando aggiornamento dinamico sinonimi per: {new_topic}")
        # in una vera implementazione, qui l'llm potrebbe suggerire sinonimi
        # e questi verrebbero aggiunti a self.synonym_map e persistiti.
        pass
