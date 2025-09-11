# main.py

import streamlit as st
import os
import time
import logging

# importa le nuove funzioni dal backend
from backend.pdf_utils.loader import get_layout_extractor
from backend.pdf_utils.preprocessor import split_sections_with_layout
from backend.pdf_utils.chunker import split_sections_into_chunks # importa il nuovo chucker

logger = logging.getLogger(__name__)

# configurazione della pagina streamlit
st.set_page_config(page_title="chat con pdf", layout="wide")

# inizializza lo stato della sessione per la cronologia della chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "layout_extractor" not in st.session_state: # inizializza l'estrattore solo una volta
    st.session_state.layout_extractor = get_layout_extractor()

# sidebar con upload pdf
with st.sidebar:
    st.header("📁 carica documento")
    uploaded_file = st.file_uploader("seleziona un pdf", type=["pdf"])
    
    if uploaded_file:
        # verifica se è un nuovo file o lo stato è stato resettato
        if not st.session_state.pdf_uploaded or (
            hasattr(st.session_state, 'uploaded_filename') and
            st.session_state.uploaded_filename != uploaded_file.name
        ):
            st.info("caricamento e elaborazione del pdf...")
            st.session_state.chat_history = [] # pulisce la cronologia se nuovo pdf
            st.session_state.processed_chunks = [] # resetta i chunk processati per il nuovo file
            st.session_state.uploaded_filename = uploaded_file.name # memorizza il nome del file caricato
            
            # --- inizio elaborazione pdf ---
            
            # legge i bytes del file
            file_bytes = uploaded_file.getvalue()
            logger.debug(f"pdf file size: {len(file_bytes)} bytes")
            
            try:
                # 1. estrazione struttura con spacy-layout
                doc = st.session_state.layout_extractor(file_bytes)
                if doc is None:
                    raise ValueError("errore nell'estrazione del documento dal pdf.")
                logger.debug("document layout extraction completed.")

                # 2. segmentazione del documento in sezioni
                sections = split_sections_with_layout(doc)
                st.session_state.processed_sections = sections # memorizza le sezioni processate
                logger.debug(f"document split into {len(sections)} logical sections.")

                # 3. suddivisione delle sezioni in chunk
                chunks = split_sections_into_chunks(sections)
                st.session_state.processed_chunks = chunks # memorizza i chunk processati
                logger.info(f"total of {len(chunks)} chunks generated from all sections.")
                
                # qui in futuro ci sarà la logica per l'indicizzazione e la creazione dell'agente
                # al momento, solo una simulazione
                st.session_state.agent = "simulated_agent" # placeholder
                st.session_state.pdf_uploaded = True
                st.success("✅ documento elaborato con successo!")
                
            except Exception as e:
                st.error(f"❌ errore durante l'elaborazione del pdf: {str(e)}")
                st.session_state.pdf_uploaded = False # resetta lo stato di caricamento in caso di errore
                st.session_state.agent = None
            
            # --- fine elaborazione pdf ---
            
            st.rerun() # aggiorna la pagina per mostrare l'interfaccia di chat
            
    # bottone per pulire la chat e resettare l'agente
    if st.session_state.pdf_uploaded:
        if st.button("🗑️ pulisci chat e resetta pdf"):
            st.session_state.chat_history = []
            st.session_state.agent = None
            st.session_state.pdf_uploaded = False
            if hasattr(st.session_state, 'uploaded_filename'):
                del st.session_state.uploaded_filename # resetta il nome del file
            if hasattr(st.session_state, 'processed_sections'):
                del st.session_state.processed_sections
            if hasattr(st.session_state, 'processed_chunks'):
                del st.session_state.processed_chunks
            st.info("chat e stato pdf resettati.")
            st.rerun()

# area centrale della pagina streamlit per la chat
if st.session_state.pdf_uploaded and st.session_state.agent:
    st.title("💬 chatta con il tuo pdf")

    # visualizza la cronologia della chat
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # campo di input per l'utente
    user_message = st.chat_input("fai una domanda sul documento...")
    
    if user_message:
        # aggiungi il messaggio dell'utente alla cronologia
        st.session_state.chat_history.append(("user", user_message))
        with st.chat_message("user"):
            st.markdown(user_message)

        # simulazione della risposta dell'agente
        with st.chat_message("assistant"):
            with st.spinner("🤔 sto elaborando la risposta..."):
                time.sleep(2) # simula il tempo di elaborazione
                # questa sarà la parte che invocherà il tuo agente reale
                response = f"hai chiesto: '{user_message}'. al momento posso solo simulare una risposta."
                st.markdown(response)
            st.session_state.chat_history.append(("assistant", response))

else:
    # messaggio iniziale di default
    st.info("👈 carica un pdf dalla sidebar per iniziare la conversazione")
