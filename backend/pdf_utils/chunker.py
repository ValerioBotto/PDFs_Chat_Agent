# backend/pdf_utils/chunker.py

import logging
from typing import List, Dict

# importa il RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# importa Document da langchain_core.documents per rappresentare i chunk
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def split_sections_into_chunks(
    sections: Dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    divide le sezioni di testo in chunk più piccoli utilizzando RecursiveCharacterTextSplitter.

    args:
        sections (dict): un dizionario dove le chiavi sono i titoli delle sezioni
                         e i valori sono i testi delle sezioni.
        chunk_size (int): la dimensione massima desiderata per ogni chunk.
        chunk_overlap (int): la dimensione della sovrapposizione tra chunk consecutivi.

    returns:
        list[Document]: una lista di oggetti Document, dove ogni Document
                        rappresenta un chunk con i suoi metadati.
    """
    
    if not sections:
        logger.warning("no sections provided for chunking.")
        return []

    # inizializza lo splitter ricorsivo
    # utilizza i separatori predefiniti per preservare la struttura del testo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # usa len() per misurare la lunghezza del chunk
    )
    
    all_chunks: List[Document] = []

    # itera su ogni sezione e la divide in chunk
    for section_title, section_text in sections.items():
        if not section_text.strip(): # salta sezioni vuote
            logger.debug(f"skipping empty section: '{section_title}'")
            continue

        try:
            # divide il testo della sezione in chunk
            chunks_for_section = text_splitter.create_documents([section_text])
            
            # aggiunge metadati a ciascun chunk per indicare la sezione di origine
            for i, chunk in enumerate(chunks_for_section):
                # i metadati useranno 'source' (nome file) e 'section' (titolo sezione)
                # il nome del file sarà aggiunto in main.py o indexer.py
                chunk.metadata["section"] = section_title
                chunk.metadata["chunk_id"] = f"{section_title}_{i}" # id univoco per il chunk
                all_chunks.append(chunk)
            
            logger.debug(f"section '{section_title}' split into {len(chunks_for_section)} chunks.")

        except Exception as e:
            logger.error(f"error chunking section '{section_title}': {e}")
            # potremmo decidere di aggiungere l'intera sezione come un singolo chunk
            # se la suddivisione fallisce, o gestirla in altro modo.
            # per ora, semplicemente logghiamo l'errore e continuiamo.
            
    logger.info(f"total of {len(all_chunks)} chunks generated from all sections.")
    return all_chunks
