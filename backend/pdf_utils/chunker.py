# backend/pdf_utils/chunker.py

import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
#importa Document da langchain_core.documents per rappresentare i chunk
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def split_sections_into_chunks(
    sections: Dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    
    #gestione delle sezioni vuote
    if not sections:
        logger.warning("no sections provided for chunking.")
        return []

    #inizializzazione dello splitter di testo
    #utilizzo i separatori predefiniti per preservare la struttura del testo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, 
    )
    
    all_chunks: List[Document] = []

    #itera su ogni sezione e la divide in chunk
    for section_title, section_text in sections.items():
        if not section_text.strip(): #salta sezioni vuote o che contengono solo spazi
            logger.debug(f"skipping empty section: '{section_title}'")
            continue

        try:
            #divide il testo della sezione in chunk
            chunks_for_section = text_splitter.create_documents([section_text])
            
            #aggiunge metadati a ciascun chunk per indicare la sezione di origine
            for i, chunk in enumerate(chunks_for_section):
                #i metadati useranno 'source' (nome file) e 'section' (titolo sezione)
                chunk.metadata["section"] = section_title
                chunk.metadata["chunk_id"] = f"{section_title}_{i}" #id univoco per il chunk
                all_chunks.append(chunk)
            
            logger.debug(f"section '{section_title}' split into {len(chunks_for_section)} chunks.")

        except Exception as e:
            logger.error(f"error chunking section '{section_title}': {e}")
            
    logger.info(f"total of {len(all_chunks)} chunks generated from all sections.")
    return all_chunks
