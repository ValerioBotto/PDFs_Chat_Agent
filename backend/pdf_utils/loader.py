# backend/pdf_utils/loader.py

import spacy
from spacy_layout import spaCyLayout
import logging

logger = logging.getLogger(__name__)

# #inizializza spacy con una pipeline vuota per la lingua italiana
# nlp = spacy.blank("it") # non carica modelli linguistici, serve solo per compatibilità con spaCyLayout

# #crea un oggetto spacyLayout che può ricevere pdf in bytes ed estrarre layout (testo+struttura)
# layout_extractor = spaCyLayout(nlp)
# #questo oggetto verrà usato nel main per estrarre blocchi strutturati dal pdf

def get_layout_extractor():
    # inizializza spacy con una pipeline vuota per la lingua italiana
    nlp = spacy.blank("it") # non carica modelli linguistici, serve solo per compatibilità con spaCyLayout
    # crea un oggetto spacyLayout che può ricevere pdf in bytes ed estrarre layout (testo+struttura)
    layout_extractor = spaCyLayout(nlp)
    logger.info("initialized spaCyLayout for PDF processing")
    return layout_extractor
