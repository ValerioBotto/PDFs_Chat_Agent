import logging

logger = logging.getLogger(__name__)

def split_sections_with_layout(doc) -> dict:
    #funzione per suddividere il documento in sezioni logiche usando le label fornite da spacylayout
    
    sections = {}
    current_title = "introduzione" # titolo iniziale predefinito per il primo blocco di testo
    sections[current_title] = ""
    #scorre tutti gli span etichettati nel livello "layout"
    for span in doc.spans.get("layout", []):
        label = span.label_.upper() 
        
        #se trova un'etichetta di sezione (header, titolo, etc.), aggiorna il titolo corrente
        if label in ("SECTION_HEADER", "TITLE", "BOLD", "BOLD_CAPTION"):
            potential_title = span.text.strip().lower() 
            #evita titoli vuoti o troppo generici 
            if len(potential_title) > 0 and potential_title not in sections:
                current_title = potential_title
                if current_title not in sections: #aggiunge la nuova sezione al dizionario
                    sections[current_title] = ""
        elif label == "TEXT":
            #aggiunge il testo sotto la sezione corrente
            sections[current_title] += span.text + "\n"

    #pulisce le sezioni vuote e rimuove spazi iniziali/finali
    cleaned_sections = {k: v.strip() for k, v in sections.items() if v.strip()}
    
    #se non c'è nessuna sezione con contenuto valido dopo la pulizia,
    #ritorna l'intero testo del documento come una sezione unica.
    if not cleaned_sections and hasattr(doc, 'text') and doc.text.strip():
        logger.warning("no distinct sections found, returning whole document as one section.")
        return {"documento_completo": doc.text.strip()}
    
    logger.info(f"document split into {len(cleaned_sections)} logical sections.")
    return cleaned_sections
