import spacy
from spacy_layout import spaCyLayout
import logging

logger = logging.getLogger(__name__)


def get_layout_extractor():
    nlp = spacy.blank("it")
    layout_extractor = spaCyLayout(nlp)
    logger.info("initialized spaCyLayout for PDF processing")
    return layout_extractor
