"""NLP methods for the identification of named entities in Wikipedia list pages."""

import spacy
from spacy.tokens import Doc

# initialization
# TODO: implement learning of model
parser = spacy.load('data_caligraph-NE/spacy-model_goldstandard_50p-all')


def parse(text: str) -> Doc:
    """Return `text` as spaCy document."""
    return parser(text)
