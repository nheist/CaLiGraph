"""NLP methods for the identification of named entities in enumerations and tables of Wikipedia articles."""

import utils
from utils import get_logger
from impl.util.spacy.training import train_ner_model
import json
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example


def parse(text: str) -> Doc:
    """Return `text` as spaCy document."""
    global __PARSER__
    if '__PARSER__' not in globals():
        __PARSER__ = _initialise_parser()

    return __PARSER__(text)


# initialization


def _initialise_parser():
    path_to_model = utils._get_cache_path('spacy_listpage_ne-tagging')
    if not path_to_model.is_dir():
        _train_parser()
    return spacy.load(str(path_to_model))


def _train_parser():
    get_logger().info('Training new spacy model for entity tagging in listings..')
    filepath_gs = utils._get_cache_path('spacy_listpage_ne-tagging')
    if not filepath_gs.is_dir():
        train_ner_model(_retrieve_training_data_gs, str(filepath_gs), model='en_core_web_lg')


def _retrieve_training_data_gs(nlp: Language):
    training_data = []
    with open(utils.get_data_file('files.listpages.goldstandard_named-entity-tagging'), mode='r') as f:
        for line in f:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                entities.append((point['start'], point['end']+1, annotation['label'][0]))
            training_data.append(Example.from_dict(nlp.make_doc(text), {'entities': entities}))
    return training_data
