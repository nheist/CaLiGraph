"""NLP methods for the identification of named entities in enumerations and tables of Wikipedia articles."""

import utils
import impl.util.rdf as rdf_util
from impl import wikipedia
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.list.store as list_store
import impl.list.mapping as list_mapping
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
    path_to_model = utils._get_cache_path('spacy_listpage_ne-tagging_GS-WLE')
    if not path_to_model.is_dir():
        _train_parser()
    return spacy.load(str(path_to_model))


def _train_parser():
    utils.get_logger().debug('LIST/NLP: Training new spacy model on GS..')
    # specialize spacy model on GS
    filepath_gs = utils._get_cache_path('spacy_listpage_ne-tagging_GS')
    if not filepath_gs.is_dir():
        train_ner_model(_retrieve_training_data_gs, str(filepath_gs), model='en_core_web_lg')

    utils.get_logger().debug('LIST/NLP: Training new spacy model on WLE..')
    # specialize spacy+GS model on WLE
    filepath_wle = utils._get_cache_path('spacy_listpage_ne-tagging_GS-WLE')
    train_ner_model(_retrieve_training_data_wle, str(filepath_wle), model=filepath_gs)


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


def _retrieve_training_data_wle(nlp: Language):
    listpages = list_store.get_parsed_listpages(wikipedia.ARTICLE_TYPE_ENUM)
    lp_to_cat_mapping = {lp: list_mapping.get_equivalent_categories(lp) | list_mapping.get_parent_categories(lp) for lp in listpages}
    lp_to_cat_mapping = {lp: cats for lp, cats in lp_to_cat_mapping.items() if cats}

    training_data = []
    # extract entities
    for lp, cats in lp_to_cat_mapping.items():
        lp_data = listpages[lp]
        for section_data in lp_data['sections']:
            for enum_data in section_data['enums']:
                for entry_data in enum_data:
                    text = entry_data['text']
                    if not text:
                        continue
                    entities = entry_data['entities']
                    if not entities:
                        continue
                    valid_entities = []
                    for entity_data in entities:
                        entity_uri = dbp_util.name2resource(entity_data['name'])
                        entity_tag = _get_tag_for_types(dbp_store.get_types(entity_uri))
                        if not entity_tag:
                            continue
                        entity_text = entity_data['text']
                        start = int(entity_data['idx'])
                        end = start + len(text)
                        if end > len(text) or text[start:end] != entity_text:
                            continue
                        valid_entities.append((start, end, entity_tag))
                    if len(entities) == len(valid_entities):
                        training_data.append(Example.from_dict(nlp.make_doc(text), {'entities': valid_entities}))
    return training_data


def _get_tag_for_types(dbp_types: set) -> str:
    type_to_tag_mapping = {tp: tag for tag, types in dbp_util.NER_LABEL_MAPPING.items() for tp in types}
    dbp_types = {t for t in dbp_types if t is not None and t != rdf_util.CLASS_OWL_THING}
    for t in dbp_types:
        if t in type_to_tag_mapping:
            return type_to_tag_mapping[t]
    parent_types = {pt for t in dbp_types for pt in dbp_store.get_supertypes(t)}
    return _get_tag_for_types(parent_types) if parent_types else None
