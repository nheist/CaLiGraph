"""NLP methods for the identification of named entities in enumerations and tables of Wikipedia articles."""

import util
import random
from pathlib import Path
import impl.util.rdf as rdf_util
import impl.dbpedia.store as dbp_store
import impl.list.store as list_store
import impl.list.mapping as list_mapping
import json
import spacy
from spacy.tokens import Doc
from spacy.util import minibatch, compounding


def parse(text: str) -> Doc:
    """Return `text` as spaCy document."""
    global __PARSER__
    if '__PARSER__' not in globals():
        __PARSER__ = _initialise_parser()

    return __PARSER__(text)


# initialization

NER_LABEL_MAPPING = {
    'PERSON': ['http://dbpedia.org/ontology/Person', 'http://dbpedia.org/ontology/Deity'],
    'NORP': ['http://dbpedia.org/ontology/PoliticalParty', 'http://dbpedia.org/ontology/Family', 'http://dbpedia.org/ontology/EthnicGroup'],
    'FAC': ['http://dbpedia.org/ontology/ArchitecturalStructure', 'http://dbpedia.org/ontology/Mine', 'http://dbpedia.org/ontology/Monument'],
    'ORG': ['http://dbpedia.org/ontology/Organisation'],
    'GPE': ['http://dbpedia.org/ontology/PopulatedPlace'],
    'LOC': ['http://dbpedia.org/ontology/Place', 'http://dbpedia.org/ontology/Location'],
    'PRODUCT': ['http://dbpedia.org/ontology/Food', 'http://dbpedia.org/ontology/MeanOfTransportation', 'http://dbpedia.org/ontology/Software', 'http://dbpedia.org/ontology/Device'],
    'EVENT': ['http://dbpedia.org/ontology/Event'],
    'WORK_OF_ART': ['http://dbpedia.org/ontology/Work', 'http://dbpedia.org/ontology/Award'],
    'LAW': ['http://dbpedia.org/ontology/Law', 'http://dbpedia.org/ontology/LegalCase', 'http://dbpedia.org/ontology/Treaty'],
    'LANGUAGE': ['http://dbpedia.org/ontology/Language'],
    'SPECIES': ['http://dbpedia.org/ontology/Species']
}


def _initialise_parser():
    path_to_model = util._get_cache_path('spacy_listpage_ne-tagging_GS-WLE')
    if not path_to_model.is_dir():
        _train_parser()
    return spacy.load(str(path_to_model))


def _train_parser():
    util.get_logger().debug('LIST/NLP: Training new spacy model on GS..')
    # specialize spacy model on GS
    filepath_gs = util._get_cache_path('spacy_listpage_ne-tagging_GS')
    if not filepath_gs.is_dir():
        training_data_gs = _retrieve_training_data_gs()
        _train_enhanced_spacy_model(training_data_gs, str(filepath_gs), model='en_core_web_lg')

    util.get_logger().debug('LIST/NLP: Training new spacy model on WLE..')
    # specialize spacy+GS model on WLE
    filepath_wle = util._get_cache_path('spacy_listpage_ne-tagging_GS-WLE')
    training_data_wle = _retrieve_training_data_wle()
    _train_enhanced_spacy_model(training_data_wle, str(filepath_wle), model=filepath_gs)


def _retrieve_training_data_gs():
    training_data = []
    with open(util.get_data_file('files.listpages.goldstandard_named-entity-tagging'), mode='r') as f:
        for line in f:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                entities.append((point['start'], point['end']+1, annotation['label'][0]))
            training_data.append((text, {'entities': entities}))
    return training_data


def _retrieve_training_data_wle():
    listpages = {lp: data for lp, data in list_store.get_parsed_listpages(list_store.LIST_TYPE_ENUM).items()}
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
                        entity_tag = _get_tag_for_types(dbp_store.get_independent_types(dbp_store.get_types(entity_data['uri'])))
                        if not entity_tag:
                            continue
                        entity_text = entity_data['text']
                        start = int(entity_data['idx'])
                        end = start + len(text)
                        if end > len(text) or text[start:end] != entity_text:
                            continue
                        valid_entities.append((start, end, entity_tag))
                    if len(entities) == len(valid_entities):
                        training_data.append((text, {'entities': valid_entities}))
    return training_data


def _get_tag_for_types(dbp_types: set) -> str:
    type_to_tag_mapping = {tp: tag for tag, types in NER_LABEL_MAPPING.items() for tp in types}
    dbp_types = {t for t in dbp_types if t is not None and t != rdf_util.CLASS_OWL_THING}
    for t in dbp_types:
        if t in type_to_tag_mapping:
            return type_to_tag_mapping[t]
    parent_types = {pt for t in dbp_types for pt in dbp_store.get_supertypes(t)}
    return _get_tag_for_types(parent_types) if parent_types else None


def _train_enhanced_spacy_model(training_data: list, output_dir: str, model=None, n_iter=50):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
    else:
        nlp = spacy.blank("en")  # create blank Language class

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    # add unkonwn labels to ner tagger
    for label in NER_LABEL_MAPPING:
        ner.add_label(label)

    # train new model
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training()  # initialize training if necessary
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )

    # save model to output directory
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
