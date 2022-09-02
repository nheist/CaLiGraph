from typing import Tuple, Dict, Set, List
from collections import defaultdict
import random
import csv
import utils
from impl import wikipedia
from impl.wikipedia import WikipediaPage
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess import sample
from impl.dbpedia.ontology import DbpOntologyStore
from impl.dbpedia.resource import DbpResourceStore
from impl.subject_entity.preprocess.pos_label import TYPE_TO_LABEL_MAPPING, POSLabel
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizedPage


def get_md_listpage_data() -> Tuple[List[WordTokenizedPage], List[WordTokenizedPage]]:
    tokenized_listpages = sample._get_tokenized_listpages_with_entity_labels()
    # split into train and validation (we use a fixed 20% for validation)
    sample_size = int(len(tokenized_listpages) * .2)
    val_sample_indices = set(random.sample([wp.idx for wp in tokenized_listpages], sample_size))
    train_pages = [tlp for tlp in tokenized_listpages if tlp.idx not in val_sample_indices]
    val_pages = [tlp for tlp in tokenized_listpages if tlp.idx in val_sample_indices]
    return train_pages, val_pages


def get_md_page_train_data() -> List[WordTokenizedPage]:
    # filter pages to contain only the ones with SEs of matching types
    train_pages = []
    entity_labels = {}
    for page in combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions()):
        subject_entities = set()
        for enum in page.get_enums():
            enum_subject_entities = [entry['subject_entity'] for entry in enum if 'subject_entity' in entry]
            enum_se_indices = {se['idx'] for se in enum_subject_entities}
            enum_se_tags = {se['tag'] for se in enum_subject_entities if se['tag'] != POSLabel.OTHER.value}
            if len(enum_se_indices) >= 5 and len(enum_se_tags) == 1:
                subject_entities.update(enum_se_indices)
        for table in page.get_tables():
            table_subject_entities = [cell['subject_entity'] for row in table['data'] for cell in row['cells'] if 'subject_entity' in cell]
            table_se_indices = {se['idx'] for se in table_subject_entities}
            table_se_tags = {se['tag'] for se in table_subject_entities if se['tag'] != POSLabel.OTHER.value}
            if len(table_se_indices) >= 5 and len(table_se_tags) == 1:
                subject_entities.update(table_se_indices)
        if subject_entities:
            train_pages.append(page)
            entity_labels[page.idx] = (subject_entities, set())
    # return tokenized pages
    return WordTokenizer(max_ents_per_item=1)(train_pages, entity_labels=entity_labels)


def get_md_page_test_data() -> List[WordTokenizedPage]:
    # load annotations
    md_gold = _load_mention_detection_goldstandard()
    # gather respective wikipedia pages
    wikipedia_pages = [wp for wp in wikipedia.get_wikipedia_pages() if wp.idx in md_gold]
    # find entity labels
    entity_labels = {wp.idx: _find_subject_entities_for_listing_labels(wp, md_gold[wp.idx]) for wp in wikipedia_pages}
    # return tokenized pages
    return WordTokenizer(max_ents_per_item=1)(wikipedia_pages, entity_labels=entity_labels)


def _load_mention_detection_goldstandard() -> Dict[int, List[str]]:
    with open(utils.get_data_file('files.listpages.goldstandard_mention-detection'), mode='r', newline='') as f:
        md_gold = defaultdict(list)
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            if not row[0]:
                continue  # not annotated (yet)
            md_gold[int(row[1])].append(row[0])
    return md_gold


def _find_subject_entities_for_listing_labels(wp: WikipediaPage, listing_labels: list) -> Tuple[Set[int], Set[int]]:
    # collect listing entities
    listing_entities = []
    for section_data in wp.sections:
        for enum_data in section_data['enums']:
            listing_entities.append([e['idx'] for entry in enum_data for e in entry['entities'] if e['idx'] >= 0])
        for table in section_data['tables']:
            listing_entities.append([e['idx'] for row in table['data'] for cell in row['cells'] for e in cell['entities'] if e['idx'] >= 0])
    # create mapping from POS labels to valid entity types
    dbo = DbpOntologyStore.instance()
    label_to_type_mapping = defaultdict(set)
    for t, label in TYPE_TO_LABEL_MAPPING.items():
        if dbo.has_class_with_name(t):
            label_to_type_mapping[label.name].add(dbo.get_class_by_name(t))
    # find valid entities based on POS labels
    dbr = DbpResourceStore.instance()
    valid_ents, invalid_ents = set(), set()
    for label, ents in zip(listing_labels, listing_entities):
        if label == POSLabel.NONE.name:
            invalid_ents.update(ents)
            continue
        valid_types = label_to_type_mapping[label]
        valid_ents.update({e for e in ents if dbr.get_resource_by_idx(e).get_transitive_types().intersection(valid_types)})
    return valid_ents, invalid_ents