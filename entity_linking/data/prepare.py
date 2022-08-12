from typing import Tuple, Dict, Set, List
from collections import defaultdict
import random
import csv
import utils
from impl.util.transformer import EntityIndex
from impl import wikipedia
from impl.wikipedia import WikipediaPage
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess import sample
from impl.dbpedia.ontology import DbpOntologyStore
from impl.dbpedia.resource import DbpResourceStore
from impl.subject_entity.preprocess.pos_label import TYPE_TO_LABEL_MAPPING, POSLabel
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizedPage


def get_md_train_and_val_data() -> Tuple[List[WordTokenizedPage], List[WordTokenizedPage]]:
    # TODO: add option to use actual page data for specialization => use only "high-quality" page data for evaluation
    tokenized_listpages = sample._get_tokenized_listpages_with_entity_labels()
    # split into train and validation (we use a fixed 20% for validation)
    sample_size = int(len(tokenized_listpages) * .2)
    val_sample_indices = set(random.sample([wp.idx for wp in tokenized_listpages], sample_size))
    train_pages = [tlp for tlp in tokenized_listpages if tlp.idx not in val_sample_indices]
    val_pages = [tlp for tlp in tokenized_listpages if tlp.idx in val_sample_indices]
    return train_pages, val_pages


def get_md_test_data() -> List[WordTokenizedPage]:
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
            listing_entities.append([e for entry in enum_data for e in entry['entities']])
        for table in section_data['tables']:
            listing_entities.append([e for row in table['data'] for cell in row for e in cell['entities']])
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


def get_mem_train_and_val_data() -> Tuple[List[WordTokenizedPage], List[WordTokenizedPage]]:
    train_pages, val_pages = utils.load_or_create_cache('entity_linking_pages', _load_page_data)
    # extract data for matching
    train_data = _create_mention_entity_matching_data(train_pages, False)
    val_data = _create_mention_entity_matching_data(val_pages, True)
    return train_data, val_data


def _load_page_data() -> Tuple[List[WikipediaPage], List[WikipediaPage]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    wiki_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # filter out pages whose main entity has no embedding
    wiki_pages = [wp for wp in wiki_pages if wp.idx in valid_res_indices]
    # filter out listings that contain only unknown entities (as those violate the distant supervision assumption)
    for wp in wiki_pages:
        wp.discard_listings_without_seen_entities()
    # filter out pages that have no labeled subject entities at all
    wiki_pages = [wp for wp in wiki_pages if wp.get_subject_entity_indices()]
    # split into train and validation (we use a fixed 5% for validation independent of training size)
    val_sample_indices = set(random.sample([wp.idx for wp in wiki_pages], int(len(wiki_pages) * .05)))
    train_pages = [wp for wp in wiki_pages if wp.idx not in val_sample_indices]
    val_pages = [wp for wp in wiki_pages if wp.idx in val_sample_indices]
    return train_pages, val_pages


def _create_mention_entity_matching_data(wiki_pages: List[WikipediaPage], include_new_entities: bool) -> List[WordTokenizedPage]:
    entity_labels = _get_subject_entity_labels(wiki_pages, include_new_entities)
    return WordTokenizer(max_ents_per_item=1)(wiki_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(wiki_pages: List[WikipediaPage], include_new_entities: bool) -> Dict[int, Tuple[Set[int], Set[int]]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    entity_labels = {}
    for wp in wiki_pages:
        subject_entity_indices = wp.get_subject_entity_indices()
        # get rid of non-entities and entities without RDF2vec embeddings (as we can't use them for training/eval)
        subject_entity_indices.intersection_update(valid_res_indices)
        if include_new_entities:
            subject_entity_indices.add(EntityIndex.NEW_ENTITY.value)
        entity_labels[wp.idx] = (subject_entity_indices, set())
    return entity_labels
