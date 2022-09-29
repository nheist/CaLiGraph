from typing import Tuple, Dict, Set, List
from collections import defaultdict
import random
import csv
import utils
from impl.util.nlp import EntityTypeLabel
from impl.wikipedia import WikiPageStore
from impl.subject_entity.mention_detection.data import MentionDetectionDataset, _prepare_labeled_chunks, chunking
from impl.subject_entity.mention_detection import labels
from impl.dbpedia.ontology import DbpOntologyStore
from impl.subject_entity.mention_detection.labels.entity_type import TYPE_TO_LABEL_MAPPING


def get_md_lp_train_dataset(tokenizer, ignore_tags: bool, negative_sample_size: float, single_item_chunks: bool) -> MentionDetectionDataset:
    prepare_train_chunks = lambda: _create_md_lp_chunks(0, negative_sample_size)
    version = f'nss-{negative_sample_size}-sic' if single_item_chunks else f'nss-{negative_sample_size}'
    train_tokens, train_labels, train_listing_types = utils.load_or_create_cache('MD_listpage_train', prepare_train_chunks, version=version)
    return MentionDetectionDataset(tokenizer, train_tokens, train_labels, train_listing_types, ignore_tags)


def get_md_lp_eval_dataset(tokenizer, ignore_tags: bool) -> MentionDetectionDataset:
    prepare_eval_chunks = lambda: _create_md_lp_chunks(1, 0.0)
    eval_tokens, eval_labels, eval_listing_types = utils.load_or_create_cache('MD_listpage_eval', prepare_eval_chunks)
    return MentionDetectionDataset(tokenizer, eval_tokens, eval_labels, eval_listing_types, ignore_tags)


def _create_md_lp_chunks(data_idx: int, negative_sample_size: float) -> Tuple[List[List[str]], List[List[int]], List[str]]:
    data = utils.load_or_create_cache('MD_listpage_data', _get_md_listpage_data)[data_idx]
    wps = WikiPageStore.instance()
    return _prepare_labeled_chunks([wps.get_page(idx) for idx in data], negative_sample_size)


def _get_md_listpage_data() -> Tuple[List[int], List[int]]:
    listpages = WikiPageStore.instance().get_listpages()
    # split into train and validation (we use a fixed 20% for validation)
    sample_size = int(len(listpages) * .2)
    val_sample_indices = set(random.sample([wp.idx for wp in listpages], sample_size))
    train_pages = [tlp.idx for tlp in listpages if tlp.idx not in val_sample_indices]
    val_pages = [tlp.idx for tlp in listpages if tlp.idx in val_sample_indices]
    return train_pages, val_pages


def get_md_page_train_dataset(tokenizer, ignore_tags: bool, negative_sample_size: float, single_item_chunks: bool) -> MentionDetectionDataset:
    prepare_train_chunks = lambda: _create_md_page_train_chunks(negative_sample_size)
    version = f'nss-{negative_sample_size}-sic' if single_item_chunks else f'nss-{negative_sample_size}'
    tokens, labels, listing_types = utils.load_or_create_cache('MD_page_train', prepare_train_chunks, version=version)
    return MentionDetectionDataset(tokenizer, tokens, labels, listing_types, ignore_tags)


def _create_md_page_train_chunks(negative_sample_size: float) -> Tuple[List[List[str]], List[List[int]], List[str]]:
    wps = WikiPageStore.instance()
    wps.set_subject_entity_mentions(utils.load_cache('subject_entity_mentions_noisy'))
    return _prepare_labeled_chunks(wps.get_pages(), negative_sample_size)


def get_md_page_test_dataset(tokenizer, ignore_tags: bool) -> MentionDetectionDataset:
    tokens, labels, listing_types = utils.load_or_create_cache('MD_page_test', _create_md_page_test_chunks)
    return MentionDetectionDataset(tokenizer, tokens, labels, listing_types, ignore_tags)


def _create_md_page_test_chunks() -> Tuple[List[List[str]], List[List[int]], List[str]]:
    md_gold = _load_mention_detection_goldstandard()
    wps = WikiPageStore.instance()
    md_gold_pages = [wps.get_page(page_idx) for page_idx in md_gold]
    md_gold_labels = labels._get_labels_for_subject_entities(md_gold_pages, _find_subject_entities_for_listing_labels(md_gold))
    return chunking.process_training_pages(md_gold_pages, md_gold_labels, 0)


def _load_mention_detection_goldstandard() -> Dict[int, Dict[int, str]]:
    with open(utils.get_data_file('files.listpages.goldstandard_mention-detection'), mode='r', newline='') as f:
        md_gold = defaultdict(dict)
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            if not row[0]:
                continue  # not annotated (yet)
            page_idx = int(row[1])
            listing_idx = int(row[5])
            md_gold[page_idx][listing_idx] = row[0]
    return md_gold


def _find_subject_entities_for_listing_labels(page_listing_labels: Dict[int, Dict[int, str]]) -> Dict[int, Dict[int, Tuple[Set[int], Set[int]]]]:
    # create mapping from entity type labels to valid DBpedia entity types
    dbo = DbpOntologyStore.instance()
    label_to_type_mapping = defaultdict(set)
    for t, label in TYPE_TO_LABEL_MAPPING.items():
        if dbo.has_class_with_name(t):
            label_to_type_mapping[label.name].add(dbo.get_class_by_name(t))
    # find positives and negatives based on entity type labels
    subject_entities = defaultdict(dict)
    wps = WikiPageStore.instance()
    for page_idx, listing_labels in page_listing_labels.items():
        wp = wps.get_page(page_idx)
        for listing in wp.get_listings():
            if listing.idx not in listing_labels:
                continue
            positives, negatives = set(), set()
            listing_entities = listing.get_mentioned_entities()
            entity_type_label = listing_labels[listing.idx]
            if entity_type_label == EntityTypeLabel.NONE.name:
                negatives.update({e.idx for e in listing_entities})
            dbpedia_types = label_to_type_mapping[entity_type_label]
            positives.update({e.idx for e in listing_entities if e.get_transitive_types().intersection(dbpedia_types)})
            subject_entities[page_idx][listing.idx] = (positives, negatives)
    return subject_entities
