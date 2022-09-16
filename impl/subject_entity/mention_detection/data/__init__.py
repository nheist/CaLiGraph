from typing import List, Tuple
import utils
from impl.wikipedia import WikiPageStore, WikiPage
from .dataset import MentionDetectionDataset
from . import chunking
from impl.subject_entity.mention_detection import labels


def get_listpage_training_dataset(tokenizer) -> MentionDetectionDataset:
    utils.get_logger().debug('Loading listpage training dataset for mention detection..')
    listpages = WikiPageStore.instance().get_listpages()
    negative_sample_size = float(utils.get_config('subject_entity.negative_sample_size'))
    return _prepare_dataset(tokenizer, listpages, False, negative_sample_size)


def get_page_training_dataset(tokenizer) -> MentionDetectionDataset:
    utils.get_logger().debug('Loading page training dataset for mention detection..')
    pages = WikiPageStore.instance().get_pages()
    negative_sample_size = float(utils.get_config('subject_entity.negative_sample_size'))
    return _prepare_dataset(tokenizer, pages, False, negative_sample_size)


def _prepare_dataset(tokenizer, pages: List[WikiPage], binary_labels: bool, negative_sample_size: float) -> MentionDetectionDataset:
    page_labels = labels.get_labels(pages)
    p_tokens, p_labels, listing_types = chunking.process_training_pages(pages, page_labels, negative_sample_size)
    return MentionDetectionDataset(tokenizer, p_tokens, p_labels, listing_types, binary_labels)


def get_page_data() -> Tuple[List[List[Tuple[int, int, int]]], List[List[str]], List[List[str]]]:
    utils.get_logger().debug('Loading page dataset for mention detection..')
    pages = WikiPageStore.instance().get_pages()
    return chunking.process_pages(pages)
