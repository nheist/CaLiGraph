from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
import utils
from impl import wikipedia
from impl.dbpedia.resource import DbpListpage
from .word_tokenize import WordTokenizer, WordTokenizedPage
from .pos_label import POSLabel, map_entities_to_pos_labels
from .heuristics import find_subject_entities_for_listpage


class MentionDetectionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_mention_detection_page_data() -> Tuple[list, list, list, list]:
    page_data = utils.load_or_create_cache('subject_entity_page_data', lambda: WordTokenizer()(wikipedia.get_wikipedia_pages()))
    return _chunk_word_tokenized_pages(page_data)


def get_mention_detection_listpage_training_dataset(tokenizer) -> MentionDetectionDataset:
    page_data = utils.load_or_create_cache('subject_entity_training_data', _get_tokenized_list_pages_with_entity_labels)
    return _get_mention_detection_dataset(page_data, tokenizer)


def _get_tokenized_list_pages_with_entity_labels() -> List[WordTokenizedPage]:
    wikipedia_listpages = [wp for wp in wikipedia.get_wikipedia_pages() if isinstance(wp.resource, DbpListpage)]
    entity_labels = {wp.idx: find_subject_entities_for_listpage(wp) for wp in wikipedia_listpages}
    return WordTokenizer()(wikipedia_listpages, entity_labels=entity_labels)


def _get_mention_detection_dataset(page_data: List[WordTokenizedPage], tokenizer) -> MentionDetectionDataset:
    negative_sample_size = float(utils.get_config('subject_entity.negative_sample_size'))
    # flatten training data into chunks and replace entities with their POS tags
    _, tokens, _, entity_indices = _chunk_word_tokenized_pages(page_data, negative_sample_size)
    labels = map_entities_to_pos_labels(entity_indices)

    train_encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    train_labels = _encode_labels(labels, train_encodings)
    train_encodings.pop('offset_mapping')  # we don't want to pass this to the model
    return MentionDetectionDataset(train_encodings, train_labels)


def _chunk_word_tokenized_pages(page_data: List[WordTokenizedPage], negative_sample_size: float = 0.0, max_items_per_chunk: int = 16, max_words_per_chunk: int = 300) -> Tuple[list, list, list, list]:
    context_chunks, token_chunks, ws_chunks, label_chunks = [], [], [], []
    for p in page_data:
        page_context_chunks, page_token_chunks, page_ws_chunks, page_label_chunks = p.to_chunks(max_items_per_chunk, max_words_per_chunk)
        context_chunks.extend(page_context_chunks)
        token_chunks.extend(page_token_chunks)
        ws_chunks.extend(page_ws_chunks)
        label_chunks.extend(page_label_chunks)
    if negative_sample_size > 0:
        pass  # TODO: implement negative sampling mechanism
    return context_chunks, token_chunks, ws_chunks, label_chunks


def _encode_labels(labels: List[List[str]], encodings) -> List[List[str]]:
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        relevant_label_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        truncated_label_length = len(doc_enc_labels[relevant_label_mask])
        if len(doc_labels) < truncated_label_length:
            # uncased tokenizers can be confused by Japanese/Chinese signs leading to an inconsistency between tokens
            # and labels after tokenization. we handle that gracefully by simply filling it up with non-empty labels.
            doc_labels += [POSLabel.NONE.value] * (truncated_label_length - len(doc_labels))
        doc_enc_labels[relevant_label_mask] = doc_labels[:truncated_label_length]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels
