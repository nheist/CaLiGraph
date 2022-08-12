from typing import Tuple, List
from collections import defaultdict
import random
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
import utils
from impl import wikipedia
from impl.dbpedia.resource import DbpListpage
from impl.util.transformer import EntityIndex
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizedPage, ListingType, WordTokenizedListing
from impl.subject_entity.preprocess.pos_label import POSLabel, map_entities_to_pos_labels
from impl.subject_entity.preprocess.heuristics import find_subject_entities_for_listpage


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
    page_data = utils.load_or_create_cache('subject_entity_training_data', _get_tokenized_listpages_with_entity_labels)
    return _get_mention_detection_dataset(page_data, tokenizer)


def _get_tokenized_listpages_with_entity_labels() -> List[WordTokenizedPage]:
    wikipedia_listpages = [wp for wp in wikipedia.get_wikipedia_pages() if isinstance(wp.resource, DbpListpage)]
    entity_labels = {wp.idx: find_subject_entities_for_listpage(wp) for wp in wikipedia_listpages}
    return WordTokenizer(max_ents_per_item=1)(wikipedia_listpages, entity_labels=entity_labels)


def _get_mention_detection_dataset(page_data: List[WordTokenizedPage], tokenizer) -> MentionDetectionDataset:
    negative_sample_size = float(utils.get_config('subject_entity.negative_sample_size'))
    # flatten training data into chunks and replace entities with their POS tags
    _, tokens, _, entity_indices = _chunk_word_tokenized_pages(page_data, negative_sample_size)
    labels = map_entities_to_pos_labels(entity_indices, False)

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
        # compute number of negative chunks to create
        num_enum_chunks = len([c for c in context_chunks if c['listing_type'] == ListingType.ENUMERATION])
        num_table_chunks = len(context_chunks) - num_enum_chunks
        num_negative_enum_chunks = int(num_enum_chunks * negative_sample_size)
        num_negative_table_chunks = int(num_table_chunks * negative_sample_size)
        # collect all contexts and items
        enum_listings = [l for p in page_data for l in p.listings if l.listing_type == ListingType.ENUMERATION]
        enum_contexts = {0: [(l.listing_type, l.topsection, l.section, l.context) for l in enum_listings]}
        enum_items = {0: [i for l in enum_listings for i in l.items]}
        table_listings = [l for p in page_data for l in p.listings if l.listing_type == ListingType.TABLE]
        table_contexts = defaultdict(list)
        table_items = defaultdict(list)
        for l in table_listings:
            table_contexts[l.column_count].append((l.listing_type, l.topsection, l.section, l.context))
            table_items[l.column_count].extend(l.items)
        # generate negatives
        for _ in range(num_negative_enum_chunks):
            n_ctx, n_tokens, n_ws, n_labels = _generate_negative_tokenized_page_chunk(enum_contexts, enum_items, max_items_per_chunk, max_words_per_chunk)
            context_chunks.append(n_ctx)
            token_chunks.append(n_tokens)
            ws_chunks.append(n_ws)
            label_chunks.append(n_labels)
        for _ in range(num_negative_table_chunks):
            n_ctx, n_tokens, n_ws, n_labels = _generate_negative_tokenized_page_chunk(table_contexts, table_items, max_items_per_chunk, max_words_per_chunk)
            context_chunks.append(n_ctx)
            token_chunks.append(n_tokens)
            ws_chunks.append(n_ws)
            label_chunks.append(n_labels)
    return context_chunks, token_chunks, ws_chunks, label_chunks


def _generate_negative_tokenized_page_chunk(contexts: dict, items: dict, max_items_per_chunk: int, max_words_per_chunk: int) -> Tuple[list, list, list, list]:
    # pick listing quantifier randomly but weighted by occurrence (for enums we have only one quantifier, making this step irrelevant)
    quantifier = random.choices(list(items), weights=[len(vals) for vals in items.values()], k=1)[0]
    # pick number of items to put in the chunk
    num_items = random.choice(range(3, max_items_per_chunk + 1))
    # draw context and items
    listing_type, listing_topsection, listing_section, listing_context = random.choice(contexts[quantifier])
    listing_items = [deepcopy(i) for i in random.choices(items[quantifier], k=num_items)]
    for i in listing_items:  # set all labels to NO_ENTITY as we are generating negatives
        i.entity_indices = [EntityIndex.NO_ENTITY.value] * len(i.entity_indices)
    # manufacture page and convert to chunks
    listing = WordTokenizedListing(listing_type, listing_context, listing_items, listing_topsection, listing_section)
    page = WordTokenizedPage(-1, [listing])
    context_chunks, token_chunks, ws_chunks, label_chunks = page.to_chunks(max_items_per_chunk, max_words_per_chunk)
    return context_chunks[0], token_chunks[0], ws_chunks[0], label_chunks[0]


def _encode_labels(labels: List[List[str]], encodings) -> List[List[str]]:
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of ignored labels
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * EntityIndex.IGNORE.value
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
