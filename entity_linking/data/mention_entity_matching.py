from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from impl.dbpedia.resource import DbpResourceStore
from entity_linking.preprocessing.blocking import WordBlocker
from impl.util.rdf import EntityIndex


class MentionEntityMatchingDataset(Dataset):
    def __init__(self, encodings: dict, mention_spans: List[List[Tuple[int, int]]], entity_indices: List[List[int]], entity_labels: List[List[int]]):
        self.encodings = encodings
        self.mention_spans = mention_spans
        self.entity_indices = entity_indices
        self.entity_labels = entity_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item.update({
            'mention_spans': torch.tensor(self.mention_spans[idx]),
            'entity_indices': torch.tensor(self.entity_indices[idx]),
            'labels': torch.tensor(self.entity_labels[idx])
        })
        return item

    def __len__(self):
        return len(self.entity_indices)


def prepare_dataset(tokens: List[List[str]], labels: List[List[int]], tokenizer, num_ents: int, items_per_chunk: int):
    entity_info = _collect_entity_info(tokens, labels)
    tokens, entity_info = _filter_truncated_entities(tokens, entity_info, tokenizer)

    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    mention_spans, entity_indices, entity_labels = _process_entity_info(entity_info, encodings['offset_mapping'], num_ents, items_per_chunk)
    encodings.pop('offset_mapping')  # we don't want to pass this to the model
    return MentionEntityMatchingDataset(encodings, mention_spans, entity_indices, entity_labels)


def _collect_entity_info(tokens: List[List[str]], labels: List[List[int]]) -> List[List[Tuple[int, Tuple[int, int], list]]]:
    entity_info = []
    for token_chunk, label_chunk in zip(tokens, labels):
        entity_info_for_chunk = []
        entity_idx, entity_start, entity_words = None, None, None
        # hint: we add some parts to the beginning and the end of every token/label chunk:
        # > beginning: we add a dummy token, as a class token is added by the tokenizer. if we ignored it here, it would mess up our span indices
        # > end: we add a dummy token so that a potential entity at the end is recorded correctly
        token_label_chunk = zip(
            [''] + token_chunk + [''],
            [EntityIndex.NO_ENTITY.value] + label_chunk + [EntityIndex.NO_ENTITY.value]
        )
        for idx, (token, label) in enumerate(token_label_chunk):
            if label <= EntityIndex.NO_ENTITY.value:  # no entity or special token
                if entity_start is not None:  # record previous entity
                    entity_info_for_chunk.append((entity_idx, (entity_start, idx), entity_words))
                entity_idx, entity_start, entity_words = None, None, None
            else:
                if label == entity_idx:  # inside entity -> only add token to current entity
                    entity_words.append(token)
                    continue
                if entity_start is not None:  # record previous entity
                    entity_info_for_chunk.append((entity_idx, (entity_start, idx), entity_words))
                entity_idx = label
                entity_start = idx
                entity_words = [token]
        entity_info.append(entity_info_for_chunk)
    return entity_info


def _filter_truncated_entities(tokens: List[List[str]], entity_info: List[List[Tuple[int, Tuple[int, int], list]]], tokenizer) -> tuple:
    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    filtered_tokens, filtered_entity_info = [], []
    for token_chunk, entity_info_chunk, offset_mapping_chunk in zip(tokens, entity_info, encodings['offset_mapping']):
        # discard entity mentions that are not in the tokenized text (due to truncation)
        tokens_after_tokenization = len([o for o in offset_mapping_chunk if o[0] == 0])
        filtered_entity_info_chunk = [eic for eic in entity_info_chunk if eic[1][1] <= tokens_after_tokenization]
        if not filtered_entity_info_chunk:
            continue  # discard whole chunk as it does not contain any labeled entities
        filtered_tokens.append(token_chunk)
        filtered_entity_info.append(filtered_entity_info_chunk)
    return filtered_tokens, filtered_entity_info


def _process_entity_info(entity_info: List[List[Tuple[int, Tuple[int, int], list]]], offset_mapping: List[List[Tuple[int, int]]], num_ents: int, items_per_chunk: int) -> tuple:
    num_chunks = len(entity_info)
    mention_spans = np.zeros((num_chunks, num_ents, 2))
    entity_indices = np.zeros((num_chunks, num_ents))
    entity_labels = np.ones((num_chunks, num_ents)) * -100  # initialize all with label: ignore
    ents_per_item = num_ents // items_per_chunk
    # prepare word blocker to retrieve entities with similar surface forms
    dbr = DbpResourceStore.instance()
    entity_surface_forms = {e_idx: dbr.get_resource_by_idx(e_idx).get_surface_forms() for e_idx in dbr.get_embedding_vectors()}
    word_blocker = WordBlocker(entity_surface_forms)
    # process entity info
    for chunk_idx, (chunk_entity_info, chunk_offset_mapping) in enumerate(zip(entity_info, offset_mapping)):
        # find start token of every word
        word_offsets = [idx for idx, offset in enumerate(chunk_offset_mapping) if offset[0] == 0]
        word_offsets += [len(chunk_offset_mapping)]  # add end index for final word
        for ent_info_idx, ent_info in enumerate(chunk_entity_info):
            ent_idx, (ent_start, ent_end), ent_words = ent_info
            item_start_idx = ent_info_idx * ents_per_item  # every item has ents_per_item slots
            item_end_idx = item_start_idx + ents_per_item
            # set all slots of the item to the same mention span
            mention_spans[chunk_idx, item_start_idx:item_end_idx] = (word_offsets[ent_start], word_offsets[ent_end])
            # retrieve related entities via surface forms
            related_ent_indices = list(word_blocker.get_entity_indices_for_words(ent_words))
            # first slot of the item is the actual entity; remaining slots are entities with similar surface forms
            if ent_idx >= 0:
                # add correct entity
                entity_indices[chunk_idx, item_start_idx] = ent_idx
                entity_labels[chunk_idx, item_start_idx] = 1  # label: correct entity
                # add related entities
                related_ents_to_add = min(len(related_ent_indices), ents_per_item - 1)
                re_start_idx = item_start_idx + 1
            else:
                # if ent_idx is negative, this means it is a new entity; so we only add related entities
                related_ents_to_add = min(len(related_ent_indices), ents_per_item)
                re_start_idx = item_start_idx
            re_end_idx = re_start_idx + related_ents_to_add
            entity_indices[chunk_idx, re_start_idx:re_end_idx] = related_ent_indices[:related_ents_to_add]
            entity_labels[chunk_idx, re_start_idx:re_end_idx] = 0  # label: incorrect entity
    return mention_spans.tolist(), entity_indices.tolist(), entity_labels.tolist()
