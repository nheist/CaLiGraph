from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from impl.dbpedia.resource import DbpResourceStore
from entity_linking.preprocessing.blocking import WordBlocker
from impl.util.rdf import EntityIndex


class EntityPredictionDataset(Dataset):
    def __init__(self, encodings: dict, mention_spans: List[List[Tuple[int, int]]], entity_indices: List[Tuple[List[int], List[int]]], num_ents: int):
        self.encodings = encodings
        self.mention_spans = mention_spans
        self.entity_indices = entity_indices
        self.num_ents = num_ents

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # add empty mention spans (0,0) to pad the spans to `num_ents`
        mention_spans = torch.tensor(self.mention_spans[idx])
        spans_to_pad = self.num_ents - len(mention_spans)
        item['mention_spans'] = torch.nn.ZeroPad2d((0, 0, 0, spans_to_pad))(mention_spans)
        item['label_ids'] = torch.tensor(self.entity_indices[idx])
        return item

    def __len__(self):
        return len(self.entity_indices)


def prepare_dataset(tokens: List[List[str]], labels: List[List[int]], tokenizer, num_ents: int):
    entity_info = _collect_entity_info(tokens, labels)
    tokens, entity_info = _filter_truncated_entities(tokens, entity_info, tokenizer)

    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    mention_spans = _get_mention_spans(entity_info, encodings['offset_mapping'])
    entity_indices = _get_entity_indices(entity_info, num_ents)
    encodings.pop('offset_mapping')  # we don't want to pass this to the model
    return EntityPredictionDataset(encodings, mention_spans, entity_indices, num_ents)


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


def _get_mention_spans(entity_info: List[List[Tuple[int, Tuple[int, int], list]]], offset_mapping: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    mention_spans = []
    for entity_info_chunk, offset_mapping_chunk in zip(entity_info, offset_mapping):
        # find start token of every word
        word_offsets = [idx for idx, offset in enumerate(offset_mapping_chunk) if offset[0] == 0]
        word_offsets += [len(offset_mapping_chunk)]  # add end index for final word
        # collect mention spans by finding token for start and end indices of entity mentions
        mention_spans.append([(word_offsets[e[1][0]], word_offsets[e[1][1]]) for e in entity_info_chunk])
    return mention_spans


def _get_entity_indices(entity_info: List[List[Tuple[int, Tuple[int, int], list]]], num_ents: int) -> List[Tuple[List[int], List[int]]]:
    dbr = DbpResourceStore.instance()
    valid_entity_indices = np.array(list(dbr.get_embedding_vectors()))
    entity_surface_forms = {e_idx: dbr.get_resource_by_idx(e_idx).get_surface_forms() for e_idx in valid_entity_indices}
    word_blocker = WordBlocker(entity_surface_forms)

    entity_indices = []
    for entity_info_chunk in entity_info:
        # first add true entities of chunk
        entity_indices_for_chunk = np.array([e[0] for e in entity_info_chunk])[:num_ents]
        # store entity status (existing=0, new=-1, no entity=-2)
        entity_status_for_chunk = [min(idx, 0) for idx in entity_indices_for_chunk]
        entity_status_for_chunk += [EntityIndex.NO_ENTITY.value] * (num_ents - len(entity_status_for_chunk))
        # replace new entities with random ones
        new_entity_mask = entity_indices_for_chunk == EntityIndex.NEW_ENTITY.value
        entity_indices_for_chunk[new_entity_mask] = np.random.choice(valid_entity_indices, size=new_entity_mask.sum(), replace=False)
        # then fill with negative entities
        # entities with similar surface forms as negatives
        surface_form_matches = {re for e in entity_info_chunk for re in word_blocker.get_entity_indices_for_words(e[2])}
        sf_entities_not_in_chunk = np.array(list(surface_form_matches.difference(set(entity_indices_for_chunk))))
        sf_entities_to_add = min(len(sf_entities_not_in_chunk), num_ents - len(entity_indices_for_chunk))
        entity_indices_for_chunk = np.concatenate([entity_indices_for_chunk, np.random.choice(sf_entities_not_in_chunk, size=sf_entities_to_add, replace=False)])
        # random entities as negatives (here we don't care whether the entities are already in the chunk
        # -> this is very unlikely with > 5M entities)
        random_entities_to_add = num_ents - len(entity_indices_for_chunk)
        entity_indices_for_chunk = np.concatenate([entity_indices_for_chunk, np.random.choice(valid_entity_indices, size=random_entities_to_add, replace=False)])
        entity_indices.append((list(entity_indices_for_chunk), entity_status_for_chunk))
    return entity_indices
