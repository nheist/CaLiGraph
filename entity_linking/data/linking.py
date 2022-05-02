from typing import List, Tuple
import random
import torch
from torch.utils.data import Dataset
from impl.dbpedia.resource import DbpResourceStore
from entity_linking.preprocessing.blocking import WordBlocker
from impl.util.rdf import EntityIndex


class LinkingDataset(Dataset):
    def __init__(self, encodings: dict, mention_spans: List[List[Tuple[int, int]]], entity_indices: List[List[int]], num_ents: int):
        self.encodings = encodings
        self.mention_spans = mention_spans
        self.entity_indices = entity_indices
        self.num_ents = num_ents
        self.all_entity_indices = torch.LongTensor([e.idx for e in DbpResourceStore.instance().get_entities()])
        self.all_entity_indices = self.all_entity_indices[torch.randperm(len(self.all_entity_indices))]  # randomize

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # add empty mention spans (0,0) to pad the spans to `num_ents`
        mention_spans = torch.tensor(self.mention_spans[idx])
        spans_to_pad = self.num_ents - len(mention_spans)
        item['mention_spans'] = torch.nn.ZeroPad2d((0, 0, 0, spans_to_pad))(mention_spans)
        # pad entity indices with the value for NO ENTITY to indicate that the remaining entities are only fillers
        entity_labels = torch.tensor(self.entity_indices[idx])
        entity_labels_to_pad = self.num_ents - len(entity_labels)
        entity_labels = torch.nn.ConstantPad1d((0, entity_labels_to_pad), EntityIndex.NO_ENTITY.value)(entity_labels)
        # get a set of random entities to use as filler embeddings for new/no entities
        start_idx = idx * len(entity_labels) % len(self.all_entity_indices)
        end_idx = start_idx + len(entity_labels) % len(self.all_entity_indices)
        random_labels = self.all_entity_indices[start_idx:end_idx]
        # pass both as labels of the item
        item['label_ids'] = torch.stack((entity_labels, random_labels))
        return item

    def __len__(self):
        return len(self.mention_spans)


def prepare_linking_dataset(tokens: List[List[str]], labels: List[List[int]], tokenizer, num_ents: int):
    entity_info = _collect_entity_info(tokens, labels)
    tokens, entity_info = _filter_truncated_entities(tokens, entity_info, tokenizer)

    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    mention_spans = _get_mention_spans(entity_info, encodings['offset_mapping'])
    entity_indices = _get_entity_indices(entity_info, num_ents)
    encodings.pop('offset_mapping')  # we don't want to pass this to the model
    return LinkingDataset(encodings, mention_spans, entity_indices, num_ents)


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


def _get_entity_indices(entity_info: List[List[Tuple[int, Tuple[int, int], list]]], num_ents: int) -> List[List[int]]:
    all_entities = DbpResourceStore.instance().get_entities()
    word_blocker = WordBlocker({e.idx: e.get_surface_forms() for e in all_entities})

    entity_indices = []
    for entity_info_chunk in entity_info:
        # first add actual entities of chunk
        entity_indices_for_chunk = [e[0] for e in entity_info_chunk]
        # then fill with entities having similar surface forms
        surface_form_matches = {re for e in entity_info_chunk for re in word_blocker.get_entities_for_words(e[2])}
        entities_not_in_chunk = list(surface_form_matches.difference(set(entity_indices_for_chunk)))
        num_entities_to_add = min(len(entities_not_in_chunk), num_ents - len(entity_indices_for_chunk))
        entity_indices_for_chunk.extend(random.sample(entities_not_in_chunk, num_entities_to_add))
        entity_indices.append(entity_indices_for_chunk)
    return entity_indices
