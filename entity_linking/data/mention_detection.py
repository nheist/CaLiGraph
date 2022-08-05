from typing import List
import torch
from torch.utils.data import Dataset
from impl.subject_entity.preprocess.pos_label import map_entities_to_pos_labels
from impl.subject_entity.preprocess import sample
from impl.subject_entity.preprocess.word_tokenize import WordTokenizedPage


class MentionDetectionDataset(Dataset):
    def __init__(self, contexts, encodings, mention_labels, type_labels=None):
        self.listing_types = [c['listing_type'].value for c in contexts]
        self.encodings = encodings
        self.mention_labels = mention_labels
        self.type_labels = type_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        mention_label = torch.tensor(self.mention_labels[idx])
        if self.type_labels:
            # with type detection per chunk
            type_label = torch.zeros_like(mention_label)
            type_label[0] = self.type_labels[idx]
            label = torch.stack((mention_label, type_label))
        else:
            # with type detection per entity
            label = mention_label
        item['label_ids'] = label
        return item

    def __len__(self):
        return len(self.mention_labels)


def prepare_dataset(page_data: List[WordTokenizedPage], tokenizer, ignore_tags: bool, predict_single_tag: bool, negative_sample_size: float = 0.0):
    contexts, tokens, _, entity_indices = sample._chunk_word_tokenized_pages(page_data, negative_sample_size)
    labels = map_entities_to_pos_labels(entity_indices, ignore_tags or predict_single_tag)

    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    type_labels = None
    if predict_single_tag:
        type_labels = [l[1] for l in labels]
        labels = [l[0] for l in labels]
    elif ignore_tags:
        labels = [l[0] for l in labels]

    mention_labels = sample._encode_labels(labels, encodings)

    encodings.pop('offset_mapping')  # we don't want to pass this to the model
    return MentionDetectionDataset(contexts, encodings, mention_labels, type_labels)
