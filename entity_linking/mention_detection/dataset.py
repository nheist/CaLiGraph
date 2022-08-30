from typing import List
import torch
from torch.utils.data import Dataset
from impl.subject_entity.preprocess.pos_label import map_entities_to_pos_labels
from impl.subject_entity.preprocess import sample
from impl.subject_entity.preprocess.word_tokenize import WordTokenizedPage


class MentionDetectionDataset(Dataset):
    def __init__(self, contexts, encodings, mention_labels):
        self.listing_types = [c['listing_type'].value for c in contexts]
        self.encodings = encodings
        self.mention_labels = mention_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label_ids'] = torch.tensor(self.mention_labels[idx])
        return item

    def __len__(self):
        return len(self.mention_labels)


def prepare_dataset(page_data: List[WordTokenizedPage], tokenizer, ignore_tags: bool, single_item_chunks: bool, negative_sample_size: float = 0.0):
    max_items_per_chunk = 1 if single_item_chunks else 16
    contexts, tokens, _, entity_indices = sample._chunk_word_tokenized_pages(page_data, negative_sample_size, max_items_per_chunk)
    labels = map_entities_to_pos_labels(entity_indices, ignore_tags)
    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    mention_labels = sample._encode_labels(labels, encodings)

    encodings.pop('offset_mapping')  # we don't want to pass this to the model
    return MentionDetectionDataset(contexts, encodings, mention_labels)
