from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import EntityIndex
from impl.subject_entity.mention_detection.labels import entity_type


class MentionDetectionDataset(Dataset):
    def __init__(self, tokenizer, tokens: List[List[str]], labels: List[List[int]], listing_types: List[str], binary_labels: bool):
        self.encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        labels = entity_type.map_entities_to_type_labels(labels, binary_labels)
        self.labels = _encode_labels(labels, self.encodings)
        self.encodings.pop('offset_mapping')  # we don't want to pass this to the model
        self.listing_types = listing_types

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label_ids'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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
            doc_labels += [EntityTypeLabel.NONE.value] * (truncated_label_length - len(doc_labels))
        doc_enc_labels[relevant_label_mask] = doc_labels[:truncated_label_length]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels
