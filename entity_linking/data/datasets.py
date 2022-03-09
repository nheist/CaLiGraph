import torch
from torch.utils.data import Dataset
from impl.subject_entity import extract


class MentionDetectionDataset(Dataset):
    def __init__(self, encodings, mention_labels, type_labels=None):
        self.encodings = encodings
        self.mention_labels = mention_labels
        self.type_labels = type_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        mention_label = self.mention_labels[idx]
        if self.type_labels:
            # with type detection per chunk
            type_label = torch.zeros_like(mention_label, dtype=torch.long)
            type_label[0] = self.type_labels[idx]
            label = torch.tensor((mention_label, type_label))
        else:
            # with type detection per entity
            label = torch.tensor(mention_label)
        item['label'] = label
        return item

    def __len__(self):
        return len(self.mention_labels)


def prepare_mentiondetection_dataset(tokens: list, labels: list, tokenizer, predict_single_tag: bool):
    train_encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    type_labels = None
    if predict_single_tag:
        type_labels = [l[1] for l in labels]
        labels = [l[0] for l in labels]
    mention_labels = extract._encode_labels(labels, train_encodings)

    train_encodings.pop('offset_mapping')  # we don't want to pass this to the model
    train_dataset = MentionDetectionDataset(train_encodings, mention_labels, type_labels)
    return train_dataset
