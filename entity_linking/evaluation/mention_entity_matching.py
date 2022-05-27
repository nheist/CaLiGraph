import torch
from transformers import EvalPrediction


class MentionEntityMatchingEvaluator:
    def __init__(self, batch_size: int, num_ents: int, items_per_chunk: int):
        self.batch_size = batch_size
        self.num_ents = num_ents
        self.items_per_chunk = items_per_chunk

    def evaluate(self, eval_prediction: EvalPrediction):
        # process results item-wise
        ents_per_item = self.num_ents // self.items_per_chunk
        labels = torch.from_numpy(eval_prediction.label_ids)  # (batches*bs, num_ents)
        labels = labels.view(-1, ents_per_item)  # (batches*bs*items_per_batch, ents_per_item)
        preds = torch.from_numpy(eval_prediction.predictions)  # (batches*bs, num_ents)
        preds = preds.view(-1, ents_per_item)  # (batches*bs*items_per_batch, ents_per_item)
        preds = torch.round(preds)
        # ignore invalid entities by setting their predictions to be always correct
        invalid_ent_mask = labels.lt(0)
        preds[invalid_ent_mask] = labels[invalid_ent_mask]
        # compare preds and labels; aggregate on item level
        itemwise_result = preds.eq(labels).all(dim=-1)
        # compute mask for known (label of first entity is 1) and unknown (label of first entity is <1) item entities
        known_entity_mask = labels[:, 0].eq(1)
        # compute metrics
        known_entity_count = known_entity_mask.sum().item()
        known_entity_correct_predictions = itemwise_result[known_entity_mask].sum().item()
        unknown_entity_count = len(known_entity_mask) - known_entity_count
        unknown_entity_correct_predictions = itemwise_result[~known_entity_mask].sum().item()
        all_entity_count = known_entity_count + unknown_entity_count
        all_entity_correct_predictions = known_entity_correct_predictions + unknown_entity_correct_predictions
        return {
            'binary_all': all_entity_correct_predictions / all_entity_count,
            'binary_known': known_entity_correct_predictions / known_entity_count,
            'binary_unknown': unknown_entity_correct_predictions / unknown_entity_count,
        }
