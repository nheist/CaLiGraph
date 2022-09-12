import torch
from transformers import EvalPrediction


class MentionEntityMatchingEvaluator:
    def __init__(self, num_ents: int, items_per_chunk: int):
        self.num_ents = num_ents
        self.items_per_chunk = items_per_chunk
        self.thresholds = [.3, .4, .5, .6, .7, .8]

    def evaluate(self, eval_prediction: EvalPrediction):
        # process results item-wise
        ents_per_item = self.num_ents // self.items_per_chunk
        labels = torch.from_numpy(eval_prediction.label_ids)  # (batches*bs, num_ents)
        labels = labels.view(-1, ents_per_item)  # (batches*bs*items_per_batch, ents_per_item)
        preds = torch.from_numpy(eval_prediction.predictions)  # (batches*bs, num_ents)
        preds = preds.view(-1, ents_per_item)  # (batches*bs*items_per_batch, ents_per_item)
        # set the prediction probabilities for undefined entities to 0
        undefined_ent_mask = labels.lt(0)
        preds[undefined_ent_mask] = 0
        # distinguish between seen and unseen entities
        seen_entity_mask = labels[:, 0].eq(1)  # (batches*bs*items_per_batch)
        trivial_unseen_entity_mask = labels[:, 0].lt(0)  # (batches*bs*items_per_batch)
        # collect entity counts
        all_entity_count = len(seen_entity_mask)
        seen_entity_count = seen_entity_mask.sum().item()
        unseen_entity_count = all_entity_count - seen_entity_count
        trivial_unseen_entity_count = trivial_unseen_entity_mask.sum().item()
        results = {
            'MEM-CNT_all': all_entity_count,
            'MEM-CNT_seen': seen_entity_count,
            'MEM-CNT_unseen': unseen_entity_count,
            'MEM-CNT_unseen_trivial': trivial_unseen_entity_count,
        }
        # evaluate entities for several thresholds
        seen_preds_max = preds[seen_entity_mask].max(dim=-1)
        unseen_preds_max = preds[~seen_entity_mask].max(dim=-1)
        for t in self.thresholds:
            seen_entity_correct_predictions = seen_preds_max.indices.eq(0) & seen_preds_max.values.ge(t)
            unseen_entity_correct_predictions = unseen_preds_max.values.lt(t)
            all_entity_correct_predictions = seen_entity_correct_predictions + unseen_entity_correct_predictions
            results.update({
                f'MEM-ACC-{t}_all': all_entity_correct_predictions / all_entity_count,
                f'MEM-ACC-{t}_seen': seen_entity_correct_predictions / seen_entity_count,
                f'MEM-ACC-{t}_unseen': unseen_entity_correct_predictions / unseen_entity_count,
            })
        return results
