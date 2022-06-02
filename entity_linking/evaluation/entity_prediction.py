import torch
from torch import Tensor
import torch.nn.functional as F
from collections import Counter, defaultdict
import statistics
from transformers import EvalPrediction
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper


class EntityPredictionEvaluator:
    def __init__(self, ent_idx2emb: EntityIndexToEmbeddingMapper, batch_size: int):
        self.ent_idx2emb = ent_idx2emb
        self.batch_size = batch_size
        self.thresholds = [.1, .2, .3, .4, .5, .6, .7]

    def evaluate(self, eval_prediction: EvalPrediction):
        results = defaultdict(list, {'EP_CNT_seen': [0], 'EP_CNT_unseen': [0], 'EP_CNT_all': [0]})
        # gather results batch-wise and then average over them
        labels = torch.from_numpy(eval_prediction.label_ids)  # (batches*bs, num_ents, 2)
        entity_vectors = torch.from_numpy(eval_prediction.predictions)  # (batches*bs, valid_ents, ent_dim)
        for label_batch, pred_batch in zip(*[torch.split(t, self.batch_size) for t in [labels, entity_vectors]]):
            entity_labels, entity_status = label_batch[:, 0].reshape(-1), label_batch[:, 1].reshape(-1)
            # compute masks for existing (status 0) and new (status -1) entities
            seen_entity_mask = entity_status.eq(0)  # (bs*valid_ents)
            seen_entity_targets = torch.arange(len(seen_entity_mask))[seen_entity_mask]  # (bs*valid_ents)
            unseen_entity_mask = entity_status.eq(-1)  # (bs*valid_ents)
            # retrieve embedding vectors for entity indices
            label_entity_vectors = self.ent_idx2emb(entity_labels)  # (bs*valid_ents, ent_dim)
            # compute cosine similarity between predictions and labels
            entity_vectors = pred_batch.view(-1, pred_batch.shape[-1])  # (bs*num_ents, ent_dim)
            entity_similarities = F.normalize(entity_vectors) @ F.normalize(label_entity_vectors).T  # (bs*valid_ents, bs*num_ents)
            # compute metrics
            seen_entity_count = seen_entity_mask.sum().item()
            seen_entity_correct_predictions = self._get_correct_predictions_for_seen_entities(entity_similarities, seen_entity_mask, seen_entity_targets)
            unseen_entity_count = unseen_entity_mask.sum().item()
            unseen_entity_correct_predictions = self._get_correct_predictions_for_unseen_entities(entity_similarities, unseen_entity_mask)
            all_entity_count = seen_entity_count + unseen_entity_count
            all_entity_correct_predictions = seen_entity_correct_predictions + unseen_entity_correct_predictions

            results['EP_CNT_seen'][0] += seen_entity_count
            results['EP_CNT_unseen'][0] += unseen_entity_count
            results['EP_CNT_all'][0] += all_entity_count
            for t in self.thresholds:
                if seen_entity_count:
                    results[f'EP_ACC-{t}_seen'].append(seen_entity_correct_predictions[t] / seen_entity_count)
                if unseen_entity_count:
                    results[f'EP_ACC-{t}_unseen'].append(unseen_entity_correct_predictions[t] / unseen_entity_count)
                results[f'EP_ACC-{t}_all'].append(all_entity_correct_predictions[t] / all_entity_count)
        return {label: statistics.fmean(vals) for label, vals in results.items()}  # average over batch results

    def _get_correct_predictions_for_seen_entities(self, entity_similarities: Tensor, mask: Tensor, targets: Tensor) -> Counter[float, int]:
        # a prediction for a seen entity is correct if its own vector is the most similar and similarity >= threshold
        result = Counter()
        for t in self.thresholds:
            seen_entity_similarities = entity_similarities[mask]
            target_most_similar = seen_entity_similarities.max(dim=-1).indices.eq(targets)
            target_greater_threshold = seen_entity_similarities[torch.arange(len(seen_entity_similarities)), targets].ge(t)
            result[t] = target_most_similar.logical_and(target_greater_threshold).sum().item()
        return result

    def _get_correct_predictions_for_unseen_entities(self, entity_similarities: Tensor, mask: Tensor) -> Counter[float, int]:
        # a prediction for a unseen entity is correct if there is no seen entity more similar than the threshold
        result = {t: entity_similarities[mask].max(dim=-1).values.lt(t).sum().item() for t in self.thresholds}
        return Counter(result)
