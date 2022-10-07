from typing import List, Tuple, Dict, Union
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, losses, InputExample
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import WikiListing
from impl.caligraph.entity import ClgEntity
from entity_linking.entity_disambiguation.data import DataCorpus


def add_special_tokens(model: Union[SentenceTransformer, CrossEncoder]):
    if isinstance(model, SentenceTransformer):
        word_embedding_model = model._first_module()
        tokenizer = word_embedding_model.tokenizer
        transformer = word_embedding_model.auto_model
    elif isinstance(model, CrossEncoder):
        tokenizer = model.tokenizer
        transformer = model.model
    else:
        raise ValueError(f'Invalid type for model: {type(model)}')
    tokenizer.add_tokens(list(SpecialToken.all_tokens()), special_tokens=True)
    transformer.resize_token_embeddings(len(tokenizer))


def get_loss_function(loss: str, model) -> nn.Module:
    if loss == 'COS':
        return losses.CosineSimilarityLoss(model=model)
    elif loss == 'RL':
        return losses.MultipleNegativesRankingLoss(model=model)
    elif loss == 'SRL':
        return losses.MultipleNegativesSymmetricRankingLoss(model=model)
    raise ValueError(f'Unknown loss identifier: {loss}')


def generate_training_data(training_set: DataCorpus, negatives: set, batch_size: int) -> DataLoader:
    source_input = prepare_listing_items(training_set.source)
    target_input = source_input if training_set.target is None else prepare_entities(training_set.target)
    input_examples = [InputExample(texts=[source_input[source_id], target_input[target_id]], label=1) for source_id, target_id in training_set.alignment]
    input_examples.extend([InputExample(texts=[source_input[source_id], target_input[target_id]], label=0) for source_id, target_id in negatives])
    return DataLoader(input_examples, shuffle=True, batch_size=batch_size)


def prepare_listing_items(listings: List[WikiListing]) -> Dict[Tuple[int, int, int], str]:
    return {(l.page_idx, l.idx, i.idx): i.subject_entity.label for l in listings for i in l.get_items() if i.subject_entity is not None}


def prepare_entities(entities: List[ClgEntity]) -> Dict[int, str]:
    return {e.idx: e.get_label() for e in entities}
