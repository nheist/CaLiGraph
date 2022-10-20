from typing import List, Tuple, Dict, Union
from itertools import cycle, islice
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, losses, InputExample
import utils
from impl.util.string import alternate_iters_to_string
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import WikiListing, WikiTable, WikiListingItem, WikiEnumEntry
from impl.dbpedia.resource import DbpResourceStore
from impl.caligraph.entity import ClgEntity
from entity_linking.entity_disambiguation.data import DataCorpus


CXS = SpecialToken.CONTEXT_SEP.value
CXE = SpecialToken.CONTEXT_END.value
COL = SpecialToken.TABLE_COL.value
ROW = SpecialToken.TABLE_ROW.value


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


def generate_training_data(training_set: DataCorpus, negatives: list, batch_size: int, add_page_context: bool, add_listing_entities: bool, add_entity_abstract: bool, add_kg_info: bool) -> DataLoader:
    source_input = prepare_listing_items(training_set.source, add_page_context, add_listing_entities)
    target_input = source_input if training_set.target is None else prepare_entities(training_set.target, add_entity_abstract, add_kg_info)
    input_examples = [InputExample(texts=[source_input[source_id], target_input[target_id]], label=1) for source_id, target_id, _ in training_set.alignment]
    input_examples.extend([InputExample(texts=[source_input[source_id], target_input[target_id]], label=0) for source_id, target_id, _ in negatives])
    return DataLoader(input_examples, shuffle=True, batch_size=batch_size)


def prepare_listing_items(listings: List[WikiListing], add_page_context: bool, add_listing_entities: bool) -> Dict[Tuple[int, int, int], str]:
    utils.get_logger().debug('Preparing listing items..')
    result = {}
    if not add_page_context and not add_listing_entities:
        for l in listings:
            for i in l.get_items():
                se = i.subject_entity
                if se is None:
                    continue
                result[(l.page_idx, l.idx, i.idx)] = f'{se.label} {SpecialToken.get_type_token(se.entity_type)}'
        return result
    for listing in listings:
        prepared_context = _prepare_listing_context(listing)
        prepared_items = [_prepare_listing_item(item) for item in listing.get_items()]
        for idx, item in enumerate(listing.get_items()):
            item_se = item.subject_entity
            if item_se is None:
                continue
            item_id = (listing.page_idx, listing.idx, item.idx)
            # add subject entity, its type, and page context
            item_content = f' {CXS} '.join([f'{item_se.label} {SpecialToken.get_type_token(item_se.entity_type)}', prepared_context])
            # add item and `add_listing_entities` subsequent items (add items from start if no subsequent items left)
            item_content += ''.join(islice(cycle(prepared_items), idx, idx + add_listing_entities + 1))
            result[item_id] = item_content
    return result


def _prepare_listing_context(listing: WikiListing) -> str:
    res = DbpResourceStore.instance().get_resource_by_idx(listing.page_idx)
    ctx = [f'{res.get_label()} {SpecialToken.get_type_token(res.get_type_label())}', listing.topsection.title, listing.section.title]
    if isinstance(listing, WikiTable):
        ctx.append(_prepare_listing_item(listing.header))
    return f' {CXS} '.join(ctx) + f' {CXE} '


def _prepare_listing_item(item: WikiListingItem) -> str:
    if isinstance(item, WikiEnumEntry):
        tokens = [SpecialToken.get_entry_by_depth(item.depth)] + item.tokens
        whitespaces = [' '] + item.whitespaces[:-1] + [' ']
    else:  # WikiTableRow
        tokens, whitespaces = [], []
        for cell_tokens, cell_whitespaces in zip(item.tokens, item.whitespaces):
            tokens += [COL] + cell_tokens
            whitespaces += [' '] + cell_whitespaces[:-1] + [' ']
        tokens[0] = ROW  # special indicator for start of table row
    return alternate_iters_to_string(tokens, whitespaces)


def prepare_entities(entities: List[ClgEntity], add_entity_abstract: bool, add_kg_info: int) -> Dict[int, str]:
    utils.get_logger().debug('Preparing entities..')
    result = {}
    for e in entities:
        ent_description = [f'{e.get_label()} {SpecialToken.get_type_token(e.get_type_label())}']
        if add_entity_abstract:
            ent_description.append((e.get_abstract() or '')[:200])
        if add_kg_info:
            kg_info = [f'type = {t.get_label()}' for t in e.get_types()]
            prop_count = max(0, add_kg_info - len(kg_info))
            if prop_count > 0:
                props = list(e.get_properties(as_tuple=True))[:prop_count]
                kg_info += [f'{pred.get_label()} = {val.get_label() if isinstance(val, ClgEntity) else val}' for pred, val in props]
        result[e.idx] = f' {CXS} '.join(ent_description)
    return result
