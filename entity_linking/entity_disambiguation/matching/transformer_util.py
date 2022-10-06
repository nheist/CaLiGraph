from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import WikiListing
from impl.caligraph.entity import ClgEntity


def add_special_tokens(model: SentenceTransformer):
    word_embedding_model = model._first_module()
    word_embedding_model.tokenizer.add_tokens(list(SpecialToken.all_tokens()), special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


def prepare_listing_items(listings: List[WikiListing]) -> Dict[Tuple[int, int, int], str]:
    return {(l.page_idx, l.idx, i.idx): i.subject_entity.label for l in listings for i in l.get_items() if i.subject_entity is not None}


def prepare_entities(entities: List[ClgEntity]) -> Dict[int, str]:
    return {e.idx: e.get_label() for e in entities}
