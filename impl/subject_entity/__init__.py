from typing import Dict
from collections import defaultdict
import torch
import utils
from . import extract, combine
from .preprocess import sample


def get_page_subject_entities() -> Dict[int, dict]:
    """Retrieve the extracted entities per page with context."""
    # TODO: discard matched entities without types to filter out invalid ones
    return combine.match_entities_with_uris(_get_subject_entity_predictions())


def _get_subject_entity_predictions() -> Dict[int, dict]:
    return defaultdict(dict, utils.load_or_create_cache('subject_entity_predictions', _make_subject_entity_predictions))


def _make_subject_entity_predictions() -> Dict[int, dict]:
    tokenizer, model = extract.get_tagging_tokenizer_and_model()
    uncleaned_predictions = extract.extract_subject_entities(tokenizer, model, sample.get_mention_detection_page_data())
    # discard single subject entities for sections and discard empty sections
    predictions = {}
    for page_idx, page_subject_entities in uncleaned_predictions.items():
        page_subject_entities = {ts: {s: ents for s, ents in ents_per_section.items() if len(ents) > 1} for ts, ents_per_section in page_subject_entities.items()}
        page_subject_entities = {ts: ents_per_section for ts, ents_per_section in page_subject_entities.items() if ents_per_section}
        predictions[page_idx] = page_subject_entities
    torch.cuda.empty_cache()  # flush GPU cache to free GPU for other purposes
    return predictions
