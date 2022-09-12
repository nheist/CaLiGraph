from typing import Dict, List, Union, Tuple, Set, Optional
from collections import defaultdict
from impl.util.transformer import EntityIndex
from impl.wikipedia.page_parser import WikiPage, WikiEnumEntry, WikiTableRow, WikiMention
from impl.subject_entity.mention_detection.labels import heuristics


def get_labels(pages: List[WikiPage]) -> Dict[int, Dict[int, Dict[int, List[Union[int, List[int]]]]]]:
    subject_entities = {page.idx: heuristics.find_subject_entities_for_page(page) for page in pages}
    return _get_labels_for_subject_entities(pages, subject_entities)


def _get_labels_for_subject_entities(pages: List[WikiPage], subject_entities: Dict[int, Dict[int, Tuple[Set[int], Set[int]]]]) -> Dict[int, Dict[int, Dict[int, List[Union[int, List[int]]]]]]:
    labels = defaultdict(lambda: defaultdict(dict))
    for page in pages:
        for listing in page.get_listings():
            if listing.idx not in subject_entities[page.idx]:
                continue
            for item in listing.get_items():
                item_subject_entities = subject_entities[page.idx][listing.idx]
                item_labels = _get_entry_labels(item, item_subject_entities) if isinstance(item, WikiEnumEntry) else _get_row_labels(item, item_subject_entities)
                if item_labels is not None:
                    labels[page.idx][listing.idx][item.idx] = item_labels
    return labels


def _get_entry_labels(item: WikiEnumEntry, subject_entities: Tuple[Set[int], Set[int]]) -> Optional[List[str]]:
    valid_subject_entities, invalid_subject_entities = subject_entities
    mention_entities = {m.entity_idx for m in item.get_mentions()}
    item_subject_entities = mention_entities.intersection(valid_subject_entities)
    has_only_known_entities = mention_entities.issubset(invalid_subject_entities)
    if not item_subject_entities and not has_only_known_entities:
        return None
    labels, _ = _get_labels_for_tokens(item.tokens, item.mentions, item_subject_entities, False)
    return labels


def _get_row_labels(item: WikiTableRow, subject_entities: Tuple[Set[int], Set[int]]) -> Optional[List[List[str]]]:
    valid_subject_entities, invalid_subject_entities = subject_entities
    mention_entities = {m.entity_idx for m in item.get_mentions()}
    item_subject_entities = mention_entities.intersection(valid_subject_entities)
    has_only_known_entities = mention_entities.issubset(invalid_subject_entities)
    if not item_subject_entities and not has_only_known_entities:
        return None

    labels = []
    found_entity = False
    for cell_tokens, cell_mentions in zip(item.tokens, item.mentions):
        cell_labels, found_entity = _get_labels_for_tokens(cell_tokens, cell_mentions, item_subject_entities, found_entity)
        labels.append(cell_labels)
    return labels


def _get_labels_for_tokens(tokens: List[str], mentions: List[WikiMention], subject_entities: Set[int], found_entity: bool) -> Tuple[List[str], bool]:
    labels = [EntityIndex.NO_ENTITY.value] * len(tokens)
    if found_entity:
        return labels, found_entity  # found subject entity already, so we do not need to find another
    for mention in mentions:
        if mention.entity_idx in subject_entities:
            labels[mention.start:mention.end] = [mention.entity_idx] * (mention.end - mention.start)
            found_entity = True
            break
    return labels, found_entity
