from typing import Dict
from collections import defaultdict
from impl.dbpedia.resource import DbpResourceStore
from impl.wikipedia import WikiPageStore
from impl.util.transformer import EntityIndex


def disambiguate_subject_entities():
    # TODO: implement a more sophisticated solution
    disambiguated_subject_entities = _disambiguate_by_creating_new_entities()
    WikiPageStore.instance().add_disambiguated_subject_entities(disambiguated_subject_entities)


def _disambiguate_by_creating_new_entities() -> Dict[int, Dict[int, Dict[int, int]]]:
    disambiguated_subject_entities = defaultdict(lambda: defaultdict(dict))
    next_entity_idx = DbpResourceStore.instance().get_highest_resource_idx() + 1
    for wp in WikiPageStore.instance().get_pages():
        for listing in wp.get_listings():
            for item in listing.get_items(has_subject_entity=True):
                if item.subject_entity.entity_idx == EntityIndex.NEW_ENTITY:
                    disambiguated_subject_entities[wp.idx][listing.idx][item.idx] = next_entity_idx
                    next_entity_idx += 1
    return disambiguated_subject_entities
