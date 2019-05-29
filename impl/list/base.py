import pandas as pd
import util
import impl.category.cat2ax as cat_axioms
from . import parser as list_parser
from . import features as list_features
from . import store as list_store


def get_listpage_entities_trainingset() -> pd.DataFrame:
    entities = None
    for lp, lp_data in get_parsed_listpages().items():
        if lp_data['type'] != list_parser.LIST_TYPE_ENUM:
            continue

        cat = list_store.get_equivalent_category(lp)
        if not cat:
            continue
        if not cat_axioms.get_axioms(cat):
            continue

        lp_entities = list_features.make_entity_features(lp_data)
        entities = entities.append(lp_entities, ignore_index=True) if entities is not None else lp_entities
    return entities


def get_parsed_listpages() -> dict:
    global __PARSED_LISTPAGES__
    if '__PARSED_LISTPAGES__' not in globals():
        __PARSED_LISTPAGES__ = util.load_or_create_cache('dbpedia_listpage_parsed', _compute_parsed_listpages)

    return __PARSED_LISTPAGES__


def _compute_parsed_listpages() -> dict:
    return {lp: list_parser.parse_listpage(lp, list_store.get_listpage_markup(lp)) for lp in list_store.get_listpages()}
