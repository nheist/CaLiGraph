import pandas as pd
from . import parser as list_parser
from . import features as list_features


def get_listpage_entity_data() -> pd.DataFrame:
    entities = None
    for lp, lp_data in list_parser.get_parsed_listpages().items():
        if lp_data['type'] != list_parser.LIST_TYPE_ENUM:
            continue

        lp_entities = list_features.make_entity_features(lp_data)
        entities = entities.append(lp_entities, ignore_index=True) if entities is not None else lp_entities

    list_features.assign_entity_labels(entities)
    return entities
