import impl.list.store as list_store
import impl.category.store as cat_store
import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util
import pandas as pd
import util


# NOISE_SECTIONS = ['See also', 'References', 'External links', 'Sources and external links']


# Rules for labels
# 1) only create labels if we have axioms for list
# 2) if entity fulfills axiom -> TRUE
# 3) if entity contradicts axiom (for types: has type and type is not at least supertype of axiom type) -> FALSE
# 4) else -> no label


def make_entity_features(listpage_uri: str, parsed_listpage: list) -> pd.DataFrame:
    listpage_axioms = set()
    category_resources = set()

    listpage_category = list_store.get_equivalent_category(listpage_uri)
    if listpage_category:
        listpage_axioms = cat_axioms.get_axioms(listpage_category)
        category_resources = cat_store.get_resources(listpage_category)

    data = []
    for entry_idx, entry in enumerate(parsed_listpage):
        section_name = entry['section_name']
        section_idx = entry['section_idx']
        section_invidx = entry['section_invidx']

        entry_depth = entry['depth']
        entry_doc = nlp_util.parse(entry['text'], skip_cache=True)
        for entity_idx, entity in enumerate(entry['entities']):
            entity_uri = entity['uri']
            entity_span = _get_span_for_entity(entry_doc, entity['text'])
            if not entity_span:
                continue

            features = {
                '_id': f'{listpage_uri}__{section_name}__{entry_idx}__{entity_uri}',
                '_listpage_uri': listpage_uri,
                '_section_name': section_name,
                '_entry_idx': entry_idx,
                '_entity_uri': entity_uri,
                'section_idx': section_idx,
                'section_invidx': section_invidx,
                'section_name': section_name,
                'entry_idx': entry_idx,
                'entry_invidx': len(parsed_listpage) - entry_idx - 1,
                'entry_depth': entry_depth,
                'entity_idx': entity_span.start,
                'entity_invidx': len(entry_doc) - entity_span.end,
                'entity_link_idx': entity_idx,
                'entity_pn': any(w.tag_ in ['NNP', 'NNPS'] for w in entity_span),
                'entity_ne': any(w.ent_type_ for w in entity_span),
                'prev_pos': entry_doc[entity_idx - 1].tag_ if entity_idx > 0 else 'START',
                'prev_ne': bool(entry_doc[entity_idx - 1].ent_type_) if entity_idx > 0 else False,
                'succ_pos': entry_doc[entity_idx + len(entity_span)].tag_ if entity_idx + len(entity_span) < len(entry_doc) else 'END',
                'succ_ne': bool(entry_doc[entity_idx + len(entity_span)].ent_type_) if entity_idx + len(entity_span) < len(entry_doc) else False,
                'comma_idx': len([w for w in entry_doc[0:entity_idx] if w.text == ','])
            }
            if entity_uri in category_resources or any(ax.accepts_resource(entity_uri) for ax in listpage_axioms):
                features['label'] = 1
            elif any(ax.rejects_resource(entity_uri) for ax in listpage_axioms):
                features['label'] = 0
            else:
                features['label'] = -1

            data.append(features)

    return pd.DataFrame(data)


def _get_span_for_entity(doc, entity_text):
    entity_doc = nlp_util.parse(entity_text, skip_cache=True)
    for i in range(len(doc) - len(entity_doc) + 1):
        span = doc[i:i+len(entity_doc)]
        if span.text == entity_doc.text:
            return span

    util.get_logger().debug(f'Could not find "{entity_text}" in "{doc}" for span retrieval.')
    return None
