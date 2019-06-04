import impl.list.store as list_store
import impl.category.store as cat_store
import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util
import pandas as pd
import numpy as np


def make_entity_features(lp_data: dict) -> pd.DataFrame:
    listpage_axioms = set()
    category_resources = set()

    listpage_category = list_store.get_equivalent_category(lp_data['uri'])
    if listpage_category:
        listpage_axioms = cat_axioms.get_axioms(listpage_category)
        category_resources = cat_store.get_resources(listpage_category)

    data = []
    sections = lp_data['sections']
    entries_per_section = [len(section['entries']) for section in sections]
    entities_per_entry = [len(entry['entities']) for section in sections for entry in section['entries']]
    depth_per_entry = [entry['depth'] for section in sections for entry in section['entries']]
    words_per_entry = [len(entry['text'].split(' ')) for section in sections for entry in section['entries']]
    chars_per_entry = [len(entry['text']) for section in sections for entry in section['entries']]
    commas_per_entry = [entry['text'].count(',') for section in sections for entry in section['entries']]

    for section_idx, section_data in enumerate(sections):
        section_name = section_data['name']

        entries = section_data['entries']
        for entry_idx, entry_data in enumerate(entries):
            entry_doc = nlp_util.parse(entry_data['text'])

            entities = entry_data['entities']
            for entity_idx, entity_data in enumerate(entities):
                entity_uri = entity_data['uri']
                entity_span = _get_span_for_entity(entry_doc, entity_data['text'])
                if not entity_span:
                    continue

                features = {
                    # ID
                    '_id': f'{lp_data["uri"]}__{section_name}__{entry_idx}__{entity_uri}',
                    '_listpage_uri': lp_data['uri'],
                    '_section_name': section_name,
                    '_entry_idx': entry_idx,
                    '_entity_uri': entity_uri,
                    # LP
                    'lp_section_count': len(sections),
                    'lp_section_entry_avg': np.average(entries_per_section),
                    'lp_section_entry_std': np.std(entries_per_section),
                    'lp_entry_depth_avg': np.average(depth_per_entry),
                    'lp_entry_depth_std': np.std(depth_per_entry),
                    'lp_entry_entity_avg': np.average(entities_per_entry),
                    'lp_entry_entity_std': np.std(entities_per_entry),
                    'lp_entry_word_avg': np.average(words_per_entry),
                    'lp_entry_word_std': np.std(words_per_entry),
                    'lp_entry_char_avg': np.average(chars_per_entry),
                    'lp_entry_char_std': np.std(chars_per_entry),
                    'lp_entry_comma_avg': np.average(commas_per_entry),
                    'lp_entry_comma_std': np.std(commas_per_entry),
                    # FEATURES
                    'section_idx': section_idx,
                    'section_invidx': len(sections) - section_idx - 1,
                    'section_name': section_name,
                    'entry_idx': entry_idx,
                    'entry_invidx': len(entries) - entry_idx - 1,
                    'entry_depth': entry_data['depth'],
                    'entity_idx': entity_span.start,
                    'entity_invidx': len(entry_doc) - entity_span.end,
                    'entity_link_idx': entity_idx,
                    'entity_pn': any(w.tag_ in ['NNP', 'NNPS'] for w in entity_span),
                    'entity_ne': any(w.ent_type_ for w in entity_span),
                    'prev_pos': entry_doc[entity_idx - 1].pos_ if entity_idx > 0 else 'START',
                    'prev_ne': bool(entry_doc[entity_idx - 1].ent_type_) if entity_idx > 0 else False,
                    'succ_pos': entry_doc[entity_idx + len(entity_span)].pos_ if entity_idx + len(entity_span) < len(entry_doc) else 'END',
                    'succ_ne': bool(entry_doc[entity_idx + len(entity_span)].ent_type_) if entity_idx + len(entity_span) < len(entry_doc) else False,
                    'comma_idx': len([w for w in entry_doc[0:entity_idx] if w.text == ','])
                }
                if entity_uri in category_resources:
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

    return None
