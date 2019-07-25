import impl.list.mapping as list_mapping
import impl.dbpedia.store as dbp_store
import impl.category.store as cat_store
import impl.category.cat2ax as cat_axioms
import impl.category.base as cat_base
import impl.util.nlp as nlp_util
import pandas as pd
import numpy as np
import util
from sklearn.preprocessing import OneHotEncoder


# COMPUTATION OF BASIC ENTITY FEATURES

def make_entity_features(lp_data: dict) -> pd.DataFrame:
    lp_uri = lp_data['uri']
    sections = lp_data['sections']

    # lp feature statistics
    lp_section_count = len(sections)
    lp_section_entries = [len(section['entries']) for section in sections]
    lp_entry_depths = []
    lp_entry_entities = []
    lp_entry_words = []
    lp_entry_chars = []
    lp_entry_commas = []
    lp_first_entity_idx = []
    lp_first_entity_pos = []

    data = []
    for section_idx, section_data in enumerate(sections):
        section_name = section_data['name']

        entries = section_data['entries']
        for entry_idx, entry_data in enumerate(entries):
            entry_doc = nlp_util.parse(entry_data['text'], disable_normalization=True)

            entities = entry_data['entities']
            lp_entry_depths.append(entry_data['depth'])
            lp_entry_entities.append(len(entities))
            lp_entry_words.append(len(entry_data['text'].split(' ')))
            lp_entry_chars.append(len(entry_data['text']))
            lp_entry_commas.append(entry_data['text'].count(','))
            for entity_idx, entity_data in enumerate(entities):
                entity_uri = entity_data['uri']
                entity_uri = entity_uri[:entity_uri.index('#')] if '#' in entity_uri else entity_uri

                entity_span = _get_span_for_entity(entry_doc, entity_data['text'], entity_data['idx'])
                if not entity_span:
                    continue

                if entity_idx == 0:
                    lp_first_entity_idx.append(entity_span.start)
                    lp_first_entity_pos.append(_get_relative_position(entity_span.start, len(entry_doc)))

                features = {
                    # ID
                    '_id': f'{lp_uri}__{section_name}__{entry_idx}__{entity_uri}',
                    '_listpage_uri': lp_uri,
                    '_section_name': section_name or '',
                    '_entry_idx': entry_idx,
                    '_entity_uri': entity_uri,
                    # ENTITY FEATURES
                    'section_pos': _get_relative_position(section_idx, len(sections)),
                    'section_invpos': _get_relative_position(section_idx, len(sections), inverse=True),
                    'entry_pos': _get_relative_position(entry_idx, len(entries)),
                    'entry_invpos': _get_relative_position(entry_idx, len(entries), inverse=True),
                    'entry_depth': entry_data['depth'],
                    'entry_leaf': entry_data['leaf'],
                    'entity_count': len(entities),
                    'entity_idx': entity_span.start,
                    'entity_invidx': len(entry_doc) - entity_span.end,
                    'entity_pos': _get_relative_position(entity_span.start, len(entry_doc)),
                    'entity_invpos': _get_relative_position(entity_span.end - 1, len(entry_doc), inverse=True),
                    'entity_link_pos': _get_relative_position(entity_idx, len(entities)),
                    'entity_link_invpos': _get_relative_position(entity_idx, len(entities), inverse=True),
                    'entity_first': entity_idx == 0,
                    'entity_last': (entity_idx + 1) == len(entities),
                    'entity_pn': any(w.tag_ in ['NNP', 'NNPS'] for w in entity_span),
                    'entity_noun': any(w.pos_ == 'NOUN' for w in entity_span),
                    'entity_ne': any(w.ent_type_ for w in entity_span),
                    'prev_pos': entry_doc[entity_span.start - 1].pos_ if entity_span.start > 0 else 'START',
                    'prev_ne': bool(entry_doc[entity_span.start - 1].ent_type_) if entity_span.start > 0 else False,
                    'succ_pos': entry_doc[entity_span.end].pos_ if entity_span.end < len(entry_doc) else 'END',
                    'succ_ne': bool(entry_doc[entity_span.end].ent_type_) if entity_span.end < len(entry_doc) else False,
                    'comma_idx': len([w for w in entry_doc[0:entity_span.start] if w.text == ','])
                }

                data.append(features)

    # LISTPAGE FEATURES
    for feature_set in data:
        feature_set['lp_section_count'] = lp_section_count
        _assign_avg_and_std_to_feature_set(feature_set, lp_section_entries, 'lp_section_entry')
        _assign_avg_and_std_to_feature_set(feature_set, lp_entry_depths, 'lp_entry_depth')
        _assign_avg_and_std_to_feature_set(feature_set, lp_entry_entities, 'lp_entry_entity')
        _assign_avg_and_std_to_feature_set(feature_set, lp_entry_words, 'lp_entry_word')
        _assign_avg_and_std_to_feature_set(feature_set, lp_entry_chars, 'lp_entry_char')
        _assign_avg_and_std_to_feature_set(feature_set, lp_entry_commas, 'lp_entry_comma')
        _assign_avg_and_std_to_feature_set(feature_set, lp_first_entity_idx, 'lp_first_entity_idx')
        _assign_avg_and_std_to_feature_set(feature_set, lp_first_entity_pos, 'lp_first_entity_pos')

    return pd.DataFrame(data)


def _get_span_for_entity(doc, text, idx):
    span = doc.char_span(idx, idx + len(text))

    if span is None or span.text != text:
        raise ValueError(f'Trying to find "{text}" in "{doc}" starting at index {idx} but failed.')

    return span


def _get_relative_position(idx, total, inverse=False):
    total = total - 1
    if total == 0:
        return 0
    return (total - idx) / total if inverse else idx / total


def _assign_avg_and_std_to_feature_set(feature_set: dict, data: list, name: str):
    feature_set[f'{name}_avg'] = np.mean(data)
    feature_set[f'{name}_std'] = np.std(data)


# COMPUTATION OF ENTITY LABELS

def assign_entity_labels(df: pd.DataFrame):
    df['label'] = df.apply(lambda row: _compute_label_for_entity(row['_listpage_uri'], row['_entity_uri']), axis=1)

    if util.get_config('list.extraction.use_negative_evidence_assumption'):
        # -- ASSUMPTION: an entry of a list page has at most one positive example --
        # locate all entries that have a positive example
        positive_examples = set()
        for _, row in df[df['label'] == 1].iterrows():
            positive_examples.add((row['_listpage_uri'], row['_section_name'], row['_entry_idx']))
        # make all candidate examples negative that appear in an entry with a positive example
        for i, row in df[df['label'] == -1].iterrows():
            if (row['_listpage_uri'], row['_section_name'], row['_entry_idx']) in positive_examples:
                df.at[i, 'label'] = 0


def _compute_label_for_entity(listpage_uri: str, entity_uri: str) -> int:
    if entity_uri not in dbp_store.get_resources():
        return 0

    listpage_axioms = set()
    category_resources = set()

    listpage_categories = list_mapping.get_equivalent_categories(listpage_uri) or list_mapping.get_parent_categories(listpage_uri)
    for cat in listpage_categories:
        for p_cat in ({cat} | cat_base.get_wikitaxonomy_graph().ancestors(cat)):
            listpage_axioms.update(cat_axioms.get_axioms(p_cat))
        for s_cat in ({cat} | cat_base.get_wikitaxonomy_graph().descendants(cat)):
            category_resources.update(cat_store.get_resources(s_cat))

    if entity_uri in category_resources:
        return 1
    elif any(ax.rejects_resource(entity_uri) for ax in listpage_axioms):
        return 0
    return -1


# COMPUTATION OF SECTION-NAME FEATURES

def with_section_name_features(df: pd.DataFrame) -> pd.DataFrame:
    # retrieve the 50 sections that appear most often in list pages
    valid_sections = df[['_section_name', '_listpage_uri']].groupby(by='_section_name')[['_listpage_uri']].nunique().sort_values(by='_listpage_uri', ascending=False).head(50).index.values
    # one-hot-encode these sections
    encoder = OneHotEncoder(categories={0: valid_sections}, handle_unknown='ignore', sparse=False)
    section_values = encoder.fit_transform(df[['_section_name']].fillna(value=''))
    section_names = encoder.get_feature_names()
    df_section = pd.DataFrame(data=section_values, columns=section_names)
    return pd.merge(left=df, right=df_section, left_index=True, right_index=True)
