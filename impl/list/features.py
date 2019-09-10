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
from collections import defaultdict


# COMPUTATION OF BASIC ENTITY FEATURES OF ENUM LISTPAGES

def make_enum_entity_features(lp_data: dict) -> list:
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

    return data


def make_table_entity_features(lp_data: dict) -> list:
    lp_uri = lp_data['uri']
    sections = lp_data['sections']

    # lp feature statistics
    lp_section_count = len(sections)
    lp_table_count = sum([len(section['tables']) for section in sections])
    lp_section_tables = [len(section['tables']) for section in sections]
    lp_table_rows = [len(table) for section in sections for table in section['tables']]
    lp_table_columns = [len(row) for section in sections for table in section['tables'] for row in table]
    lp_table_column_words = []
    lp_table_column_chars = []
    lp_table_row_entities = []
    lp_table_column_entities = []
    lp_table_first_entity_column = []

    data = []
    for section_idx, section_data in enumerate(sections):
        section_name = section_data['name']

        tables = section_data['tables']
        for table_idx, table in enumerate(tables):

            for row_idx, row in enumerate(table):
                lp_table_row_entities.append(sum([len(col['entities']) for col in row]))

                for column_idx, column_data in enumerate(row):
                    if column_data['entities']:
                        lp_table_first_entity_column.append(column_idx)
                        break

                for column_idx, column_data in enumerate(row):
                    column_text = column_data['text']
                    lp_table_column_words.append(len(column_text.split(' ')))
                    lp_table_column_chars.append(len(column_text))

                    column_doc = nlp_util.parse(column_text, disable_normalization=True)

                    column_entities = column_data['entities']
                    lp_table_column_entities.append(len(column_entities))

                    for entity_idx, entity_data in enumerate(column_entities):
                        entity_uri = entity_data['uri']
                        entity_uri = entity_uri[:entity_uri.index('#')] if '#' in entity_uri else entity_uri

                        entity_span = _get_span_for_entity(column_doc, entity_data['text'], entity_data['idx'])
                        if not entity_span:
                            continue

                        features = {
                            # ID
                            '_id': f'{lp_uri}__{section_name}__{table_idx}__{row_idx}__{column_idx}__{entity_uri}',
                            '_listpage_uri': lp_uri,
                            '_section_name': section_name or '',
                            '_table_idx': table_idx,
                            '_row_idx': row_idx,
                            '_column_idx': column_idx,
                            '_entity_uri': entity_uri,
                            # ENTITY FEATURES
                            'section_pos': _get_relative_position(section_idx, len(sections)),
                            'section_invpos': _get_relative_position(section_idx, len(sections), inverse=True),
                            'table_pos': _get_relative_position(table_idx, len(tables)),
                            'table_invpos': _get_relative_position(table_idx, len(tables), inverse=True),
                            'table_count': len(tables),
                            'row_pos': _get_relative_position(row_idx, len(table)),
                            'row_invpos': _get_relative_position(row_idx, len(table), inverse=True),
                            'row_count': len(table),
                            'column_pos': _get_relative_position(column_idx, len(row)),
                            'column_invpos': _get_relative_position(column_idx, len(row), inverse=True),
                            'column_count': len(row),
                            'entity_link_pos': _get_relative_position(entity_idx, len(column_entities)),
                            'entity_link_invpos': _get_relative_position(entity_idx, len(column_entities), inverse=True),
                            'entity_first': entity_idx == 0,
                            'entity_last': (entity_idx + 1) == len(column_entities),
                            'entity_count': len(column_entities),
                            'entity_idx': entity_span.start,
                            'entity_invidx': len(column_doc) - entity_span.end,
                            'entity_pos': _get_relative_position(entity_span.start, len(column_doc)),
                            'entity_invpos': _get_relative_position(entity_span.end - 1, len(column_doc), inverse=True),
                            'entity_pn': any(w.tag_ in ['NNP', 'NNPS'] for w in entity_span),
                            'entity_noun': any(w.pos_ == 'NOUN' for w in entity_span),
                            'entity_ne': any(w.ent_type_ for w in entity_span),
                            'prev_pos': column_doc[entity_span.start - 1].pos_ if entity_span.start > 0 else 'START',
                            'prev_ne': bool(column_doc[entity_span.start - 1].ent_type_) if entity_span.start > 0 else False,
                            'succ_pos': column_doc[entity_span.end].pos_ if entity_span.end < len(column_doc) else 'END',
                            'succ_ne': bool(column_doc[entity_span.end].ent_type_) if entity_span.end < len(column_doc) else False
                        }

                        data.append(features)

    # LISTPAGE FEATURES
    for feature_set in data:
        feature_set['lp_section_count'] = lp_section_count
        feature_set['lp_table_count'] = lp_table_count
        _assign_avg_and_std_to_feature_set(feature_set, lp_section_tables, 'lp_section_tables')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_rows, 'lp_table_rows')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_row_entities, 'lp_table_row_entities')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_columns, 'lp_table_columns')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_column_words, 'lp_table_column_words')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_column_chars, 'lp_table_column_chars')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_column_entities, 'lp_table_column_entities')
        _assign_avg_and_std_to_feature_set(feature_set, lp_table_first_entity_column, 'lp_table_first_entity_column')

    return data


def _get_span_for_entity(doc, text, idx):
    span = doc.char_span(idx, idx + len(text))
    if span is None or span.text != text:
        return None
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
    listpage_valid_resources = {}
    listpage_axioms = defaultdict(set)

    listpage_uris = set(df['_listpage_uri'].unique())
    for idx, listpage_uri in enumerate(listpage_uris):
        if idx % 1000 == 0:
            util.get_logger().debug(f'List-Entities: Processed {idx} of {len(listpage_uris)}.')
        listpage_resources = set(df[df['_listpage_uri'] == listpage_uri]['_entity_uri'].unique())
        listpage_category_resources = {res for cat in _get_category_descendants_for_list(listpage_uri) for res in cat_store.get_resources(cat)}
        listpage_valid_resources[listpage_uri] = listpage_resources.intersection(listpage_category_resources)

        axioms = {ax for cat in _get_category_ancestors_for_list(listpage_uri) for ax in cat_axioms.get_axioms(cat)}
        listpage_axioms[listpage_uri] = axioms

    df['label'] = df.apply(lambda row: _compute_label_for_entity(row['_listpage_uri'], row['_entity_uri'], listpage_valid_resources, listpage_axioms), axis=1)

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


def _compute_label_for_entity(listpage_uri: str, entity_uri: str, lp_valid_resources: dict, lp_axioms: dict) -> int:
    if entity_uri in lp_valid_resources[listpage_uri]:
        return 1
    if not dbp_store.is_possible_resource(entity_uri):
        return 0
    if any(ax.rejects_resource(entity_uri) for ax in lp_axioms[listpage_uri]):
        return 0
    if any(ax.accepts_resource(entity_uri) for ax in lp_axioms[listpage_uri]):
        return 1
    return -1


def _get_category_ancestors_for_list(listpage_uri: str) -> set:
    categories = set()
    cat_graph = cat_base.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    ancestor_categories = {ancestor for cat in mapped_categories for ancestor in cat_graph.ancestors(cat)}
    for cat in mapped_categories | ancestor_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_category_descendants_for_list(listpage_uri: str) -> set:
    categories = set()
    cat_graph = cat_base.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    descendant_categories = {descendant for cat in mapped_categories for descendant in cat_graph.descendants(cat)}
    for cat in mapped_categories | descendant_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_categories_for_list(listpage_uri: str) -> set:
    return list_mapping.get_equivalent_categories(listpage_uri) | list_mapping.get_parent_categories(listpage_uri)


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
