"""Extraction of features and labels for different list page layouts."""

import impl.list.mapping as list_mapping
import impl.list.util as list_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.heuristics as dbp_heur
import impl.category.store as cat_store
import impl.category.base as cat_base
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hyper_util
import impl.util.rdf as rdf_util
import pandas as pd
import numpy as np
import util
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import operator
from _collections import Counter
import spacy


# COMPUTATION OF BASIC ENTITY FEATURES OF ENUM LISTPAGES

def make_enum_entity_features(lp_data: dict) -> list:
    """Return a set of features for every entity in a enumeration list page."""
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
            if not entry_data['text'].strip():
                continue

            # load enum-specific NE-tagging model and use only first sentence of every entry
            # TODO: proper wrapping of NE model
            nlp = spacy.load('data_caligraph-NE/spacy-model_goldstandard_26p-full')
            entry_doc = nlp.parse(entry_data['text'])
            entry_doc = list(entry_doc.sents)[0]
            entry_text = entry_doc.text

            entities = entry_data['entities']
            # add link type (blue/red) to entities and collect entity boundaries
            entity_character_idxs = set()
            for entity_data in entities:
                entity_data['link_type'] = 'blue' if entity_data['uri'] in dbp_store.get_raw_resources() else 'red'
                start = entity_data['idx']
                end = start + len(entity_data['text'])
                entity_character_idxs.update(range(start, end))

            # find previously unlinked entities
            for ent in entry_doc.ents:
                start = ent.start_char
                end = ent.end_char
                text = ent.text
                if not entity_character_idxs.intersection(set(range(start, end))):
                    uri = rdf_util.name2uri(text, lp_uri + '#')
                    entities.append({'uri': uri, 'text': text, 'idx': start, 'link_type': 'grey'})
            entities = sorted(entities, key=lambda x: x['idx'])

            # collect features
            lp_entry_depths.append(entry_data['depth'])
            lp_entry_entities.append(len(entities))
            lp_entry_words.append(len(entry_text.split(' ')))
            lp_entry_chars.append(len(entry_text))
            lp_entry_commas.append(entry_text.count(','))
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
                    '_link_type': entity_data['link_type'],
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
                    'entity_ne': _extract_ne_tag(entity_span),
                    'prev_pos': entry_doc[entity_span.start - 1].pos_ if entity_span.start > 0 else 'START',
                    'prev_ne': bool(entry_doc[entity_span.start - 1].ent_type_) if entity_span.start > 0 else False,
                    'succ_pos': entry_doc[entity_span.end].pos_ if entity_span.end < len(entry_doc) else 'END',
                    'succ_ne': bool(entry_doc[entity_span.end].ent_type_) if entity_span.end < len(entry_doc) else False,
                    'comma_idx': len([w for w in entry_doc[0:entity_span.start] if w.text == ','])
                }

                data.append(features)

    for feature_set in data:
        # ENTITY-STATS FEATURES
        feature_set['entity_occurrence_count'] = len([fs for fs in data if fs['_entity_uri'] == feature_set['_entity_uri']]) - 1
        feature_set['entity_occurrence'] = feature_set['entity_occurrence_count'] > 0

        # LISTPAGE FEATURES
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
    """Return a set of features for every entity in a table list page."""
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
                    column_name = table[0][column_idx]['text'] if len(table[0]) > column_idx else ''
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
                            '_column_name': column_name,
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
                            'column_list_similar': _compute_column_list_similarity(operator.eq, lp_uri, column_name),
                            'column_list_synonym': _compute_column_list_similarity(hyper_util.is_synonym, lp_uri, column_name),
                            'column_list_hypernym': _compute_column_list_similarity(_is_hyper, lp_uri, column_name),
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
                            'entity_ne': _extract_ne_tag(entity_span),
                            'prev_pos': column_doc[entity_span.start - 1].pos_ if entity_span.start > 0 else 'START',
                            'prev_ne': bool(column_doc[entity_span.start - 1].ent_type_) if entity_span.start > 0 else False,
                            'succ_pos': column_doc[entity_span.end].pos_ if entity_span.end < len(column_doc) else 'END',
                            'succ_ne': bool(column_doc[entity_span.end].ent_type_) if entity_span.end < len(column_doc) else False
                        }

                        data.append(features)

    for feature_set in data:
        # ENTITY-STATS FEATURES
        feature_set['entity_in_same_table_count'] = len([fs for fs in data if fs['_entity_uri'] == feature_set['_entity_uri'] and fs['table_pos'] == feature_set['table_pos']]) - 1
        feature_set['entity_in_same_table'] = feature_set['entity_in_same_table_count'] > 0
        feature_set['entity_in_other_table_count'] = len([fs for fs in data if fs['_entity_uri'] == feature_set['_entity_uri'] and fs['table_pos'] != feature_set['table_pos']])
        feature_set['entity_in_other_table'] = feature_set['entity_in_other_table_count'] > 0

        # LISTPAGE FEATURES
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


def _extract_ne_tag(entity_span) -> str:
    """Return the (most common) named entity tag of an entity span."""
    tags = [w.ent_type_ for w in entity_span]
    if not tags:
        return 'NONE'
    tag_count = Counter(reversed(tags))  # reverse order to return tag of last entity (at same count)
    return sorted(dict(tag_count.items(), key=lambda x: x[1], reverse=True))[0][0]


def _compute_column_list_similarity(sim_func, listpage_uri, column_name):
    """Return similarity value of column header and list page name."""
    listpage_name = list_util.listpage2name(listpage_uri)
    listpage_lemmas = {w.lemma_ for w in nlp_util.parse(listpage_name, disable_normalization=True)}
    column_name_lemmas = {w.lemma_ for w in nlp_util.parse(str(column_name))}
    return 1 if any(sim_func(l, c) for l in listpage_lemmas for c in column_name_lemmas) else 0


def _is_hyper(word_a: str, word_b: str) -> bool:
    """Return True, if there is any kind of hypernymy-relationship between the two words."""
    return hyper_util.is_hypernym(word_a, word_b) or hyper_util.is_hypernym(word_b, word_a)


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

def assign_entity_labels(graph, df: pd.DataFrame):
    """Derive labels (1 for True, 0 for False, -1 for Undefined) for all entities in the data frame."""
    listpage_valid_resources = {}
    listpage_types = defaultdict(set)

    listpage_uris = set(df['_listpage_uri'].unique())
    for idx, listpage_uri in enumerate(listpage_uris):
        if idx % 1000 == 0:
            util.get_logger().debug(f'List-Entities: Processed {idx} of {len(listpage_uris)}.')
        listpage_resources = set(df[df['_listpage_uri'] == listpage_uri]['_entity_uri'].unique())
        listpage_category_resources = {res for cat in _get_category_descendants_for_list(listpage_uri) for res in cat_store.get_resources(cat)}
        listpage_valid_resources[listpage_uri] = listpage_resources.intersection(listpage_category_resources)

        caligraph_nodes = graph.get_nodes_for_part(listpage_uri)
        for n in caligraph_nodes:
            listpage_types[listpage_uri].update(dbp_store.get_independent_types(graph.get_dbpedia_types(n)))

    # assign positive and negative labels that are induced directly from the taxonomy
    df['label'] = df.apply(lambda row: _compute_label_for_entity(row['_listpage_uri'], row['_entity_uri'], listpage_valid_resources, listpage_types), axis=1)

    if util.get_config('list.extraction.use_negative_evidence_assumption'):
        # -- ASSUMPTION: an entry of a list page has at most one positive example --
        # locate all entries that have a positive example
        positive_examples = set()
        for _, row in df[df['label'] == 1].iterrows():
            positive_examples.add(_get_entry_id(df, row))
        # make all candidate examples negative that appear in an entry with a positive example
        for i, row in df[df['label'] == -1].iterrows():
            if _get_entry_id(df, row) in positive_examples:
                df.at[i, 'label'] = 0


def _get_entry_id(df: pd.DataFrame, row: pd.Series) -> tuple:
    if '_entry_idx' in df.columns:
        return row['_listpage_uri'], row['_section_name'], row['_entry_idx']
    else:
        return row['_listpage_uri'], row['_section_name'], row['_table_idx'], row['_row_idx']


def _compute_label_for_entity(listpage_uri: str, entity_uri: str, lp_valid_resources: dict, lp_types: dict) -> int:
    """Return a label for the entity based on links in the taxonomy graph."""
    entity_types = dbp_store.get_types(entity_uri)
    if entity_uri in lp_valid_resources[listpage_uri]:
        return 1
    if not dbp_store.is_possible_resource(entity_uri) or any(entity_types.intersection(dbp_heur.get_disjoint_types(t)) for t in lp_types[listpage_uri]):
        return 0
    return -1


def _get_category_descendants_for_list(listpage_uri: str) -> set:
    """Return the category that is most closely related to the given list page as well as all of its children."""
    categories = set()
    cat_graph = cat_base.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    descendant_categories = {descendant for cat in mapped_categories for descendant in cat_graph.descendants(cat)}
    for cat in mapped_categories | descendant_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_categories_for_list(listpage_uri: str) -> set:
    """Return category that is mapped to the list page."""
    return list_mapping.get_equivalent_categories(listpage_uri) | list_mapping.get_parent_categories(listpage_uri)


# ONE-HOT ENCODE FEATURE BASED ON LISTPAGE FREQUENCY

def onehotencode_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    # retrieve the 50 values that appear most often in list pages
    frequent_values = df[[feature_name, '_listpage_uri']].groupby(by=feature_name)[['_listpage_uri']].nunique().sort_values(by='_listpage_uri', ascending=False).head(50).index.values
    # one-hot-encode the most frequent values
    encoder = OneHotEncoder(categories={0: frequent_values}, handle_unknown='ignore', sparse=False)
    values = encoder.fit_transform(df[[feature_name]].fillna(value=''))
    names = encoder.get_feature_names()
    df_feature = pd.DataFrame(data=values, columns=names)
    return pd.merge(left=df, right=df_feature, left_index=True, right_index=True)
