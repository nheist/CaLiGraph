"""Extraction of features for entities in enumerations and tables."""

import impl.list.util as list_util
import impl.list.nlp as list_nlp
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.hypernymy as hyper_util
import impl.util.rdf as rdf_util
import pandas as pd
import numpy as np
import util
from sklearn.preprocessing import OneHotEncoder
from collections import Counter, defaultdict
import operator


def make_enum_entity_features(page: tuple) -> list:
    """Return a set of features for every entity in an enumeration of a page."""
    page_uri, page_data = page
    sections = page_data['sections']
    top_section_name = ''

    # page feature statistics
    page_section_count = len(sections)
    page_section_enums = [len(section['enums']) for section in sections]
    page_section_entries = [sum([len(enum) for enum in section['enums']]) for section in sections]
    page_entry_depths = []
    page_entry_entities = []
    page_entry_words = []
    page_entry_chars = []
    page_entry_commas = []
    page_first_entity_idx = []
    page_first_entity_pos = []

    data = []
    entity_page_index = 0
    line_index = -1
    for section_idx, section_data in enumerate(sections):
        section_name = section_data['name']
        if section_data['level'] <= 2:
            top_section_name = section_name

        enums = section_data['enums']
        for enum_index, entries in enumerate(enums):
            for entry_idx, entry_data in enumerate(entries):
                line_index += 1
                entry_text = entry_data['text']
                entry_doc = list_nlp.parse(entry_text)

                entities = entry_data['entities']
                # add link type (blue/red) to entities and collect entity boundaries
                entity_character_idxs = set()
                for entity_data in entities:
                    if entity_data['uri'] in dbp_store.get_raw_resources():
                        entity_data['link_type'] = 'blue'
                    else:
                        entity_data['link_type'] = 'red'
                        # red link must be disambiguated, if user is not providing specific details of linked entity
                        if dbp_util.name2resource(entity_data['text']) == entity_data['uri']:
                            entity_data['uri'] = rdf_util.name2uri(entity_data['text'], page_uri + '__')
                    start = entity_data['idx']
                    end = start + len(entity_data['text'])
                    entity_character_idxs.update(range(start, end))

                # find previously unlinked entities
                if util.get_config('page.extraction.extract_unlinked_entities'):
                    for ent in entry_doc.ents:
                        start = ent.start_char
                        end = ent.end_char
                        text = ent.text
                        if not entity_character_idxs.intersection(set(range(start, end))) and len(text) > 1:
                            uri = rdf_util.name2uri(text, page_uri + '__')
                            entities.append({'uri': uri, 'text': text, 'idx': start, 'link_type': 'grey'})
                entities = sorted(entities, key=lambda x: x['idx'])

                # collect features
                page_entry_depths.append(entry_data['depth'])
                page_entry_entities.append(len(entities))
                page_entry_words.append(len(entry_text.split(' ')))
                page_entry_chars.append(len(entry_text))
                page_entry_commas.append(entry_text.count(','))
                for entity_idx, entity_data in enumerate(entities):
                    entity_uri = entity_data['uri']
                    entity_uri = entity_uri[:entity_uri.index('#')] if '#' in entity_uri else entity_uri

                    entity_span = _get_span_for_entity(entry_doc, entity_data['text'], entity_data['idx'])
                    if not entity_span:
                        continue

                    if entity_idx == 0:
                        page_first_entity_idx.append(entity_span.start)
                        page_first_entity_pos.append(_get_relative_position(entity_span.start, len(entry_doc)))

                    features = {
                        # ID
                        '_id': f'{page_uri}__{section_name}__{enum_index}__{entry_idx}__{entity_uri}',
                        '_page_uri': page_uri,
                        '_top_section_name': top_section_name,
                        '_section_name': section_name or '',
                        '_line_idx': line_index,
                        '_enum_idx': enum_index,
                        '_entry_idx': entry_idx,
                        '_entity_uri': entity_uri,
                        '_entity_page_idx': entity_page_index,
                        '_entity_line_idx': entity_idx,
                        '_link_type': entity_data['link_type'],
                        '_text': entity_data['text'],
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
                    entity_page_index += 1

    for feature_set in data:
        # ENTITY-STATS FEATURES
        feature_set['entity_occurrence_count'] = len([fs for fs in data if fs['_entity_uri'] == feature_set['_entity_uri']]) - 1
        feature_set['entity_occurrence'] = feature_set['entity_occurrence_count'] > 0

        # PAGE FEATURES
        feature_set['page_section_count'] = page_section_count
        _assign_avg_and_std_to_feature_set(feature_set, page_section_enums, 'page_section_enums')
        _assign_avg_and_std_to_feature_set(feature_set, page_section_entries, 'page_section_entry')
        _assign_avg_and_std_to_feature_set(feature_set, page_entry_depths, 'page_entry_depth')
        _assign_avg_and_std_to_feature_set(feature_set, page_entry_entities, 'page_entry_entity')
        _assign_avg_and_std_to_feature_set(feature_set, page_entry_words, 'page_entry_word')
        _assign_avg_and_std_to_feature_set(feature_set, page_entry_chars, 'page_entry_char')
        _assign_avg_and_std_to_feature_set(feature_set, page_entry_commas, 'page_entry_comma')
        _assign_avg_and_std_to_feature_set(feature_set, page_first_entity_idx, 'page_first_entity_idx')
        _assign_avg_and_std_to_feature_set(feature_set, page_first_entity_pos, 'page_first_entity_pos')

    return [tuple(d.values()) for d in data]


def get_enum_feature_names() -> list:
    return [
        '_id', '_page_uri', '_top_section_name', '_section_name', '_line_idx', '_enum_idx', '_entry_idx', '_entity_uri',
        '_entity_page_idx', '_entity_line_idx', '_link_type', '_text', 'section_pos', 'section_invpos', 'entry_pos',
        'entry_invpos', 'entry_depth', 'entry_leaf', 'entity_count', 'entity_idx', 'entity_invidx', 'entity_pos',
        'entity_invpos', 'entity_link_pos', 'entity_link_invpos', 'entity_first', 'entity_last', 'entity_pn',
        'entity_noun', 'entity_ne', 'prev_pos', 'prev_ne', 'succ_pos', 'succ_ne', 'comma_idx', 'entity_occurrence_count',
        'entity_occurrence', 'page_section_count', 'page_section_enums_avg', 'page_section_entry_avg',
        'page_entry_depth_avg', 'page_entry_entity_avg', 'page_entry_word_avg', 'page_entry_char_avg',
        'page_entry_comma_avg', 'page_first_entity_idx_avg', 'page_first_entity_pos_avg', 'page_section_enums_std',
        'page_section_entry_std', 'page_entry_depth_std', 'page_entry_entity_std', 'page_entry_word_std',
        'page_entry_char_std', 'page_entry_comma_std', 'page_first_entity_idx_std', 'page_first_entity_pos_std'
    ]


def make_table_entity_features(page: tuple) -> list:
    """Return a set of features for every entity in a table of the page."""
    page_uri, page_data = page
    sections = page_data['sections']
    top_section_name = ''

    # page feature statistics
    page_section_count = len(sections)
    page_table_count = sum([len(section['tables']) for section in sections])
    page_section_tables = [len(section['tables']) for section in sections]
    page_table_rows = [len(table['data']) for section in sections for table in section['tables']]
    page_table_columns = [len(row) for section in sections for table in section['tables'] for row in table['data']]
    page_table_column_words = []
    page_table_column_chars = []
    page_table_row_entities = []
    page_table_column_entities = []
    page_table_first_entity_column = []

    # compute lemmas for page-name / column-name similarity
    page_name = list_util.listpage2name(page_uri) if list_util.is_listpage(page_uri) else dbp_util.resource2name(page_uri)
    page_lemmas = {w.lemma_ for w in list_nlp.parse(page_name)}

    data = []
    entity_page_index = 0
    line_index = -1
    for section_idx, section_data in enumerate(sections):
        section_name = section_data['name']
        if section_data['level'] <= 2:
            top_section_name = section_name

        tables = section_data['tables']
        for table_idx, table in enumerate(tables):
            table_header = table['header']
            table_data = table['data']
            for row_idx, row in enumerate(table_data):
                entity_line_index = 0
                line_index += 1
                first_entity_column_found = False
                page_table_row_entities.append(sum([len(col['entities']) for col in row]))

                for column_idx, column_data in enumerate(row):
                    column_name = table_header[column_idx]['text'] if len(table_header) > column_idx else ''
                    column_name_lemmas = {w.lemma_ for w in list_nlp.parse(str(column_name))}
                    column_text = column_data['text']
                    column_doc = list_nlp.parse(column_text)
                    column_page_similar = _compute_column_page_similarity(operator.eq, page_lemmas, column_name_lemmas)
                    column_page_synonym = _compute_column_page_similarity(hyper_util.is_synonym, page_lemmas, column_name_lemmas)
                    column_page_hypernym = _compute_column_page_similarity(_is_hyper, page_lemmas, column_name_lemmas)

                    page_table_column_words.append(len(column_text.split(' ')))
                    page_table_column_chars.append(len(column_text))

                    column_entities = column_data['entities']
                    # add link type (blue/red) to entities and collect entity boundaries
                    entity_character_idxs = set()
                    for entity_data in column_entities:
                        if entity_data['uri'] in dbp_store.get_raw_resources():
                            entity_data['link_type'] = 'blue'
                        else:
                            entity_data['link_type'] = 'red'
                            # red link must be disambiguated, if user is not providing specific details of linked entity
                            if dbp_util.name2resource(entity_data['text']) == entity_data['uri']:
                                entity_data['uri'] = rdf_util.name2uri(entity_data['text'], page_uri + '__')
                        start = entity_data['idx']
                        end = start + len(entity_data['text'])
                        entity_character_idxs.update(range(start, end))
                    # find previously unlinked entities
                    if util.get_config('page.extraction.extract_unlinked_entities'):
                        for ent in column_doc.ents:
                            start = ent.start_char
                            end = ent.end_char
                            text = ent.text
                            if not entity_character_idxs.intersection(set(range(start, end))) and len(text) > 1:
                                uri = rdf_util.name2uri(text, page_uri + '__')
                                column_entities.append({'uri': uri, 'text': text, 'idx': start, 'link_type': 'grey'})
                    column_entities = sorted(column_entities, key=lambda x: x['idx'])

                    if not first_entity_column_found and column_entities:
                        page_table_first_entity_column.append(column_idx)
                        first_entity_column_found = True

                    page_table_column_entities.append(len(column_entities))
                    for entity_idx, entity_data in enumerate(column_entities):
                        entity_uri = entity_data['uri']
                        entity_uri = entity_uri[:entity_uri.index('#')] if '#' in entity_uri else entity_uri

                        entity_span = _get_span_for_entity(column_doc, entity_data['text'], entity_data['idx'])
                        if not entity_span:
                            continue

                        features = {
                            # ID
                            '_id': f'{page_uri}__{section_name}__{table_idx}__{row_idx}__{column_idx}__{entity_uri}',
                            '_page_uri': page_uri,
                            '_top_section_name': top_section_name,
                            '_section_name': section_name or '',
                            '_line_idx': line_index,
                            '_table_idx': table_idx,
                            '_row_idx': row_idx,
                            '_column_idx': column_idx,
                            '_column_name': column_name,
                            '_entity_uri': entity_uri,
                            '_entity_page_idx': entity_page_index,
                            '_entity_line_idx': entity_line_index,
                            '_link_type': entity_data['link_type'],
                            '_text': entity_data['text'],
                            # ENTITY FEATURES
                            'section_pos': _get_relative_position(section_idx, len(sections)),
                            'section_invpos': _get_relative_position(section_idx, len(sections), inverse=True),
                            'table_pos': _get_relative_position(table_idx, len(tables)),
                            'table_invpos': _get_relative_position(table_idx, len(tables), inverse=True),
                            'table_count': len(tables),
                            'row_pos': _get_relative_position(row_idx, len(table_data)),
                            'row_invpos': _get_relative_position(row_idx, len(table_data), inverse=True),
                            'row_count': len(table_data),
                            'row_isheader': column_name == column_text,
                            'column_pos': _get_relative_position(column_idx, len(row)),
                            'column_invpos': _get_relative_position(column_idx, len(row), inverse=True),
                            'column_count': len(row),
                            'column_page_similar': column_page_similar,
                            'column_page_synonym': column_page_synonym,
                            'column_page_hypernym': column_page_hypernym,
                            'entity_link_pos': _get_relative_position(entity_idx, len(column_entities)),
                            'entity_link_invpos': _get_relative_position(entity_idx, len(column_entities), inverse=True),
                            'entity_line_first': entity_line_index == 0,
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
                        entity_page_index += 1
                        entity_line_index += 1

    # compute distribution of entities in tables for entity-stats features
    entities_per_table = defaultdict(lambda: defaultdict(int))
    for feature_set in data:
        table_pos = feature_set['table_pos']
        entity_uri = feature_set['_entity_uri']
        entities_per_table[table_pos][entity_uri] += 1

    for feature_set in data:
        # ENTITY-STATS FEATURES
        table_pos = feature_set['table_pos']
        entity_uri = feature_set['_entity_uri']
        feature_set['entity_in_same_table_count'] = entities_per_table[table_pos][entity_uri] - 1
        feature_set['entity_in_same_table'] = feature_set['entity_in_same_table_count'] > 0
        feature_set['entity_in_other_table_count'] = sum(entities_per_table[tp][entity_uri] for tp in entities_per_table if tp != table_pos)
        feature_set['entity_in_other_table'] = feature_set['entity_in_other_table_count'] > 0

        # LISTPAGE FEATURES
        feature_set['page_section_count'] = page_section_count
        feature_set['page_table_count'] = page_table_count
        _assign_avg_and_std_to_feature_set(feature_set, page_section_tables, 'page_section_tables')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_rows, 'page_table_rows')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_row_entities, 'page_table_row_entities')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_columns, 'page_table_columns')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_column_words, 'page_table_column_words')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_column_chars, 'page_table_column_chars')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_column_entities, 'page_table_column_entities')
        _assign_avg_and_std_to_feature_set(feature_set, page_table_first_entity_column, 'page_table_first_entity_column')

    return [tuple(d.values()) for d in data]


def get_table_feature_names() -> list:
    return [
        '_id', '_page_uri', '_top_section_name', '_section_name', '_line_idx', '_table_idx', '_row_idx', '_column_idx',
        '_column_name', '_entity_uri', '_entity_page_idx', '_entity_line_idx', '_link_type', '_text', 'section_pos',
        'section_invpos', 'table_pos', 'table_invpos', 'table_count', 'row_pos', 'row_invpos', 'row_count',
        'row_isheader', 'column_pos', 'column_invpos', 'column_count', 'column_page_similar', 'column_page_synonym',
        'column_page_hypernym', 'entity_link_pos', 'entity_link_invpos', 'entity_line_first', 'entity_first',
        'entity_last', 'entity_count', 'entity_idx', 'entity_invidx', 'entity_pos', 'entity_invpos', 'entity_pn',
        'entity_noun', 'entity_ne', 'prev_pos', 'prev_ne', 'succ_pos', 'succ_ne', 'entity_in_same_table_count',
        'entity_in_same_table', 'entity_in_other_table_count', 'entity_in_other_table', 'page_section_count',
        'page_table_count', 'page_section_tables_avg', 'page_table_rows_avg', 'page_table_row_entities_avg',
        'page_table_columns_avg', 'page_table_column_words_avg', 'page_table_column_chars_avg',
        'page_table_column_entities_avg', 'page_table_first_entity_column_avg', 'page_section_tables_std',
        'page_table_rows_std', 'page_table_row_entities_std', 'page_table_columns_std',
        'page_table_column_words_std', 'page_table_column_chars_std', 'page_table_column_entities_std',
        'page_table_first_entity_column_std'
    ]


def _extract_ne_tag(entity_span) -> str:
    """Return the (most common) named entity tag of an entity span."""
    tags = [w.ent_type_ for w in entity_span]
    if not tags:
        return ''
    tag_count = Counter(reversed(tags))  # reverse order to return tag of last entity (at same count)
    return sorted(dict(tag_count).items(), key=lambda x: x[1], reverse=True)[0][0]


def _compute_column_page_similarity(sim_func, page_lemmas: set, column_name_lemmas: set) -> int:
    """Return similarity value of column header and page name."""
    return 1 if any(sim_func(l, c) for l in page_lemmas for c in column_name_lemmas) else 0


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


def onehotencode_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    # retrieve the 50 values that appear most often in the extracted pages
    frequent_values = df[[feature_name, '_page_uri']].groupby(by=feature_name)[['_page_uri']].nunique().sort_values(by='_page_uri', ascending=False).head(50).index.values
    # one-hot-encode the most frequent values
    encoder = OneHotEncoder(categories={0: frequent_values}, handle_unknown='ignore', sparse=False)
    values = encoder.fit_transform(df[[feature_name]].fillna(value=''))
    names = encoder.get_feature_names()
    df_feature = pd.DataFrame(data=values, columns=names)
    return pd.merge(left=df, right=df_feature, left_index=True, right_index=True)
