from typing import Tuple, Dict, Set
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import get_logger
from impl.wikipedia import WikiPageStore
from impl.dbpedia.ontology import DbpType, DbpOntologyStore
from impl.dbpedia.resource import DbpListpage, DbpResourceStore
from impl.caligraph.ontology import ClgType, ClgOntologyStore
from impl.caligraph.entity import ClgEntity, ClgEntityStore


PAGE_TYPE_LIST = -1
PAGE_TYPE_OTHER = -2


# RETRIEVE ENTITY CONTEXT ON PAGES

def retrieve_page_entity_context() -> Tuple[pd.DataFrame, Dict[int, Dict[Tuple[int, int], str]]]:
    """Retrieve all subject entities on pages together with their context information (i.e. where they occur)"""
    # gather data about subject entities in listings
    get_logger().debug('Initializing context..')
    section_title_ids = {}
    entity_labels = defaultdict(dict)
    columns = ['P', 'L', 'TS_id', 'TS_ent', 'S_id', 'S_ent', 'E_tag', 'E_ent']
    data = []
    for wp in WikiPageStore.instance().get_pages():
        for listing in wp.get_listings():
            topsection = listing.topsection
            section = listing.section
            if len(topsection.title) <= 1 or len(section.title) <= 1:
                # get rid of all single-letter (sub)sections distorting results
                continue
            if topsection.title not in section_title_ids:
                section_title_ids[topsection.title] = len(section_title_ids)
            if section.title not in section_title_ids:
                section_title_ids[section.title] = len(section_title_ids)
            for se_mention in listing.get_subject_entities():
                entity_labels[se_mention.entity_idx][(wp.idx, listing.idx)] = se_mention.label
                data.append((
                    wp.idx,
                    listing.idx,
                    section_title_ids[topsection.title],
                    topsection.entity_idx or np.nan,
                    section_title_ids[section.title],
                    section.entity_idx or np.nan,
                    se_mention.entity_type.value,
                    se_mention.entity_idx
                ))
    # create frame and add further features
    df = pd.DataFrame(data=np.asarray(data), columns=columns)
    df['idx'] = df.index.values
    df = _assign_entity_types_for_section(df, 'TS')
    df = _assign_entity_types_for_section(df, 'S')
    df = _align_section_entity_types(df)
    df = _assign_pagetypes(df)
    return df, entity_labels


def _assign_entity_types_for_section(df: pd.DataFrame, section_id: str) -> pd.DataFrame:
    """Retrieve the types of section entities."""
    clge = ClgEntityStore.instance()
    section_ent = f'{section_id}_ent'
    section_types = {}
    for ent_idx in df[section_ent].unique():
        if not clge.has_entity_with_idx(ent_idx):
            continue
        types = clge.get_entity_by_idx(ent_idx).get_independent_types()
        if types:
            section_types[ent_idx] = sorted(types)[0].idx
    section_types = pd.Series(section_types, name=f'{section_id}_enttype')
    return pd.merge(how='left', left=df, right=section_types, left_on=section_ent, right_index=True)


def _align_section_entity_types(df: pd.DataFrame) -> pd.DataFrame:
    """Align the types of section entities to the most common entity type aggregated by top-section."""
    clgo = ClgOntologyStore.instance()
    clge = ClgEntityStore.instance()
    
    section_types = {}
    for ts, s_df in df.groupby('TS_id'):
        type_counter = Counter()
        section_ent_indices = s_df['S_ent'].unique()
        for s_ent in section_ent_indices:
            if not clge.has_entity_with_idx(s_ent):
                continue
            type_counter.update(clge.get_entity_by_idx(s_ent).get_transitive_types())
        if not type_counter:
            continue
        max_type_count = type_counter.most_common(1)[0][1]
        top_types = clgo.get_independent_types({t for t, cnt in type_counter.items() if cnt == max_type_count})
        if top_types:
            top_type = sorted(top_types)[0]
            section_types.update({(ts, sei): top_type.idx for sei in section_ent_indices if clge.has_entity_with_idx(sei) and top_type in clge.get_entity_by_idx(sei).get_transitive_types()})
    section_types = pd.Series(section_types, name='S_enttype_new')
    df = pd.merge(how='left', left=df, right=section_types, left_on=['TS_id', 'S_ent'], right_index=True)
    df['S_enttype_new'].fillna(df['S_enttype'], inplace=True)
    return df.drop(columns='S_enttype').rename(columns={'S_enttype_new': 'S_enttype'})


def _assign_pagetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the most basic page types to the existing dataframe."""
    dbr = DbpResourceStore.instance()
    data = []
    for page_idx in df['P'].unique():
        res = dbr.get_resource_by_idx(page_idx)
        if isinstance(res, DbpListpage):
            data.append((page_idx, PAGE_TYPE_LIST))
            continue
        page_types = res.get_independent_types(exclude_root=True)
        if not page_types:
            data.append((page_idx, PAGE_TYPE_OTHER))
            continue
        page_type = sorted(page_types)[0]
        data.append((page_idx, _get_basetype(page_type).idx))
    return pd.merge(left=df, right=pd.DataFrame(data, columns=['P', 'P_type']), on='P')


def _get_basetype(dbp_type: DbpType) -> DbpType:
    """Retrieve the base type of a DBpedia type (i.e., the type below owl:Thing or dbo:Agent)."""
    dbo = DbpOntologyStore.instance()
    toplevel_types = {dbo.get_type_root(), dbo.get_class_by_name('Agent')}
    parent_types = {dbp_type}
    while parent_types and not parent_types.intersection(toplevel_types):
        dbp_type = sorted(parent_types)[0]
        parent_types = dbo.get_supertypes(dbp_type)
    return dbp_type


# ENTITY TYPES


def get_valid_tags_for_entity_types(df: pd.DataFrame, ent_types: Dict[int, Set[int]], threshold: float) -> Dict[int, Set[int]]:
    """Compute NE tags that are acceptable for a given type. E.g. for the type Building we would want the tag FAC."""
    # find most likely tags (if probability higher than threshold)
    valid_tags = defaultdict(set)
    for t, tag_probas in _get_tag_probabilities(df, ent_types).items():
        for tag, proba in tag_probas.items():
            if proba >= threshold:
                valid_tags[t].add(tag)
    # assign tags of parents to types without tags (to avoid inconsistencies)
    for type_idx in set(valid_tags):
        valid_tags[type_idx] = _compute_valid_tags_for_type(type_idx, valid_tags)
    return valid_tags


def _get_tag_probabilities(df: pd.DataFrame, ent_types: Dict[int, Set[int]]) -> Dict[int, Dict[int, float]]:
    """Compute simple tag probabilities (frequencies of tags per entity type)."""
    # collect tag and entity counts per type
    tag_counter = defaultdict(Counter)
    entity_counter = Counter()
    for ent_idx, df_tag in tqdm(df[['E_ent', 'E_tag']].groupby('E_ent'), total=df['E_ent'].nunique(), desc='Tag probabilities'):
        tag_count = Counter(df_tag.value_counts('E_tag').to_dict())
        ent_count = len(df_tag)
        for t in ent_types[ent_idx]:
            tag_counter[t] += tag_count
            entity_counter[t] += ent_count
    # compute tag probabilities per type
    tag_probabilities = {t: {tag: tag_count / ent_count for tag, tag_count in tag_counter[t].items()} for t, ent_count in entity_counter.items()}
    return tag_probabilities


def _compute_valid_tags_for_type(type_idx: int, valid_tags: Dict[int, Set[int]]) -> Set[int]:
    clgo = ClgOntologyStore.instance()
    if type_idx not in valid_tags:
        return set()
    if not valid_tags[type_idx]:
        clg_type = clgo.get_class_by_idx(type_idx)
        valid_tags[type_idx] = {tag for ptype in clgo.get_supertypes(clg_type, include_root=False) for tag in _compute_valid_tags_for_type(ptype.idx, valid_tags)}
    return valid_tags[type_idx]


# ENTITY RELATIONS


def get_entity_relations():
    clge = ClgEntityStore.instance()
    data = [(sub.idx, pred.idx, val.idx) for sub, props in clge.get_entity_properties().items() for pred, vals in props.items() for val in vals if isinstance(val, ClgEntity)]
    return pd.DataFrame(data, columns=['sub', 'pred', 'obj'])
