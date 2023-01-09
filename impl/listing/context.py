from typing import Set, Dict
from collections import Counter
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

def retrieve_page_entity_context() -> pd.DataFrame:
    """Retrieve all subject entities on pages together with their context information (i.e. where they occur)"""
    # gather data about subject entities in listings
    get_logger().debug('Initializing context..')
    columns = ['P', 'TS_text', 'TS_ent', 'S_text', 'S_ent', 'E_text', 'E_tag', 'E_ent']
    data = []
    for wp in WikiPageStore.instance().get_pages():
        for listing in wp.get_listings():
            for se_mention in listing.get_subject_entities():
                data.append((
                    wp.idx,
                    listing.topsection.title,
                    listing.topsection.entity_idx,
                    listing.section.title,
                    listing.section.entity_idx,
                    se_mention.label,
                    se_mention.entity_type.value,
                    se_mention.entity_idx
                ))
    # create frame and add further features
    df = pd.DataFrame(data=data, columns=columns)
    df['idx'] = df.index.values
    df = _assign_entity_types_for_section(df, 'TS')
    df = _assign_entity_types_for_section(df, 'S')
    df = _align_section_entity_types(df)
    df = _assign_pagetypes(df)
    # get rid of all single-letter (sub)sections distorting results
    df = df.drop(index=df[(df['TS_text'].str.len() <= 1) | (df['S_text'].str.len() <= 1)].index)
    return df


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
    for ts, s_df in df.groupby('TS_text'):
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
    df = pd.merge(how='left', left=df, right=section_types, left_on=['TS_text', 'S_ent'], right_index=True)
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
        page_types = res.get_independent_types()
        if not page_types:
            data.append((page_idx, PAGE_TYPE_OTHER))
            continue
        page_type = sorted(page_types)[0]
        page_basetype = _get_basetype(page_type)
        data.append((page_idx, page_basetype.idx))
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


def get_entity_types(df: pd.DataFrame) -> pd.DataFrame:
    clge = ClgEntityStore.instance()
    return pd.DataFrame([{'E_ent': ent_idx, 'E_enttype': t.idx} for ent_idx in df['E_ent'].unique() for t in clge.get_entity_by_idx(ent_idx).get_transitive_types()])


def get_valid_tags_for_entity_types(dft: pd.DataFrame, threshold: float) -> Dict[str, Set[str]]:
    """Compute NE tags that are acceptable for a given type. E.g. for the type Building we would want the tag FAC."""
    clgo = ClgOntologyStore.instance()

    tag_probabilities = _get_tag_probabilities(dft)
    valid_tags = tag_probabilities[tag_probabilities['tag_fit'] >= threshold].groupby('E_enttype')['E_tag'].apply(lambda x: x.values.tolist()).to_dict()
    for type_idx in set(valid_tags):  # assign tags of parents to types without tags (to avoid inconsistencies)
        valid_tags[type_idx] = _compute_valid_tags_for_type(clgo.get_class_by_idx(type_idx), valid_tags)
    return valid_tags


def _get_tag_probabilities(dft: pd.DataFrame) -> pd.DataFrame:
    """Compute simple tag probabilities (frequencies of tags per entity type)."""
    tag_count = dft.groupby('E_enttype')['E_tag'].value_counts().rename('tag_count').reset_index('E_tag')
    entity_count = dft.groupby('E_enttype')['idx'].nunique().rename('entity_count')
    simple_tag_probabilities = pd.merge(left=tag_count, right=entity_count, left_index=True, right_index=True)
    simple_tag_probabilities['tag_proba'] = simple_tag_probabilities['tag_count'] / simple_tag_probabilities['entity_count']
    simple_tag_proba_dict = {grp: df.set_index('E_tag')['tag_proba'].to_dict() for grp, df in simple_tag_probabilities.groupby('E_enttype')}
    tag_probabilities = [(t, tag, proba) for t, tag_probas in simple_tag_proba_dict.items() for tag, proba in tag_probas.items()]
    return pd.DataFrame(data=tag_probabilities, columns=['E_enttype', 'E_tag', 'tag_fit'])


def _compute_valid_tags_for_type(clg_type: ClgType, valid_tags: Dict[int, Set[str]]) -> Set[str]:
    clgo = ClgOntologyStore.instance()

    if clg_type.idx not in valid_tags:
        return set()
    if not valid_tags[clg_type.idx]:
        valid_tags[clg_type.idx] = {tag for ptype in clgo.get_supertypes(clg_type, include_root=False) for tag in _compute_valid_tags_for_type(ptype, valid_tags)}
    return valid_tags[clg_type.idx]


# ENTITY RELATIONS


def get_entity_relations():
    clge = ClgEntityStore.instance()
    data = [(sub.idx, pred.idx, val.idx) for sub, props in clge.get_entity_properties().items() for pred, vals in props.items() for val in vals if isinstance(val, ClgEntity)]
    return pd.DataFrame(data, columns=['sub', 'pred', 'obj'])
