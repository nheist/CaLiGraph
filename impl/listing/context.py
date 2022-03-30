from typing import Set, Dict
from collections import defaultdict
import pandas as pd
from impl import subject_entity
import impl.caligraph.util as clg_util
from impl.dbpedia.ontology import DbpType, DbpOntologyStore
from impl.dbpedia.resource import DbpListpage, DbpEntity, DbpResourceStore


# RETRIEVE ENTITY CONTEXT ON PAGES


def retrieve_page_entity_context(graph) -> pd.DataFrame:
    """Retrieve all subject entities on pages together with their context information (i.e. where they occur)"""
    # gather data
    df_data = []
    for res, ts_data in subject_entity.get_page_subject_entities(graph).items():
        for ts, s_data in ts_data.items():
            for s, entities in s_data.items():
                for e, e_data in entities.items():
                    df_data.append((res.idx, e, e_data['text'], e_data['tag'], ts, e_data['TS_entidx'], s, e_data['S_entidx']))

    # create frame and add further features
    df = pd.DataFrame(data=df_data, columns=['P_residx', 'E_ent', 'E_text', 'E_tag', 'TS_text', 'TS_entidx', 'S_text', 'S_entidx'])
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
    dbr = DbpResourceStore.instance()
    section_types = {}
    for ent_idx in df[f'{section_id}_entidx'].unique():
        types = dbr.get_resource_by_idx(ent_idx).get_independent_types()
        if types:
            section_types[ent_idx] = sorted(types)[0].name
    section_types = pd.Series(section_types, name=f'{section_id}_enttype')
    return pd.merge(how='left', left=df, right=section_types, left_on=section_id, right_index=True)


def _align_section_entity_types(df: pd.DataFrame) -> pd.DataFrame:
    """Align the types of section entities to the most common entity type aggregated by top-section."""
    dbo = DbpOntologyStore.instance()
    dbr = DbpResourceStore.instance()
    
    section_types = {}
    for ts, s_df in df.groupby('TS_text'):
        type_counter = defaultdict(int)
        section_ent_indices = s_df['S_entidx'].unique()
        for s_entidx in section_ent_indices:
            for t in dbr.get_resource_by_idx(s_entidx).get_transitive_types():
                type_counter[t] += 1
        top_types = dbo.get_independent_types({t for t, cnt in type_counter.items() if cnt == max(type_counter.values())})
        if top_types:
            top_type = sorted(top_types)[0]
            section_types.update({(ts, sei): top_type.name for sei in section_ent_indices if top_type in dbr.get_resource_by_idx(sei).get_transitive_types()})
    section_types = pd.Series(section_types, name='S_enttype_new')
    df = pd.merge(how='left', left=df, right=section_types, left_on=['TS_text', 'S_entidx'], right_index=True)
    df['S_enttype_new'].fillna(df['S_enttype'], inplace=True)
    return df.drop(columns='S_enttype').rename(columns={'S_enttype_new': 'S_enttype'})


def _assign_pagetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign (most basic and most specific) page types to the existing dataframe."""
    dbr = DbpResourceStore.instance()
    data = []
    for page_idx in df['P_residx'].unique():
        res = dbr.get_resource_by_idx(page_idx)
        if isinstance(res, DbpListpage):
            data.append((page_idx, 'List', 'List'))
            continue
        page_types = res.get_independent_types()
        if not page_types:
            data.append((page_idx, 'Other', 'Other'))
            continue
        page_type = sorted(page_types)[0]
        page_basetype = _get_basetype(page_type)
        data.append((page_idx, page_type.name, page_basetype.name))
    return pd.merge(left=df, right=pd.DataFrame(data, columns=['P_residx', 'P_type', 'P_basetype']), on='P_residx')


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


def get_entity_types(df: pd.DataFrame, graph) -> pd.DataFrame:
    return pd.DataFrame([{'E_ent': ent, 'E_enttype': t} for ent in df['E_ent'].unique() for t in _get_transitive_types(ent, graph)])


def _get_transitive_types(ent: str, graph) -> Set[str]:
    ent_uri = clg_util.name2clg_resource(ent)
    clg_nodes = graph.get_nodes_for_resource(ent_uri)
    transitive_clg_nodes = (clg_nodes | {an for n in clg_nodes for an in graph.ancestors(n)}).difference({graph.root_node})
    return {clg_util.clg_class2name(n) for n in transitive_clg_nodes}


def get_valid_tags_for_entity_types(dft: pd.DataFrame, graph, threshold: float) -> Dict[str, Set[str]]:
    """Compute NE tags that are acceptable for a given type. E.g. for the type Building we would want the tag FAC."""
    tag_probabilities = _get_tag_probabilities(dft)
    valid_tags = tag_probabilities[tag_probabilities['tag_fit'] >= threshold].groupby('E_enttype')['E_tag'].apply(lambda x: x.values.tolist()).to_dict()
    for ent_type in set(valid_tags):  # assign tags of parents to types without tags (to avoid inconsistencies)
        valid_tags[ent_type] = _compute_valid_tags_for_type(ent_type, valid_tags, graph)
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


def _compute_valid_tags_for_type(ent_type: str, valid_tags: Dict[str, Set[str]], graph) -> Set[str]:
    if ent_type not in valid_tags:
        return set()
    if not valid_tags[ent_type]:
        valid_tags[ent_type] = {tag for ptype in _get_supertypes(ent_type, graph) for tag in _compute_valid_tags_for_type(ptype, valid_tags, graph)}
    return valid_tags[ent_type]


def _get_supertypes(ent_type: str, graph) -> Set[str]:
    clg_type = clg_util.name2clg_type(ent_type)
    return {clg_util.clg_class2name(t) for t in graph.parents(clg_type).difference({graph.root_node})}


# ENTITY RELATIONS


def get_entity_relations():
    """Retrieve all existing relation triples in the knowledge graph.
    As DBpedia and CaLiGraph have the same base set of triples, we can retrieve it directly from DBpedia.
    """
    dbr = DbpResourceStore.instance()
    data = [(sub.name, pred.idx, val.name) for sub, props in dbr.get_entity_properties() for pred, vals in props.items() for val in vals if isinstance(val, DbpEntity)]
    return pd.DataFrame(data, columns=['sub', 'predidx', 'obj'])
