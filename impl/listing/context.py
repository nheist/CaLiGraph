from collections import defaultdict
import pandas as pd
from impl import subject_entity
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util
import impl.caligraph.util as clg_util


# RETRIEVE ENTITY CONTEXT ON PAGES


def retrieve_page_entity_context(graph) -> pd.DataFrame:
    """Retrieve all subject entities on pages together with their context information (i.e. where they occur)"""
    # gather data
    df_data = []
    for page_uri, ts_data in subject_entity.get_page_subject_entities(graph).items():
        p = dbp_util.resource2name(page_uri)
        for ts, s_data in ts_data.items():
            for s, entities in s_data.items():
                for e, e_data in entities.items():
                    df_data.append((p, e, e_data['text'], e_data['tag'], ts, e_data['TS_ent'], s, e_data['S_ent']))

    # create frame and add further features
    df = pd.DataFrame(data=df_data, columns=['P', 'E_ent', 'E_text', 'E_tag', 'TS_text', 'TS_ent', 'S_text', 'S_ent'])
    df['E_id'] = df.index.values
    df = _assign_entity_types_for_section(df, 'TS_ent')
    df = _assign_entity_types_for_section(df, 'S_ent')
    df = _align_section_entity_types(df)
    df = _assign_pagetypes(df)

    # get rid of all single-letter (sub)sections distorting results
    df = df.drop(index=df[(df['TS_text'].str.len() <= 1) | (df['S_text'].str.len() <= 1)].index)
    # get rid of entities that are lists, files, etc.
    valid_entities = {e for e in df['E_ent'].unique() if not e.startswith(('List of', 'File:', 'Image:'))}
    df = df.loc[df['E_ent'].isin(valid_entities), :]

    return df


def _assign_entity_types_for_section(df, section_entity):
    """Retrieve the types of section entities."""
    section_types = {}
    for ent in df[section_entity].unique():
        types = dbp_store.get_independent_types(dbp_store.get_types(dbp_util.name2resource(str(ent))))
        if types:
            section_types[ent] = dbp_util.type2name(list(types)[0])
    section_types = pd.Series(section_types, name=f'{section_entity}type')
    return pd.merge(how='left', left=df, right=section_types, left_on=section_entity, right_index=True)


def _align_section_entity_types(df):
    """Align the types of section entities to the most common entity type aggregated by top-section."""
    section_types = {}
    for ts, s_df in df.groupby('TS_text'):
        section_ents = set(s_df['S_ent'].unique())
        type_counter = defaultdict(int)
        for s_ent in section_ents:
            for t in dbp_store.get_transitive_types(dbp_util.name2resource(str(s_ent))):
                type_counter[t] += 1
        top_types = dbp_store.get_independent_types({t for t, cnt in type_counter.items() if cnt == max(type_counter.values())})
        if top_types:
            top_type = dbp_util.type2name(list(top_types)[0])
            section_types.update({(ts, se): top_type for se in section_ents if top_type in dbp_store.get_transitive_types(dbp_util.name2resource(str(se)))})
    section_types = pd.Series(section_types, name='SE_enttype_new')
    df = pd.merge(how='left', left=df, right=section_types, left_on=['TS_text', 'S_ent'], right_index=True)
    df['S_enttype_new'].fillna(df['S_enttype'], inplace=True)
    return df.drop(columns='S_enttype').rename(columns={'S_enttype_new': 'S_enttype'})


def _assign_pagetypes(df):
    """Assign (most basic and most specific) page types to the existing dataframe."""
    data = []
    for page_name in df['P'].unique():
        if page_name.startswith('List of'):
            data.append((page_name, 'List', 'List'))
            continue
        page_uri = dbp_util.name2resource(page_name)
        P_types = dbp_store.get_independent_types(dbp_store.get_types(page_uri))
        if not P_types:
            data.append((page_name, 'Other', 'Other'))
            continue
        P_type = sorted(P_types)[0]
        P_basetype = _get_basetype(P_type)
        data.append((page_name, dbp_util.type2name(P_type), dbp_util.type2name(P_basetype)))
    return pd.merge(left=df, right=pd.DataFrame(data, columns=['P', 'P_type', 'P_basetype']), on='P')


def _get_basetype(dbp_type: str):
    """Retrieve the base type of a DBpedia type (i.e., the type below owl:Thing or dbo:Agent)."""
    parent_types = {dbp_type}
    while parent_types and not parent_types.intersection({rdf_util.CLASS_OWL_THING, f'{dbp_util.NAMESPACE_DBP_ONTOLOGY}Agent'}):
        dbp_type = sorted(parent_types)[0]
        parent_types = dbp_store.get_supertypes(dbp_type)
    return dbp_type


# ENTITY TYPES


def get_entity_types(df: pd.DataFrame, graph) -> pd.DataFrame:
    return pd.DataFrame([{'E_ent': ent, 'E_enttype': t} for ent in df['E_ent'].unique() for t in _get_transitive_types(ent, graph)])


def _get_transitive_types(resource: str, graph) -> set:
    resource_uri = clg_util.name2clg_resource(str(resource))
    clg_nodes = graph.get_nodes_for_resource(resource_uri)
    transitive_clg_nodes = (clg_nodes | {an for n in clg_nodes for an in graph.ancestors(n)}).difference({rdf_util.CLASS_OWL_THING})
    return {clg_util.clg_type2name(n) for n in transitive_clg_nodes}


def get_valid_tags_for_entity_types(dft: pd.DataFrame, graph, threshold) -> dict:
    """Compute NE tags that are acceptable for a given type. E.g. for the type Building we would want the tag FAC."""
    tag_probabilities = _get_tag_probabilities(dft)
    valid_tags = tag_probabilities[tag_probabilities['tag_fit'] >= threshold].groupby('E_enttype')['E_tag'].apply(lambda x: x.values.tolist()).to_dict()
    for ent_type in set(valid_tags):  # assign tags of parents to types without tags (to avoid inconsistencies)
        valid_tags[ent_type] = _compute_valid_tags_for_type(ent_type, valid_tags, graph)
    return valid_tags


def _get_tag_probabilities(dft: pd.DataFrame) -> pd.DataFrame:
    """Compute simple tag probabilities (frequencies of tags per entity type)."""
    tag_count = dft.groupby('E_enttype')['E_tag'].value_counts().rename('tag_count').reset_index('E_tag')
    entity_count = dft.groupby('E_enttype')['E_id'].nunique().rename('entity_count')
    simple_tag_probabilities = pd.merge(left=tag_count, right=entity_count, left_index=True, right_index=True)
    simple_tag_probabilities['tag_proba'] = simple_tag_probabilities['tag_count'] / simple_tag_probabilities['entity_count']
    simple_tag_proba_dict = {grp: df.set_index('E_tag')['tag_proba'].to_dict() for grp, df in simple_tag_probabilities.groupby('E_enttype')}
    tag_probabilities = [(t, tag, proba) for t, tag_probas in simple_tag_proba_dict.items() for tag, proba in tag_probas.items()]
    return pd.DataFrame(data=tag_probabilities, columns=['E_enttype', 'E_tag', 'tag_fit'])


def _compute_valid_tags_for_type(ent_type: str, valid_tags: dict, graph) -> set:
    if ent_type not in valid_tags:
        return set()
    if not valid_tags[ent_type]:
        valid_tags[ent_type] = {tag for ptype in _get_supertypes(ent_type, graph) for tag in _compute_valid_tags_for_type(ptype, valid_tags, graph)}
    return valid_tags[ent_type]


def _get_supertypes(ent_type: str, graph) -> set:
    clg_type = clg_util.name2clg_type(ent_type)
    return {clg_util.clg_type2name(t) for t in graph.parents(clg_type).difference({rdf_util.CLASS_OWL_THING})}


# ENTITY RELATIONS


def get_entity_relations():
    """Retrieve all existing relation triples in the knowledge graph.
    As DBpedia and CaLiGraph have the same base set of triples, we can retrieve it directly from DBpedia.
    """
    rpm = dbp_store.get_resource_property_mapping()
    data = [(dbp_util.resource2name(sub), pred, dbp_util.resource2name(obj)) for sub in rpm for pred in rpm[sub] for obj in rpm[sub][pred] if dbp_util.is_dbp_resource(obj)]
    return pd.DataFrame(data, columns=['sub', 'pred', 'obj'])
