from impl import subject_entity
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util
from collections import defaultdict
import pandas as pd


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
                    df_data.append((p, e, e_data['text'], e_data['tag'], ts, s))

    # create frame and add further features
    df = pd.DataFrame(data=df_data, columns=['P', 'E', 'E_text', 'E_tag', 'TS_text', 'S_text'])
    df['E_id'] = df.index.values
    df['TS_ent'] = df['TS_text'].str.extract(r'.*\[\[([^\]|]+)(?:|[^\]]+)?\]\].*')
    df['S_ent'] = df['S_text'].str.extract(r'.*\[\[([^\]|]+)(?:|[^\]]+)?\]\].*')
    df = _assign_entity_types_for_section(df, 'TS_ent')
    df = _assign_entity_types_for_section(df, 'S_ent')
    df = _align_section_entity_types(df)
    df = _assign_pagetypes(df)

    # get rid of all single-letter (sub)sections distorting results
    df = df.drop(index=df[(df['TS_text'].str.len() <= 1) | (df['S_text'].str.len() <= 1)].index)

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
            top_type = list(top_types)[0]
            section_types.update({(ts, se): dbp_util.type2name(top_type) for se in section_ents if top_type in dbp_store.get_transitive_types(dbp_util.name2resource(str(se)))})
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


# ENRICH ENTITY CONTEXT WITH TYPE INFORMATION


def add_type_information(df: pd.DataFrame, graph) -> pd.DataFrame:
    df_types = pd.DataFrame([{'E': ent, 'E_type': t} for ent in df['E'].unique() for t in get_transitive_types(ent)])


# ENRICH ENTITY CONTEXT WITH RELATION INFORMATION


def add_relation_information(df: pd.DataFrame, graph) -> pd.DataFrame:
    pass
