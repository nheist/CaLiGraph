from typing import Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import utils
from utils import get_logger
from impl.listing import context
from impl.dbpedia.resource import DbpResource, DbpResourceStore
from impl.caligraph.ontology import ClgOntologyStore
from impl.caligraph.entity import ClgEntity, ClgEntityStore

RULE_PATTERNS = {
    'pattern_TS': ['P_basetype', 'TS_text'],
    'pattern_TSent': ['P_basetype', 'TS_enttype'],
    'pattern_TSent-S': ['P_basetype', 'TS_enttype', 'S_text'],
    'pattern_TS-S': ['P_basetype', 'TS_text', 'S_text'],
    'pattern_TS-Sent': ['P_basetype', 'TS_text', 'S_enttype']
}


def extract_listing_entity_information() -> Dict[ClgEntity, Dict[Tuple[DbpResource, str], dict]]:
    get_logger().info('Extracting types and relations for page entities..')
    dbr = DbpResourceStore.instance()
    clgo = ClgOntologyStore.instance()

    # format: entity -> origin -> (labels, types, in, out)
    page_entities = defaultdict(lambda: defaultdict(lambda: {'labels': set(), 'types': set(), 'in': set(), 'out': set()}))

    df = context.retrieve_page_entity_context()

    # extract list page entities
    get_logger().debug('Extracting types of list page entities..')
    df_lps = df[df['P_type'] == 'List']
    for lp_idx, df_lp in df_lps.groupby(by='P'):
        origin = _get_origin(lp_idx, None)
        lp = dbr.get_resource_by_idx(lp_idx)
        clg_types = {t.idx for t in clgo.get_types_for_associated_dbp_resource(lp)}
        if clg_types:
            for _, row in df_lp.iterrows():
                ent_idx = row['E_ent']
                page_entities[ent_idx][origin]['labels'].add(row['E_text'])
                page_entities[ent_idx][origin]['types'].update(clg_types)

    df = df.loc[df['P_type'] != 'List']  # ignore list pages in subsequent steps

    # compute valid combinations of types and NE tags
    df_types = context.get_entity_types(df)
    dft = pd.merge(left=df, right=df_types, on='E_ent')
    valid_tags = context.get_valid_tags_for_entity_types(dft, utils.get_config('listing.valid_tag_threshold'))

    # extract types
    get_logger().debug('Extracting types of page entities..')
    df_new_types = _compute_new_types(df, dft, df_types, valid_tags)
    for ent_idx, df_ent in df_new_types.groupby(by='E_ent'):
        for (page_idx, section_text), df_entorigin in df_ent.groupby(by=['P', 'S_text']):
            origin = _get_origin(page_idx, section_text)
            page_entities[ent_idx][origin]['labels'].update(set(df_entorigin['E_text'].unique()))
            new_types = {clgo.get_class_by_idx(t_idx) for t_idx in set(df_entorigin['E_enttype'].unique())}
            new_types = new_types.difference({tt for t in new_types for tt in clgo.get_transitive_supertypes(t, include_root=True)})
            page_entities[ent_idx][origin]['types'].update({t.idx for t in new_types})

    # extract relations
    get_logger().debug('Extracting relations of page entities..')
    df_rels = context.get_entity_relations()
    df_new_relations = _compute_new_relations(df, df_rels, 'P', valid_tags)
    df_new_relations = pd.concat([df_new_relations, _compute_new_relations(df, df_rels, 'TS_ent', valid_tags)])
    df_new_relations = pd.concat([df_new_relations, _compute_new_relations(df, df_rels, 'S_ent', valid_tags)])
    for ent_idx, df_ent in df_new_relations.groupby(by='E_ent'):
        for (page_idx, section_text), df_entorigin in df_ent.groupby(by=['P', 'S_text']):
            origin = _get_origin(page_idx, section_text)
            page_entities[ent_idx][origin]['labels'].update(set(df_entorigin['E_text'].unique()))
            rels_in = set(map(tuple, df_entorigin[~df_entorigin['inv']][['pred', 'target']].values))
            page_entities[ent_idx][origin]['in'].update(rels_in)
            rels_out = set(map(tuple, df_entorigin[df_entorigin['inv']][['pred', 'target']].values))
            page_entities[ent_idx][origin]['out'].update(rels_out)

    return {e: dict(origin_data) for e, origin_data in page_entities.items()}  # convert defaultdicts to plain dicts


def _get_origin(page_idx: int, section_text: Optional[str]) -> Tuple[int, str]:
    section_name = section_text if section_text and section_text != 'Main' else None
    return page_idx, section_name


# EXTRACT TYPES


def _compute_new_types(df: pd.DataFrame, dft: pd.DataFrame, df_types: pd.DataFrame, valid_tags: Dict[str, Set[str]]) -> pd.DataFrame:
    """Compute all new type assertions."""
    rule_dfs = {}
    for rule_name, rule_pattern in RULE_PATTERNS.items():
        dft_by_page = _aggregate_types_by_page(dft, rule_pattern)
        rule_dfs[rule_name] = _aggregate_types_by_section(dft_by_page, rule_pattern)
    new_types = _extract_new_types_with_threshold(df, df_types, rule_dfs)
    filtered_types = _filter_new_types_by_tag(new_types, valid_tags)
    return filtered_types


def _aggregate_types_by_page(dft: pd.DataFrame, section_grouping: list) -> pd.DataFrame:
    """Aggregate the type data by Wikipedia page."""
    dft = dft.dropna(subset=section_grouping)
    page_grouping = section_grouping + ['P']
    # compute type count
    dftP = dft.groupby(page_grouping)['E_enttype'].value_counts().rename('type_count').reset_index(level='E_enttype')
    # compute entity count
    ent_counts = dft.groupby(page_grouping)['idx'].nunique().rename('ent_count')
    dftP = pd.merge(dftP, ent_counts, left_index=True, right_index=True, copy=False)
    # compute type confidence
    dftP['type_conf'] = (dftP['type_count'] / dftP['ent_count']).fillna(0).clip(0, 1)
    return dftP[dftP['ent_count'] > 2]


def _aggregate_types_by_section(dfp: pd.DataFrame, section_grouping: list) -> pd.DataFrame:
    """Aggregate the type data by (top-)section."""
    page_grouping = section_grouping + ['P']
    grp = section_grouping + ['E_enttype']
    dfp = dfp.reset_index()
    # compute micro mean
    section_type_count = dfp.groupby(grp)['type_count'].sum().reset_index(level='E_enttype')
    section_ent_count = dfp.drop_duplicates(page_grouping).groupby(section_grouping)['ent_count'].sum()
    result = pd.merge(section_type_count, section_ent_count, left_index=True, right_index=True).reset_index()
    result['micro_mean'] = (result['type_count'] / result['ent_count']).clip(0, 1)
    result.drop(columns=['type_count', 'ent_count'], inplace=True)
    # compute macro mean
    section_type_confidences = dfp.groupby(grp)['type_conf'].sum().rename('section_type_conf').reset_index(level='E_enttype')
    section_page_count = dfp.groupby(section_grouping)['P'].nunique().rename('section_page_count')
    macro_df = pd.merge(section_type_confidences, section_page_count, left_index=True, right_index=True).reset_index().set_index(grp)
    macro_df['macro_mean'] = macro_df['section_type_conf'] / macro_df['section_page_count']
    result = pd.merge(left=result, right=macro_df['macro_mean'], left_on=grp, right_index=True)
    # add page count
    page_count = dfp.groupby(section_grouping)['P'].nunique().rename('page_count').reset_index()
    result = pd.merge(left=result, right=page_count, on=section_grouping)
    # compute micro_std
    confidence_deviations = pd.merge(how='left', left=dfp, right=result, on=grp)
    confidence_deviations['dev'] = np.absolute(confidence_deviations['micro_mean'] - confidence_deviations['type_conf'])
    micro_std = confidence_deviations.groupby(grp).apply(lambda x: (x['dev'].sum() + (x['page_count'].mean() - len(x)) * x['micro_mean'].mean()) / x['page_count'].mean()).rename('micro_std').reset_index()
    result = pd.merge(left=result, right=micro_std, on=grp)
    return result.set_index(grp)


def _extract_new_types_with_threshold(df: pd.DataFrame, df_types: pd.DataFrame, rule_dfs: dict) -> pd.DataFrame:
    """Extract new types from rules based on confidence and consistency thresholds."""
    mean_threshold = utils.get_config('listing.type_mean_threshold')
    std_threshold = utils.get_config('listing.type_std_threshold')
    valid_rule_dfs = [rule_dfs[rule_name].query(f'micro_mean > {mean_threshold} & micro_std < {std_threshold}').reset_index()[rule_pattern + ['E_enttype', 'micro_mean', 'micro_std']].drop_duplicates() for rule_name, rule_pattern in RULE_PATTERNS.items()]
    return _extract_new_types(valid_rule_dfs, df, df_types)


def _extract_new_types(valid_rule_dfs: list, source_df: pd.DataFrame, existing_types: pd.DataFrame) -> pd.DataFrame:
    """Apply valid rules to the initial df to extract new type assertions."""
    # extract new types
    result_df = pd.DataFrame()
    for dfr in valid_rule_dfs:
        key_cols = list(set(dfr.columns).difference({'E_enttype', 'micro_mean', 'micro_std'}))
        result_df = pd.concat([result_df, pd.merge(how='left', left=dfr, right=source_df, on=key_cols)])
    # filter out duplicate extractions
    result_df = result_df.drop_duplicates(['E_ent', 'E_enttype'])
    # filter out existing types
    result_df = pd.merge(result_df, existing_types, on=['E_ent', 'E_enttype'], indicator=True, how='outer').query('_merge=="left_only"').drop(columns='_merge')
    return result_df


def _filter_new_types_by_tag(df: pd.DataFrame, valid_tags: Dict[str, Set[str]]) -> pd.DataFrame:
    """Filter out entities with a NE tag that is not in `valid_tags`."""
    return df[df.apply(lambda row: row['E_enttype'] in valid_tags and row['E_tag'] in valid_tags[row['E_enttype']], axis=1)]


# EXTRACT RELATIONS


def _compute_new_relations(df: pd.DataFrame, df_rels: pd.DataFrame, target_identifier: str, valid_tags: Dict[str, Set[str]]) -> pd.DataFrame:
    """Retrieve relation assertions from the initial dataframe."""
    rule_dfs = {}
    dfr = _create_relation_df(df, df_rels, target_identifier)
    dfr_types = _create_relation_type_df(dfr)
    for rule_name, rule_pattern in RULE_PATTERNS.items():
        dfr_by_page = _aggregate_relations_by_page(df, dfr, df_rels, rule_pattern)
        rule_dfs[rule_name] = _aggregate_relations_by_section(dfr_by_page, rule_pattern)
    new_relations = _extract_new_relations_with_threshold(df, dfr_types, df_rels, target_identifier, rule_dfs)
    filtered_relations = _filter_new_relations_by_tag(new_relations, valid_tags)
    return filtered_relations


def _create_relation_df(df: pd.DataFrame, df_rels: pd.DataFrame, target_identifier: str) -> pd.DataFrame:
    """Join the original dataframe with the existing relations in the knowledge graph."""
    df = df.dropna(subset=[target_identifier])
    dfr_sub = pd.merge(how='inner', left=df, right=df_rels.rename(columns={'sub': target_identifier, 'obj': 'E_ent'}), on=[target_identifier, 'E_ent'])
    dfr_sub['inv'] = False
    dfr_obj = pd.merge(how='inner', left=df, right=df_rels.rename(columns={'obj': target_identifier, 'sub': 'E_ent'}), on=[target_identifier, 'E_ent'])
    dfr_obj['inv'] = True
    dfr = pd.concat([dfr_sub, dfr_obj])
    dfr['rel'] = dfr.apply(_predinv_to_rel_row, axis=1)
    return dfr


def _create_relation_type_df(dfr: pd.DataFrame) -> pd.DataFrame:
    """Retrieve domains and ranges for predicates."""
    clgo = ClgOntologyStore.instance()

    data = []
    for _, row in dfr[['pred', 'inv']].drop_duplicates().iterrows():
        pred = clgo.get_class_by_idx(row['pred'])
        e_type = pred.get_domain() if row['inv'] else pred.get_range()
        data.append({'pred': row['pred'], 'inv': row['inv'], 'E_predtype': e_type.idx})
    return pd.DataFrame(data)


def _aggregate_relations_by_page(df: pd.DataFrame, dfr: pd.DataFrame, df_rels: pd.DataFrame, section_grouping: list) -> pd.DataFrame:
    """Aggregate the df on a page-level."""
    dfr = dfr.dropna(subset=section_grouping)
    page_grouping = section_grouping + ['P']
    # compute rel_count
    dfrP = dfr.groupby(page_grouping + ['rel']).agg({'idx': 'count'}).rename(columns={'idx': 'rel_count'}).reset_index()
    # initialize all rel_counts of a section with 0 if not existing in page
    all_relations = pd.merge(left=dfrP[section_grouping + ['rel']].drop_duplicates(), right=dfrP[page_grouping].drop_duplicates(), on=section_grouping)
    dfrP = pd.merge(how='right', left=dfrP, right=all_relations, on=page_grouping + ['rel']).fillna(0)
    # compute pred_count
    ## first create (inverse) counts for relations per entity
    entity_preds = defaultdict(list, df_rels.groupby('sub')['pred'].unique().to_dict())
    entity_invpreds = defaultdict(list, df_rels.groupby('obj')['pred'].unique().to_dict())
    ## then assign predicate counts
    data = []
    for grp, df_grp in df.groupby(page_grouping):
        rels = Counter()
        invrels = Counter()
        for _, ent in df_grp['E_ent'].iteritems():
            rels.update(entity_preds[ent])
            invrels.update(entity_invpreds[ent])
        for r, cnt in rels.items():
            data.append((*grp, r, True, cnt))
        for r, cnt in invrels.items():
            data.append((*grp, r, False, cnt))
    df_relcounts = pd.DataFrame(data=data, columns=page_grouping + ['pred', 'inv', 'pred_count'])
    df_relcounts['rel'] = df_relcounts.apply(_predinv_to_rel_row, axis=1)
    df_relcounts.drop(columns=['pred', 'inv'], inplace=True)
    dfrP = pd.merge(how='left', left=dfrP, right=df_relcounts, on=page_grouping + ['rel']).fillna(0)
    # compute rel_conf on P-level
    dfrP['rel_conf'] = (dfrP['rel_count'] / dfrP['pred_count']).fillna(0).clip(0, 1)
    return dfrP


def _aggregate_relations_by_section(dfrP: pd.DataFrame, section_grouping: list) -> pd.DataFrame:
    """Aggregate the df by (top-)section."""
    relation_grouping = section_grouping + ['rel']
    dfrP = dfrP.drop_duplicates(subset=relation_grouping + ['P'])
    # compute mean, std
    result = dfrP.groupby(relation_grouping).agg({'rel_count': 'sum', 'pred_count': 'sum', 'rel_conf': ['mean', 'std']}).set_axis(['rel_count', 'pred_count', 'macro_mean', 'macro_std'], axis=1)
    result['micro_mean'] = result['rel_count'] / result['pred_count']
    # compute micro_std
    micro_std = pd.merge(how='left', left=dfrP, right=result, on=relation_grouping)
    micro_std['micro_std'] = np.absolute(micro_std['micro_mean'] - micro_std['rel_conf'])
    micro_std = micro_std.groupby(relation_grouping)['micro_std'].mean()
    result = pd.merge(left=result, right=micro_std, on=relation_grouping)
    # add page_count
    page_count = dfrP.groupby(relation_grouping)['P'].nunique().rename('page_count')
    result = pd.merge(left=result, right=page_count, on=relation_grouping)
    # filter low-pred rules
    result = result[result['pred_count'] > 2]
    result.drop(columns=['rel_count', 'pred_count'], inplace=True)
    return result


def _extract_new_relations_with_threshold(df: pd.DataFrame, dfr_types: pd.DataFrame, df_rels: pd.DataFrame, target_identifier: str, rule_dfs: dict) -> pd.DataFrame:
    """Extract new relations from rules based on support, confidence, and consistency thresholds."""
    mean_threshold = utils.get_config('listing.relation_mean_threshold')
    std_threshold = utils.get_config('listing.relation_std_threshold')
    valid_rule_dfs = [rule_dfs[rule_name].query(f'page_count > 2 & micro_mean > {mean_threshold} & micro_std < {std_threshold}').reset_index()[rule_pattern + ['rel', 'micro_mean', 'micro_std']].drop_duplicates() for rule_name, rule_pattern in RULE_PATTERNS.items()]
    return _extract_new_relations(valid_rule_dfs, target_identifier, df, dfr_types, df_rels)


def _extract_new_relations(rule_dfs: list, target_identifier: str, source_df: pd.DataFrame, dfr_types: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    """Apply valid rules to the initial dataframe to extract new relation assertions."""
    # extract new relations
    result_df = pd.DataFrame()
    for df_rule in rule_dfs:
        key_cols = list(set(df_rule.columns).difference({'rel', 'micro_mean', 'micro_std'}))
        result_df = pd.concat([result_df, pd.merge(how='left', left=df_rule, right=source_df, on=key_cols)])
    # filter out invalid target values
    result_df = result_df.dropna(subset=[target_identifier])
    result_df = _remove_relations_with_invalid_targets(result_df, target_identifier)
    # filter out existing relations
    predicate_mapping = _get_predicate_mapping()
    result_df['pred'] = result_df.apply(lambda x: predicate_mapping[x['rel']], axis=1)
    result_df['inv'] = result_df['rel'].str.startswith('<')
    result_df = pd.merge(left=result_df, right=dfr_types, on=['pred', 'inv'])

    result_df['target'] = result_df[target_identifier]
    return pd.concat([
        pd.merge(indicator=True, how='left', on=['target', 'pred', 'E_ent'], left=result_df[~result_df['inv']], right=df_rels.rename(columns={'sub': 'target', 'obj': 'E_ent'})),
        pd.merge(indicator=True, how='left', on=['target', 'pred', 'E_ent'], left=result_df[result_df['inv']], right=df_rels.rename(columns={'sub': 'E_ent', 'obj': 'target'}))
    ]).query('_merge=="left_only"').drop(columns='_merge')


def _remove_relations_with_invalid_targets(df: pd.DataFrame, target_identifier: str) -> pd.DataFrame:
    valid_targets = {ent.idx for ent in ClgEntityStore.instance().get_entities()}
    return df[df[target_identifier].isin(valid_targets)]


def _filter_new_relations_by_tag(df: pd.DataFrame, valid_tags: Dict[str, Set[str]]) -> pd.DataFrame:
    """Filter out entities with a NE tag that is not in `valid_tags` of the domain/range."""
    return df[df.apply(lambda row: row['E_predtype'] in valid_tags and row['E_tag'] in valid_tags[row['E_predtype']], axis=1)]


def _get_predicate_mapping() -> Dict[str, int]:
    clgo = ClgOntologyStore.instance()
    predicate_mapping = {_predinv_to_rel(p.idx, True): p.idx for p in clgo.get_predicates()}
    predicate_mapping.update({_predinv_to_rel(p.idx, False): p.idx for p in clgo.get_predicates()})
    return predicate_mapping


def _predinv_to_rel_row(row) -> str:
    return _predinv_to_rel(row['pred'], row['inv'])


def _predinv_to_rel(pred_idx: int, inv: bool) -> str:
    return f'< {pred_idx} <' if inv else f'> {pred_idx} >'
