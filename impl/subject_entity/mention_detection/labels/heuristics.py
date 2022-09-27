"""Heuristic retrieval of subject entities in list pages using the mapping to DBpedia categories."""

from typing import Tuple, Set, Dict, List
from collections import defaultdict
import utils
import impl.dbpedia.heuristics as dbp_heur
from impl import category
from impl import listpage
import impl.listpage.mapping as list_mapping
from impl.dbpedia.ontology import DbpOntologyStore
from impl.dbpedia.resource import DbpEntity, DbpListpage
from impl.dbpedia.category import DbpCategory
from impl.caligraph.ontology import ClgOntologyStore
from impl.wikipedia import WikiPage, WikiPageStore


def find_subject_entities_for_pages(pages: List[WikiPage]) -> Dict[int, Dict[int, Tuple[Set[int], Set[int]]]]:
    lp_cache = None
    if any(isinstance(page.resource, DbpListpage) for page in pages):
        lp_cache = defaultdict(dict, utils.load_or_create_cache('subject_entity_listpage_labels', _find_subject_entities_for_listpages))
    return {p.idx: lp_cache[p.idx] if isinstance(p.resource, DbpListpage) else _find_subject_entities_for_page(p) for p in pages}


def _find_subject_entities_for_page(wp: WikiPage) -> Dict[int, Tuple[Set[int], Set[int]]]:
    """Return indices of all subject entities that have consistent type labels for fine-tuning."""
    subject_entities_per_listing = {}
    for listing in wp.get_listings():
        se_indices = {se.entity_idx for se in listing.get_subject_entities()}
        se_tags = {se.entity_type for se in listing.get_subject_entities()}
        if len(se_indices) >= 5 and len(se_tags) == 1:
            subject_entities_per_listing[listing.idx] = (se_indices, set())
    return subject_entities_per_listing


def _find_subject_entities_for_listpages() -> Dict[int, Dict[int, Tuple[Set[int], Set[int]]]]:
    """Retrieve all entities of list pages that are subject entities or explicit non-subject entities."""
    subject_entities = {}
    for page in WikiPageStore.instance().get_listpages():
        clgo = ClgOntologyStore.instance()
        dbo = DbpOntologyStore.instance()
        # compute potential subject entities for list page
        subject_entity_candidates = {ent for cat in _get_category_descendants_for_list(page.resource) for ent in cat.get_entities()}
        # find dbp types of listpage based on the caligraph ontology
        page_dbp_types = {dt for ct in clgo.get_types_for_associated_dbp_resource(page.resource) for dt in dbo.get_independent_types(ct.get_all_dbp_types())}
        page_disjoint_types = {dt for t in page_dbp_types for dt in dbp_heur.get_all_disjoint_types(t)}
        # retrieve subject entity ids per listing
        subject_entities_per_listing = {}
        for listing in page.get_listings():
            positives, negatives = set(), set()
            for ent in listing.get_mentioned_entities():
                if not isinstance(ent, DbpEntity):
                    negatives.add(ent.idx)
                elif ent in subject_entity_candidates:
                    positives.add(ent.idx)
                elif page_disjoint_types.intersection(ent.get_types()):
                    negatives.add(ent.idx)
            if positives or negatives:
                subject_entities_per_listing[listing.idx] = (positives, negatives)
        # filter out listings having only one or two positves (as this means a single entity is repeating multiple times)
        subject_entities_per_listing = {listing_idx: (positives, negatives) for listing_idx, (positives, negatives) in subject_entities_per_listing.items() if len(positives) == 0 or len(positives) > 2}
        if subject_entities_per_listing:
            subject_entities[page.idx] = subject_entities_per_listing
    return subject_entities


def _get_category_descendants_for_list(lp: DbpListpage) -> Set[DbpCategory]:
    """Return the categories that are most closely related to the given listpage as well as all of its children."""
    list_graph = listpage.get_merged_listgraph()
    cat_graph = category.get_merged_graph()
    category_nodes = {cat_node for list_node in list_graph.get_nodes_for_list(lp) for cat_node in list_mapping.get_related_category_nodes(list_node)}
    category_nodes = category_nodes | {dn for cn in category_nodes for dn in cat_graph.descendants(cn)}

    categories = {cat for cn in category_nodes for cat in cat_graph.get_categories(cn)}
    return categories
