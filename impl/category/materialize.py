import impl.category.base as cat_base
import impl.category.store as cat_store


__MATERIALIZED_RESOURCE_CACHE__ = {}


def get_materialized_resources(category: str) -> set:
    if category not in __MATERIALIZED_RESOURCE_CACHE__:
        __MATERIALIZED_RESOURCE_CACHE__[category] = _compute_materialized_resources(category)
    return __MATERIALIZED_RESOURCE_CACHE__[category]


def _compute_materialized_resources(category: str) -> set:
    cat_graph = cat_base.get_wikitaxonomy_graph()
    child_resources = {r for c in cat_graph.children(category) for r in get_materialized_resources(c)}
    return cat_store.get_resources(category) | child_resources
