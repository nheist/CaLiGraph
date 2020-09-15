from pytest_check import check_func
import impl.caligraph.base as cali_base
from impl.caligraph.util import NAMESPACE_CLG_ONTOLOGY as CLGO
from impl.dbpedia.util import NAMESPACE_DBP_RESOURCE as DBR


def test_class_hierarchy():
    _is_parent_of(f'{CLGO}Air_force', f'{CLGO}Disbanded_air_force')
    _is_no_parent_of(f'{CLGO}Air_force', f'{CLGO}Air_force_personnel')


@check_func
def _is_parent_of(parent: str, child: str):
    G = cali_base.get_axiom_graph()
    assert child in G.children(parent)


@check_func
def _is_no_parent_of(parent: str, child: str):
    G = cali_base.get_axiom_graph()
    assert child not in G.children(parent)


def test_node_parts():
    _is_part_of(f'{DBR}Category:Organization', f'{CLGO}Organisation')


@check_func
def _is_part_of(part: str, node: str):
    G = cali_base.get_axiom_graph()
    assert part in G.get_parts(node)
