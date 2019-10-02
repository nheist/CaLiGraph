from owlready2 import *
import util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.caligraph.util as cali_util
import bz2
from typing import Optional
import os
import types
import urllib.parse


def serialize_graph(graph):
    # initialise ontology
    caligraph_onto = get_ontology(cali_util.NAMESPACE_CLG_ONTOLOGY)
    caligraph_resource_ns = caligraph_onto.get_namespace(cali_util.NAMESPACE_CLG_RESOURCE)

    dbpedia_onto = get_ontology(util.get_config('files.dbpedia.taxonomy_owl.url')).load()
    dbpedia_resource_ns = dbpedia_onto.get_namespace(dbp_util.NAMESPACE_DBP_RESOURCE)
    dbpedia_onto.metadata.versionInfo = util.get_config('caligraph.version')

    get_ontology('https://www.w3.org/ns/prov-o.owl').load()  # load prov onto to be able to use wasDerivedFrom property

    # define classes
    for node in graph.traverse_topdown():
        name = _get_item_name(node, cali_util.NAMESPACE_CLG_ONTOLOGY)
        label = graph.get_label(node)
        parents = {_encode_item_name(p) for p in graph.parents(node)} or {Thing.iri}
        equivalents = {_encode_item_name(t) for t in graph.get_parts(node) if dbp_util.is_dbp_type(t)}
        for eq in equivalents:
            if not IRIS[eq]:  # make sure that the equivalent dbpedia classes are initialised
                _add_class(dbpedia_onto, eq[len(dbp_util.NAMESPACE_DBP_ONTOLOGY):], None, {Thing.iri}, set())
        _add_class(caligraph_onto, name, label, parents, equivalents)

    # define properties
    for prop in graph.get_all_properties():
        name = _get_item_name(prop, cali_util.NAMESPACE_CLG_ONTOLOGY)
        dbp_property = dbp_util.NAMESPACE_DBP_ONTOLOGY + name
        _add_property(caligraph_onto, name, dbp_property)

    # define resources
    axiom_resources = {ax[1] for n in graph.nodes for ax in graph.get_axioms(n, transitive=False)}
    for res in graph.get_all_resources() | axiom_resources:
        name = _get_item_name(res, cali_util.NAMESPACE_CLG_RESOURCE)
        classes = graph.get_nodes_for_resource(res) or {Thing.iri}
        equivalent = _get_dbpedia_resource(_encode_item_name(cali_util.clg_resource2dbp_resource(res)), dbpedia_resource_ns) if cali_util.clg_resource2dbp_resource(res) in dbp_store.get_resources() else None
        provenance = {_encode_item_name(_get_dbpedia_resource(prov, dbpedia_resource_ns)) for prov in graph.get_resource_provenance(res)}
        label = graph.get_label(res)
        _add_resource(caligraph_resource_ns, name, label, classes, equivalent, provenance)

    # define restrictions
    for node in graph.nodes:
        for prop, val in graph.get_axioms(node, transitive=False):
            _add_restriction(node, prop, val)

    # intermediate persist
    result_filepath = util.get_results_file('results.caligraph.complete')
    tmp_filepath = result_filepath[:-4]
    caligraph_onto.save(file=tmp_filepath, format="ntriples")

    # define metadata (including type provenance data)
    metadata_lines = _get_metadata(graph)

    # write metadata and graph data
    with bz2.open(result_filepath, mode='wt') as f:
        f.writelines(metadata_lines)
        with open(tmp_filepath, mode='r') as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_filepath)


def _add_class(onto, cls_name: str, label: Optional[str], parents: set, equivalents: set):
    with onto:
        cls = types.new_class(cls_name, tuple([IRIS[p] for p in parents]))
    cls.equivalent_to.extend([IRIS[eq] for eq in equivalents])
    if label:
        cls.label = label


def _add_property(onto, prop_name: str, equivalent_prop: str):
    with onto:
        prop = types.new_class(prop_name, (Property,))
    prop.equivalent_to.append(IRIS[equivalent_prop])


def _add_resource(resource_namespace, res_name: str, label: str, classes: set, equivalent: Optional[str], provenance: set):
    main_class = IRIS[classes.pop()]
    res = main_class(res_name, namespace=resource_namespace)
    res.is_a.extend([IRIS[c] for c in classes])
    res.label = label
    if equivalent:
        res.equivalent_to.append(equivalent)
    res.wasDerivedFrom.extend(provenance)


def _add_restriction(cls_iri: str, prop_iri: str, val: str):
    cls = IRIS[cls_iri]
    prop = IRIS[prop_iri]
    val = _encode_item_name(IRIS[val]) if cali_util.is_clg_resource(val) else val
    cls.is_a.append(prop.value(val))


def _get_dbpedia_resource(resource_iri: str, resource_ns):
    resource = IRIS[resource_iri]
    if not resource:
        name = _get_item_name(resource_iri, dbp_util.NAMESPACE_DBP_RESOURCE)
        resource = Thing(name, namespace=resource_ns)
    return resource


def _get_metadata(graph) -> list:
    void_entity = '<http://caligraph.org/.well-known/void>'
    description = 'The CaLiGraph is a large-scale general-purpose knowledge graph that extends DBpedia with a more fine-grained and restrictive ontology as well as additional resources extracted from Wikipedia Listpages.'
    entity_count = len(graph.get_all_resources())
    class_count = len(graph.nodes)
    property_count = len(graph.get_all_properties())

    metadata = [
        f'{void_entity} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://rdfs.org/ns/void#Dataset> .\n',
        f'{void_entity} <http://purl.org/dc/elements/1.1/title> "CaLiGraph" .\n',
        f'{void_entity} <http://www.w3.org/2000/01/rdf-schema#label> "CaLiGraph" .\n',
        f'{void_entity} <http://purl.org/dc/elements/1.1/description> "{description}" .\n',
        f'{void_entity} <http://purl.org/dc/terms/license> <http://www.gnu.org/copyleft/fdl.html> .\n',
        f'{void_entity} <http://purl.org/dc/terms/license> <http://creativecommons.org/licenses/by-sa/3.0/> .\n',
        f'{void_entity} <http://purl.org/dc/terms/creator> "Nicolas Heist" .\n',
        f'{void_entity} <http://purl.org/dc/terms/creator> "Nicolas Heist" .\n',
        f'{void_entity} <http://purl.org/dc/terms/creator> "Heiko Paulheim" .\n',
        f'{void_entity} <http://purl.org/dc/terms/created> "2019-10-01"^^<http://www.w3.org/2001/XMLSchema#date> .\n',
        f'{void_entity} <http://purl.org/dc/terms/publisher> "Nicolas Heist" .\n',
        f'{void_entity} <http://purl.org/dc/terms/publisher> "Heiko Paulheim" .\n',
        f'{void_entity} <http://rdfs.org/ns/void#uriSpace> "{cali_util.NAMESPACE_CLG_RESOURCE}" .\n',
        f'{void_entity} <http://rdfs.org/ns/void#entities> "{entity_count}"^^<http://www.w3.org/2001/XMLSchema#integer> .\n',
        f'{void_entity} <http://rdfs.org/ns/void#classes> "{class_count}"^^<http://www.w3.org/2001/XMLSchema#integer> .\n',
        f'{void_entity} <http://rdfs.org/ns/void#properties> "{property_count}"^^<http://www.w3.org/2001/XMLSchema#integer> .\n',
        f'{void_entity} <http://purl.org/dc/terms/source> <http://dbpedia.org/resource/DBpedia> .\n',
        f'{void_entity} <http://purl.org/dc/terms/source> <http://dbpedia.org/resource/Wikipedia> .\n',
        f'{void_entity} <http://xmlns.com/foaf/0.1/homepage> <http://caligraph.org> .\n',
        f'{void_entity} <http://rdfs.org/ns/void#sparqlEndpoint> <http://caligraph.org/sparql> .\n',
    ]

    for node in graph.nodes:
        for part in graph.get_parts(node):
            metadata.append(f'<{node}> <http://www.w3.org/ns/prov#wasDerivedFrom> <{part}> .\n')

    return metadata


def _get_item_name(item: str, namespace: str) -> str:
    return _encode_item_name(item[len(namespace):])


def _encode_item_name(name: str) -> str:
    for c in ['\'', '"', '´', '`']:
        name = name.replace(c, urllib.parse.quote_plus(c))
    return name
