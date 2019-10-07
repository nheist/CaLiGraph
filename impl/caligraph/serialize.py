import util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.caligraph.util as cali_util
import bz2
from typing import Optional
import impl.util.serialize as serialize_util
import impl.util.rdf as rdf_util
import datetime


def serialize_graph(graph):
    lines = []

    # void metadata
    lines.extend(_get_metadata(graph))

    # ontology metadata
    ontology_resource = 'http://caligraph.org/ontology'
    lines.extend([
        serialize_util.as_object_triple(ontology_resource, rdf_util.PREDICATE_TYPE, 'http://www.w3.org/2002/07/owl#Ontology'),
        serialize_util.as_literal_triple(ontology_resource, 'http://purl.org/dc/terms/created', _get_creation_date()),
        serialize_util.as_literal_triple(ontology_resource, rdf_util.PREDICATE_LABEL, 'CaLiGraph Ontology'),
        serialize_util.as_literal_triple(ontology_resource, 'http://www.w3.org/2002/07/owl#versionInfo', util.get_config('caligraph.version')),
    ])

    # classes
    for node in graph.traverse_topdown():
        label = graph.get_label(node)
        parents = graph.parents(node) or {rdf_util.CLASS_OWL_THING}
        equivalents = {t for t in graph.get_parts(node) if dbp_util.is_dbp_type(t)}
        sources = graph.get_parts(node)
        lines.extend(_serialize_class(node, label, parents, equivalents, sources))

    # properties
    for prop in graph.get_all_properties():
        equivalent_property = cali_util.clg_type2dbp_type(prop)
        _serialize_property(prop, equivalent_property)

    # resources
    axiom_resources = {ax[1] for n in graph.nodes for ax in graph.get_axioms(n, transitive=False)}
    for res in graph.get_all_resources() | axiom_resources:
        label = graph.get_label(res)
        types = graph.get_nodes_for_resource(res)
        equivalent = cali_util.clg_resource2dbp_resource(res) if cali_util.clg_resource2dbp_resource(res) in dbp_store.get_resources() else None
        provenance = graph.get_resource_provenance(res)
        _serialize_resource(res, label, types, equivalent, provenance)

    # restrictions
    blank_node_counter = 1
    for node in graph.nodes:
        for prop, val in graph.get_axioms(node, transitive=False):
            _serialize_restriction(node, prop, val, blank_node_counter)
            blank_node_counter += 1

    # persist minimal version
    result_filepath = util.get_results_file('results.caligraph.minimal')
    with bz2.open(result_filepath, mode='wt') as f:
        f.writelines(lines)

    # transitive classes, transitive types, materialized axioms
    for node in graph.traverse_topdown():
        direct_parents = graph.parents(node)
        transitive_parents = {tp for p in direct_parents for tp in graph.ancestors(p)}.difference(direct_parents)
        lines.extend([serialize_util.as_object_triple(node, rdf_util.PREDICATE_SUBCLASS_OF, tp) for tp in transitive_parents])

    for res in graph.get_all_resources():
        direct_types = graph.get_nodes_for_resource(res)
        transitive_types = {tt for t in direct_types for tt in graph.ancestors(t)}.difference(direct_types)
        lines.extend([serialize_util.as_object_triple(res, rdf_util.PREDICATE_TYPE, tt) for tt in transitive_types])

        axioms = {a for t in direct_types for a in graph.get_axioms(t)}
        for prop, val in axioms:
            if val.startswith(cali_util.NAMESPACE_CLG_RESOURCE):
                lines.append(serialize_util.as_object_triple(res, prop, val))
            else:
                lines.append(serialize_util.as_literal_triple(res, prop, val))

    # persist full version
    result_filepath = util.get_results_file('results.caligraph.materialized')
    with bz2.open(result_filepath, mode='wt') as f:
        f.writelines(lines)


def _get_metadata(graph) -> list:
    void_resource = 'http://caligraph.org/.well-known/void'
    description = 'The CaLiGraph is a large-scale general-purpose knowledge graph that extends DBpedia with a more fine-grained and restrictive ontology as well as additional resources extracted from Wikipedia Listpages.'
    entity_count = len(graph.get_all_resources())
    class_count = len(graph.nodes)
    property_count = len(graph.get_all_properties())

    return [
        serialize_util.as_object_triple(void_resource, rdf_util.PREDICATE_TYPE, 'http://rdfs.org/ns/void#Dataset'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/elements/1.1/title', 'CaLiGraph'),
        serialize_util.as_literal_triple(void_resource, rdf_util.PREDICATE_LABEL, 'CaLiGraph'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/elements/1.1/description', description),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/license', 'http://www.gnu.org/copyleft/fdl.html'),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/license', 'http://creativecommons.org/licenses/by-sa/3.0/'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/creator', 'Nicolas Heist'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/creator', 'Heiko Paulheim'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/created', _get_creation_date()),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/publisher', 'Nicolas Heist'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/publisher', 'Heiko Paulheim'),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#uriSpace', cali_util.NAMESPACE_CLG_RESOURCE),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#entities', entity_count),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#classes', class_count),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#properties', property_count),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/source', 'http://dbpedia.org/resource/DBpedia'),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/source', 'http://dbpedia.org/resource/Wikipedia'),
        serialize_util.as_object_triple(void_resource, 'http://xmlns.com/foaf/0.1/homepage', 'http://caligraph.org'),
        serialize_util.as_object_triple(void_resource, 'http://rdfs.org/ns/void#sparqlEndpoint', 'http://caligraph.org/sparql'),
    ]


def _get_creation_date() -> datetime.datetime:
    return datetime.datetime.strptime(util.get_config('caligraph.creation_date'), '%Y-%m-%d')


def _serialize_class(class_iri: str, label: str, parents: set, equivalents: set, sources: set) -> list:
    result = [serialize_util.as_object_triple(class_iri, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_CLASS)]
    if label:
        result.append(serialize_util.as_literal_triple(class_iri, rdf_util.PREDICATE_LABEL, label))
    result.extend([serialize_util.as_object_triple(class_iri, rdf_util.PREDICATE_SUBCLASS_OF, p) for p in parents])
    result.extend([serialize_util.as_object_triple(class_iri, rdf_util.PREDICATE_EQUIVALENT_CLASS, e) for e in equivalents])
    result.extend([serialize_util.as_object_triple(class_iri, 'http://www.w3.org/ns/prov#wasDerivedFrom', s) for s in sources])
    return result


def _serialize_property(prop_iri: str, equivalent_property_iri: str):
    return [
        serialize_util.as_object_triple(prop_iri, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_CLASS),
        serialize_util.as_object_triple(prop_iri, rdf_util.PREDICATE_SUBPROPERTY_OF, rdf_util.CLASS_PROPERTY),
        serialize_util.as_object_triple(prop_iri, rdf_util.PREDICATE_EQUIVALENT_PROPERTY, equivalent_property_iri),
    ]


def _serialize_resource(resource_iri: str, label: str, types: set, equivalent: Optional[str], provenance_iris: set):
    result = [serialize_util.as_object_triple(resource_iri, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_NAMED_INDIVIDUAL)]
    if label:
        result.append(serialize_util.as_literal_triple(resource_iri, rdf_util.PREDICATE_LABEL, label))
    result.extend([serialize_util.as_object_triple(resource_iri, rdf_util.PREDICATE_TYPE, t) for t in types])
    result.extend([serialize_util.as_object_triple(resource_iri, 'http://www.w3.org/ns/prov#wasDerivedFrom', p) for p in provenance_iris])
    if equivalent:
        result.append(serialize_util.as_object_triple(resource_iri, rdf_util.PREDICATE_SAME_AS, equivalent))
    return result


def _serialize_restriction(class_iri: str, prop_iri: str, val: str, blank_node_index: int):
    blank_node_iri = f'_:{blank_node_index}'
    result = [
        serialize_util.as_object_triple(blank_node_iri, rdf_util.PREDICATE_TYPE, 'http://www.w3.org/2002/07/owl#Restriction'),
        serialize_util.as_object_triple(blank_node_iri, 'http://www.w3.org/2002/07/owl#onProperty', prop_iri),
        serialize_util.as_object_triple(class_iri, rdf_util.PREDICATE_SUBCLASS_OF, blank_node_iri)
    ]
    if val.startswith(cali_util.NAMESPACE_CLG_RESOURCE):
        result.append(serialize_util.as_object_triple(blank_node_iri, 'http://www.w3.org/2002/07/owl#hasValue', val))
    else:
        result.append(serialize_util.as_literal_triple(blank_node_iri, 'http://www.w3.org/2002/07/owl#hasValue', val))
    return result
