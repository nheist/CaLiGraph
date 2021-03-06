"""Functionality to serialize the individual parts of CaLiGraph."""

import utils
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.caligraph.util as cali_util
import bz2
import impl.util.serialize as serialize_util
import impl.util.rdf as rdf_util
import datetime
from collections import defaultdict


def serialize_graph(graph):
    """Serialize the complete graph as individual files."""
    _write_lines_to_file(_get_lines_metadata(graph), 'results.caligraph.metadata')
    _write_lines_to_file(_get_lines_ontology(graph), 'results.caligraph.ontology')
    _write_lines_to_file(_get_lines_ontology_dbpedia_mapping(graph), 'results.caligraph.ontology_dbpedia-mapping')
    _write_lines_to_file(_get_lines_ontology_provenance(graph), 'results.caligraph.ontology_provenance')
    _write_lines_to_file(_get_lines_instances_types(graph), 'results.caligraph.instances_types')
    _write_lines_to_file(_get_lines_instances_transitive_types(graph), 'results.caligraph.instances_transitive-types')
    _write_lines_to_file(_get_lines_instances_labels(graph), 'results.caligraph.instances_labels')
    _write_lines_to_file(_get_lines_instances_relations(graph), 'results.caligraph.instances_relations')
    _write_lines_to_file(_get_lines_instances_dbpedia_mapping(graph), 'results.caligraph.instances_dbpedia-mapping')
    _write_lines_to_file(_get_lines_instances_provenance(graph), 'results.caligraph.instances_provenance')

    _write_lines_to_file(_get_lines_dbpedia_instances(graph), 'results.caligraph.dbpedia_instances')
    _write_lines_to_file(_get_lines_dbpedia_instance_types(graph), 'results.caligraph.dbpedia_instance-types')
    _write_lines_to_file(_get_lines_dbpedia_instance_caligraph_types(graph), 'results.caligraph.dbpedia_instance-caligraph-types')
    _write_lines_to_file(_get_lines_dbpedia_instance_transitive_caligraph_types(graph), 'results.caligraph.dbpedia_instance-transitive-caligraph-types')
    _write_lines_to_file(_get_lines_dbpedia_instance_relations(graph), 'results.caligraph.dbpedia_instance-relations')


def _write_lines_to_file(lines: list, filepath_config: str):
    filepath = utils.get_results_file(filepath_config)
    with bz2.open(filepath, mode='wt') as f:
        f.writelines(lines)


def _get_lines_metadata(graph) -> list:
    """Serialize metadata."""
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


def _get_lines_ontology(graph) -> list:
    """Serialize the ontology."""
    lines_ontology = []
    # metadata
    ontology_resource = 'http://caligraph.org/ontology'
    lines_ontology.extend([
        serialize_util.as_object_triple(ontology_resource, rdf_util.PREDICATE_TYPE, 'http://www.w3.org/2002/07/owl#Ontology'),
        serialize_util.as_literal_triple(ontology_resource, 'http://purl.org/dc/terms/created', _get_creation_date()),
        serialize_util.as_literal_triple(ontology_resource, rdf_util.PREDICATE_LABEL, 'CaLiGraph Ontology'),
        serialize_util.as_literal_triple(ontology_resource, 'http://www.w3.org/2002/07/owl#versionInfo', utils.get_config('caligraph.version')),
    ])
    # classes
    for node in graph.traverse_nodes_topdown():
        if node == rdf_util.CLASS_OWL_THING:
            continue
        lines_ontology.append(serialize_util.as_object_triple(node, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_CLASS))
        label = graph.get_label(node)
        if label:
            lines_ontology.append(serialize_util.as_literal_triple(node, rdf_util.PREDICATE_LABEL, label))
        parents = graph.parents(node) or {rdf_util.CLASS_OWL_THING}
        lines_ontology.extend([serialize_util.as_object_triple(node, rdf_util.PREDICATE_SUBCLASS_OF, p) for p in parents])
    # properties
    for prop in graph.get_all_properties():
        lines_ontology.append(serialize_util.as_object_triple(prop, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_PROPERTY))
    # disjointnesses
    for node in graph.nodes:
        for disjoint_node in graph.get_disjoint_nodes(node):
            if node < disjoint_node:  # make sure that disjointnesses are only serialized once
                lines_ontology.append(serialize_util.as_object_triple(node, rdf_util.PREDICATE_DISJOINT_WITH, disjoint_node))
    # restrictions
    defined_restrictions = set()
    for node in graph.nodes:
        for prop, val in graph.get_axioms(node, transitive=False):
            restriction_is_defined = (prop, val) in defined_restrictions
            lines_ontology.extend(_serialize_restriction(node, prop, val, restriction_is_defined))
            defined_restrictions.add((prop, val))
    return lines_ontology


def _get_creation_date() -> datetime.datetime:
    return datetime.datetime.strptime(utils.get_config('caligraph.creation_date'), '%Y-%m-%d')


def _serialize_restriction(class_iri: str, prop_iri: str, val: str, restriction_is_defined: bool) -> list:
    """Serialize the restrictions (i.e. relation axioms)."""
    prop_id = prop_iri[len(cali_util.NAMESPACE_CLG_ONTOLOGY):]
    val_id = val[len(cali_util.NAMESPACE_CLG_RESOURCE):] if cali_util.is_clg_resource(val) else val
    restriction_iri = f'{cali_util.NAMESPACE_CLG_ONTOLOGY}RestrictionHasValue_{prop_id}_{val_id}'

    if restriction_is_defined:
        result = []
    else:
        result = [
            serialize_util.as_object_triple(restriction_iri, rdf_util.PREDICATE_TYPE, 'http://www.w3.org/2002/07/owl#Restriction'),
            serialize_util.as_literal_triple(restriction_iri, rdf_util.PREDICATE_LABEL, f'Restriction onProperty={prop_id} hasValue={val_id}'),
            serialize_util.as_object_triple(restriction_iri, 'http://www.w3.org/2002/07/owl#onProperty', prop_iri),
        ]
        if cali_util.is_clg_resource(val):
            result.append(serialize_util.as_object_triple(restriction_iri, 'http://www.w3.org/2002/07/owl#hasValue', val))
        else:
            result.append(serialize_util.as_literal_triple(restriction_iri, 'http://www.w3.org/2002/07/owl#hasValue', val))

    result.append(serialize_util.as_object_triple(class_iri, rdf_util.PREDICATE_SUBCLASS_OF, restriction_iri))
    return result


def _get_lines_ontology_dbpedia_mapping(graph) -> list:
    """Serialize the DBpedia mapping for types and predicates."""
    lines_ontology_dbpedia_mapping = []
    for node in graph.traverse_nodes_topdown():
        if node == rdf_util.CLASS_OWL_THING:
            continue
        equivalents = {t for t in graph.get_parts(node) if dbp_util.is_dbp_type(t)}
        lines_ontology_dbpedia_mapping.extend([serialize_util.as_object_triple(node, rdf_util.PREDICATE_SUBCLASS_OF, e) for e in equivalents])
    for prop in graph.get_all_properties():
        eq_prop = cali_util.clg_type2dbp_type(prop)
        lines_ontology_dbpedia_mapping.append(serialize_util.as_object_triple(prop, rdf_util.PREDICATE_EQUIVALENT_PROPERTY, eq_prop))
    return lines_ontology_dbpedia_mapping


def _get_lines_ontology_provenance(graph) -> list:
    """Serialize provenance information of the ontology."""
    lines_ontology_provenance = []
    for node in graph.traverse_nodes_topdown():
        if node == rdf_util.CLASS_OWL_THING:
            continue
        sources = {dbp_util.dbp_resource2wikipedia_resource(p) for p in graph.get_parts(node)}
        lines_ontology_provenance.extend([serialize_util.as_object_triple(node, rdf_util.PREDICATE_WAS_DERIVED_FROM, s) for s in sources])
    return lines_ontology_provenance


def _get_lines_instances_types(graph) -> list:
    """Serialize types of resources."""
    lines_instances_types = []

    caligraph_ancestors = defaultdict(set)
    for n in graph.traverse_nodes_topdown():
        parents = graph.parents(n)
        caligraph_ancestors[n] = parents | {a for p in parents for a in caligraph_ancestors[p]}

    axiom_resources = {ax[1] for n in graph.nodes for ax in graph.get_axioms(n, transitive=False) if cali_util.is_clg_resource(ax[1])}
    for res in graph.get_all_resources() | axiom_resources:
        lines_instances_types.append(serialize_util.as_object_triple(res, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_NAMED_INDIVIDUAL))
        types = graph.get_nodes_for_resource(res)
        direct_types = types.difference({a for t in types for a in caligraph_ancestors[t]})
        lines_instances_types.extend([serialize_util.as_object_triple(res, rdf_util.PREDICATE_TYPE, t) for t in direct_types])
    return lines_instances_types


def _get_lines_instances_transitive_types(graph) -> list:
    """Serialize transitive types of resources."""
    lines_instances_transitive_types = []

    caligraph_ancestors = defaultdict(set)
    for n in graph.traverse_nodes_topdown():
        parents = graph.parents(n)
        caligraph_ancestors[n] = parents | {a for p in parents for a in caligraph_ancestors[p]}

    for res in graph.get_all_resources():
        types = graph.get_nodes_for_resource(res)
        direct_types = types.difference({a for t in types for a in caligraph_ancestors[t]})
        transitive_types = {tt for t in direct_types for tt in graph.ancestors(t)}.difference(direct_types | {rdf_util.CLASS_OWL_THING})
        lines_instances_transitive_types.extend([serialize_util.as_object_triple(res, rdf_util.PREDICATE_TYPE, tt) for tt in transitive_types])
    return lines_instances_transitive_types


def _get_lines_instances_labels(graph) -> list:
    """Serialize resource labels."""
    lines_instances_labels = []
    axiom_resources = {ax[1] for n in graph.nodes for ax in graph.get_axioms(n, transitive=False) if cali_util.is_clg_resource(ax[1])}
    for res in graph.get_all_resources() | axiom_resources:
        label = graph.get_label(res)
        if label:
            lines_instances_labels.append(serialize_util.as_literal_triple(res, rdf_util.PREDICATE_LABEL, label))
            lines_instances_labels.append(serialize_util.as_literal_triple(res, rdf_util.PREDICATE_PREFLABEL, label))
            altlabels = [serialize_util.as_literal_triple(res, rdf_util.PREDICATE_ALTLABEL, l) for l in graph.get_altlabels(res) if l != label]
            lines_instances_labels.extend(altlabels)
    return lines_instances_labels


def _get_lines_instances_relations(graph) -> list:
    """Serialize resource facts."""
    lines_instances_relations = []
    instance_relations = set()
    for node in graph.nodes:
        for prop, val in graph.get_axioms(node):
            instance_relations.update({(res, prop, val) for res in graph.get_resources(node)})
    for s, p, o in instance_relations:
        if cali_util.is_clg_resource(o):
            lines_instances_relations.append(serialize_util.as_object_triple(s, p, o))
        else:
            lines_instances_relations.append(serialize_util.as_literal_triple(s, p, o))
    return lines_instances_relations


def _get_lines_instances_dbpedia_mapping(graph) -> list:
    """Serialize DBpedia mapping for resources."""
    lines_instances_dbpedia_mapping = []
    axiom_resources = {ax[1] for n in graph.nodes for ax in graph.get_axioms(n, transitive=False) if cali_util.is_clg_resource(ax[1])}
    for res in graph.get_all_resources() | axiom_resources:
        equivalent_res = cali_util.clg_resource2dbp_resource(res)
        if equivalent_res in dbp_store.get_resources():
            lines_instances_dbpedia_mapping.append(serialize_util.as_object_triple(res, rdf_util.PREDICATE_SAME_AS, equivalent_res))
    return lines_instances_dbpedia_mapping


def _get_lines_instances_provenance(graph) -> list:
    """Serialize provenance information for resources."""
    lines_instances_provenance = []
    for res in graph.get_all_resources():
        provenance_data = {dbp_util.dbp_resource2wikipedia_resource(p) for p in graph.get_resource_provenance(res)}
        lines_instances_provenance.extend([serialize_util.as_object_triple(res, rdf_util.PREDICATE_WAS_DERIVED_FROM, p) for p in provenance_data])
    return lines_instances_provenance


def _get_lines_dbpedia_instances(graph) -> list:
    """Serialize new DBpedia resources in DBpedia namespace."""
    lines_dbpedia_instances = []
    new_instances = {cali_util.clg_resource2dbp_resource(res) for res in graph.get_all_resources()}.difference(dbp_store.get_resources())
    for inst in new_instances:
        lines_dbpedia_instances.append(serialize_util.as_object_triple(inst, rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_NAMED_INDIVIDUAL))
        label = graph.get_label(cali_util.dbp_resource2clg_resource(inst))
        if label:
            lines_dbpedia_instances.append(serialize_util.as_literal_triple(inst, rdf_util.PREDICATE_LABEL, label))
    return lines_dbpedia_instances


def _get_lines_dbpedia_instance_types(graph) -> list:
    """Serialize new types for DBpedia resources in DBpedia namespace."""
    new_dbpedia_types = defaultdict(set)
    for node in graph.nodes:
        node_types = graph.get_transitive_dbpedia_types(node, force_recompute=True)
        transitive_node_types = {tt for t in node_types for tt in dbp_store.get_transitive_supertype_closure(t)}.difference({rdf_util.CLASS_OWL_THING})
        for res in graph.get_resources(node):
            dbp_res = cali_util.clg_resource2dbp_resource(res)
            if dbp_res in dbp_store.get_resources():
                new_dbpedia_types[dbp_res].update(transitive_node_types.difference(dbp_store.get_transitive_types(dbp_res)))
            else:
                new_dbpedia_types[dbp_res].update(transitive_node_types)
    return [serialize_util.as_object_triple(res, rdf_util.PREDICATE_TYPE, t) for res, types in new_dbpedia_types.items() for t in types]


def _get_lines_dbpedia_instance_caligraph_types(graph) -> list:
    """Serialize CaLiGraph types for DBpedia resources."""
    instance_clg_types = []

    caligraph_ancestors = defaultdict(set)
    for n in graph.traverse_nodes_topdown():
        parents = graph.parents(n)
        caligraph_ancestors[n] = parents | {a for p in parents for a in caligraph_ancestors[p]}

    for res in graph.get_all_resources():
        dbp_res = cali_util.clg_resource2dbp_resource(res)
        if dbp_res not in dbp_store.get_resources():
            continue

        types = graph.get_nodes_for_resource(res)
        direct_types = types.difference({a for t in types for a in caligraph_ancestors[t]})
        instance_clg_types.extend([serialize_util.as_object_triple(dbp_res, rdf_util.PREDICATE_TYPE, t) for t in direct_types])
    return instance_clg_types


def _get_lines_dbpedia_instance_transitive_caligraph_types(graph) -> list:
    """Serialize transitive CaLiGraph types for DBpedia resources."""
    instance_transitive_clg_types = []

    caligraph_ancestors = defaultdict(set)
    for n in graph.traverse_nodes_topdown():
        parents = graph.parents(n)
        caligraph_ancestors[n] = parents | {a for p in parents for a in caligraph_ancestors[p]}

    for res in graph.get_all_resources():
        dbp_res = cali_util.clg_resource2dbp_resource(res)
        if dbp_res not in dbp_store.get_resources():
            continue

        types = graph.get_nodes_for_resource(res)
        direct_types = types.difference({a for t in types for a in caligraph_ancestors[t]})
        transitive_types = {tt for t in direct_types for tt in graph.ancestors(t)}.difference(direct_types | {rdf_util.CLASS_OWL_THING})
        instance_transitive_clg_types.extend([serialize_util.as_object_triple(dbp_res, rdf_util.PREDICATE_TYPE, tt) for tt in transitive_types])
    return instance_transitive_clg_types


def _get_lines_dbpedia_instance_relations(graph) -> list:
    """Serialize new facts for DBpedia resources in DBpedia namespace."""
    new_instance_relations = set()
    for node in graph.nodes:
        for prop, val in graph.get_axioms(node):
            dbp_prop = cali_util.clg_type2dbp_type(prop)
            dbp_val = cali_util.clg_resource2dbp_resource(val) if cali_util.is_clg_resource(val) else val
            for res in graph.get_resources(node):
                dbp_res = cali_util.clg_resource2dbp_resource(res)
                if dbp_res not in dbp_store.get_resources() or dbp_prop not in dbp_store.get_properties(dbp_res) or dbp_val not in dbp_store.get_properties(dbp_res)[dbp_prop]:
                    new_instance_relations.add((dbp_res, dbp_prop, dbp_val))
    lines_dbpedia_instance_relations = []
    for s, p, o in new_instance_relations:
        if dbp_util.is_dbp_resource(o):
            lines_dbpedia_instance_relations.append(serialize_util.as_object_triple(s, p, o))
        else:
            lines_dbpedia_instance_relations.append(serialize_util.as_literal_triple(s, p, o))
    return lines_dbpedia_instance_relations
