"""Functionality to serialize the individual parts of CaLiGraph."""

from typing import Union, Dict
import bz2
import json
import utils
import random
import datetime
import numpy as np
from collections import Counter
import impl.util.rdf as rdf_util
import impl.dbpedia.util as dbp_util
import impl.util.serialize as serialize_util
from impl.util.rdf import RdfClass, RdfPredicate, Namespace
from impl.dbpedia.resource import DbpListpage
from impl.dbpedia.category import DbpCategory, DbpListCategory
from impl.caligraph.ontology import ClgType, ClgPredicate, ClgObjectPredicate, ClgOntologyStore
from impl.caligraph.entity import ClgEntity, ClgEntityStore


def run_serialization():
    """Serialize the complete graph as individual files."""
    clgo = ClgOntologyStore.instance()
    clge = ClgEntityStore.instance()

    _write_lines_to_file(_get_lines_metadata(clgo, clge), 'results.caligraph.metadata')
    _write_lines_to_file(_get_lines_ontology(clgo, clge), 'results.caligraph.ontology')
    _write_lines_to_file(_get_lines_ontology_dbpedia_mapping(clgo), 'results.caligraph.ontology_dbpedia-mapping')
    _write_lines_to_file(_get_lines_ontology_provenance(clgo), 'results.caligraph.ontology_provenance')
    _write_lines_to_file(_get_lines_instances_types(clge), 'results.caligraph.instances_types')
    _write_lines_to_file(_get_lines_instances_transitive_types(clge), 'results.caligraph.instances_transitive-types')
    _write_lines_to_file(_get_lines_instances_labels(clge), 'results.caligraph.instances_labels')
    _write_lines_to_file(_get_lines_instances_relations(clge), 'results.caligraph.instances_relations')
    _write_lines_to_file(_get_lines_instances_restriction_relations(clge), 'results.caligraph.instances_restriction-relations')
    _write_lines_to_file(_get_lines_instances_dbpedia_mapping(clge), 'results.caligraph.instances_dbpedia-mapping')
    _write_lines_to_file(_get_lines_instances_provenance(clge), 'results.caligraph.instances_provenance')

    _write_lines_to_file(_get_lines_dbpedia_instances(clge), 'results.caligraph.dbpedia_instances')
    _write_lines_to_file(_get_lines_dbpedia_instance_types(clge), 'results.caligraph.dbpedia_instance-types')
    _write_lines_to_file(_get_lines_dbpedia_instance_caligraph_types(clge), 'results.caligraph.dbpedia_instance-caligraph-types')
    _write_lines_to_file(_get_lines_dbpedia_instance_transitive_caligraph_types(clge), 'results.caligraph.dbpedia_instance-transitive-caligraph-types')
    _write_lines_to_file(_get_lines_dbpedia_instance_relations(clge), 'results.caligraph.dbpedia_instance-relations')

    _serialize_type_distribution(clgo, clge, 'results.caligraph.sunburst_type_distribution')

    utils.get_logger().info(_print_statistics(clgo, clge))


def _write_lines_to_file(lines: list, filepath_config: str):
    filepath = utils.get_results_file(filepath_config)
    with bz2.open(filepath, mode='wt') as f:
        f.writelines(lines)


def _get_lines_metadata(clgo, clge) -> list:
    """Serialize metadata."""
    void_resource = 'http://caligraph.org/.well-known/void'
    description = 'CaLiGraph is a large-scale general-purpose knowledge graph that extends DBpedia with a more fine-grained and restrictive ontology as well as additional resources extracted from Wikipedia list pages.'
    entity_count = len(clge.get_entities())
    class_count = len(clgo.get_types())
    predicate_count = len(clgo.get_predicates())
    return [
        serialize_util.as_object_triple(void_resource, RdfPredicate.TYPE, 'http://rdfs.org/ns/void#Dataset'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/elements/1.1/title', 'CaLiGraph'),
        serialize_util.as_literal_triple(void_resource, RdfPredicate.LABEL, 'CaLiGraph'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/elements/1.1/description', description),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/license', 'http://www.gnu.org/copyleft/fdl.html'),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/license', 'http://creativecommons.org/licenses/by-sa/4.0/'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/creator', 'Nicolas Heist'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/creator', 'Heiko Paulheim'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/created', _get_creation_date()),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/publisher', 'Nicolas Heist'),
        serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/publisher', 'Heiko Paulheim'),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#uriSpace', Namespace.CLG_RESOURCE),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#entities', entity_count),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#classes', class_count),
        serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#properties', predicate_count),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/source', 'http://dbpedia.org/resource/DBpedia'),
        serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/source', 'http://dbpedia.org/resource/Wikipedia'),
        serialize_util.as_object_triple(void_resource, 'http://xmlns.com/foaf/0.1/homepage', 'http://caligraph.org'),
        serialize_util.as_object_triple(void_resource, 'http://rdfs.org/ns/void#sparqlEndpoint', 'http://caligraph.org/sparql'),
    ]


def _get_lines_ontology(clgo, clge) -> list:
    """Serialize the ontology."""
    lines_ontology = []
    # metadata
    ontology_resource = 'http://caligraph.org/ontology'
    lines_ontology.extend([
        serialize_util.as_object_triple(ontology_resource, RdfPredicate.TYPE, 'http://www.w3.org/2002/07/owl#Ontology'),
        serialize_util.as_literal_triple(ontology_resource, 'http://purl.org/dc/terms/created', _get_creation_date()),
        serialize_util.as_literal_triple(ontology_resource, RdfPredicate.LABEL, 'CaLiGraph Ontology'),
        serialize_util.as_literal_triple(ontology_resource, 'http://www.w3.org/2002/07/owl#versionInfo', utils.get_config('caligraph.version')),
    ])
    # classes
    for ct in clgo.get_types(include_root=False):
        lines_ontology.append(serialize_util.as_object_triple(ct, RdfPredicate.TYPE, RdfClass.OWL_CLASS))
        lines_ontology.append(serialize_util.as_literal_triple(ct, RdfPredicate.LABEL, ct.get_label()))
        lines_ontology.extend([serialize_util.as_object_triple(ct, RdfPredicate.SUBCLASS_OF, st) for st in clgo.get_supertypes(ct)])
    # predicates
    for pred in clgo.get_predicates():
        pred_type = RdfClass.OWL_OBJECT_PROPERTY if isinstance(pred, ClgObjectPredicate) else RdfClass.OWL_DATATYPE_PROPERTY
        lines_ontology.append(serialize_util.as_object_triple(pred, RdfPredicate.TYPE, pred_type))
    # disjointnesses
    for ct in clgo.get_types(include_root=False):
        for dct in clgo.get_disjoint_types(ct):
            if ct.idx < dct.idx:  # make sure that disjointnesses are only serialized once
                lines_ontology.append(serialize_util.as_object_triple(ct, RdfPredicate.DISJOINT_WITH, dct))
    # restrictions
    defined_restrictions = set()
    for ct in clgo.get_types(include_root=False):
        for pred, val in clge.get_axioms(ct, transitive=False):
            restriction_is_defined = (pred, val) in defined_restrictions
            lines_ontology.extend(_serialize_restriction(ct, pred, val, restriction_is_defined))
            defined_restrictions.add((pred, val))
    return lines_ontology


def _get_creation_date() -> datetime.datetime:
    return datetime.datetime.strptime(utils.get_config('caligraph.creation_date'), '%Y-%m-%d')


def _serialize_restriction(sub: ClgEntity, pred: ClgPredicate, val: Union[ClgEntity, str], restriction_is_defined: bool) -> list:
    """Serialize the restrictions (i.e. relation axioms)."""
    val_label = val.get_label() if isinstance(val, ClgEntity) else val
    restriction_iri = f'{Namespace.CLG_ONTOLOGY.value}RestrictionHasValue_{pred.name}_{val_label}'
    restriction_label = f'Restriction onProperty={pred.name} hasValue={val_label}'

    if restriction_is_defined:
        result = []
    else:
        result = [
            serialize_util.as_object_triple(restriction_iri, RdfPredicate.TYPE, 'http://www.w3.org/2002/07/owl#Restriction'),
            serialize_util.as_literal_triple(restriction_iri, RdfPredicate.LABEL, restriction_label),
            serialize_util.as_object_triple(restriction_iri, 'http://www.w3.org/2002/07/owl#onProperty', pred),
        ]
        if isinstance(val, ClgEntity):
            result.append(serialize_util.as_object_triple(restriction_iri, 'http://www.w3.org/2002/07/owl#hasValue', val))
        else:
            result.append(serialize_util.as_literal_triple(restriction_iri, 'http://www.w3.org/2002/07/owl#hasValue', val))

    result.append(serialize_util.as_object_triple(sub, RdfPredicate.SUBCLASS_OF, restriction_iri))
    return result


def _get_lines_ontology_dbpedia_mapping(clgo) -> list:
    """Serialize the DBpedia mapping for types and predicates."""
    lines_ontology_dbpedia_mapping = []
    for ct in clgo.get_types(include_root=False):
        lines_ontology_dbpedia_mapping.extend([serialize_util.as_object_triple(ct, RdfPredicate.SUBCLASS_OF, dt) for dt in ct.get_direct_dbp_types()])
    for pred in clgo.get_predicates():
        dbp_pred = clgo.get_dbp_predicate(pred)
        lines_ontology_dbpedia_mapping.append(serialize_util.as_object_triple(pred, RdfPredicate.EQUIVALENT_PROPERTY, dbp_pred))
    return lines_ontology_dbpedia_mapping


def _get_lines_ontology_provenance(clgo) -> list:
    """Serialize provenance information of the ontology."""
    lines_ontology_provenance = []
    for ct in clgo.get_types(include_root=False):
        sources = {rdf_util.res2wiki_iri(res) for res in ct.get_associated_dbp_resources()}
        lines_ontology_provenance.extend([serialize_util.as_object_triple(ct, RdfPredicate.WAS_DERIVED_FROM, s) for s in sources])
    return lines_ontology_provenance


def _get_lines_instances_types(clge) -> list:
    """Serialize types of resources."""
    lines_instances_types = []
    for ent in clge.get_entities():
        lines_instances_types.append(serialize_util.as_object_triple(ent, RdfPredicate.TYPE, RdfClass.OWL_NAMED_INDIVIDUAL))
        lines_instances_types.extend([serialize_util.as_object_triple(ent, RdfPredicate.TYPE, t) for t in ent.get_types()])
    return lines_instances_types


def _get_lines_instances_transitive_types(clge) -> list:
    """Serialize transitive types of resources."""
    lines_instances_transitive_types = []
    for ent in clge.get_entities():
        transitive_types = ent.get_transitive_types(include_root=False).difference(ent.get_types())
        lines_instances_transitive_types.extend([serialize_util.as_object_triple(ent, RdfPredicate.TYPE, tt) for tt in transitive_types])
    return lines_instances_transitive_types


def _get_lines_instances_labels(clge) -> list:
    """Serialize resource labels."""
    lines_instances_labels = []
    for ent in clge.get_entities():
        label = ent.get_label()
        lines_instances_labels.append(serialize_util.as_literal_triple(ent, RdfPredicate.LABEL, label))
        lines_instances_labels.append(serialize_util.as_literal_triple(ent, RdfPredicate.PREFLABEL, label))
        altlabels = [serialize_util.as_literal_triple(ent, RdfPredicate.ALTLABEL, sf) for sf in ent.get_surface_forms() if sf != label]
        lines_instances_labels.extend(altlabels)
    return lines_instances_labels


def _get_lines_instances_relations(clge) -> list:
    """Serialize resource facts."""
    lines_instances_relations = []
    for ent in clge.get_entities():
        for pred, vals in ent.get_properties().items():
            if isinstance(pred, ClgObjectPredicate):
                lines_instances_relations.extend([serialize_util.as_object_triple(ent, pred, val) for val in vals])
            else:
                lines_instances_relations.extend([serialize_util.as_literal_triple(ent, pred, val) for val in vals])
    return lines_instances_relations


def _get_lines_instances_restriction_relations(clge) -> list:
    """Serialize resource facts (only from restrictions)."""
    lines_instances_relations = []
    for ent in clge.get_entities():
        for pred, vals in ent.get_axiom_properties().items():
            if isinstance(pred, ClgObjectPredicate):
                lines_instances_relations.extend([serialize_util.as_object_triple(ent, pred, val) for val in vals])
            else:
                lines_instances_relations.extend([serialize_util.as_literal_triple(ent, pred, val) for val in vals])
    return lines_instances_relations


def _get_lines_instances_dbpedia_mapping(clge) -> list:
    """Serialize DBpedia mapping for resources."""
    lines_instances_dbpedia_mapping = []
    for ent in clge.get_entities():
        dbp_ent = ent.get_dbp_entity()
        if dbp_ent is not None:
            lines_instances_dbpedia_mapping.append(serialize_util.as_object_triple(ent, RdfPredicate.SAME_AS, dbp_ent))
    return lines_instances_dbpedia_mapping


def _get_lines_instances_provenance(clge) -> list:
    """Serialize provenance information for resources."""
    lines_instances_provenance = []
    for ent in clge.get_entities():
        provenance_data = {rdf_util.res2wiki_iri(res) for res in ent.get_provenance_resources()}
        lines_instances_provenance.extend([serialize_util.as_object_triple(ent, RdfPredicate.WAS_DERIVED_FROM, p) for p in provenance_data])
    return lines_instances_provenance


def _get_lines_dbpedia_instances(clge) -> list:
    """Serialize new DBpedia resources in DBpedia namespace."""
    lines_dbpedia_instances = []
    new_instances = {dbp_util.res2dbp_iri(ent): ent for ent in clge.get_entities() if ent.get_dbp_entity() is None}
    for dbp_iri, ce in new_instances.items():
        lines_dbpedia_instances.append(serialize_util.as_object_triple(dbp_iri, RdfPredicate.TYPE, RdfClass.OWL_NAMED_INDIVIDUAL))
        lines_dbpedia_instances.append(serialize_util.as_literal_triple(dbp_iri, RdfPredicate.LABEL, ce.get_label()))
    return lines_dbpedia_instances


def _get_lines_dbpedia_instance_types(clge) -> list:
    """Serialize new types for DBpedia resources in DBpedia namespace."""
    lines_dbpedia_types = []
    for ent in clge.get_entities():
        all_dbp_types = ent.get_all_dbp_types(add_transitive_closure=True)
        dbp_ent = ent.get_dbp_entity()
        if dbp_ent is not None:
            new_dbp_types = all_dbp_types.difference(dbp_ent.get_transitive_types(include_root=True))
            lines_dbpedia_types.extend([serialize_util.as_object_triple(dbp_ent, RdfPredicate.TYPE, t) for t in new_dbp_types])
        else:
            dbp_ent_iri = dbp_util.res2dbp_iri(ent)
            lines_dbpedia_types.extend([serialize_util.as_object_triple(dbp_ent_iri, RdfPredicate.TYPE, t) for t in all_dbp_types])
    return lines_dbpedia_types


def _get_lines_dbpedia_instance_caligraph_types(clge) -> list:
    """Serialize CaLiGraph types for DBpedia resources."""
    instance_clg_types = []
    for ent in clge.get_entities():
        dbp_ent = ent.get_dbp_entity()
        if dbp_ent is not None:
            instance_clg_types.extend([serialize_util.as_object_triple(dbp_ent, RdfPredicate.TYPE, t) for t in ent.get_types()])
    return instance_clg_types


def _get_lines_dbpedia_instance_transitive_caligraph_types(clge) -> list:
    """Serialize transitive CaLiGraph types for DBpedia resources."""
    instance_transitive_clg_types = []
    for ent in clge.get_entities():
        dbp_ent = ent.get_dbp_entity()
        if dbp_ent is not None:
            transitive_types = ent.get_transitive_types(include_root=False).difference(ent.get_types())
            instance_transitive_clg_types.extend([serialize_util.as_object_triple(dbp_ent, RdfPredicate.TYPE, tt) for tt in transitive_types])
    return instance_transitive_clg_types


def _get_lines_dbpedia_instance_relations(clge) -> list:
    """Serialize new facts for DBpedia resources in DBpedia namespace."""
    lines_dbpedia_instance_relations = []
    for ent in clge.get_entities():
        dbp_ent = ent.get_dbp_entity()
        dbp_ent_or_iri = dbp_ent or dbp_util.res2dbp_iri(ent)
        for pred, vals in ent.get_properties().items():
            dbp_pred = pred.get_dbp_predicate()
            for val in vals:
                dbp_val = val.get_dbp_entity() if isinstance(val, ClgEntity) else val
                if dbp_ent is not None and dbp_pred in dbp_ent.get_properties() and dbp_val is not None and dbp_val in dbp_ent.get_properties()[dbp_pred]:
                    continue
                if isinstance(val, ClgEntity):
                    val_iri = dbp_util.res2dbp_iri(val)
                    lines_dbpedia_instance_relations.append(serialize_util.as_object_triple(dbp_ent_or_iri, dbp_pred, val_iri))
                else:
                    lines_dbpedia_instance_relations.append(serialize_util.as_literal_triple(dbp_ent_or_iri, dbp_pred, val))
    return lines_dbpedia_instance_relations


def _serialize_type_distribution(clgo, clge, filepath_config: str):
    type_counts = Counter()
    for ent in clge.get_entities():
        random_type = random.choice(list(ent.get_types()))
        for t in clgo.get_transitive_supertypes(random_type, include_self=True):
            type_counts[t] += 1

    type_distribution = _create_type_distribution(type_counts, clgo.get_type_root(), clgo, clge)
    normalized_type_distribution = _normalize_type_distribution(type_distribution, type_distribution['value'])
    with open(utils.get_results_file(filepath_config), mode='wt') as f:
        json.dump(normalized_type_distribution, f)


def _create_type_distribution(type_counts: Dict[ClgType, int], current_node: ClgType, clgo, clge):
    result = {'name': current_node.get_label(), 'value': type_counts[current_node]}
    children = [_create_type_distribution(type_counts, c, clgo, clge) for c in clgo.get_subtypes(current_node)]
    if children:
        result['children'] = children
    return result


def _normalize_type_distribution(type_distribution, node_weight, level=0):
    remaining_nodes_name = '...' if level == 0 else '-other-'
    threshold = .15 if level > 3 else (.1 if level > 0 else .005)

    if 'children' not in type_distribution:
        return {'name': type_distribution['name'], 'value': round(node_weight)}
    node_value = max(type_distribution['value'], sum(c['value'] for c in type_distribution['children']))
    valid_children = [c for c in type_distribution['children'] if c['value'] > 0 and c['value'] / node_value > threshold]
    remaining_value = node_value - sum(c['value'] for c in valid_children)
    if remaining_value > 0:
        valid_children.append({'name': remaining_nodes_name, 'value': remaining_value})
    normalized_children = [_normalize_type_distribution(c, node_weight * c['value'] / node_value, level+1) for c in valid_children]
    return {'name': type_distribution['name'], 'children': normalized_children}


def _print_statistics(clgo, clge) -> str:
    """Return statistics of CaLiGraph in a printable format."""
    types_connected_to_dbpedia_count = len({t for t in clgo.get_types() if t.get_all_dbp_types()})
    axiom_predicate_count = len({pred for ent in clge.get_entities() for pred in clge.get_axiom_properties(ent)})
    cat_parts_count = len({cat for t in clgo.get_types() for cat in t.get_associated_dbp_resources() if isinstance(cat, DbpCategory) and not isinstance(cat, DbpListCategory)})
    list_parts_count = len({lp for t in clgo.get_types() for lp in t.get_associated_dbp_resources() if isinstance(lp, DbpListpage)})
    listcat_parts_count = len({lc for t in clgo.get_types() for lc in t.get_associated_dbp_resources() if isinstance(lc, DbpListCategory)})
    leaf_types = {t for t in clgo.get_types() if not clgo.get_subtypes(t)}
    classtree_depth_avg = np.mean([t.get_depth() for t in leaf_types])
    branching_factor_avg = np.mean([d for _, d in clgo.graph.out_degree if d > 0])
    axiom_count = sum([len(node_axioms) for node_axioms in clge.get_all_axioms().values()])
    resource_axiom_count = len([ax for axioms in clge.get_all_axioms().values() for ax in axioms if isinstance(ax[1], ClgEntity)])
    literal_axiom_count = axiom_count - resource_axiom_count
    direct_type_axiom_count = len({t for t in clgo.get_types() if clge.get_axioms(t, transitive=False)})
    type_axiom_count = len({t for t in clgo.get_types() if clge.get_axioms(t, transitive=True)})

    entities = clge.get_entities()
    types_per_entity = np.mean([len(clge.get_transitive_types(e, include_root=True)) for e in entities])
    relations = {(e, p, v) for e, props in clge.get_entity_properties() for p, v in props.items()}
    resource_relation_count = len({r for r in relations if isinstance(r[2], ClgEntity)})
    literal_relation_count = len(relations) - resource_relation_count
    in_degree = resource_relation_count / len(entities)
    out_degree = len(relations) / len(entities)

    return '\n'.join([
        '{:^40}'.format('STATISTICS'),
        '=' * 40,
        '{:<30} | {:>7}'.format('types', len(clgo.get_types())),
        '{:<30} | {:>7}'.format('types below root', len(clgo.get_type_root().get_subtypes())),
        '{:<30} | {:>7}'.format('types connected to DBpedia', types_connected_to_dbpedia_count),
        '{:<30} | {:>7}'.format('subtype relations', len(clgo.graph.edges)),
        '{:<30} | {:>7}'.format('predicates', len(clgo.get_predicates())),
        '{:<30} | {:>7}'.format('axiom predicates', axiom_predicate_count),
        '{:<30} | {:>7}'.format('category parts', cat_parts_count),
        '{:<30} | {:>7}'.format('list parts', list_parts_count),
        '{:<30} | {:>7}'.format('listcat parts', listcat_parts_count),
        '{:<30} | {:>7.2f}'.format('classtree depth', classtree_depth_avg),
        '{:<30} | {:>7.2f}'.format('branching factor', branching_factor_avg),
        '{:<30} | {:>7}'.format('axioms', axiom_count),
        '{:<30} | {:>7}'.format('resource axioms', resource_axiom_count),
        '{:<30} | {:>7}'.format('literal axioms', literal_axiom_count),
        '{:<30} | {:>7}'.format('types with direct axiom', direct_type_axiom_count),
        '{:<30} | {:>7}'.format('types with axiom', type_axiom_count),
        '-' * 40,
        '{:<30} | {:>7}'.format('entities', len(entities)),
        '{:<30} | {:>7}'.format('types per entities', types_per_entity),
        '{:<30} | {:>7}'.format('relations', len(relations)),
        '{:<30} | {:>7}'.format('resource relations', resource_relation_count),
        '{:<30} | {:>7}'.format('literal relations', literal_relation_count),
        '{:<30} | {:>7}'.format('entities in-degree', in_degree),
        '{:<30} | {:>7}'.format('entities out-degree', out_degree),
        ])
