from typing import Set, Union, Tuple, Optional
from collections import defaultdict
import utils
from impl.util.rdf import Namespace, RdfResource, name2label
from impl.util.singleton import Singleton
import impl.dbpedia.heuristics as dbp_heur
from impl.dbpedia.ontology import DbpType, DbpPredicate, DbpObjectPredicate, DbpOntologyStore
from impl.dbpedia.resource import DbpListpage
from impl.dbpedia.category import DbpCategory
from .graph import CaLiGraph


class ClgClass(RdfResource):
    @classmethod
    def get_namespace(cls) -> str:
        return Namespace.CLG_ONTOLOGY.value

    @classmethod
    def _get_store(cls):
        return ClgOntologyStore.instance()


class ClgType(ClgClass):
    def get_depth(self) -> int:
        return self._get_store().get_depth(self)

    def get_direct_dbp_types(self) -> Set[DbpType]:
        return self._get_store().get_direct_dbp_types(self)

    def get_all_dbp_types(self, add_transitive_closure=False) -> Set[DbpType]:
        return self._get_store().get_all_dbp_types(self, add_transitive_closure=add_transitive_closure)

    def get_associated_dbp_resources(self) -> Set[Union[DbpListpage, DbpCategory]]:
        return self._get_store().get_associated_dbp_resources(self)


class ClgPredicate(ClgClass):
    def get_dbp_predicate(self) -> DbpPredicate:
        return self._get_store().get_dbp_predicate(self)

    def get_domain(self) -> ClgType:
        return self._get_store().get_domain(self)

    def get_range(self):
        return self._get_store().get_range(self)


class ClgObjectPredicate(ClgPredicate):
    pass


class ClgDatatypePredicate(ClgPredicate):
    pass


class ClgClassNotExistingException(KeyError):
    pass


@Singleton
class ClgOntologyStore:
    def __init__(self):
        self.dbo = DbpOntologyStore.instance()

        all_classes, graph = utils.load_or_create_cache('caligraph_ontology', self._init_class_cache)
        self.classes_by_idx = {c.idx: c for c in all_classes}
        self.classes_by_name = {c.name: c for c in all_classes}
        self.graph = graph
        self.type_depths = None
        self.disjoint_types = None
        self.transitive_disjoint_Types = None

    def _init_class_cache(self) -> Tuple[Set[ClgClass], CaLiGraph]:
        graph = CaLiGraph.build_graph().merge_ontology().remove_transitive_edges()

        all_classes = {ClgType(0, 'Thing', False)}  # initialize root type with 0
        # initialize all predicates exactly as in DBpedia
        for pred in self.dbo.get_predicates():
            if isinstance(pred, DbpObjectPredicate):
                all_classes.add(ClgObjectPredicate(pred.idx, pred.name, False))
            else:
                all_classes.add(ClgDatatypePredicate(pred.idx, pred.name, False))
        # define all types from graph
        for idx, node in enumerate(graph.traverse_nodes_topdown(), start=len(all_classes)):
            if node == graph.root_node:
                continue
            all_classes.add(ClgType(idx, node, False))

        return all_classes, graph

    def get_class_by_idx(self, idx: int) -> ClgClass:
        if idx not in self.classes_by_idx:
            raise ClgClassNotExistingException(f'Could not find class for index: {idx}')
        return self.classes_by_idx[idx]

    def get_class_by_name(self, name: str) -> ClgClass:
        if name not in self.classes_by_name:
            raise ClgClassNotExistingException(f'Could not find class for name: {name}')
        return self.classes_by_name[name]

    def get_type_root(self):
        return self.get_class_by_idx(0)  # root is initialized as 0

    def get_label(self, clg_type: ClgType) -> str:
        return name2label(clg_type.name)

    def get_types(self, include_root=True) -> Set[ClgType]:
        types = {c for c in self.classes_by_idx.values() if isinstance(c, ClgType)}
        types = types if include_root else types.difference({self.get_type_root()})
        return types

    def get_supertypes(self, clg_type: ClgType, include_root=True) -> Set[ClgType]:
        supertypes = {self.get_class_by_name(p) for p in self.graph.parents(clg_type.name)}
        supertypes = supertypes if include_root else supertypes.difference({self.get_type_root()})
        return supertypes

    def get_transitive_supertypes(self, clg_type: ClgType, include_root=True, include_self=False) -> Set[ClgType]:
        tt = {self.get_class_by_name(p) for p in self.graph.ancestors(clg_type.name)}
        tt = tt | {clg_type} if include_self else tt
        tt = tt if include_root else tt.difference({self.get_type_root()})
        return tt

    def get_subtypes(self, clg_type: ClgType) -> Set[ClgType]:
        return {self.get_class_by_name(c) for c in self.graph.children(clg_type.name)}

    def get_independent_types(self, types: Set[ClgType]) -> Set[ClgType]:
        transitive_types = {tt for t in types for tt in self.get_transitive_supertypes(t)}
        return types.difference(transitive_types)

    def get_depth(self, clg_type: ClgType) -> int:
        if self.type_depths is None:
            node_depths = self.graph.depths()
            self.type_depths = {self.get_class_by_name(node): depth for node, depth in node_depths.items()}
        return self.type_depths[clg_type]

    def get_direct_dbp_types(self, clg_type: ClgType) -> Set[DbpType]:
        return self.graph.get_type_parts(clg_type.name)

    def get_all_dbp_types(self, clg_type: ClgType, add_transitive_closure=False) -> Set[DbpType]:
        nodes = {clg_type.name} | self.graph.ancestors(clg_type.name)
        dbp_types = {tp for n in nodes for tp in self.graph.get_type_parts(n)}
        if add_transitive_closure:
            dbp_types = {tt for t in dbp_types for tt in self.dbo.get_transitive_supertypes(t, include_self=True)}
        return dbp_types

    def get_disjoint_types(self, clg_type: ClgType) -> Set[ClgType]:
        if self.disjoint_types is None:
            self.disjoint_types = defaultdict(set)
            for t in self.get_types(include_root=True):
                disjoint_dbp_types = {dt for dbp_type in self.get_direct_dbp_types(t) for dt in dbp_heur.get_direct_disjoint_types(dbp_type)}
                self.disjoint_types[t] = {ct for dt in disjoint_dbp_types for ct in self.get_types_for_associated_dbp_type(dt)}
            # get rid of transitive disjointnesses by first applying full transitivity and then reducing from bottom up
            for node in self.graph.traverse_nodes_topdown():
                node_type = self.get_class_by_name(node)
                for subtype in self.get_subtypes(node_type):
                    self.disjoint_types[subtype].update(self.disjoint_types[node_type])
            for node in self.graph.traverse_nodes_bottomup():
                node_type = self.get_class_by_name(node)
                supertype_disjoint_types = {dt for st in self.get_supertypes(node_type) for dt in self.disjoint_types[st]}
                self.disjoint_types[node_type].difference_update(supertype_disjoint_types)
        return self.disjoint_types[clg_type]

    def get_types_for_associated_dbp_type(self, dbp_type: DbpType) -> Set[ClgType]:
        return {self.get_class_by_name(node) for node in self.graph.get_nodes_for_part(dbp_type)}

    def get_associated_dbp_resources(self, clg_type: ClgType) -> Set[Union[DbpListpage, DbpCategory]]:
        return self.graph.get_list_parts(clg_type.name) | self.graph.get_category_parts(clg_type.name)

    def get_types_for_associated_dbp_resource(self, dbp_res: Union[DbpListpage, DbpCategory]) -> Set[ClgType]:
        return {self.get_class_by_name(node) for node in self.graph.get_nodes_for_part(dbp_res)}

    def get_predicates(self) -> Set[ClgPredicate]:
        return {c for c in self.classes_by_idx.values() if isinstance(c, ClgPredicate)}

    def get_dbp_predicate(self, pred: ClgPredicate) -> DbpPredicate:
        return self.dbo.get_class_by_idx(pred.idx)

    def get_predicate_for_dbp_predicate(self, dbp_pred: DbpPredicate) -> ClgPredicate:
        return self.get_class_by_idx(dbp_pred.idx)

    def get_domain(self, pred: ClgPredicate) -> ClgType:
        dbp_domain = dbp_heur.get_domain(pred.get_dbp_predicate())
        clg_types = self.get_types_for_associated_dbp_type(dbp_domain)
        return sorted(clg_types)[0] if clg_types else self.get_type_root()

    def get_range(self, pred: ClgPredicate):
        dbp_range = dbp_heur.get_range(pred.get_dbp_predicate())
        if not isinstance(dbp_range, DbpType):
            return self.get_type_root() if isinstance(pred, ClgObjectPredicate) else dbp_range
        clg_types = self.get_types_for_associated_dbp_type(dbp_range)
        return sorted(clg_types)[0] if clg_types else self.get_type_root()
