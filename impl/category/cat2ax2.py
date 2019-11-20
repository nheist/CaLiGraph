import util
from collections import defaultdict
import operator
import numpy as np
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.category.category_set as cat_set
import impl.category.nlp as cat_nlp
import impl.category.store as cat_store
import impl.util.nlp as nlp_util
import impl.util.rdf as rdf_util


PATTERN_CONF = util.get_config('cat2ax.pattern_confidence')


class Axiom:
    def __init__(self, predicate: str, value: str, confidence: float):
        self.predicate = predicate
        self.value = value
        self.confidence = confidence

    def implies(self, other):
        return self.predicate == other.predicate and self.value == other.value

    def contradicts(self, other):
        raise NotImplementedError("Please use the subclasses.")

    def accepts_resource(self, dbp_resource: str) -> bool:
        raise NotImplementedError("Please use the subclasses.")

    def rejects_resource(self, dbp_resource: str) -> bool:
        raise NotImplementedError("Please use the subclasses.")


class TypeAxiom(Axiom):
    def __init__(self, value: str, confidence: float):
        super().__init__(rdf_util.PREDICATE_TYPE, value, confidence)

    def implies(self, other):
        return super().implies(other) or other.value in dbp_store.get_transitive_supertype_closure(self.value)

    def contradicts(self, other):
        return other.value in dbp_heur.get_disjoint_types(self.value)

    def accepts_resource(self, dbp_resource: str) -> bool:
        return self.value in dbp_store.get_transitive_types(dbp_resource)

    def rejects_resource(self, dbp_resource: str) -> bool:
        return self.value in {dt for t in dbp_store.get_types(dbp_resource) for dt in dbp_heur.get_disjoint_types(t)}


class RelationAxiom(Axiom):
    def contradicts(self, other):
        if self.predicate != other.predicate or not dbp_store.is_functional(self.predicate):
            return False
        return dbp_store.resolve_redirect(self.value) != dbp_store.resolve_redirect(other.value)

    def accepts_resource(self, dbp_resource: str) -> bool:
        props = dbp_store.get_properties(dbp_resource)
        return self.predicate in props and (self.value in props[self.predicate] or dbp_store.resolve_redirect(self.value) in props[self.predicate])

    def rejects_resource(self, dbp_resource: str) -> bool:
        if not dbp_store.is_functional(self.predicate):
            return False
        props = dbp_store.get_properties(dbp_resource)
        return self.predicate in props and self.value not in props[self.predicate] and dbp_store.resolve_redirect(self.value) not in props[self.predicate]


def get_axioms(category: str) -> set:
    global __CATEGORY_AXIOMS_2__
    if '__CATEGORY_AXIOMS_2__' not in globals():
        __CATEGORY_AXIOMS_2__ = util.load_cache('cat2ax2_axioms')
        if not __CATEGORY_AXIOMS_2__:
            raise ValueError('cat2ax_axioms not initialised. Run axiom extraction once to create the necessary cache!')

    return __CATEGORY_AXIOMS_2__[category]


def extract_category_axioms(category_graph, pattern_confidence):
    candidate_sets = cat_set.get_category_sets()
    patterns = _extract_patterns(category_graph, candidate_sets)
    return _extract_axioms(category_graph, candidate_sets, patterns, pattern_confidence)


# --- PATTERN EXTRACTION ---

def _extract_patterns(category_graph, candidate_sets):
    util.get_logger().debug('Cat2Ax2: Extracting patterns..')
    patterns = defaultdict(lambda: {'preds': defaultdict(list), 'types': defaultdict(list)})

    for parent, children, (first_words, last_words) in candidate_sets:
        predicate_frequencies = defaultdict(list)
        type_frequencies = defaultdict(list)
        type_surface_scores = _get_type_surface_scores(first_words + last_words)

        categories_with_matches = {cat: _get_match_for_category(cat, first_words, last_words) for cat in children}
        categories_with_matches = {cat: match for cat, match in categories_with_matches.items() if category_graph.has_node(cat) and match}
        for cat, match in categories_with_matches.items():
            # compute predicate frequencies
            statistics = cat_store.get_statistics(cat)
            possible_vals = _get_resource_surface_scores(match)
            for (pred, val), freq in statistics['property_frequencies'].items():
                if val in possible_vals:
                    predicate_frequencies[pred].append(freq * possible_vals[val])
            for t, tf in statistics['type_frequencies'].items():
                type_frequencies[t].append(tf * type_surface_scores[t])
        if predicate_frequencies:
            # pad frequencies to get the correct median
            predicate_frequencies = {pred: freqs + ([0]*(len(categories_with_matches)-len(freqs))) for pred, freqs in predicate_frequencies.items()}
            pred, freqs = max(predicate_frequencies.items(), key=lambda x: np.median(x[1]))
            med = np.median(freqs)
            if dbp_util.is_dbp_type(pred) and med > 0:
                for _ in categories_with_matches:
                    patterns[(tuple(first_words), tuple(last_words))]['preds'][pred].append(med)
        if type_frequencies:
            # pad frequencies to get the correct median
            type_frequencies = {t: freqs + ([0]*(len(categories_with_matches)-len(freqs))) for t, freqs in type_frequencies.items()}
            max_median = max(np.median(freqs) for freqs in type_frequencies.values())
            types = {t for t, freqs in type_frequencies.items() if np.median(freqs) >= max_median}
            if max_median > 0:
                for _ in categories_with_matches:
                    for t in types:
                        patterns[(tuple(first_words), tuple(last_words))]['types'][t].append(max_median)

    util.get_logger().debug(f'Cat2Ax2: Extracted {len(patterns)} patterns.')
    return patterns


def _get_match_for_category(category: str, first_words: tuple, last_words: tuple) -> str:
    doc = nlp_util.remove_by_phrase(cat_nlp.parse_category(category))
    return doc[len(first_words):len(doc)-len(last_words)].text


def _get_resource_surface_scores(text):
    resource_surface_scores = {}
    if not text:
        return resource_surface_scores
    resource_surface_scores[text] = 1
    direct_match = dbp_store.resolve_redirect(dbp_util.name2resource(text))
    if direct_match in dbp_store.get_resources():
        resource_surface_scores[direct_match] = 1
    for surface_match, frequency in sorted(dbp_store.get_inverse_lexicalisations(text.lower()).items(), key=operator.itemgetter(1)):
        resource_surface_scores[surface_match] = frequency
    return resource_surface_scores


def _get_type_surface_scores(words):
    lexicalisation_scores = defaultdict(lambda: 0)
    word_lemmas = [nlp_util.parse(w)[0].lemma_ for w in words]
    for lemma in word_lemmas:
        for t, score in dbp_store.get_type_lexicalisations(lemma).items():
            lexicalisation_scores[t] += score
    total_scores = sum(lexicalisation_scores.values())
    type_surface_scores = defaultdict(lambda: 0.0, {t: score / total_scores for t, score in lexicalisation_scores.items()})

    # make sure that exact matches get at least appropriate probability
    for lemma in word_lemmas:
        word_types = dbp_store.get_types_by_name(lemma)
        for word_type in word_types:
            min_word_type_score = 1 / len(words)
            type_surface_scores[word_type] = max(type_surface_scores[word_type], min_word_type_score)

    return type_surface_scores


# --- PATTERN APPLICATION ---


def _extract_axioms(category_graph, candidate_sets, patterns, pattern_confidence):
    util.get_logger().debug('Cat2Ax2: Extracting axioms..')
    category_axioms = defaultdict(set)

    front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, False).items():
        _fill_dict(front_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, False, True).items():
        _fill_dict(back_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, True).items():
        _fill_dict(enclosing_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    # add categories that are not in existing category sets as new sets with a single category
    category_sets = [cs.children for cs in candidate_sets]
    cats_in_category_sets = {c for cset in category_sets for c in cset}
    category_sets.extend([{c} for c in category_graph.nodes.difference(cats_in_category_sets)])

    for category_set in category_sets:

        # gather scores for possible axioms of category set
        category_set_property_axioms = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        category_set_type_axioms = defaultdict(lambda: defaultdict(float))
        for cat in category_set:
            cat_doc = nlp_util.remove_by_phrase(cat_nlp.parse_category(cat))
            cat_prop_axioms = []
            cat_type_axioms = []

            front_prop_axiom, front_type_axiom = _find_axioms(pattern_confidence, front_pattern_dict, cat, cat_doc)
            if front_prop_axiom:
                cat_prop_axioms.append(front_prop_axiom)
            if front_type_axiom:
                cat_type_axioms.append(front_type_axiom)

            back_prop_axiom, back_type_axiom = _find_axioms(pattern_confidence, back_pattern_dict, cat, cat_doc)
            if back_prop_axiom:
                cat_prop_axioms.append(back_prop_axiom)
            if back_type_axiom:
                cat_type_axioms.append(back_type_axiom)

            enclosing_prop_axiom, enclosing_type_axiom = _find_axioms(pattern_confidence, enclosing_pattern_dict, cat, cat_doc)
            if enclosing_prop_axiom:
                cat_prop_axioms.append(enclosing_prop_axiom)
            if enclosing_type_axiom:
                cat_type_axioms.append(enclosing_type_axiom)

            for pa in cat_prop_axioms:
                pred, val, score = pa[1], pa[2], pa[3]
                category_set_property_axioms[pred][cat][val] = score
            for ta in cat_type_axioms:
                t, score = ta[2], ta[3]
                category_set_type_axioms[t][cat] = score

        # identify valid axioms
        for prop in category_set_property_axioms:
            scores = [max(category_set_property_axioms[prop][cat].values(), default=0) for cat in category_set_property_axioms[prop]]
            scores += [0] * (len(category_set) - len(scores))  # add 0 for categories that did not find the property axiom at all
            if np.median(scores) > 0:
                for cat in category_set:
                    val, score = max(category_set_property_axioms[prop][cat].items(), key=operator.itemgetter(1), default=(None, 0))
                    if val is not None:
                        category_axioms[cat].add(RelationAxiom(prop, val, score))

        type_scores = {}
        for t, category_scores in category_set_type_axioms.items():
            scores = list(category_scores.values())
            scores += [0] * (len(category_set) - len(scores))  # add 0 for categories that did not find the type axiom at all
            type_scores[t] = np.median(scores)
        best_type = None
        best_score = 0
        for t, score in sorted(type_scores.items(), key=operator.itemgetter(1), reverse=True):
            if score == 0:
                break
            if not best_type or t in dbp_store.get_transitive_subtypes(best_type):
                best_type = t
                best_score = score
        if best_type:
            for cat in category_set:
                category_axioms[cat].add(TypeAxiom(best_type, best_score))

    util.get_logger().debug(f'Cat2Ax2: Extracted {sum(len(axioms) for axioms in category_axioms.values())} axioms for {len(category_axioms)} categories.')
    return category_axioms


def _get_confidence_pattern_set(pattern_set, has_front, has_back):
    result = {}
    for pattern, axiom_patterns in pattern_set.items():
        if bool(pattern[0]) == has_front and bool(pattern[1]) == has_back:
            preds, types = axiom_patterns['preds'], axiom_patterns['types']
            preds_sum = sum(len(freqs) for freqs in preds.values())
            types_sum = sum(len(freqs) for freqs in types.values())
            result[pattern] = {
                'preds': defaultdict(lambda: 0, {p: len(freqs) / preds_sum for p, freqs in preds.items()}),
                'types': defaultdict(lambda: 0, {t: len(freqs) / types_sum for t, freqs in types.items()})
            }
    return result


MARKER_HIT = '_marker_hit_'
MARKER_REVERSE = '_marker_reverse_'
def _fill_dict(dictionary, elements, leaf):
    if not elements:
        if callable(leaf):
            if MARKER_REVERSE not in dictionary:
                dictionary[MARKER_REVERSE] = {}
            leaf(dictionary[MARKER_REVERSE])
        else:
            dictionary[MARKER_HIT] = leaf
    else:
        if elements[0] not in dictionary:
            dictionary[elements[0]] = {}
        _fill_dict(dictionary[elements[0]], elements[1:], leaf)


def _detect_pattern(pattern_dict, words):
    pattern_length = 0
    ctx = pattern_dict
    for word in words:
        if word in ctx:
            ctx = ctx[word]
            pattern_length += 1
            continue
        if MARKER_HIT in ctx:
            return ctx[MARKER_HIT], pattern_length
        if MARKER_REVERSE in ctx:
            preds, back_pattern_length = _detect_pattern(ctx[MARKER_REVERSE], list(reversed(words)))
            return preds, (pattern_length, back_pattern_length)
        return None, None
    return None, None


def _get_axioms_for_cat(pattern_confidence, axiom_patterns, cat, text_diff, words_same):
    prop_axiom = None
    type_axiom = None

    statistics = cat_store.get_statistics(cat)
    pred_patterns = axiom_patterns['preds']
    possible_values = _get_resource_surface_scores(text_diff)
    props_scores = {(p, v): freq * pred_patterns[p] * possible_values[v] for (p, v), freq in statistics['property_frequencies'].items() if p in pred_patterns and v in possible_values}
    prop, max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    if max_prop_score >= pattern_confidence:
        pred, val = prop
        prop_axiom = (cat, pred, val, max_prop_score)

    type_patterns = axiom_patterns['types']
    type_surface_scores = _get_type_surface_scores(words_same)
    types_scores = {t: freq * type_patterns[t] * type_surface_scores[t] for t, freq in statistics['type_frequencies'].items() if t in type_patterns}
    t, max_type_score = max(types_scores.items(), key=operator.itemgetter(1), default=(None, 0))
    if max_type_score >= pattern_confidence:
        type_axiom = (cat, rdf_util.PREDICATE_TYPE, t, max_type_score)

    return prop_axiom, type_axiom


def _find_axioms(pattern_confidence, pattern_dict, cat, cat_doc):
    cat_words = [w.text for w in cat_doc]
    axiom_patterns, pattern_lengths = _detect_pattern(pattern_dict, cat_words)
    if axiom_patterns:
        front_pattern_idx = pattern_lengths[0] or None
        back_pattern_idx = -1 * pattern_lengths[1] or None
        text_diff = cat_doc[front_pattern_idx:back_pattern_idx].text
        words_same = []
        if front_pattern_idx:
            words_same += cat_words[:front_pattern_idx]
        if back_pattern_idx:
            words_same += cat_words[back_pattern_idx:]
        return _get_axioms_for_cat(pattern_confidence, axiom_patterns, cat, text_diff, words_same)
    return None, None


# --- AXIOM APPLICATION & POST-FILTERING ---


def _extract_assertions(relation_axioms, type_axioms):
    util.get_logger().debug('Cat2Ax: Applying axioms..')

    relation_assertions = {(res, pred, val) for cat, pred, val, _ in relation_axioms for res in cat_store.get_resources(cat)}
    new_relation_assertions = {(res, pred, val) for res, pred, val in relation_assertions if pred not in dbp_store.get_properties(res) or val not in dbp_store.get_properties(res)[pred]}

    type_assertions = {(res, rdf_util.PREDICATE_TYPE, t) for cat, pred, t, _ in type_axioms for res in cat_store.get_resources(cat)}
    new_type_assertions = {(res, pred, t) for res, pred, t in type_assertions if t not in {tt for t in dbp_store.get_types(res) for tt in dbp_store.get_transitive_supertype_closure(t)} and t != rdf_util.CLASS_OWL_THING}
    new_type_assertions_transitive = {(res, pred, tt) for res, pred, t in new_type_assertions for tt in dbp_store.get_transitive_supertype_closure(t) if tt not in {ott for t in dbp_store.get_types(res) for ott in dbp_store.get_transitive_supertype_closure(t)} and tt != rdf_util.CLASS_OWL_THING}

    # post-filtering
    filtered_new_relation_assertions = {(res, pred, val) for res, pred, val in new_relation_assertions if pred not in dbp_store.get_properties(res) or not dbp_store.is_functional(pred)}
    filtered_new_type_assertions = {(res, pred, t) for res, pred, t in new_type_assertions_transitive if not dbp_heur.get_disjoint_types(t).intersection(dbp_store.get_transitive_types(res))}

    util.get_logger().debug(f'Cat2Ax: Generated {len(filtered_new_relation_assertions)} new relation assertions and {len(filtered_new_type_assertions)} new type assertions.')
    return filtered_new_relation_assertions, filtered_new_type_assertions
