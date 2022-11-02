from typing import Set, Tuple, Dict, List
from collections import namedtuple, defaultdict
import json
import itertools
import utils
from impl.util.transformer import SpecialToken, EntityIndex
from impl.dbpedia.resource import DbpResourceStore, DbpEntity
from impl.dbpedia.category import DbpCategoryStore
from impl.caligraph.entity import ClgEntity, ClgEntityStore
from impl.wikipedia import MentionId
from .util import DataCorpus, Pair, CXS, CXE, TXS


NEW_ENT = EntityIndex.NEW_ENTITY.value


NilkExample = namedtuple('NilkExample', ['mention_id', 'label', 'left_text', 'right_text', 'page_res_id', 'wikidata_id', 'ent_id', 'is_nil_entity'])


class NilkDataCorpus(DataCorpus):
    def __init__(self, examples: List[NilkExample], entity_indices: Set[int]):
        self.mention_labels = {ex.mention_id: ex.label for ex in examples}
        self.mention_left_text = {ex.mention_id: ex.left_text for ex in examples}
        self.mention_right_text = {ex.mention_id: ex.right_text for ex in examples}
        self.mention_pages = {ex.mention_id: ex.page_res_id for ex in examples}
        self.entity_indices = entity_indices
        self.mm_alignment, self.me_alignment = self._init_alignments(examples)

    @classmethod
    def _init_alignments(cls, examples: List[NilkExample]) -> Tuple[Set[Pair], Set[Pair]]:
        mm_alignment = set()
        example_groups = defaultdict(set)
        for ex in examples:
            example_groups[ex.wikidata_id].add(ex)
        for ex_grp in example_groups.values():
            mm_alignment.update({Pair(*sorted([ex_a.mention_id, ex_b.mention_id]), 1) for ex_a, ex_b in itertools.combinations(ex_grp, 2)})
        me_alignment = {Pair(ex.mention_id, ex.ent_id, 1) for ex in examples if not ex.is_nil_entity}
        return mm_alignment, me_alignment

    def get_mention_labels(self) -> Dict[MentionId, str]:
        return self.mention_labels

    def get_mention_input(self, add_page_context: bool, add_text_context: int) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
        utils.get_logger().debug('Preparing listing items..')
        result = {}
        result_ent_known = {}
        for m_id, label in self.mention_labels.items():
            result[m_id] = label
            if add_page_context:
                result[m_id] += self._prepare_page_context(self.mention_pages[m_id])
            if add_text_context:
                result[m_id] += f' {CXE} {self.mention_left_text[m_id]} {TXS} {self.mention_right_text[m_id]}'
            result_ent_known[m_id] = True
        return result, result_ent_known

    @classmethod
    def _prepare_page_context(cls, page_idx: int) -> str:
        res = DbpResourceStore.instance().get_resource_by_idx(page_idx)
        res_type = SpecialToken.get_type_token(res.get_type_label())
        cats = list(DbpCategoryStore.instance().get_categories_for_resource(res.idx))[:3]
        cats_text = ' | '.join([cat.get_label() for cat in cats])
        return f' {CXS} {res.get_label()} {res_type} {CXS} {cats_text}'

    def get_entities(self) -> Set[ClgEntity]:
        clge = ClgEntityStore.instance()
        return {clge.get_entity_by_idx(idx) for idx in self.entity_indices}


def _init_nilk_data_corpora() -> Tuple[NilkDataCorpus, NilkDataCorpus, NilkDataCorpus]:
    clge = ClgEntityStore.instance()
    train_examples = _load_valid_examples_from_jsonl(utils.get_data_file('files.nilk.train'))
    eval_examples = _load_valid_examples_from_jsonl(utils.get_data_file('files.nilk.eval'))
    test_examples = _load_valid_examples_from_jsonl(utils.get_data_file('files.nilk.test'))
    # collect valid entities (i.e., all entities that are not flagged as NIL)
    invalid_entity_ids = {ex.ent_id for ex in train_examples + eval_examples + test_examples if ex.is_nil_entity}
    entity_ids = {e.idx for e in clge.get_entities() if e.idx not in invalid_entity_ids}
    return NilkDataCorpus(train_examples, entity_ids), NilkDataCorpus(eval_examples, entity_ids), NilkDataCorpus(test_examples, entity_ids)


def _load_valid_examples_from_jsonl(filepath: str) -> List[NilkExample]:
    result = []
    dbr = DbpResourceStore.instance()
    with open(filepath, mode='r') as f:
        for line in f:
            example = json.loads(line)
            ex_id = example['id']
            label = example['mention']
            context = example['context']
            label_start_idx = example['offset']
            label_end_idx = label_start_idx + example['length']
            page_id = int(example['wikipedia_page_id'])
            wikidata_id = example['wikidata_id']
            is_nil = example['nil']
            # filter out invalid examples
            page_res = dbr.get_resource_by_page_id(page_id)
            if page_res is None:
                continue  # discard examples where the Wiki page of their occurrence is not existing or not an entity
            wikidata_res = dbr.get_resource_by_wikidata_id(wikidata_id)
            if not is_nil and not isinstance(wikidata_res, DbpEntity):
                continue  # discard examples for which the mapped resource should exist (non-NIL) but does not
            ent_id = wikidata_res.idx if wikidata_res else NEW_ENT
            # create mentionId with Pos0 = example id, Pos1 = indicator for NILK dataset, Pos2 = -1 if new entity else 0
            mention_id = MentionId(ex_id, NEW_ENT, NEW_ENT if is_nil else 0)
            # get text context
            left_text = context[:label_start_idx].strip()
            right_text = context[label_end_idx:].strip()
            result.append(NilkExample(mention_id, label, left_text, right_text, page_res.idx, wikidata_id, ent_id, is_nil))
    return result
