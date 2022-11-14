from typing import Set, Tuple, Dict, List
from collections import namedtuple, defaultdict
import json
import random
from tqdm import tqdm
import utils
from impl.util.transformer import SpecialToken, EntityIndex
from impl.dbpedia.resource import DbpResourceStore, DbpEntity
from impl.dbpedia.category import DbpCategoryStore
from impl.caligraph.entity import ClgEntity, ClgEntityStore
from impl.wikipedia import MentionId
from .util import DataCorpus, Alignment, CXS, CXE, TXS


NilkExample = namedtuple('NilkExample', ['mention_id', 'label', 'left_text', 'right_text', 'page_res_id', 'ent_id', 'is_nil_entity'])


class NilkDataCorpus(DataCorpus):
    def __init__(self, examples: List[NilkExample], known_entities: Set[int]):
        self.mention_labels = {ex.mention_id: ex.label for ex in examples}
        self.mention_left_text = {ex.mention_id: ex.left_text for ex in examples}
        self.mention_right_text = {ex.mention_id: ex.right_text for ex in examples}
        self.mention_pages = {ex.mention_id: ex.page_res_id for ex in examples}
        self.known_entities = known_entities
        self.alignment = self._init_alignment(examples, known_entities)

    @classmethod
    def _init_alignment(cls, examples: List[NilkExample], known_entities: Set[int]) -> Alignment:
        entity_to_mention_mapping = defaultdict(set)
        for ex in examples:
            entity_to_mention_mapping[ex.ent_id].add(ex.mention_id)
        return Alignment(entity_to_mention_mapping, known_entities)

    def get_mention_labels(self, discard_unknown: bool = False) -> Dict[MentionId, str]:
        return self.mention_labels

    def get_mention_input(self, add_page_context: bool, add_text_context: int) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
        result = {}
        result_ent_known = {}
        for m_id, label in tqdm(self.mention_labels.items(), desc='Preparing mention input'):
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
        return {clge.get_entity_by_idx(idx) for idx in self.known_entities}


def _init_nilk_data_corpora(sample_size: int) -> Tuple[NilkDataCorpus, NilkDataCorpus, NilkDataCorpus]:
    clge = ClgEntityStore.instance()
    utils.get_logger().debug('NILK: Loading train examples..')
    train_examples = _load_valid_examples_from_jsonl(utils.get_data_file('files.nilk.train'))
    train_sample = random.sample(train_examples, int(len(train_examples) * min(10, sample_size) / 100))
    utils.get_logger().debug('NILK: Loading eval examples..')
    eval_examples = _load_valid_examples_from_jsonl(utils.get_data_file('files.nilk.eval'))
    eval_sample = random.sample(eval_examples, int(len(eval_examples) * min(10, sample_size) / 100))
    utils.get_logger().debug('NILK: Loading test examples..')
    test_examples = _load_valid_examples_from_jsonl(utils.get_data_file('files.nilk.test'))
    test_sample = random.sample(test_examples, int(len(test_examples) * sample_size / 100))
    # collect valid entities (i.e., all entities that are not flagged as NIL)
    utils.get_logger().debug('NILK: Collecting valid entities..')
    unknown_entities = {ex.ent_id for ex in train_examples + eval_examples + test_examples if ex.is_nil_entity}
    known_entities = {e.idx for e in clge.get_entities() if e.idx not in unknown_entities}
    return NilkDataCorpus(train_sample, known_entities), NilkDataCorpus(eval_sample, known_entities), NilkDataCorpus(test_sample, known_entities)


def _load_valid_examples_from_jsonl(filepath: str) -> List[NilkExample]:
    result = []
    dbr = DbpResourceStore.instance()
    new_ent_id = 50 * 10**6  # let the entity index of unknown entities start with 50M so we can be sure that they do not overlap with existing ones
    new_ent_dict = {}
    with open(filepath, mode='r') as f:
        for line in tqdm(f, desc='Parsing examples'):
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
            if wikidata_res:
                ent_id = wikidata_res.idx
            else:
                if wikidata_id not in new_ent_dict:
                    new_ent_dict[wikidata_id] = new_ent_id
                    new_ent_id += 1
                ent_id = new_ent_dict[wikidata_id]
            # create mentionId with Pos0 = example id, Pos1 = indicator for NILK dataset, Pos2 = -1 if new entity else 0
            mention_id = MentionId(ex_id, EntityIndex.NEW_ENTITY, EntityIndex.NEW_ENTITY if is_nil else 0)
            # get text context
            left_text = context[:label_start_idx].strip()
            right_text = context[label_end_idx:].strip()
            result.append(NilkExample(mention_id, label, left_text, right_text, page_res.idx, ent_id, is_nil))
    return result
