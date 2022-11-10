from typing import List, Tuple, Union, Dict, Optional, Set, Iterable, TypeVar
from collections import defaultdict
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from enum import Enum
from scipy.special import comb
import itertools
from impl.util.transformer import SpecialToken, EntityIndex
from impl.wikipedia import MentionId, WikiPageStore
from impl.caligraph.entity import ClgEntity


CXS = SpecialToken.CONTEXT_SEP.value
CXE = SpecialToken.CONTEXT_END.value
COL = SpecialToken.TABLE_COL.value
ROW = SpecialToken.TABLE_ROW.value
TXS = SpecialToken.TEXT_SEP.value


class CorpusType(Enum):
    LIST = 'LIST'
    NILK = 'NILK'


Pair = TypeVar('Pair', bound=Tuple[MentionId, Union[MentionId, int]])


class Alignment:
    def __init__(self, entity_to_mention_mapping: Dict[int, Set[MentionId]], known_entities: Set[int]):
        self.entity_to_mention_mapping = entity_to_mention_mapping
        self.mention_to_entity_mapping = {m_id: e_id for e_id, m_ids in entity_to_mention_mapping.items() for m_id in m_ids}
        self.known_entities = known_entities

    def __contains__(self, pair: Pair) -> bool:
        source, target = pair
        if isinstance(target, MentionId):
            if source not in self.mention_to_entity_mapping or target not in self.mention_to_entity_mapping:
                return False
            return self.mention_to_entity_mapping[source] == self.mention_to_entity_mapping[target]
        return source in self.mention_to_entity_mapping and self.mention_to_entity_mapping[source] == target

    def mention_count(self, nil_flag: Optional[bool] = None) -> int:
        if nil_flag is None:
            return len(self.mention_to_entity_mapping)
        elif nil_flag:
            return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id not in self.known_entities])
        else:
            return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities])

    def entity_count(self, nil_flag: Optional[bool] = None) -> int:
        if nil_flag is None:
            return len(self.entity_to_mention_mapping)
        elif nil_flag:
            return len(set(self.entity_to_mention_mapping).difference(self.known_entities))
        else:
            return len(set(self.entity_to_mention_mapping).intersection(self.known_entities))

    def sample_mm_matches(self, sample_size: int) -> List[Pair]:
        sample_size *= 10 ** 6
        mm_matches = []
        if sample_size > self.get_mm_match_count(None):  # return all
            for mention_group in self.entity_to_mention_mapping.values():
                mm_matches.extend([tuple(pair) for pair in itertools.combinations(mention_group, 2)])
        else:  # return sample
            sample_grps = [grp for grp in self.entity_to_mention_mapping.values() if len(grp) > 1]
            sample_grp_weights = list(itertools.accumulate([len(grp) for grp in sample_grps]))
            for _ in tqdm(range(sample_size), desc='Sampling mention-mention matches'):
                grp = random.choices(sample_grps, cum_weights=sample_grp_weights)[0]
                mm_matches.append(tuple(random.sample(grp, 2)))
        return mm_matches

    def sample_me_matches(self, sample_size: int) -> List[Pair]:
        sample_size *= 10 ** 6
        known_mentions = [m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities]
        if sample_size > len(known_mentions):  # return all
            return [(m_id, self.mention_to_entity_mapping[m_id]) for m_id in known_mentions]
        sample_mentions = random.sample(known_mentions, sample_size)
        return [(m_id, self.mention_to_entity_mapping[m_id]) for m_id in sample_mentions]

    def get_mm_match_count(self, nil_flag: Optional[bool] = None) -> int:
        if nil_flag is None:
            grps = list(self.entity_to_mention_mapping.values())
        elif nil_flag:
            grps = [grp for e_id, grp in self.entity_to_mention_mapping.items() if e_id not in self.known_entities]
        else:
            grps = [grp for e_id, grp in self.entity_to_mention_mapping.items() if e_id in self.known_entities]
        return sum(comb(len(grp), 2) for grp in grps)

    def get_me_match_count(self, nil_flag: Optional[bool] = None) -> int:
        if nil_flag:
            return 0
        return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities])


class CandidateAlignment:
    def __init__(self):
        self.mention_to_target_mapping = defaultdict(dict)

    def __contains__(self, pair: Pair) -> bool:
        if isinstance(pair[1], MentionId):
            mention_a, mention_b = sorted(pair)
            return mention_a in self.mention_to_target_mapping and mention_b in self.mention_to_target_mapping[mention_a]
        mention, ent = pair
        return mention in self.mention_to_target_mapping and ent in self.mention_to_target_mapping[mention]

    def add_candidate(self, pair: Pair, score: float):
        item_a, item_b = sorted(pair) if isinstance(pair[1], MentionId) else pair
        self.mention_to_target_mapping[item_a][item_b] = score

    def get_mm_candidates(self, include_score: bool, nil_flag: Optional[bool] = None) -> Iterable[Union[Pair, Tuple[Pair, float]]]:
        yield from self._get_candidates(MentionId, include_score, nil_flag)

    def get_me_candidates(self, include_score: bool, nil_flag: Optional[bool] = None) -> Iterable[Union[Pair, Tuple[Pair, float]]]:
        yield from self._get_candidates(int, include_score, nil_flag)

    def _get_candidates(self, target_type, include_score: bool, nil_flag: Optional[bool]) -> Iterable[Union[Pair, Tuple[Pair, float]]]:
        for m_id, targets in self.mention_to_target_mapping.items():
            for t_id, score in targets.items():
                pair = (m_id, t_id)
                if isinstance(t_id, target_type) and self._is_consistent_with_nil_flag(pair, nil_flag):
                    yield ((m_id, t_id), score) if include_score else (m_id, t_id)

    def get_mm_candidate_count(self, nil_flag: Optional[bool] = None) -> int:
        return sum(1 for pair in self.get_mm_candidates(False) if self._is_consistent_with_nil_flag(pair, nil_flag))

    def get_me_candidate_count(self, nil_flag: Optional[bool] = None) -> int:
        return sum(1 for pair in self.get_me_candidates(False) if self._is_consistent_with_nil_flag(pair, nil_flag))

    def get_mm_overlap(self, alignment: Alignment, nil_flag: Optional[bool] = None) -> int:
        return sum(1 for pair in self.get_mm_candidates(False, nil_flag) if pair in alignment)

    def get_me_overlap(self, alignment: Alignment, nil_flag: Optional[bool] = None) -> int:
        return sum(1 for pair in self.get_me_candidates(False, nil_flag) if pair in alignment)

    @classmethod
    def _is_consistent_with_nil_flag(cls, pair: Pair, nil_flag: Optional[bool]) -> bool:
        if nil_flag is None:
            return True
        if nil_flag:  # True, if any mention id is nil
            item_a, item_b = pair
            return cls._is_nil_mention(item_a) or isinstance(item_b, MentionId) and cls._is_nil_mention(item_b)
        else:  # True, if any mention id is not nil
            item_a, item_b = pair
            return not cls._is_nil_mention(item_a) or isinstance(item_b, MentionId) and not cls._is_nil_mention(item_b)

    @classmethod
    def _is_nil_mention(cls, mention_id: MentionId) -> bool:
        if mention_id[1] == EntityIndex.NEW_ENTITY:  # NILK dataset
            return mention_id[2] == EntityIndex.NEW_ENTITY
        else:  # LIST dataset
            return WikiPageStore.instance().get_subject_entity(mention_id).entity_idx == EntityIndex.NEW_ENTITY


class DataCorpus(ABC):
    alignment: Alignment

    @abstractmethod
    def get_mention_labels(self, discard_unknown: bool = False) -> Dict[MentionId, str]:
        pass

    @abstractmethod
    def get_mention_input(self, add_page_context: bool, add_text_context: bool) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
        pass

    @abstractmethod
    def get_entities(self) -> Set[ClgEntity]:
        pass

    def get_entity_input(self, add_entity_abstract: bool, add_kg_info: int) -> Dict[int, str]:
        result = {}
        for e in tqdm(self.get_entities(), desc='Preparing entities'):
            ent_description = [f'{e.get_label()} {SpecialToken.get_type_token(e.get_type_label())}']
            if add_entity_abstract:
                ent_description.append((e.get_abstract() or '')[:200])
            if add_kg_info:
                kg_info = [f'type = {t.get_label()}' for t in e.get_types()]
                prop_count = max(0, add_kg_info - len(kg_info))
                if prop_count > 0:
                    props = list(e.get_properties(as_tuple=True))[:prop_count]
                    kg_info += [f'{pred.get_label()} = {val.get_label() if isinstance(val, ClgEntity) else val}' for pred, val in props]
            result[e.idx] = f' {CXS} '.join(ent_description)
        return result
