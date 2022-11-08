from typing import List, Tuple, Union, Dict, Optional, Set, Iterable, TypeVar
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from enum import Enum
from scipy.special import comb
import itertools
from impl.util.transformer import SpecialToken, EntityIndex
from impl.wikipedia.page_parser import MentionId
from impl.caligraph.entity import ClgEntity
from entity_linking.entity_disambiguation.matching.util import MatchingScenario


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
        self.sample_size = int(1e6)

    def __contains__(self, pair: Pair) -> bool:
        source, target = pair
        if isinstance(target, MentionId):
            if source not in self.mention_to_entity_mapping or target not in self.mention_to_entity_mapping:
                return True
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

    def sample_matches(self, scenario: MatchingScenario) -> List[Pair]:
        return self._sample_mm_matches() if scenario.is_MM() else self._sample_me_matches()

    def _sample_mm_matches(self) -> List[Pair]:
        mm_matches = []
        if self.sample_size > self._get_mm_match_count(None):  # return all
            for mention_group in self.entity_to_mention_mapping.values():
                mm_matches.extend([tuple(pair) for pair in itertools.combinations(mention_group, 2)])
        else:  # return sample
            sample_grps = [grp for grp in self.entity_to_mention_mapping.values() if len(grp) > 1]
            sample_grp_weights = list(itertools.accumulate([len(grp) for grp in sample_grps]))
            for _ in tqdm(range(self.sample_size), desc='Sampling mention-mention matches'):
                grp = random.choices(sample_grps, cum_weights=sample_grp_weights)[0]
                mm_matches.append(tuple(random.sample(grp, 2)))
        return mm_matches

    def _sample_me_matches(self) -> List[Pair]:
        known_mentions = [m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities]
        if self.sample_size > len(known_mentions):  # return all
            return [(m_id, self.mention_to_entity_mapping[m_id]) for m_id in known_mentions]
        sample_mentions = random.sample(known_mentions, self.sample_size)
        return [(m_id, self.mention_to_entity_mapping[m_id]) for m_id in sample_mentions]

    def get_match_count(self, scenario: MatchingScenario, nil_flag: Optional[bool] = None) -> int:
        return self._get_mm_match_count(nil_flag) if scenario.is_MM() else self._get_me_match_count(nil_flag)

    def _get_mm_match_count(self, nil_flag: Optional[bool]) -> int:
        if nil_flag is None:
            grps = list(self.entity_to_mention_mapping.values())
        elif nil_flag:
            grps = [grp for e_id, grp in self.entity_to_mention_mapping.items() if e_id not in self.known_entities]
        else:
            grps = [grp for e_id, grp in self.entity_to_mention_mapping.items() if e_id in self.known_entities]
        return sum(comb(len(grp), 2) for grp in grps)

    def _get_me_match_count(self, nil_flag: Optional[bool]) -> int:
        if nil_flag:
            return 0
        return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities])


class CandidateAlignment:
    def __init__(self):
        pass

    def add_candidate(self, pair: Pair, score: float):
        pass

    def get_mm_candidates(self) -> Iterable[Tuple[Pair, float]]:
        pass

    def get_me_candidates(self) -> Iterable[Tuple[Pair, float]]:
        pass

    def get_candidate_count(self, scenario: MatchingScenario = None, nil_flag: Optional[bool] = None) -> int:
        if scenario is None:
            return self._get_mm_candidate_count(nil_flag) + self._get_me_candidate_count(nil_flag)
        return self._get_mm_candidate_count(nil_flag) if scenario.is_MM() else self._get_me_candidate_count(nil_flag)

    def _get_mm_candidate_count(self, nil_flag: Optional[bool]) -> int:
        pass

    def _get_me_candidate_count(self, nil_flag: Optional[bool]) -> int:
        pass

    def get_overlap(self, alignment: Alignment, scenario: MatchingScenario, nil_flag: Optional[bool] = None) -> int:
        pass

    def _get_mm_overlap(self, alignment: Alignment, nil_flag: Optional[bool]) -> int:
        pass

    def _get_me_overlap(self, alignment: Alignment, nil_flag: Optional[bool]) -> int:
        pass

    def _is_nil_mention(self, mention_id: MentionId) -> bool:
        if mention_id[1] == EntityIndex.NEW_ENTITY:  # NILK dataset
            return mention_id[2] == EntityIndex.NEW_ENTITY
        else:  # LIST dataset
            return self.wps.get_subject_entity(mention_id).entity_idx == EntityIndex.NEW_ENTITY


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
