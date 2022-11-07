import random
from typing import List, Tuple, Union, NamedTuple, Dict, Optional, Set
from tqdm import tqdm
from abc import ABC, abstractmethod
from enum import Enum
from scipy.special import comb
import itertools
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import MentionId
from impl.caligraph.entity import ClgEntity


CXS = SpecialToken.CONTEXT_SEP.value
CXE = SpecialToken.CONTEXT_END.value
COL = SpecialToken.TABLE_COL.value
ROW = SpecialToken.TABLE_ROW.value
TXS = SpecialToken.TEXT_SEP.value


class CorpusType(Enum):
    LIST = 'LIST'
    NILK = 'NILK'


class Pair(NamedTuple):
    source: MentionId
    target: Union[MentionId, int]
    confidence: float

    def __eq__(self, other) -> bool:
        return self.source == other.source and self.target == other.target

    def __hash__(self):
        return self.source.__hash__() + self.target.__hash__()


class Alignment:
    def __init__(self, entity_to_mention_mapping: Dict[int, Set[MentionId]], known_entities: Set[int]):
        self.entity_to_mention_mapping = entity_to_mention_mapping
        self.mention_to_entity_mapping = {m_id: e_id for e_id, m_ids in entity_to_mention_mapping.items() for m_id in m_ids}
        self.known_entities = known_entities
        self.sample_size = int(1e6)

    def __contains__(self, item) -> bool:
        if not isinstance(item, Pair):
            return False
        if isinstance(item.target, MentionId):
            if item.source not in self.mention_to_entity_mapping or item.target not in self.mention_to_entity_mapping:
                return False
            return self.mention_to_entity_mapping[item.source] == self.mention_to_entity_mapping[item.target]
        return item.source in self.mention_to_entity_mapping and self.mention_to_entity_mapping[item.source] == item.target

    def mention_count(self, nil_partition: Optional[bool]) -> int:
        if nil_partition is None:
            return len(self.mention_to_entity_mapping)
        elif nil_partition:
            return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id not in self.known_entities])
        else:
            return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities])

    def entity_count(self, nil_partition: Optional[bool]) -> int:
        if nil_partition is None:
            return len(self.entity_to_mention_mapping)
        elif nil_partition:
            return len(set(self.entity_to_mention_mapping).difference(self.known_entities))
        else:
            return len(set(self.entity_to_mention_mapping).intersection(self.known_entities))

    def sample_mm_matches(self) -> List[Pair]:
        mm_matches = []
        if self.sample_size > self.mm_match_count():  # return all
            for mention_group in self.entity_to_mention_mapping.values():
                mm_matches.extend([Pair(*sorted(item_pair), 1) for item_pair in itertools.combinations(mention_group, 2)])
        else:  # return sample
            sample_grps = [grp for grp in self.entity_to_mention_mapping.values() if len(grp) > 1]
            sample_grp_weights = list(itertools.accumulate([len(grp) for grp in sample_grps]))
            for _ in tqdm(range(self.sample_size), desc='Sampling mention-mention matches'):
                grp = random.choices(sample_grps, cum_weights=sample_grp_weights)[0]
                mm_matches.append(Pair(*sorted(random.sample(grp, 2)), 1))
        return mm_matches

    def mm_match_count(self, nil_partition: Optional[bool] = None) -> int:
        if nil_partition is None:
            grps = list(self.entity_to_mention_mapping.values())
        elif nil_partition:
            grps = [grp for e_id, grp in self.entity_to_mention_mapping.items() if e_id not in self.known_entities]
        else:
            grps = [grp for e_id, grp in self.entity_to_mention_mapping.items() if e_id in self.known_entities]
        return sum(comb(len(grp), 2) for grp in grps)

    def sample_me_matches(self) -> List[Pair]:
        known_mentions = [m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities]
        if self.sample_size > len(known_mentions):  # return all
            return [Pair(m_id, self.mention_to_entity_mapping[m_id], 1) for m_id in known_mentions]
        sample_mentions = random.sample(known_mentions, self.sample_size)
        return [Pair(m_id, self.mention_to_entity_mapping[m_id], 1) for m_id in sample_mentions]

    def me_match_count(self, nil_partition: Optional[bool] = None) -> int:
        if nil_partition:
            return 0
        return len([m_id for m_id, e_id in self.mention_to_entity_mapping.items() if e_id in self.known_entities])


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
