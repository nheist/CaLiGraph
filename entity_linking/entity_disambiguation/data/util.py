from typing import Set, Tuple, Union, NamedTuple, Dict
from abc import ABC, abstractmethod
from enum import Enum
import utils
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import MentionId
from impl.caligraph.entity import ClgEntity


CXS = SpecialToken.CONTEXT_SEP.value
CXE = SpecialToken.CONTEXT_END.value
COL = SpecialToken.TABLE_COL.value
ROW = SpecialToken.TABLE_ROW.value


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


class DataCorpus(ABC):
    mm_alignment: Set[Pair]
    me_alignment: Set[Pair]

    @abstractmethod
    def get_mention_labels(self) -> Dict[MentionId, str]:
        pass

    @abstractmethod
    def get_mention_input(self, add_page_context: bool, add_text_context: bool) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
        pass

    @abstractmethod
    def get_entities(self) -> Set[ClgEntity]:
        pass

    def get_entity_input(self, add_entity_abstract: bool, add_kg_info: int) -> Dict[int, str]:
        utils.get_logger().debug('Preparing entities..')
        result = {}
        for e in self.get_entities():
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
