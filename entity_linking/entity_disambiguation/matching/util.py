from enum import Enum


class MatchingScenario(Enum):
    MENTION_MENTION = 'MM'
    MENTION_ENTITY = 'ME'
    FUSION = 'F'

    def is_MM(self) -> bool:
        return self in [self.MENTION_MENTION, self.FUSION]

    def is_ME(self) -> bool:
        return self in [self.MENTION_ENTITY, self.FUSION]


class MatchingApproach(Enum):
    EXACT = 'exact'
    WORD = 'word'
    BM25 = 'bm25'
    BIENCODER = 'biencoder'
    CROSSENCODER = 'crossencoder'
    POPULARITY = 'popularity'  # ME only!
    TOP_DOWN_FUSION = 'tdf'
    BOTTOM_UP_FUSION = 'buf'
