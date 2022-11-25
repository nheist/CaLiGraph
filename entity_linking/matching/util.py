from enum import Enum


class MatchingScenario(Enum):
    MENTION_MENTION = 'MM'
    MENTION_ENTITY = 'ME'
    FULL = 'F'

    def is_mention_mention(self) -> bool:
        return self in [self.MENTION_MENTION, self.FULL]

    def is_mention_entity(self) -> bool:
        return self in [self.MENTION_ENTITY, self.FULL]


class MatchingApproach(Enum):
    EXACT = 'exact'
    WORD = 'word'
    BM25 = 'bm25'
    BIENCODER = 'biencoder'
    CROSSENCODER = 'crossencoder'
    POPULARITY = 'popularity'  # ME only!
    NASTY_LINKER = 'nl'
    EDIN = 'edin'
