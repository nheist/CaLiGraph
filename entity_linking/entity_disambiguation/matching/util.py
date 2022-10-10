from enum import Enum
import os
import pickle
import utils


class MatchingScenario(Enum):
    MENTION_MENTION = 'MM'
    MENTION_ENTITY = 'ME'


class MatchingApproach(Enum):
    EXACT = 'exact'
    WORD = 'word'
    POPULARITY = 'popularity'
    BIENCODER = 'biencoder'
    CROSSENCODER = 'crossencoder'


def store_candidates(approach_id: str, candidates: dict):
    with open(_get_approach_path(approach_id), mode='wb') as f:
        return pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_candidates(approach_id: str) -> dict:
    with open(_get_approach_path(approach_id), mode='rb') as f:
        return pickle.load(f)


def _get_approach_path(approach_id: str) -> str:
    return os.path.join(utils._get_root_path(), 'entity_linking', 'data', f'{approach_id}.p')
