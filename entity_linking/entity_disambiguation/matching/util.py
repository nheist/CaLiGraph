from enum import Enum
import os
import pickle
import utils


class MatchingScenario(Enum):
    MENTION_MENTION = 'MM'
    MENTION_ENTITY = 'ME'
    FUSION = 'F'

    def is_MM(self) -> bool:
        return self in [self.MENTION_MENTION, self.FUSION]

    def is_ME(self) -> bool:
        return self in [self.MENTION_ENTITY, self.FUSION]


class MatchingApproach(Enum):
    # MM/ME
    EXACT = 'exact'
    WORD = 'word'
    BIENCODER = 'biencoder'
    CROSSENCODER = 'crossencoder'
    # ME
    POPULARITY = 'popularity'
    # F
    WEAKEST_MENTION = 'wmention'
    WEAKEST_ENTITY = 'wentity'
    WEAKEST_LINK = 'wlink'
    PRECISION_WEIGHTED_WEAKEST_LINK = 'pwlink'


def store_candidates(approach_name: str, candidates: dict):
    utils.get_logger().debug(f'Storing candidates for matcher with name "{approach_name}"..')
    with open(_get_approach_path(approach_name), mode='wb') as f:
        return pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_candidates(approach_id: str, scenario: MatchingScenario) -> dict:
    utils.get_logger().debug(f'Loading candidates from matcher with id "{approach_id}"..')
    with open(_get_approach_path_by_id(approach_id), mode='rb') as f:
        candidates = pickle.load(f)
        candidates = {eval_mode: candidates_by_scenario[scenario] for eval_mode, candidates_by_scenario in candidates.items()}
        return candidates


def _get_approach_path(approach_name: str) -> str:
    return os.path.join(utils._get_root_path(), 'entity_linking', 'data', f'{approach_name}.p')


def _get_approach_path_by_id(approach_id: str) -> str:
    approach_dir = os.path.join(utils._get_root_path(), 'entity_linking', 'data')
    for filename in os.listdir(approach_dir):
        filepath = os.path.join(approach_dir, filename)
        if os.path.isfile(filepath) and filename.startswith(approach_id):
            return filepath
    raise FileNotFoundError(f'Could not find file for approach with ID {approach_id}.')
