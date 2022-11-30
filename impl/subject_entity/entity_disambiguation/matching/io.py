from typing import Dict
import os
import pickle
import utils
from impl.subject_entity.entity_disambiguation.data import CandidateAlignment


def store_candidate_alignment(approach_name: str, candidates: Dict[str, CandidateAlignment]):
    utils.get_logger().debug(f'Storing candidates for matcher with name "{approach_name}"..')
    with open(get_cache_path(approach_name) + '.p', mode='wb') as f:
        return pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_candidate_alignment(approach_id: str) -> Dict[str, CandidateAlignment]:
    utils.get_logger().debug(f'Loading candidates from matcher with id "{approach_id}"..')
    with open(_get_approach_path_by_id(approach_id, '.p'), mode='rb') as f:
        return pickle.load(f)


def get_cache_path(approach_name: str) -> str:
    return os.path.join(utils._get_root_path(), 'data', 'cache', 'ED', approach_name)


def _get_approach_path_by_id(approach_id: str, file_ending: str) -> str:
    data_dir = os.path.join(utils._get_root_path(), 'data', 'cache', 'ED')
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath) and filename.startswith(approach_id) and filename.endswith(file_ending):
            return filepath
    raise FileNotFoundError(f'Could not find file for approach with ID {approach_id}.')
