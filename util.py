import yaml
import logging
import os
import datetime
import pickle
from pathlib import Path


# CONFIGURATION
def get_config(path: str):
    try:
        value = __CONFIG__
        for part in path.split('.'):
            value = value.get(part)
    except KeyError:
        raise KeyError('Could not find configuration value for path "{}"!'.format(path))

    return value


with open('config.yaml', 'r') as config_file:
    __CONFIG__ = yaml.load(config_file)


# FILES
def get_root_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def get_data_file(config_path: str) -> str:
    return os.path.join(get_root_path(), 'data', get_config(config_path))


def get_cache_file(config_path: str) -> str:
    cache_file_config = get_config(config_path)
    filename = cache_file_config['filename'].format(cache_file_config['version'])
    return os.path.join(get_root_path(), 'data', 'cache', filename)


def get_results_file(config_path: str) -> str:
    return os.path.join(get_root_path(), 'results', get_config(config_path))


# CACHING
def from_cache(cache_identifier: str, init_func):
    cache_config = get_config('cache.' + cache_identifier)
    cache_path = Path(os.path.join(get_root_path(), 'data', 'cache', cache_config['filename'].format(cache_config['version'])))
    if cache_path.exists():
        with cache_path.open(encoding='utf-8', mode='r') as cache_file:
            return pickle.load(cache_file)
    else:
        cache_obj = init_func()
        with cache_path.open(encoding='utf-8', mode='w') as cache_file:
            pickle.dump(cache_obj, cache_file)
        return cache_obj


# LOGGING
def get_logger():
    return logging.getLogger('caligraph')


log_format = '%(asctime)s %(levelname)s: %(message)s'
log_level = get_config('logging.level')
if get_config('logging.to_file'):
    log_filename = '{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), get_config('logging.filename'))
    log_filepath = os.path.join(get_root_path(), 'logs', log_filename)
    log_file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_file_handler.setFormatter(logging.Formatter(log_format))
    logger = get_logger()
    logger.addHandler(log_file_handler)
    logger.setLevel(log_level)
else:
    logging.basicConfig(format=log_format, level=log_level)
