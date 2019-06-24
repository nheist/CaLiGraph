import yaml
import logging
import os
import sys
import datetime
import pickle
import bz2
from pathlib import Path
import urllib.request


# CONFIGURATION
def get_config(path: str):
    """Return the defined configuration value from 'config.yaml' of `path`."""
    try:
        value = __CONFIG__
        for part in path.split('.'):
            value = value[part]
    except KeyError:
        raise KeyError('Could not find configuration value for path "{}"!'.format(path))

    return value


def set_config(path: str, val):
    """Override the configuration value at `path`."""
    try:
        current = __CONFIG__
        path_segments = path.split('.')
        for s in path_segments[:-1]:
            current = current[s]
        current[path_segments[-1]] = val
    except KeyError:
        raise KeyError('Could not find configuration value for path "{}"!'.format(path))


with open('config.yaml', 'r') as config_file:
    __CONFIG__ = yaml.load(config_file)


# FILES
def get_data_file(config_path: str) -> str:
    """Return the file path where the data file is located (given the `config_path` where it is specified)."""
    file_config = get_config(config_path)

    # download file if not existing
    filepath = os.path.join(_get_root_path(), 'data', file_config['filename'])
    if not os.path.isfile(filepath):
        url = file_config['url']
        get_logger().info(f'Download file from {url}..')
        urllib.request.urlretrieve(url, filepath)
        get_logger().info(f'Finished downloading from {url}')

    return filepath


def get_results_file(config_path: str) -> str:
    """Return the file path where the results file is located (given the `config_path` where it is specified)."""
    return os.path.join(_get_root_path(), 'results', get_config(config_path))


# CACHING
def load_or_create_cache(cache_identifier: str, init_func, version=None):
    """Return the object cached at `cache_identifier` if existing - otherwise initialise it first using `init_func`."""
    cache_obj = load_cache(cache_identifier, version=version)
    if cache_obj is None:
        cache_obj = init_func()
        update_cache(cache_identifier, cache_obj, version=version)
    return cache_obj


def load_cache(cache_identifier: str, version=None):
    """Return the object cached at `cache_identifier`."""
    cache_path = _get_cache_path(cache_identifier, version=version)
    if not cache_path.exists():
        return None
    open_func = _get_cache_open_func(cache_identifier)
    with open_func(cache_path, mode='rb') as cache_file:
        return pickle.load(cache_file)


def update_cache(cache_identifier: str, cache_obj, version=None):
    """Update the object cached at `cache_identifier` with `cache_obj`."""
    open_func = _get_cache_open_func(cache_identifier)
    with open_func(_get_cache_path(cache_identifier, version=version), mode='wb') as cache_file:
        pickle.dump(cache_obj, cache_file)


def _get_cache_path(cache_identifier: str, version=None) -> Path:
    config = get_config('cache.{}'.format(cache_identifier))
    filename = config['filename']
    version = version or config['version']
    fileformat = '.p' + ('.bz2' if _should_compress_cache(cache_identifier) else '')
    return Path(os.path.join(_get_root_path(), 'data', 'cache', f'{filename}_v{version}{fileformat}'))


def _get_cache_open_func(cache_identifier: str):
    return bz2.open if _should_compress_cache(cache_identifier) else open


def _should_compress_cache(cache_identifier: str) -> bool:
    config = get_config('cache.{}'.format(cache_identifier))
    return 'compress' in config and config['compress']


def _get_root_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


# LOGGING
def get_logger():
    return logging.getLogger('impl')


log_format = '%(asctime)s %(levelname)s: %(message)s'
log_level = get_config('logging.level')
if get_config('logging.to_file') and 'ipykernel' not in sys.modules:
    log_filename = '{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), get_config('logging.filename'))
    log_filepath = os.path.join(_get_root_path(), 'logs', log_filename)
    log_file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_file_handler.setFormatter(logging.Formatter(log_format))
    logger = get_logger()
    logger.addHandler(log_file_handler)
    logger.setLevel(log_level)
else:
    logging.basicConfig(format=log_format, level=log_level)
