"""Basic utilities for configuration, file management, caching, and logging."""

import yaml
import logging
import os
import sys
import datetime
import pickle
import bz2
from pathlib import Path
import urllib.request
import pandas as pd


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
    __CONFIG__ = yaml.full_load(config_file)


# FILES
def get_data_file(config_path: str) -> str:
    """Return the file path where the data file is located (given the `config_path` where it is specified)."""
    config = get_config(config_path)

    # download file if not existing
    filepath = os.path.join(_get_root_path(), 'data', config['filename'])
    if not os.path.isfile(filepath):
        url = config['url']
        log_info(f'Download file from {url}..')
        urllib.request.urlretrieve(url, filepath)
        log_info(f'Finished downloading from {url}')

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
    config = get_config('cache.{}'.format(cache_identifier))
    cache_path = _get_cache_path(cache_identifier, version=version)
    if not cache_path.exists():
        return None
    if _should_store_as_hdf(config):
        return pd.read_hdf(cache_path, key='df')
    if _should_store_as_csv(config):
        return pd.read_csv(cache_path, sep=';', index_col=0)
    else:
        open_func = _get_cache_open_func(config)
        with open_func(cache_path, mode='rb') as cache_file:
            return pickle.load(cache_file)


def update_cache(cache_identifier: str, cache_obj, version=None):
    """Update the object cached at `cache_identifier` with `cache_obj`."""
    config = get_config('cache.{}'.format(cache_identifier))
    cache_path = _get_cache_path(cache_identifier, version=version)
    if _should_store_as_hdf(config):
        cache_obj.to_hdf(cache_path, key='df', mode='w')
    elif _should_store_as_csv(config):
        cache_obj.to_csv(cache_path, sep=';')
    else:
        open_func = _get_cache_open_func(config)
        with open_func(cache_path, mode='wb') as cache_file:
            pickle.dump(cache_obj, cache_file, protocol=4)


def _get_cache_path(cache_identifier: str, version=None) -> Path:
    config = get_config('cache.{}'.format(cache_identifier))
    filename = config['filename']
    version = version or config['version']
    if _should_store_as_folder(config):
        base_fileformat = ''
    elif _should_store_as_hdf(config):
        base_fileformat = '.h5'
    elif _should_store_as_csv(config):
        base_fileformat = '.csv'
    else:
        base_fileformat = '.p'
    fileformat = base_fileformat + ('.bz2' if _should_compress_cache(config) else '')
    return Path(os.path.join(_get_root_path(), 'data', 'cache', f'{filename}_v{version}{fileformat}'))


def _get_cache_open_func(config: dict):
    return bz2.open if _should_compress_cache(config) else open


def _should_compress_cache(config: dict) -> bool:
    return 'compress' in config and config['compress']


def _should_store_as_folder(config: dict) -> bool:
    return 'store_as_folder' in config and config['store_as_folder']


def _should_store_as_hdf(config: dict) -> bool:
    return 'store_as_hdf' in config and config['store_as_hdf']


def _should_store_as_csv(config: dict) -> bool:
    return 'store_as_csv' in config and config['store_as_csv']


def _get_root_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


# LOGGING
def log_debug(msg: str):
    _get_logger().debug(msg)


def log_info(msg: str):
    _get_logger().info(msg)


def log_error(msg: str):
    _get_logger().error(msg)


def _get_logger():
    return logging.getLogger('impl')


log_format = '%(asctime)s|%(levelname)s|%(pathname)s->%(funcName)s: %(message)s'
log_level = get_config('logging.level')
if get_config('logging.to_file') and 'ipykernel' not in sys.modules:
    log_filename = '{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), get_config('logging.filename'))
    log_filepath = os.path.join(_get_root_path(), 'logs', log_filename)
    log_file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_file_handler.setFormatter(logging.Formatter(log_format))
    logger = _get_logger()
    logger.addHandler(log_file_handler)
    logger.setLevel(log_level)
else:
    logging.basicConfig(format=log_format, level=log_level)
