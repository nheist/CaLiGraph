import yaml
import logging
import os
import datetime


# FILES
def get_root_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_data_file(filename):
    return os.path.join(get_root_path(), 'data', filename)


def get_cache_file(filename):
    return os.path.join(get_root_path(), 'data', 'cache', filename)


def get_results_file(filename):
    return os.path.join(get_root_path(), 'results', filename)


def get_logs_file(filename):
    return os.path.join(get_root_path(), 'logs', filename)


# CONFIGURATION
def get_config(path: str):
    try:
        value = __CONFIG__
        for part in path.split('.'):
            value = value.get(part)
    except KeyError:
        raise KeyError('Could not find configuration value for path {}!'.format(path))

    return value


with open('config.yaml', 'r') as config_file:
    __CONFIG__ = yaml.parse(config_file)


# LOGGING
def get_logger():
    return logging.getLogger('caligraph')


log_format = '%(asctime)s %(levelname)s: %(message)s'
log_level = get_config('logging.level')
if get_config('logging.to_file'):
    log_filename = '{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), get_config('logging.filename'))
    log_file_handler = logging.FileHandler(get_logs_file(log_filename), 'a', 'utf-8')
    log_file_handler.setFormatter(logging.Formatter(log_format))
    logger = get_logger()
    logger.addHandler(log_file_handler)
    logger.setLevel(log_level)
else:
    logging.basicConfig(format=log_format, level=log_level)
