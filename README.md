# CaLiGraph

TODO: Intro-Text

## Purpose
todo

## Configuration
### Prerequisites
- Python 3
- pipenv (https://pipenv.readthedocs.io/en/latest/)

### Setup

- Create virtual environment with pipenv
```
pipenv install
```

- Download the spacy corpus:
```
pipenv run python -m spacy download en_core_web_lg
```

- Download the wordnet corpus of nltk (run in python):
```
import nltk
nltk.download('wordnet')
```

### Basic Configuration Options

Use `config.yaml` for configuration of the application.

## Usage

- Run the application with pipenv:
```
pipenv run .
```

## License
MIT.
https://opensource.org/licenses/MIT