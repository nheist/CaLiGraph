# CaLiGraph

\- A Large Semantic Knowledge Graph from Wikipedia Categories and Listpages \-

For information about the general idea, extraction statistics, and resources of CaLiGraph, visit the [CaLiGraph website](http://caligraph.org).

## Configuration
### Prerequisites
- Python 3
- pipenv (https://pipenv.readthedocs.io/en/latest/)

Note: If you have problems with your pipenv installation, you can also run the code directly via python. Just make sure to install all the dependencies given in `Pipfile` and `Pipfile.lock`. 

### System Requirements
- You need a machine with at least 100 GB of RAM as we load most of DBpedia in memory to speed up the extraction
- During the first execution of an extraction you need a stable internet connection as the required DBpedia files are downloaded automatically 

### Setup
- In the project source directory, create and initialize a virtual environment with pipenv (run in terminal):

```
pipenv install
```

- If you have not downloaded them already, you have to fetch the latest corpora for spaCy and nltk-wordnet (run in terminal):
```
pipenv run python -m spacy download en_core_web_lg            # download the most recent corpus of spaCy
pipenv run python -c 'import nltk; nltk.download("wordnet")'  # download the wordnet corpus of ntlk
```

### Basic Configuration Options

You can configure the application-specific parameters as well as logging- and file-related parameters in `config.yaml`. 

## Usage

Run the extraction with pipenv:

```
pipenv run python3 .
```

All the required resources, like DBpedia files, will be downloaded automatically during execution.
CaLiGraph is serialized in N-Triple format. The resulting files are placed in the `results` folder.
