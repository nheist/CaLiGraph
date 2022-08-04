# CaLiGraph
**A Large Semantic Knowledge Graph from Wikipedia Categories and Listings**

For information about the general idea, extraction statistics, and resources of CaLiGraph, visit the [CaLiGraph website](http://caligraph.org).

## Configuration
### System Requirements
- At least 300 GB of RAM as we load most of DBpedia in memory to speed up the extraction
- At least one GPU to run [transformers](https://huggingface.co/transformers/)
- During the first execution of an extraction you need a stable internet connection as the required DBpedia files are downloaded automatically 

### Prerequisites
- Environment manager: [conda](https://docs.continuum.io/anaconda/install/)
- Dependency manager: [poetry](https://python-poetry.org/docs/#installation)

### Setup
- In the project root, create a conda environment with: `conda env create -f environment.yaml`

- Activate the environment with `conda activate caligraph`

- Install dependencies with `poetry install`
- Install PyTorch for your specific cuda version with `poetry run poe autoinstall-torch-cuda`

- If you have not downloaded them already, you have to fetch the latest corpora for spaCy and nltk (run in terminal):
```
# download the most recent corpus of spaCy
python -m spacy download en_core_web_lg
# download wordnet & words corpora of nltk
python -c 'import nltk; nltk.download("wordnet"); nltk.download("words"); nltk.download("omw-1.4")'
```

### Basic Configuration Options

You can configure the application-specific parameters as well as logging- and file-related parameters in `config.yaml`. 

## Usage

In the project root, run the extraction with `python .`

All the required resources, like DBpedia files, will be downloaded automatically during execution.
CaLiGraph is serialized in N-Triple format. The resulting files are placed in the `results` folder.


## Tests

In the project root, run tests with `pytest`