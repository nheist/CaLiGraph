import argparse
import entity_linking.util as el_util
from entity_linking.vecpred.fixed.data import extract


def run_extraction(parts: int, embedding_type: str, extract_validation: bool):
    extract.extract_training_data(parts, embedding_type, extract_validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the extraction of training data.')
    parser.add_argument('dataset_id', help='ID of the dataset')
    parser.add_argument('parts', type=int, choices=list(range(1, 11)), help='specify how many parts of the dataset (max: 10) are extracted')
    parser.add_argument('embedding_type', help='embeddings used to encode DBpedia')
    parser.add_argument('-ev', '--extract_validation', action="store_true", help='specify whether the validation set should be extracted')
    args = parser.parse_args()

    el_util.DATASET_ID = args.dataset_id

    run_extraction(args.parts, args.embedding_type, args.extract_validation)
