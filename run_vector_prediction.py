import argparse
import entity_linking.util as el_util
from entity_linking.vecpred.loss import LOSS_BCE, LOSS_TYPES
from entity_linking.vecpred.fixed.data import load
from entity_linking.vecpred.fixed.model import FCNN, FCNN3, ACTIVATION_FUNCS
from entity_linking.vecpred.fixed import train
from entity_linking.vecpred.baseline import evaluate_baselines


def run_evaluation(loss: str, parts: int, embedding_type: str, learning_rate: float, activation_function: str, epochs: int, batch_size: int, hard_negatives: bool, with_baselines: bool):
    print('Retrieving entity vectors..')
    entity_vectors, idx2ent, ent2idx = load.get_entity_vectors(parts, embedding_type)
    embedding_size = len(entity_vectors[0])

    print('Preparing data loaders..')
    X_train, Y_train, X_val, Y_val = load.get_train_and_val_data(parts, embedding_type, ent2idx, True)
    if hard_negatives:
        sf_to_entity_word_mapping = el_util.load_data('sf-to-entity-word-mapping.p', parts=parts)
        train_loader = load.get_data_loader_with_hard_negatives(X_train, Y_train, ent2idx, sf_to_entity_word_mapping, batch_size=batch_size)
    else:
        train_loader = load.get_data_loader(X_train, Y_train, batch_size=batch_size)
    val_loader = load.get_data_loader(X_val, Y_val)
    n_features = X_train.shape[-1]

    if with_baselines:
        print('Evaluating baselines..')
        evaluate_baselines(val_loader, entity_vectors, idx2ent, epochs)

    models = [
        FCNN(n_features, embedding_size, activation_function),
        FCNN3(n_features, embedding_size, activation_function)
    ]
    for model in models:
        label = _create_label(model, parts, embedding_type, loss, learning_rate, activation_function, epochs, batch_size, hard_negatives)
        print(f'Running training for {label}')
        train.train(label, train_loader, val_loader, entity_vectors, model, learning_rate, loss, epochs)


def run_evaluation_binary(loss: str, parts: int, embedding_type: str, learning_rate: float, activation_function: str, epochs: int, batch_size: int):
    print('Retrieving entity vectors..')
    entity_vectors, idx2ent, ent2idx = load.get_entity_vectors(parts, embedding_type)
    embedding_size = len(entity_vectors[0])

    print('Preparing data loaders..')
    X_train, Y_train, X_val, Y_val = load.get_train_and_val_data(parts, embedding_type, ent2idx, False)
    train_loader = load.get_data_loader_for_binary_matching(X_train, Y_train, entity_vectors, batch_size=batch_size)
    val_loader = load.get_data_loader(X_val, Y_val)
    n_features = X_train.shape[-1] + embedding_size

    models = [
        FCNN(n_features, 1, activation_function, last_activation_function='sigmoid'),
        FCNN3(n_features, 1, activation_function, last_activation_function='sigmoid')
    ]
    for model in models:
        label = _create_label(model, parts, embedding_type, loss, learning_rate, activation_function, epochs, batch_size, False)
        print(f'Running training for {label}')
        train.train_binary(label, train_loader, val_loader, entity_vectors, model, learning_rate, loss, epochs)


def _create_label(model, p: int, et: str, loss: str, lr: float, af: str, e: int, bs: int, hn: bool) -> str:
    return f'{type(model).__name__}_p-{p}_et-{et}_loss-{loss}_lr-{lr}_af-{af}_e-{e}_bs-{bs}_hn-{hn}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of embedding vector prediction on DBpedia.')
    parser.add_argument('loss', choices=LOSS_TYPES, help='loss function used for training')
    parser.add_argument('-d', '--dataset_id', default='clgv21-v1', help='ID of the dataset used')
    parser.add_argument('-p', '--parts', type=int, choices=list(range(1, 11)), default=3, help='specify how many parts of the dataset (max: 10) are used')
    parser.add_argument('-et', '--embedding_type', default='rdf2vec', help='embeddings used to encode DBpedia')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='learning rate used during training')
    parser.add_argument('-af', '--activation_function', choices=ACTIVATION_FUNCS.keys(), default='relu', help='activation function used in fixed layers')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='size of batches for training')
    parser.add_argument('-hn', '--hard_negatives', action="store_true", help='put similar examples together in the same batch')
    parser.add_argument('-wb', '--with_baselines', action="store_true", help='additionally evaluate baselines')
    args = parser.parse_args()

    el_util.DATASET_ID = args.dataset_id

    if args.loss == LOSS_BCE:
        run_evaluation_binary(args.loss, args.parts, args.embedding_type, args.learning_rate, args.activation_function, args.epochs, args.batch_size)
    else:
        run_evaluation(args.loss, args.parts, args.embedding_type, args.learning_rate, args.activation_function, args.epochs, args.batch_size, args.hard_negatives, args.with_baselines)
