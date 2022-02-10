import argparse
import entity_linking.util as el_util
from entity_linking.vecpred.loss import LOSS_TYPES
from entity_linking.vecpred.fixed.data.load import get_entity_vectors, get_train_and_val_data, get_data_loader
from entity_linking.vecpred.fixed.model import FCNN, FCNN3, ACTIVATION_FUNCS
from entity_linking.vecpred.fixed.train import train
from entity_linking.vecpred.baseline import evaluate_baselines


# run evaluation

def run_evaluation(parts: int, loss: str, learning_rate: float, activation_function: str, epochs: int, batch_size: int, hard_negatives: bool, ignore_singles: bool, with_baselines: bool):
    # retrieve entity vectors
    print('Retrieving entity vectors..')
    entity_vectors, idx2ent, ent2idx = get_entity_vectors(parts)
    embedding_size = len(entity_vectors[0])

    # retrieve entity blocks if necessary
    print('Retrieving entity blocks..')
    sf_to_entity_word_mapping = el_util.load_data('sf-to-entity-word-mapping.p', parts=parts) if hard_negatives else None

    # prepare data loaders
    print('Preparing data loaders..')
    X_train, Y_train, X_val, Y_val = get_train_and_val_data(parts, ent2idx)
    train_loader = get_data_loader(X_train, Y_train, ent2idx, batch_size=batch_size, hard_negative_blocks=sf_to_entity_word_mapping, ignore_singles=ignore_singles)
    val_loader = get_data_loader(X_val, Y_val, ent2idx)
    n_features = X_train.shape[-1]

    if with_baselines:
        print('Evaluating baselines..')
        evaluate_baselines(val_loader, entity_vectors, idx2ent, epochs)

    # run training
    for model in [FCNN(n_features, embedding_size, activation_function), FCNN3(n_features, embedding_size, activation_function)]:
        label = _create_label(model, parts, learning_rate, activation_function, epochs, batch_size, hard_negatives)
        print(f'Running training for {label}')
        train(label, train_loader, val_loader, entity_vectors, model, learning_rate, loss, epochs)


def _create_label(model, p: int, lr: float, af: str, e: int, bs: int, hn: bool) -> str:
    return f'{type(model).__name__}_p-{p}_lr-{lr}_af-{af}_e-{e}_bs-{bs}_hn-{hn}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of embedding vector prediction on DBpedia.')
    parser.add_argument('loss', choices=LOSS_TYPES, help='loss function used for training')
    parser.add_argument('-p', '--parts', type=int, choices=list(range(1, 11)), default=3, help='specify how many parts of the dataset (max: 10) are used')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='learning rate used during training')
    parser.add_argument('-af', '--activation_function', choices=ACTIVATION_FUNCS.keys(), default='relu', help='activation function used in fixed layers')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='size of batches for training')
    parser.add_argument('-hn', '--hard_negatives', action="store_true", help='put similar examples together in the same batch')
    parser.add_argument('-is', '--ignore_singles', action="store_true", help='ignore examples without similar entities')
    parser.add_argument('-wb', '--with_baselines', action="store_true", help='additionally evaluate baselines')
    args = parser.parse_args()

    run_evaluation(args.loss, args.parts, args.learning_rate, args.activation_function, args.epochs, args.batch_size, args.hard_negatives, args.ignore_singles, args.with_baselines)
