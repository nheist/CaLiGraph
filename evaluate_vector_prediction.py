import os
os.environ['DISABLE_SPACY_CACHE'] = '1'

import argparse
import random
import numpy as np
import torch
from entity_linking.entity_prediction import run_prediction


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of vector prediction.')
    parser.add_argument('model_name', help='Huggingface Transformer model used for prediction')
    parser.add_argument('-s', '--sample', type=int, default=10, help='Percentage of dataset used')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='Epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size used in train/eval')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate used during training')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Warmup steps during learning')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='Weight decay during learning')
    parser.add_argument('-ne', '--num_ents', type=int, default=64, help='Number of candidate entities per sentence')
    parser.add_argument('-ed', '--ent_dim', type=int, default=200, help='Number of dimensions of the entity embedding')
    parser.add_argument('-ipc', '--items_per_chunk', type=int, default=15, help='Maximum number of items in a chunk')
    parser.add_argument('-cp', '--cls_predictor', action='store_true', help='Use CLS token as mention vector')
    args = parser.parse_args()

    if args.cls_predictor:
        assert args.items_per_chunk == 1, 'Can only evaluate a single item at once if using CLS token as mention vector'

    run_prediction(args.model_name, args.sample, args.epochs, args.batch_size, args.learning_rate, args.warmup_steps,
                   args.weight_decay, args.num_ents, args.ent_dim, args.items_per_chunk, args.cls_predictor)
