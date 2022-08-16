import argparse
import os


if __name__ == '__main__':
    # first parse all the arguments
    parser = argparse.ArgumentParser(description='Run the evaluation of subject entity mention detection.')
    parser.add_argument('gpu', type=int, choices=range(8), help='Number of GPU to use')
    parser.add_argument('model_name', help='Huggingface Transformer model used for tagging')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size used in train/eval')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='learning rate used during training')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='warmup steps during learning')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay during learning')
    parser.add_argument('-it', '--ignore_tags', action="store_true", help='Only predict entity mentions and ignore POS tags')
    parser.add_argument('-st', '--predict_single_tag', action="store_true", help='Predict only a single POS tag per chunk')
    parser.add_argument('-nss', '--negative_sample_size', type=float, default=0.0, help='Ratio of artificial negative examples')
    parser.add_argument('-sic', '--single_item_chunks', action="store_true", help='Use only one item per chunk')
    args = parser.parse_args()
    # then set necessary environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['DISABLE_SPACY_CACHE'] = '1'
    # then fix all seeds
    import random
    import numpy as np
    import torch
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # then import application-specific code and run it
    from entity_linking.mention_detection import run_evaluation
    assert not (args.ignore_tags and args.predict_single_tag), 'Can\'t ignore tags AND predict a single tag.'
    run_evaluation(args.model_name, args.epochs, args.batch_size, args.learning_rate, args.warmup_steps, args.weight_decay, args.ignore_tags, args.predict_single_tag, args.negative_sample_size, args.single_item_chunks)
