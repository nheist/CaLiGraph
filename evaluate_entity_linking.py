import argparse
import os


# TODO: For final EL, make sure that extraction also works from list pages (they may not have an embedding vector)


if __name__ == '__main__':
    # first parse all the arguments
    parser = argparse.ArgumentParser(description='Run the evaluation of entity linking.')
    parser.add_argument('gpu', choices=range(8), help='Number of GPU to use')
    parser.add_argument('type', choices=['EP', 'MEM'], help='Type of prediction')
    parser.add_argument('model_name', help='Huggingface Transformer model used for prediction')
    parser.add_argument('-s', '--sample', type=int, default=10, help='Percentage of dataset used')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='Epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size used in train/eval')
    parser.add_argument('-l', '--loss', type=str, default='NPAIR', choices=['NPAIR', 'MSE'], help='Loss used in train/eval')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate used during training')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Warmup steps during learning')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='Weight decay during learning')
    parser.add_argument('-ne', '--num_ents', type=int, default=64, help='Number of candidate entities per sentence')
    parser.add_argument('-ed', '--ent_dim', type=int, default=200, help='Number of dimensions of the entity embedding')
    parser.add_argument('-ipc', '--items_per_chunk', type=int, default=15, help='Maximum number of items in a chunk')
    parser.add_argument('-cp', '--cls_predictor', action='store_true', help='Use CLS token as mention embedding')
    parser.add_argument('-isp', '--include_source_page', action='store_false', help='Add embedding of page as feature for prediction')
    args = parser.parse_args()
    # then set necessary environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
    from entity_linking import entity_prediction as ep
    from entity_linking import mention_entity_matching as mem
    if args.cls_predictor:
        assert args.items_per_chunk == 1, 'Can only evaluate a single item at once if using CLS token as mention vector'
    match args.type:
        case 'MEM':
            mem.run_prediction(args.model_name, args.sample, args.epochs, args.batch_size, args.learning_rate,
                               args.warmup_steps, args.weight_decay, args.num_ents, args.ent_dim, args.items_per_chunk,
                               args.cls_predictor, args.include_source_page)
        case 'EP':
            ep.run_prediction(args.model_name, args.sample, args.epochs, args.batch_size, args.loss, args.learning_rate,
                              args.warmup_steps, args.weight_decay, args.num_ents, args.ent_dim, args.items_per_chunk,
                              args.cls_predictor, args.include_source_page)
