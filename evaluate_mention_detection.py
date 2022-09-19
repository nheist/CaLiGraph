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
    parser.add_argument('-it', '--ignore_tags', action=argparse.BooleanOptionalAction, default=False, help='Only predict entity mentions and ignore POS tags')
    parser.add_argument('-nss', '--negative_sample_size', type=float, default=0.0, help='Ratio of artificial negative examples')
    parser.add_argument('-dns', '--disable_negative_sampling', action=argparse.BooleanOptionalAction, default=False, help='Disable negative sampling in fine-tuning on pages')
    parser.add_argument('-sic', '--single_item_chunks', action=argparse.BooleanOptionalAction, default=False, help='Use only one item per chunk')
    parser.add_argument('-tlp', '--train_on_listpages', action=argparse.BooleanOptionalAction, default=True, help='Train on list page data')
    parser.add_argument('-tp', '--train_on_pages', action=argparse.BooleanOptionalAction, default=False, help='Train on page data')
    parser.add_argument('-sa', '--save_as', type=str, default=None, help='Name of the model to save as')
    parser.add_argument('-gm', '--gpu_memory', type=int, default=47, help='Amount of GPU memory to reserve')
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
    # reserve gpu until dataset is loaded
    import utils
    utils.reserve_gpu(args.gpu_memory)
    # then import application-specific code and run it
    from impl.subject_entity.mention_detection.evaluation import run_evaluation
    run_evaluation(args.model_name, args.epochs, args.batch_size, args.learning_rate, args.warmup_steps, args.weight_decay, args.ignore_tags, args.negative_sample_size, args.disable_negative_sampling, args.single_item_chunks, args.train_on_listpages, args.train_on_pages, args.save_as)
