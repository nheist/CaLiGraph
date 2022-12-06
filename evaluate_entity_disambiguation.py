import argparse
import configargparse
import os


VERSION = 1


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Run the evaluation of entity disambiguation approaches.')
    # config
    parser.add_argument('-c', '--config', is_config_file=True, help='Path to config file')
    # machine-specific
    parser.add_argument('gpu', type=int, choices=range(-1, 8), help='Number of GPU to use')
    parser.add_argument('-gm', '--gpu_memory', type=int, default=46, help='Amount of GPU memory to reserve')
    # general matching
    parser.add_argument('scenario', type=str, choices=['MM', 'ME', 'F'], help='Mention-mention, mention-entity, or fusion matching')
    parser.add_argument('approach', type=str, help='Approach used for matching')
    parser.add_argument('corpus', type=str, choices=['LIST', 'NILK'], help='Data corpus to use for the experiments')
    parser.add_argument('-sa', '--save_alignment', action=argparse.BooleanOptionalAction, default=False, help='Whether to save the produced alignment for train/val/test')
    parser.add_argument('-sta', '--save_test_alignment', action=argparse.BooleanOptionalAction, default=False, help='Whether to save the produced alignment for test')
    parser.add_argument('-ss', '--sample_size', type=int, choices=list(range(5, 101, 5)), default=5, help='Percentage of dataset to use')
    # bi-encoder
    parser.add_argument('-l', '--loss', type=str, choices=['COS', 'RL', 'SRL'], default='SRL', help='Loss function for training (only for bi-encoder)')
    parser.add_argument('-k', '--top_k', type=int, default=4, help='Number of ME matches to return per input (only for bi-encoder)')
    parser.add_argument('-ans', '--approximate_neighbor_search', action=argparse.BooleanOptionalAction, default=False, help='Use approximate nearest neighbor search')
    # bi/cross-encoder
    parser.add_argument('-bm', '--base_model', type=str, default='all-MiniLM-L12-v2', help='Base model used for the bi/cross-encoder')
    parser.add_argument('-ts', '--train_sample', type=int, default=1, help='Sample size to use for training (in millions)')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs for training the bi/cross-encoder')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Number of warmup steps for training the bi/cross-encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training the bi/cross-encoder')
    parser.add_argument('-apc', '--add_page_context', action=argparse.BooleanOptionalAction, default=True, help='Use page context for disambiguation (M)')
    parser.add_argument('-atc', '--add_text_context', type=int, default=0, help='Other listing entities to append for disambiguation (M)')
    parser.add_argument('-aea', '--add_entity_abstract', action=argparse.BooleanOptionalAction, default=True, help='Use entity abstract for disambiguation (E)')
    parser.add_argument('-aki', '--add_kg_info', type=int, default=0, help='Types/properties to add from KG for disambiguation (E)')
    # matchers needing candidates
    parser.add_argument('--mm_approach', type=str, help='Mention-mention approach (ID) used for candidate generation')
    parser.add_argument('--me_approach', type=str, help='Mention-entity approach (ID) used for candidate generation')
    # nastylinker / edin
    parser.add_argument('--mm_threshold', type=float, default=0.0, help="Confidence threshold to filter MM predictions.")
    parser.add_argument('--me_threshold', type=float, default=0.0, help="Confidence threshold to filter ME predictions.")
    parser.add_argument('--path_threshold', type=float, default=0.0, help="Confidence threshold to filter graph paths (NastyLinker only).")
    parser.add_argument('--me_cluster_threshold', type=float, default=0.0, help="Confidence threshold to filter graph paths (EDIN only).")

    args = parser.parse_args()
    # and set necessary environment variables
    os.environ['DISABLE_SPACY_CACHE'] = '1'  # make sure that spaCy cache is disabled
    import utils
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        utils.reserve_gpu(args.gpu_memory)
    # then fix all seeds
    import random
    import numpy as np
    import torch
    approach_id = f'{args.corpus}{args.sample_size}v{VERSION}{args.scenario}{args.approach[:2]}{str(random.randint(0, 999)).zfill(3)}'
    SEED = 310
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # initialize parameters
    from impl.subject_entity.entity_disambiguation.matching import MatchingScenario, MatchingApproach
    from impl.subject_entity.entity_disambiguation.data import CorpusType
    scenario = MatchingScenario(args.scenario)
    approach = MatchingApproach(args.approach)
    corpus_type = CorpusType(args.corpus)
    params = {
        'id': approach_id,
        'base_model': args.base_model,
        'train_sample': args.train_sample,
        'loss': args.loss,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'batch_size': args.batch_size,
        'top_k': args.top_k,
        'approximate_neighbor_search': args.approximate_neighbor_search,
        'add_page_context': args.add_page_context,
        'add_text_context': args.add_text_context,
        'add_entity_abstract': args.add_entity_abstract,
        'add_kg_info': args.add_kg_info,
        'mm_approach': args.mm_approach,
        'me_approach': args.me_approach,
        'mm_threshold': args.mm_threshold,
        'me_threshold': args.me_threshold,
        'path_threshold': args.path_threshold,
        'me_cluster_threshold': args.me_cluster_threshold,
    }
    # then import application-specific code and run it
    from impl.subject_entity import mention_detection
    from impl.subject_entity.entity_disambiguation.matching import initialize_matcher
    from impl.subject_entity.entity_disambiguation.data import get_data_corpora
    from impl.subject_entity.entity_disambiguation.matching.io import store_candidate_alignment
    if corpus_type == CorpusType.LIST:  # make sure subject entities are initialized
        mention_detection.detect_mentions()
    matcher = initialize_matcher(scenario, approach, params)
    train_data, _, test_data = get_data_corpora(corpus_type, args.sample_size)

    alignments = matcher.train(train_data, args.save_alignment)
    alignments |= matcher.test(test_data)
    if args.save_alignment or args.save_test_alignment:
        store_candidate_alignment(matcher.get_approach_name(), alignments)
    utils.get_logger().info('Evaluation successful.')
