import argparse
import configargparse
import os

# TODO: merging of MM and ME results (remove edges by confidence, MM/ME first, classifier, graph partitioning algorithm?)
# TODO: lexical model with more recall (levenshtein distance, n-grams) or other baselines
# TODO: symbolic approach from Manni?
if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Run the evaluation of entity disambiguation approaches.')
    # config
    parser.add_argument('-c', '--config', is_config_file=True, help='Path to config file')
    # machine-specific
    parser.add_argument('gpu', type=int, choices=range(-1, 8), help='Number of GPU to use')
    parser.add_argument('-gm', '--gpu_memory', type=int, default=47, help='Amount of GPU memory to reserve')
    # general matching
    parser.add_argument('scenario', type=str, help='Whether to match mention-mention or mention-entity')
    parser.add_argument('approach', type=str, help='Approach used for matching')
    parser.add_argument('-sa', '--save_alignment', action=argparse.BooleanOptionalAction, default=False, help='Whether to save the produced alignment for train/val/test')
    # matchers needing candidates
    parser.add_argument('-ba', '--blocking_approach', type=str, help='Matcher (ID) to retrieve candidates from')
    # bi/cross-encoder
    parser.add_argument('-bm', '--base_model', type=str, default='all-MiniLM-L12-v2', help='Base model used for the bi/cross-encoder')
    parser.add_argument('-l', '--loss', type=str, choices=['COS', 'RL', 'SRL'], default='RL', help='Loss function for training the bi/cross-encoder')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs for training the bi/cross-encoder')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Number of warmup steps for training the bi/cross-encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training the bi/cross-encoder')
    parser.add_argument('-k', '--top_k', type=int, default=3, help='Number of matches to return per input (only for bi-encoder)')
    parser.add_argument('-apc', '--add_page_context', action=argparse.BooleanOptionalAction, default=False, help='Use page context for disambiguation (M)')
    parser.add_argument('-ale', '--add_listing_entities', type=int, default=0, help='Other listing entities to append for disambiguation (M)')
    parser.add_argument('-aea', '--add_entity_abstract', action=argparse.BooleanOptionalAction, default=False, help='Use entity abstract for disambiguation (E)')
    parser.add_argument('-aki', '--add_kg_info', type=int, default=0, help='Types/properties to add from KG for disambiguation (E)')

    args = parser.parse_args()
    # and set necessary environment variables
    os.environ['DISABLE_SPACY_CACHE'] = '1'  # make sure that spaCy cache is disabled
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        import utils
        utils.reserve_gpu(args.gpu_memory)
    # then fix all seeds
    import random
    import numpy as np
    import torch
    SEED = 310
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # initialize parameters
    from entity_linking.entity_disambiguation.matching.util import MatchingScenario, MatchingApproach
    scenario = MatchingScenario(args.scenario)
    approach = MatchingApproach(args.approach)
    params = {
        'version': 1,
        'blocking_approach': args.blocking_approach,
        'base_model': args.base_model,
        'loss': args.loss,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'batch_size': args.batch_size,
        'top_k': args.top_k,
        'add_page_context': args.add_page_context,
        'add_listing_entities': args.add_listing_entities,
        'add_entity_abstract': args.add_entity_abstract,
        'add_kg_info': args.add_kg_info,
    }
    # then import application-specific code and run it
    from entity_linking import entity_disambiguation
    entity_disambiguation.run_evaluation(scenario, approach, params, args.save_alignment)
