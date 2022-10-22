import argparse
import configargparse
import os


VERSION = 4

# TODO: Fusion (graph partitioning algorithm -> Holland-sameAs) -> how big is the "problem"?
# Holland-sameAs: https://link.springer.com/chapter/10.1007/978-3-030-00671-6_23
# Implementation: https://github.com/dwslab/melt/blob/e94287f1349217e04cdb3a6b6565f3345f216b45/matching-jena-matchers/src/main/java/de/uni_mannheim/informatik/dws/melt/matching_jena_matchers/multisource/clustering/ComputeErrDegree.java
# TODO: Filtering (consistent alignment for every listing) -> before or after Fusion? e.g. apply WWW-Approach before
# TODO: lexical model with more recall (levenshtein distance, n-grams) or other baselines
if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Run the evaluation of entity disambiguation approaches.')
    # config
    parser.add_argument('-c', '--config', is_config_file=True, help='Path to config file')
    parser.add_argument('-id', '--approach_id', type=str, help='ID to identify the approach')
    # machine-specific
    parser.add_argument('gpu', type=int, choices=range(-1, 8), help='Number of GPU to use')
    parser.add_argument('-gm', '--gpu_memory', type=int, default=46, help='Amount of GPU memory to reserve')
    # general matching
    parser.add_argument('scenario', type=str, choices=['MM', 'ME', 'F'], help='Mention-mention, mention-entity, or fusion matching')
    parser.add_argument('approach', type=str, help='Approach used for matching')
    parser.add_argument('-sa', '--save_alignment', action=argparse.BooleanOptionalAction, default=False, help='Whether to save the produced alignment for train/val/test')
    # matchers needing candidates
    parser.add_argument('-ba', '--blocking_approach', type=str, help='Matcher (ID) to retrieve candidates from')
    # bi/cross-encoder
    parser.add_argument('-bm', '--base_model', type=str, default='all-MiniLM-L12-v2', help='Base model used for the bi/cross-encoder')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs for training the bi/cross-encoder')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Number of warmup steps for training the bi/cross-encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training the bi/cross-encoder')
    parser.add_argument('-apc', '--add_page_context', action=argparse.BooleanOptionalAction, default=False, help='Use page context for disambiguation (M)')
    parser.add_argument('-acc', '--add_category_context', action=argparse.BooleanOptionalAction, default=False, help='Use category context for disambiguation (M)')
    parser.add_argument('-ale', '--add_listing_entities', type=int, default=0, help='Other listing entities to append for disambiguation (M)')
    parser.add_argument('-aea', '--add_entity_abstract', action=argparse.BooleanOptionalAction, default=False, help='Use entity abstract for disambiguation (E)')
    parser.add_argument('-aki', '--add_kg_info', type=int, default=0, help='Types/properties to add from KG for disambiguation (E)')
    # bi-encoder
    parser.add_argument('-l', '--loss', type=str, choices=['COS', 'RL', 'SRL'], default='RL', help='Loss function for training (only for bi-encoder)')
    parser.add_argument('-k', '--top_k', type=int, default=3, help='Number of matches to return per input (only for bi-encoder)')
    # cross-encoder
    parser.add_argument('-t', '--confidence_threshold', type=float, default=.5, help="Confidence threshold to filter predictions.")
    # fusion
    parser.add_argument('-mma', '--mm_approach', type=str, help='Mention-mention approach (ID) used for fusion')
    parser.add_argument('-mea', '--me_approach', type=str, help='Mention-entity approach (ID) used for fusion')
    parser.add_argument('-mmw', '--mm_weight', type=str, help='Weight of mention-mention approach used for fusion')
    parser.add_argument('-mew', '--me_weight', type=str, help='Weight of mention-entity approach used for fusion')


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
    approach_id = args.approach_id or f'V{VERSION}{args.scenario}{args.approach[:2]}{str(random.randint(0, 999)).zfill(3)}'
    SEED = 310
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # initialize parameters
    from entity_linking.entity_disambiguation.matching.util import MatchingScenario, MatchingApproach
    scenario = MatchingScenario(args.scenario)
    approach = MatchingApproach(args.approach)
    params = {
        'id': approach_id,
        'blocking_approach': args.blocking_approach,
        'base_model': args.base_model,
        'loss': args.loss,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'batch_size': args.batch_size,
        'top_k': args.top_k,
        'add_page_context': args.add_page_context,
        'add_category_context': args.add_category_context,
        'add_listing_entities': args.add_listing_entities,
        'add_entity_abstract': args.add_entity_abstract,
        'add_kg_info': args.add_kg_info,
        'confidence_threshold': args.confidence_threshold,
        'mm_approach': args.mm_approach,
        'me_approach': args.me_approach,
        'mm_weight': args.mm_weight,
        'me_weight': args.me_weight,
    }
    # then import application-specific code and run it
    from entity_linking import entity_disambiguation
    entity_disambiguation.run_evaluation(scenario, approach, params, args.save_alignment)
