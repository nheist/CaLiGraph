import argparse
import configargparse
import os


VERSION = 1

# TODO: ANALYSIS
# > investigate the errors made
# TODO: SEQUENCE CONTEXT
# > experiment with different item/entity layouts (E4.1/4.2/4.4)
# TODO: Bi-Encoder
# > Implement hard-sampling (lexically similar but unrelated items/ents batched together to get fewer candidates)
# > analyze currently used pooling function in bi-encoder (possible improvements in E4.4)
# TODO: FUSION
# Top-Down Fusion
# > graph partitioning algorithm -> Holland-sameAs
# Holland-sameAs: https://link.springer.com/chapter/10.1007/978-3-030-00671-6_23
# Implementation: https://github.com/dwslab/melt/blob/e94287f1349217e04cdb3a6b6565f3345f216b45/matching-jena-matchers/src/main/java/de/uni_mannheim/informatik/dws/melt/matching_jena_matchers/multisource/clustering/ComputeErrDegree.java
# > Filtering (consistent alignment for every listing) -> before or after Fusion?
#  > e.g. apply WWW-Approach before
#  > e.g. for every listing, collect intermediate entity representation and go for majority
# TODO: COMPARSION
# > apply to NILK dataset
#   > generalize implementation to mention_id and mention_context
#   > convert NILK data (https://zenodo.org/record/6607514#.Y1JyOi8Rr0o) to processable format
#   > compare approach
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
    parser.add_argument('corpus', type=str, choices=['LIST', 'NILK'], help='Data corpus to use for the experiments')
    parser.add_argument('-sa', '--save_alignment', action=argparse.BooleanOptionalAction, default=False, help='Whether to save the produced alignment for train/val/test')
    parser.add_argument('-ss', '--sample_size', type=int, choices=list(range(101)), default=5, help='Percentage of dataset to use')
    # matchers needing candidates
    parser.add_argument('--mm_approach', type=str, help='Mention-mention approach (ID) used for candidate generation')
    parser.add_argument('--me_approach', type=str, help='Mention-entity approach (ID) used for candidate generation')
    # bi/cross-encoder
    parser.add_argument('-bm', '--base_model', type=str, default='all-MiniLM-L12-v2', help='Base model used for the bi/cross-encoder')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs for training the bi/cross-encoder')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Number of warmup steps for training the bi/cross-encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training the bi/cross-encoder')
    parser.add_argument('-apc', '--add_page_context', action=argparse.BooleanOptionalAction, default=True, help='Use page context for disambiguation (M)')
    parser.add_argument('-atc', '--add_text_context', type=int, default=0, help='Other listing entities to append for disambiguation (M)')
    parser.add_argument('-aea', '--add_entity_abstract', action=argparse.BooleanOptionalAction, default=True, help='Use entity abstract for disambiguation (E)')
    parser.add_argument('-aki', '--add_kg_info', type=int, default=0, help='Types/properties to add from KG for disambiguation (E)')
    # bi-encoder
    parser.add_argument('-l', '--loss', type=str, choices=['COS', 'RL', 'SRL'], default='SRL', help='Loss function for training (only for bi-encoder)')
    parser.add_argument('-k', '--top_k', type=int, default=3, help='Number of ME matches to return per input (only for bi-encoder)')
    # cross-encoder
    parser.add_argument('--mm_threshold', type=float, default=.5, help="Confidence threshold to filter MM predictions.")
    parser.add_argument('--me_threshold', type=float, default=.5, help="Confidence threshold to filter ME predictions.")
    # top-down fusion
    parser.add_argument('--mm_weight', type=float, help='Weight of mention-mention approach used for fusion')
    parser.add_argument('--me_weight', type=float, help='Weight of mention-entity approach used for fusion')
    # bottom-up fusion
    parser.add_argument('--cluster_comparisons', type=int, default=2, help='Number of mentions/entities per cluster that are considered for a merge')
    parser.add_argument('--cluster_threshold', type=float, default=.5, help='Confidence threshold to filter cluster merges')

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
    approach_id = args.approach_id or f'{args.corpus}{args.sample_size}v{VERSION}{args.scenario}{args.approach[:2]}{str(random.randint(0, 999)).zfill(3)}'
    SEED = 310
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # initialize parameters
    from entity_linking.entity_disambiguation.matching import MatchingScenario, MatchingApproach
    from entity_linking.entity_disambiguation.data import CorpusType
    scenario = MatchingScenario(args.scenario)
    approach = MatchingApproach(args.approach)
    corpus_type = CorpusType(args.corpus)
    params = {
        'id': approach_id,
        'base_model': args.base_model,
        'loss': args.loss,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'batch_size': args.batch_size,
        'top_k': args.top_k,
        'add_page_context': args.add_page_context,
        'add_text_context': args.add_text_context,
        'add_entity_abstract': args.add_entity_abstract,
        'add_kg_info': args.add_kg_info,
        'mm_approach': args.mm_approach,
        'me_approach': args.me_approach,
        'mm_threshold': args.mm_threshold,
        'me_threshold': args.me_threshold,
        'mm_weight': args.mm_weight,
        'me_weight': args.me_weight,
        'cluster_comparisons': args.cluster_comparisons,
        'cluster_threshold': args.cluster_threshold,
    }
    # then import application-specific code and run it
    from entity_linking import entity_disambiguation
    entity_disambiguation.run_evaluation(scenario, approach, corpus_type, args.sample_size, params, args.save_alignment)
