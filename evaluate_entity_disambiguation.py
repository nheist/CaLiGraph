import argparse
import os


if __name__ == '__main__':
    os.environ['DISABLE_SPACY_CACHE'] = '1'  # make sure that spaCy cache is disabled
    # make relevant imports and parse all the arguments
    from entity_linking.entity_disambiguation.matching import MatchingScenario, MatchingApproach
    parser = argparse.ArgumentParser(description='Run the evaluation of entity disambiguation approaches.')
    parser.add_argument('gpu', type=int, choices=range(8), help='Number of GPU to use')
    parser.add_argument('scenario', type=str, choices=[s.value for s in MatchingScenario], help='Whether to match mention-mention or mention-entity')
    parser.add_argument('approach', type=str, choices=[a.value for a in MatchingApproach], help='Approach used for matching')
    parser.add_argument('-ba', '--blocking_approach', type=str, help='Matcher (ID) to retrieve candidates from')
    parser.add_argument('-sa', '--save_alignment', action=argparse.BooleanOptionalAction, default=False, help='Whether to save the produced alignment for train/val/test')
    args = parser.parse_args()
    # and set necessary environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # then fix all seeds
    import random
    import numpy as np
    import torch
    SEED = 310
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # initialize parameters
    scenario = MatchingScenario(args.scenario)
    approach = MatchingApproach(args.approach)
    params = {'version': 1}
    if args.blocking_approach is not None:
        params['blocking_approach'] = args.blocking_approach
    # then import application-specific code and run it
    from entity_linking import entity_disambiguation
    entity_disambiguation.run_evaluation(scenario, approach, params, args.save_alignment)
