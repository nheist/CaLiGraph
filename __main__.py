import traceback
import util
import mailer
import impl.list.base as list_base
import impl.category.axioms as cat_axioms
# from impl.category.evaluation import test_metrics

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        # TASK: extraction of category axioms
        #cat_axioms.extract_axioms_and_relation_assertions()
        # TASK: create cataxioms data
        def get_data_X_y(version):
            candidates = util.load_or_create_cache('cataxioms_candidates', cat_axioms._compute_candidate_axioms, version=version)
            X, y = cat_axioms._create_goldstandard(candidates)
            return candidates, X, y
        get_data_X_y('2b')

        # TASK: evaluation of assigned dbp-types to catgraph
        # graph = util.load_cache('catgraph_cyclefree')
        # graph._assign_resource_type_counts()
        # test_metrics(graph)

        # TASK: listpage markup extraction
        #list_base.get_parsed_listpages()

        mailer.send_success('extracted cataxioms markup.')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
