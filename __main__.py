import traceback
import util
import mailer
import impl.category.axioms as cat_axioms
from impl.category.evaluation import test_metrics

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        # TASK: extraction of category axioms
        cat_axioms.extract_axioms_and_relation_assertions()

        # TASK: evaluation of assigned dbp-types to catgraph
        # graph = util.load_cache('catgraph_cyclefree')
        # graph._assign_resource_type_counts()
        # test_metrics(graph)

        mailer.send_success()
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
