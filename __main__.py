import util
import relation_test
import caligraph.category.base as cat_base

if __name__ == '__main__':
    util.get_logger().info('Starting CaLiGraph extraction..')
    relation_test.evaluate_parameters()

    # G = cat_base.get_dbp_typed_category_graph()
    # util.get_logger().info(G.statistics)

    util.get_logger().info('Finished CaLiGraph extraction.')
