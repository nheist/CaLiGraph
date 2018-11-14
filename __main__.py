import util
import mailer
import relation_test
import caligraph.category.base as cat_base
import caligraph.dbpedia.heuristics as dbp_heuristics

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        # util.get_logger().info('TEST: {}'.format(dbp_heuristics.get_disjoint_types('http://dbpedia.org/ontology/Person')))

        relation_test.evaluate_parameters()

        # G = cat_base.get_dbp_typed_category_graph()
        # util.get_logger().info(G.statistics)

        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = str(e)
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
