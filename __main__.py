import traceback
import util
import mailer
#import impl.list.base as list_base
import impl.dbpedia.heuristics as dbp_heur

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        util.get_logger().info(dbp_heur.get_disjoint_types('http://dbpedia.org/ontology/School'))

        mailer.send_success('FINISHED heuristics extraction')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
