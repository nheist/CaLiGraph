import traceback
import util
import mailer
#import impl.list.base as list_base
#import impl.dbpedia.heuristics as dbp_heur
import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        util.get_logger().info(cat_axioms.get_axioms('http://dbpedia.org/resource/Category:Writers'))
        nlp_util.persist_cache()

        mailer.send_success('FINISHED heuristics extraction')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
