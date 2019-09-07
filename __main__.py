import traceback
import util
import mailer
import impl.category.base as cat_base
import impl.category.cat2ax as cat_axioms
import impl.util.hypernymy as hypernymy_util
import impl.list.store as list_store
import impl.list.base as list_base
import impl.list.parser as list_parser
import impl.list.mapping as list_mapping
#import impl.dbpedia.heuristics as dbp_heur
#import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util
import impl.caligraph.base as cali_base


def setup():
    category_graph = cat_base.get_conceptual_category_graph()

    # initialise cat2ax axioms
    cat2ax_confidence = util.get_config('cat2ax.pattern_confidence')
    cat2ax_axioms = cat_axioms.extract_category_axioms(category_graph, cat2ax_confidence)
    util.update_cache('cat2ax_axioms', cat2ax_axioms)

    # initialise wikitaxonomy hypernyms
    wikitaxonomy_hypernyms = hypernymy_util.compute_hypernyms(category_graph)
    util.update_cache('wikitaxonomy_hypernyms', wikitaxonomy_hypernyms)


if __name__ == '__main__':
    try:
        util.get_logger().info('Starting caligraph merge..')

        caligraph = cali_base.get_base_graph()
        util.get_logger().info(caligraph.statistics)

        #cat_graph = cat_base.get_merged_graph()
        #util.get_logger().info('catgraph done.')
        #nlp_util.persist_cache()
        #util.get_logger().info('cache persist done.')
        #list_graph = list_base.get_merged_listgraph()
        #util.get_logger().info('listgraph done.')
        #nlp_util.persist_cache()
        #util.get_logger().info('cache persist done.')
        #list_mapping.get_parent_categories('http://dbpedia.org/resource/Category:Lists_of_NASCAR_broadcasters')
        #util.get_logger().info('mapping done.')
        #nlp_util.persist_cache()
        #util.get_logger().info('cache persist done.')
        #list_base.get_listpage_entity_features()
        #util.get_logger().info('extraction done.')
        #nlp_util.persist_cache()
        #util.get_logger().info('cache persist done.')

        mailer.send_success(f'FINISHED caligraph merge')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
