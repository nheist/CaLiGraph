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
import impl.list.features as list_features
#import impl.dbpedia.heuristics as dbp_heur
#import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util
import impl.caligraph.base as cali_base
import impl.caligraph.serialize as cali_serialize


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
        util.get_logger().info('Starting caligraph v11..')

        #graph = cali_base.get_axiom_graph()
        # recompute entity labels
        #enum_features = list_base.get_enum_listpage_entity_features(graph)
        #util.get_logger().debug('Before relabeling (enum)')
        #util.get_logger().debug(f'True: {len(enum_features[enum_features["label"] == 1])}')
        #util.get_logger().debug(f'False: {len(enum_features[enum_features["label"] == 0])}')
        #util.get_logger().debug(f'New: {len(enum_features[enum_features["label"] == -1])}')
#
        #list_features.assign_entity_labels(graph, enum_features)
        #util.get_logger().debug('After relabeling (enum)')
        #util.get_logger().debug(f'True: {len(enum_features[enum_features["label"] == 1])}')
        #util.get_logger().debug(f'False: {len(enum_features[enum_features["label"] == 0])}')
        #util.get_logger().debug(f'New: {len(enum_features[enum_features["label"] == -1])}')
        #util.update_cache('dbpedia_listpage_enum_features', enum_features, version=10)
#
        #table_features = list_base.get_table_listpage_entity_features(graph)
        #util.get_logger().debug('Before relabeling (table)')
        #util.get_logger().debug(f'True: {len(table_features[table_features["label"] == 1])}')
        #util.get_logger().debug(f'False: {len(table_features[table_features["label"] == 0])}')
        #util.get_logger().debug(f'New: {len(table_features[table_features["label"] == -1])}')
#
        #list_features.assign_entity_labels(graph, table_features)
        #util.get_logger().debug('After relabeling (table)')
        #util.get_logger().debug(f'True: {len(table_features[table_features["label"] == 1])}')
        #util.get_logger().debug(f'False: {len(table_features[table_features["label"] == 0])}')
        #util.get_logger().debug(f'New: {len(table_features[table_features["label"] == -1])}')
        #util.update_cache('dbpedia_listpage_table_features', table_features, version=10)



        # extract table features
        #list_base.get_table_listpage_entity_features()
        #nlp_util.persist_cache()


        # extract complete caligraph
        setup()
        cali_base.serialize_final_graph()

        #cat_graph = cat_base.get_merged_graph()
        #util.get_logger().info('catgraph done.')
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

        mailer.send_success(f'FINISHED caligraph v11')
        util.get_logger().info('Finished caligraph v11.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
