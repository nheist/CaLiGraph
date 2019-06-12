import traceback
import util
import mailer
import impl.list.store as list_store
import impl.list.hierarchy as list_hierarchy
import impl.list.base as list_base
#import impl.dbpedia.heuristics as dbp_heur
#import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util
import impl.category.wikitaxonomy as cat_wikitax

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        #entity_data = list_base.get_listpage_entity_data()
        #nlp_util.persist_cache()
        #entity_data.to_csv('data_caligraph/entities_train_v5.csv', index=False)

        #edges_found = len(cat_wikitax.get_valid_edges())

        list_hierarchy.get_equivalent_listpage('')

        mailer.send_success(f'FINISHED equivalent listpage extraction')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
