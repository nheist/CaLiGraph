import util
import relation_test
import caligraph.category.base as cat_base
import caligraph.wikidata.store as wikidata_store

if __name__ == '__main__':
    util.get_logger().info('Starting CaLiGraph extraction..')
    # relation_test.evaluate_category_relations()

    # G = cat_base.get_dbp_typed_category_graph()
    # util.get_logger().info(G.statistics)

    print('RESULT: {}'.format(wikidata_store.resource_has_type('http://dbpedia.org/resource/Cristiano_Ronaldo', 'http://dbpedia.org/ontology/Person')))

    util.get_logger().info('Finished CaLiGraph extraction.')
