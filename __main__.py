import relation_test
import caligraph.category.base as cat_base
import caligraph.wikidata.store as wikidata_store
import util

if __name__ == '__main__':
    # relation_test.evaluate_category_relations()

    G = cat_base.get_dbp_typed_category_graph()
    util.get_logger().info(G.statistics)

    # print('RESULT: {}'.format(wikidata_store.resource_has_type('http://dbpedia.org/resource/Cristiano_Ronaldo', 'http://dbpedia.org/ontology/Person')))
