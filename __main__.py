import relation_test
from caligraph.category.base import get_cycle_free_category_graph
import caligraph.wikidata.store as wikidata_store


if __name__ == '__main__':
    # relation_test.evaluate_category_relations()

    # G = get_cycle_free_category_graph()
    # G.assign_dbp_types()
    # print(G.statistics)

    print('RESULT: {}'.format(wikidata_store.resource_has_type('http://dbpedia.org/resource/Cristiano_Ronaldo', 'http://dbpedia.org/ontology/Person')))
