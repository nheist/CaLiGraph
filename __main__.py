import relation_test
from caligraph.category.base import get_cycle_free_category_graph


if __name__ == '__main__':
    # relation_test.evaluate_category_relations()
    G = get_cycle_free_category_graph()
    G.assign_dbp_types()
    print(G.statistics)
