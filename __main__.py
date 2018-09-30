import caligraph.base as caligraph_base
import caligraph.category.evaluation as catgraph_eval


if __name__ == '__main__':
    cat_graph = caligraph_base.get_cycle_free_category_graph()
    cat_graph._assign_resource_type_counts()

    catgraph_eval.test_settings(cat_graph)
