import traceback
import util
import mailer
import impl.category.base as cat_base
import impl.category.cat2ax as cat_axioms
import impl.util.hypernymy as hypernymy_util
import impl.caligraph.base as cali_base
import impl.util.nlp as nlp_util


def _setup_hypernyms():
    """Initialisation of hypernyms that are extracted from Wikipedia categories using Cat2Ax axioms."""
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
        util.get_logger().info('Starting extraction of CaLiGraph version 1.')

        _setup_hypernyms()  # initialise hypernyms
        cali_base.serialize_final_graph()  # run the complete extraction cycle and end with serializing CaLiGraph
        nlp_util.persist_cache()  # persist the spaCy cache to rerun extraction more quickly

        success_msg = 'Starting extraction of CaLiGraph version 1.'
        mailer.send_success(success_msg)
        util.get_logger().info(success_msg)
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
