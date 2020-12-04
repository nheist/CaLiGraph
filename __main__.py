import traceback
import utils
import mailer
from impl import wikipedia
import impl.category.base as cat_base
import impl.category.cat2ax as cat_axioms
import impl.util.hypernymy as hypernymy_util
import impl.caligraph.base as cali_base


def _setup_hypernyms():
    """Initialisation of hypernyms that are extracted from Wikipedia categories using Cat2Ax axioms."""
    if utils.load_cache('wikitaxonomy_hypernyms') is not None:
        return  # only compute hypernyms if they are not existing already
    ccg = cat_base.get_conceptual_category_graph()
    # initialise cat2ax axioms
    cat2ax_axioms = cat_axioms.extract_category_axioms(ccg)
    utils.update_cache('cat2ax_axioms', cat2ax_axioms)
    # initialise wikitaxonomy hypernyms
    wikitaxonomy_hypernyms = hypernymy_util.compute_hypernyms(ccg)
    utils.update_cache('wikitaxonomy_hypernyms', wikitaxonomy_hypernyms)


if __name__ == '__main__':
    try:
        utils.get_logger().info('Starting extraction of CaLiGraph.')

        # prepare resources like type lexicalisations from hearst patterns and wikitaxonomy hypernyms
        utils.get_logger().info('Preparing resources..')
        wikipedia.extract_wiki_corpus_resources()
        _setup_hypernyms()

        # run the complete extraction cycle and end with serializing CaLiGraph
        utils.get_logger().info('Running extraction..')
        cali_base.serialize_final_graph()

        success_msg = 'Finished extraction of CaLiGraph.'
        mailer.send_success(success_msg)
        utils.get_logger().info(success_msg)
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        utils.get_logger().error(error_msg)
        raise e
