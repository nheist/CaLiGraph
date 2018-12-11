import traceback
import util
import mailer
import impl.category.axioms as cat_axioms

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        cat_axioms.extract_axioms_and_relation_assertions()

        mailer.send_success()
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
