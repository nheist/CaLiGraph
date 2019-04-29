import traceback
import util
import mailer
from impl.category import cat2ax

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        cat2ax.run_extraction()

        mailer.send_success('FINISHED Cat2Ax')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
