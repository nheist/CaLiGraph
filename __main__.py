import traceback
import util
import mailer
import impl.list.base as list_base

if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        mailer.send_success('WHAT WHAT WHAT')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
