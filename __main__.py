import traceback
import util
import mailer
import impl.caligraph.base as cali_base


if __name__ == '__main__':
    try:
        util.get_logger().info('Starting serialization of caligraph.')

        cali_base.serialize_final_graph()  # run the complete extraction cycle and end with serializing CaLiGraph

        success_msg = 'Finished serialization of caligraph.'
        mailer.send_success(success_msg)
        util.get_logger().info(success_msg)
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e
