import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
#'/zed/zed_node/rgb/color/rect/image'

class RGBImageSubscriber(Node):

    def __init__(self):
        super().__init__('rgb_image_subscriber')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10
        )

        self.get_logger().info("RGB Image Subscriber Node Started")

    def image_callback(self, msg):
        try:
            # Convert ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Show image
            cv2.imshow("ZED RGB Image", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RGBImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
