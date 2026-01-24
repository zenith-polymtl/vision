#!/usr/bin/env python3  
import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from std_msgs.msg import Bool  
from cv_bridge import CvBridge  
import cv2  
import numpy as np  
  
class ZedImageSubscriber(Node):  
    def __init__(self):  
        super().__init__('zed_image_subscriber')  
          
        # Initialize CV Bridge  
        self.bridge = CvBridge()  
          
        # Subscribe to the rectified RGB color image topic  
        self.subscription = self.create_subscription(  
            Image,  
            '/zed/zed_node/rgb/image_rect_color',  
            self.image_callback,  
            10  
        )  
  
        self.download_sub = self.create_subscription(  
            Bool,  
            '/vision/download_image',  
            self.download_callback,  
            10  
        )  
        
        self.image_number = 0  
        self.last_image = None  
        self.get_logger().info('ZED RGB Image Subscriber started')  
      
    def download_callback(self, msg):  
        if msg.data and self.last_image is not None:  
            # Save the last received image to a file  
            filename = f'shoot_confirmation_{self.image_number}.png'  
            self.save_image(self.last_image, filename)  
            self.image_number += 1  
              
    def save_image(self, ros_image, filename):  
        """Convert ROS Image to OpenCV and save to file"""  
        try:  
            # Convert ROS Image to OpenCV format  
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')  
              
            # Save the image  
            cv2.imwrite(filename, cv_image)  
            self.get_logger().info(f'Saved image: {filename}')  
              
        except Exception as e:  
            self.get_logger().error(f'Failed to save image: {e}')  
      
    def image_callback(self, msg):  
        # Store the last received image  
        self.last_image = msg  
        self.get_logger().info(  
            f'Received RGB image: {msg.width}x{msg.height}, '  
            f'encoding: {msg.encoding}, '  
            f'step: {msg.step}'  
        )  
  
def main(args=None):  
    rclpy.init(args=args)  
    node = ZedImageSubscriber()  
      
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        pass  
    finally:  
        node.destroy_node()  
        rclpy.shutdown()  
  
if __name__ == '__main__':  
    main()