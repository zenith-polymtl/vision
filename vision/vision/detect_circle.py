#!/usr/bin/env python3  
import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from std_msgs.msg import Bool  
from cv_bridge import CvBridge  
import cv2  
import numpy as np  
from custom_interfaces.msg import Pixels
  
class ZedImageSubscriber(Node):  
    def __init__(self):  
        super().__init__('zed_image_subscriber')  
          
        # Initialize CV Bridge  
        self.bridge = CvBridge()  

        self.detection_active = False
          
        # Subscribe to the rectified RGB color image topic  
        self.subscription = self.create_subscription(  
            Image,  
            '/zed/zed_node/rgb/image_rect_color',  
            self.image_callback,  
            10  
        )  

        self.activation_sub = self.create_subscription(  
            Bool,  
            '/vision/detect_circle_activate',  
            self.activation_callback,  
            10  
        )

        self.pixels_pub = self.create_publisher(  
            Pixels,  
            '/vision/detected_circle_pixels',  
            10  
        )

        self.get_logger().info('Circle detection node started')  
    
    def activation_callback(self, msg):  
        self.detection_active = msg.data
        if self.detection_active:
            self.get_logger().info('Circle detection activated')
        else:
            self.get_logger().info('Circle detection deactivated')
              
    def extract_circles(self, cv_image):
        # DO CIRCLE DETECTION and output pixels
        # TODO: Implement circle detection logic here
        if True:
            pixels = (0, 0)  # Placeholder
            return pixels
        else:
            return None
    
    def image_callback(self, msg):  
        # Store the last received image
        if self.detection_active:
            self.last_image = msg  
            self.get_logger().info(  
                f'Received RGB image: {msg.width}x{msg.height}, '  
                f'encoding: {msg.encoding}, '  
                f'step: {msg.step}'  
            )  

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
            pixels = self.extract_circles(cv_image) 

            if pixels is not None:
                self.get_logger().info(f'Circle detected at pixels: {pixels}') 
                pixel_msg = Pixels()
                pixel_msg.x = pixels[0]
                pixel_msg.y = pixels[1]
                self.pixels_pub.publish(pixel_msg) 
            
              


  
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