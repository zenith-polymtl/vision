#!/usr/bin/env python3  
import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from std_msgs.msg import Bool  
from cv_bridge import CvBridge  
import cv2  
import numpy as np 
from pathlib import Path 
  
class ZedImageSubscriber(Node):  
    def __init__(self):  
        super().__init__('zed_image_subscriber')  
          
        # Initialize CV Bridge  
        self.bridge = CvBridge()  
          
        # Subscribe to the rectified RGB color image topic  
        self.subscription = self.create_subscription(  
            Image,  
            '/zed/zed_node/rgb/color/rect/image',  
            self.image_callback,  
            10  
        )  
  
        self.download_sub = self.create_subscription(  
            Bool,  
            '/vision/download_image',  
            self.download_callback,  
            10  
        )

        self.output_dir = Path("saved_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)  
        
        self.image_number = 0  
        self.last_image = None  
        self.get_logger().info('ZED RGB Image Subscriber started')  
      
    def download_callback(self, msg):
        if msg.data and self.last_image is not None:
            filename = f"training_{self.image_number}.png"
            full_path = self.output_dir / filename 
            self.save_image(self.last_image, str(full_path))
            self.image_number += 1

    def save_image(self, ros_image, filepath):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

            ok = cv2.imwrite(filepath, cv_image)
            if ok:
                self.get_logger().info(f"Saved image: {filepath}")
            else:
                self.get_logger().error(f"cv2.imwrite failed for: {filepath}")

        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
      
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