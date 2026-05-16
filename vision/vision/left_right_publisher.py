import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Header, Empty, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interfaces.msg import UiMessage, LeftRightScene

import cv2
import numpy as np
import message_filters
from ultralytics import YOLO
import os

class StereoYOLONode(Node):

    def __init__(self):
        super().__init__('stereo_yolo_node')
        
        qos_reliable = self._create_qos_profile(QoSReliabilityPolicy.RELIABLE)
        qos_best_effort = self._create_qos_profile(QoSReliabilityPolicy.BEST_EFFORT)

        # --- Trigger Logic ---
        self.trigger_requested = False
        self.trigger_sub = self.create_subscription(
            Bool,
            '/aeac/external/describe_scene',
            self.trigger_callback,
            10
        )
        
        self.ui_message_pub = self.create_publisher(
            UiMessage,
            '/aeac/external/send_to_ui',
            qos_reliable
        )
        
        self.left_right_pub = self.create_publisher(
            LeftRightScene,
            '/aeac/external/left_right_scene',
            qos_best_effort
        )

        self.left_sub = message_filters.Subscriber(
            self, CompressedImage, '/zed/zed_node/left/color/rect/image'
        )
        self.right_sub = message_filters.Subscriber(
            self, CompressedImage, '/zed/zed_node/right/color/rect/image'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], 
            queue_size=10, 
            slop=0.05 
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Stereo YOLO Node Started. Waiting for trigger on /aeac/internal/describe_scene...")


    @staticmethod
    def _create_qos_profile(reliability_policy):
        return QoSProfile(
            reliability=reliability_policy,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

    def trigger_callback(self, msg):
        self.get_logger().info("Trigger received! Processing next available stereo pair...")
        self.trigger_requested = True

    def sync_callback(self, left_msg, right_msg):
        if not self.trigger_requested:
            return
        
        msg = LeftRightScene()
        msg.left_image = left_msg
        msg.right_msg = right_msg
        self.left_right_pub.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = StereoYOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()