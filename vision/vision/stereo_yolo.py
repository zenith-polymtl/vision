import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Header, Empty, Bool
from zed_msgs.msg import Object, ObjectsStamped, BoundingBox2Di
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interfaces.msg import UiMessage

import cv2
import numpy as np
import message_filters
from ultralytics import YOLO
import os

class StereoYOLONode(Node):

    def __init__(self):
        super().__init__('stereo_yolo_node')
        self.bridge = CvBridge()

        # --- Trigger Logic ---
        self.trigger_requested = False
        self.trigger_sub = self.create_subscription(
            Empty,
            '/aeac/internal/auto_approach/detect_target',
            self.trigger_callback,
            10
        )
        
        # Save Configuration
        self.save_dir = os.path.expanduser('/water_ws/Pictures/Stereo')
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0

        # Camera Params
        self.BASELINE = 0.12
        self.f_pixel = None
        self.cx = None
        self.cy = None
        
        qos_reliable = self._create_qos_profile(QoSReliabilityPolicy.RELIABLE)

        
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/left/color/rect/camera_info',
            self.info_callback,
            10
        )

        self.objects_stamped_pub = self.create_publisher(
            ObjectsStamped, 
            '/aeac/internal/auto_approach/target_detected', 
            10
        )
        
        self.auto_approach_pub = self.create_publisher(
            Bool,
            '/aeac/external/auto_approach/start',
            qos_reliable
        )
        
        self.ui_message_pub = self.create_publisher(
            UiMessage,
            '/aeac/external/send_to_ui',
            qos_reliable
        )

        # Load YOLO model
        pkg_share = get_package_share_directory('vision') 
        model_path = os.path.join(pkg_share, 'models', 'best-medium.pt')
        self.get_logger().info(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)

        self.left_sub = message_filters.Subscriber(
            self, Image, '/zed/zed_node/left/color/rect/image'
        )
        self.right_sub = message_filters.Subscriber(
            self, Image, '/zed/zed_node/right/color/rect/image'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], 
            queue_size=10, 
            slop=0.05 
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Stereo YOLO Node Started. Waiting for trigger on /aeac/trigger_detection...")


    @staticmethod
    def _create_qos_profile(reliability_policy):
        return QoSProfile(
            reliability=reliability_policy,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

    def trigger_callback(self, msg):
        """When a message hits this topic, we 'arm' the detector for one frame."""
        self.get_logger().info("Trigger received! Processing next available stereo pair...")
        self.trigger_requested = True

    def info_callback(self, msg):
        if self.f_pixel is None:
            self.f_pixel = msg.k[0]  
            self.cx = msg.k[2]       
            self.cy = msg.k[5]       
            
    def calculate_3d_position(self, u_left, v_left, u_right, v_right):
        disparity = u_left - u_right
        if disparity <= 0:
            return None  
        z_depth = (self.f_pixel * self.BASELINE) / disparity
        x_pos = (u_left - self.cx) * z_depth / self.f_pixel
        y_pos = (v_left - self.cy) * z_depth / self.f_pixel
        return (x_pos, y_pos, z_depth)

    def sync_callback(self, left_msg, right_msg):
        # --- Guard Clause ---
        if not self.trigger_requested:
            return # Do nothing if we haven't been triggered
        
        if self.f_pixel is None:
            self.get_logger().warn("Triggered, but waiting for CameraInfo...")
            return

        objs_stamped_msg = ObjectsStamped()
        objs_stamped_msg.objects = []
        
        try:
            # Reset trigger immediately so we only run once per command
            self.trigger_requested = False

            img_L = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            img_R = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            # Run YOLO (This is the heavy part we are now saving)
            results_L = self.model(img_L, verbose=False)[0]
            results_R = self.model(img_R, verbose=False)[0]

            # ... [Rest of your detection and 3D logic remains the same] ...
            # (Keeping the original logic here)
            
            boxes_L = results_L.boxes.data.cpu().numpy() if len(results_L.boxes) > 0 else []
            boxes_R = results_R.boxes.data.cpu().numpy() if len(results_R.boxes) > 0 else []

            annotated_L = img_L.copy()
            detection_saved = False
            
            for box_L in boxes_L:
                x1_L, y1_L, x2_L, y2_L, conf_L, cls_L = box_L
                center_L = (int((x1_L + x2_L) / 2), int((y1_L + y2_L) / 2))

                best_match = None
                min_y_diff = 1000 
                
                for box_R in boxes_R:
                    x1_R, y1_R, x2_R, y2_R, conf_R, cls_R = box_R
                    center_R = (int((x1_R + x2_R) / 2), int((y1_R + y2_R) / 2))
                    
                    if int(cls_L) != int(cls_R): continue
                        
                    y_diff = abs(center_L[1] - center_R[1])
                    if y_diff < 20 and center_L[0] > center_R[0]:
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_match = center_R

                if best_match:
                    pos_3d = self.calculate_3d_position(center_L[0], center_L[1], best_match[0], best_match[1])
                    if pos_3d:
                        x, y, z = pos_3d
                        class_name = self.model.names[int(cls_L)]
                        
                        objs_stamped_msg.header.stamp = left_msg.header.stamp
                        objs_stamped_msg.header.frame_id = left_msg.header.frame_id

                        obj_msg = Object()
                        obj_msg.label = class_name
                        obj_msg.confidence = float(conf_L)
                        obj_msg.position = [float(x), float(y), float(z)]
                        # ... box corners ...
                        objs_stamped_msg.objects.append(obj_msg)
                        detection_saved = True
            
            self.objects_stamped_pub.publish(objs_stamped_msg) 
            if detection_saved:
                ui_msg = UiMessage()
                ui_msg.message = f"Objects Detected. First one is {objs_stamped_msg.objects[0].label}"
                ui_msg.is_success = True
                self.ui_message_pub.publish(ui_msg)
            else:
                self.get_logger().info("Triggered, but no objects found in this frame.")
                ui_msg = UiMessage()
                ui_msg.message = f"No object detected. Canceling auto aproach"
                ui_msg.is_success = False
                self.ui_message_pub.publish(ui_msg)

        except Exception as e:
            self.get_logger().error(f"Error in stereo sync_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = StereoYOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()