import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

from std_msgs.msg import Bool
from custom_interfaces.msg import AimError

class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('rgb_yolo')

        # 1. Parameters
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('activation_topic', '/aeac/internal/auto_shoot/start_hr_aiming')
        self.declare_parameter('gimbal_error_topic', '/aeac/internal/gimbal/target_error')
        self.declare_parameter('model_name', 'best_nano.engine')
        self.declare_parameter('min_confidence', 0.8)

        # 2. Attributes
        self.bridge = CvBridge()
        self.is_activated = True
        self.last_log_time = self.get_clock().now() # Timer for 1s logging
        self.frame_count = 0
        
        pkg_share = get_package_share_directory('vision')
        model_path = os.path.join(pkg_share, 'models', self.get_parameter('model_name').value)

        # Load Model (Segmentation-compatible)
        self.model = YOLO(model_path, task='segment') 
        self.min_confidence = self.get_parameter('min_confidence').value
        
        # Setup Constants on GPU
        self.device = torch.device('cuda:0')
        self.get_logger().info(f"YOLO Engine loaded. Classes: {self.model.names}")

        # 3. Topics
        qos = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, qos)
        self.create_subscription(Bool, self.get_parameter('activation_topic').value, self.activation_callback, qos)
        self.error_publisher = self.create_publisher(AimError, self.get_parameter('gimbal_error_topic').value, qos)

    def activation_callback(self, msg):
        self.is_activated = msg.data
        self.get_logger().info(f"Targeting System: {'ON' if self.is_activated else 'OFF'}")

    def image_callback(self, msg):
        if not self.is_activated:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_image.shape[:2]

            # GPU Inference with Stream=True
            results = self.model.predict(
                source=cv_image,
                device=0,
                imgsz=640,
                conf=self.min_confidence,
                half=True,
                verbose=False,
                stream=True
            )

            img_center = torch.tensor([w / 2, h / 2], device=self.device, dtype=torch.float16)
            
            error_pitch, error_yaw = 0.0, 0.0
            found_target = False
            current_frame_classes = set()

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # Collect detected class names for logging
                # result.boxes.cls is on GPU; move to CPU only for the 1s log check
                class_ids = result.boxes.cls.int().cpu().tolist()
                for cid in class_ids:
                    current_frame_classes.add(self.model.names[cid])

                centers = result.boxes.xywh[:, :2]
                confs = result.boxes.conf

                mask = confs >= self.min_confidence
                valid_centers = centers[mask]

                if valid_centers.shape[0] > 0:
                    dist = torch.norm(valid_centers - img_center, dim=1)
                    closest_idx = torch.argmin(dist)
                    target_center = valid_centers[closest_idx]

                    delta = target_center - img_center
                    error_yaw = float(delta[0].item())
                    error_pitch = -float(delta[1].item())
                    found_target = True

            # --- 1-Second Log Logic ---
            now = self.get_clock().now()
            if (now - self.last_log_time).nanoseconds >= 1e9: # 1 second in nanoseconds
                if current_frame_classes:
                    self.get_logger().info(f"Detections: {', '.join(current_frame_classes)}")
                else:
                    self.get_logger().info("No targets detected.")
                self.last_log_time = now
            # Increment the frame counter
            self.frame_count += 1

            # --- 1-Second Log & FPS Logic ---
            now = self.get_clock().now()
            elapsed_ns = (now - self.last_log_time).nanoseconds
            
            if elapsed_ns >= 1e9: # If 1 second has passed
                elapsed_sec = elapsed_ns / 1e9
                fps = self.frame_count / elapsed_sec
                
                # Format the log string
                class_str = ', '.join(current_frame_classes) if current_frame_classes else "None"
                self.get_logger().info(f"⚡ FPS: {fps:.1f} | Detections: {class_str}")
                
                # Reset timers and counters for the next second
                self.last_log_time = now
                self.frame_count = 0
            if found_target:
                msg_err = AimError()
                msg_err.pitch_error = error_pitch
                msg_err.yaw_error = error_yaw
                self.error_publisher.publish(msg_err)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()