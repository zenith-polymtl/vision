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
from geometry_msgs.msg import Vector3
from custom_interfaces.msg import AimError


def compute_error(yolo_results, img_width, img_height, min_confidence, offset_x=0, offset_y=0):
    """
    Compute pitch/yaw error to the closest detected target above confidence threshold.

    Uses vectorised NumPy operations instead of Python loops for efficiency:
      - All qualifying box centres are gathered into a (N, 2) array.
      - Squared Euclidean distances are computed in one broadcast operation.
      - np.argmin picks the closest box without iterating.

    Returns (error_pitch, error_yaw) in pixels, or (0.0, 0.0) when no target found.
    """
    img_center = np.array(
        [img_width / 2 + offset_x, img_height / 2 - offset_y],
        dtype=np.float32,
    )

    # Collect centres of all boxes that pass the confidence gate
    centers = []
    for result in yolo_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # boxes.conf and boxes.xyxy are torch tensors; convert once to numpy
        confs = boxes.conf.cpu().numpy()          # shape (N,)
        xyxy  = boxes.xyxy.cpu().numpy()          # shape (N, 4)

        mask = confs >= min_confidence
        if not mask.any():
            continue

        valid_xyxy = xyxy[mask]                   # shape (M, 4)
        # Centre = mean of x1,x2 and y1,y2 — computed columnwise
        cx = (valid_xyxy[:, 0] + valid_xyxy[:, 2]) / 2.0   # shape (M,)
        cy = (valid_xyxy[:, 1] + valid_xyxy[:, 3]) / 2.0   # shape (M,)
        centers.append(np.stack([cx, cy], axis=1))          # shape (M, 2)

    if not centers:
        return 0.0, 0.0

    all_centers = np.concatenate(centers, axis=0)           # shape (N, 2)

    # Vectorised squared-distance from image centre; argmin avoids a sqrt
    deltas = all_centers - img_center                       # shape (N, 2)
    sq_distances = (deltas ** 2).sum(axis=1)                # shape (N,)
    best_idx = int(np.argmin(sq_distances))

    error_yaw   =  float(deltas[best_idx, 0])
    error_pitch = -float(deltas[best_idx, 1])               # pitch: up is positive

    return error_pitch, error_yaw


class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('rgb_yolo')

        self.initialize_parameters()
        self.initialize_attributes()
        self.initialize_topics()

        self.get_logger().info("YOLO Node Started")

    def initialize_parameters(self):
        self.declare_parameter('image_topic',      '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('activation_topic', '/aeac/internal/auto_shoot/start_hr_aiming')
        self.declare_parameter('gimbal_error_topic', '/aeac/internal/gimbal/target_error')
        self.declare_parameter('image_save_dir',   '/vision_ws/Pictures/yolo_without_distances')
        self.declare_parameter('model_name',       'yolo_m_100_epoch.pt')
        self.declare_parameter('initial_offset_x', 0.0)
        self.declare_parameter('initial_offset_y', 0.0)
        self.declare_parameter('min_confidence',   0.8)

        gp = self.get_parameter
        self.image_topic         = gp('image_topic').value
        self.activation_topic    = gp('activation_topic').value
        self.gimbal_error_topic  = gp('gimbal_error_topic').value
        self.model_named         = gp('model_name').value
        self.save_dir            = gp('image_save_dir').value
        self.offset_x            = gp('initial_offset_x').value
        self.offset_y            = gp('initial_offset_y').value
        self.min_confidence      = gp('min_confidence').value

        self.frame_count = 0

        self.get_logger().info("Parameters initialized:")
        self.get_logger().info(f"  Model:          {self.model_named}")
        self.get_logger().info(f"  Min confidence: {self.min_confidence}")

    def initialize_attributes(self):
        self.bridge       = CvBridge()
        self.is_activated = False

        pkg_share  = get_package_share_directory('vision')
        model_path = os.path.join(pkg_share, 'models', self.model_named)

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"YOLO inference device: {self.device}")

        self.model = YOLO(model_path)
        self.get_logger().info(f"Model class names: {self.model.names}")

        # Uncomment to enable local frame saving:
        # os.makedirs(self.save_dir, exist_ok=True)

    def initialize_topics(self):
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Image, self.image_topic,      self.image_callback,      qos_profile=qos)
        self.create_subscription(Bool,  self.activation_topic, self.activation_callback, qos_profile=qos)

        self.error_publisher = self.create_publisher(AimError, self.gimbal_error_topic, qos_profile=qos)
        self.create_subscription(Vector3, '/aeac/external/gimbal_offset', self.gimbal_offset_callback, qos_profile=qos)

    def activation_callback(self, msg):
        self.is_activated = msg.data
        state = 'Activated' if self.is_activated else 'Deactivated'
        self.get_logger().info(f"Activation status changed: {state}")

    def gimbal_offset_callback(self, msg):
        self.offset_x = -msg.x
        self.offset_y = -msg.y

    def image_callback(self, msg):
        if not self.is_activated:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.model.predict(
                source=cv_image,
                device=self.device,
                imgsz=640,
                conf=self.min_confidence,
                verbose=False,
            )

            height, width = cv_image.shape[:2]
            error_pitch, error_yaw = compute_error(
                results,
                width,
                height,
                self.min_confidence,
                self.offset_x,
                self.offset_y,
            )

            target_error             = AimError()
            target_error.pitch_error = error_pitch
            target_error.yaw_error   = error_yaw
            self.error_publisher.publish(target_error)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {repr(e)}")

        # Uncomment to save an annotated frame every 20 callbacks:
        # if self.frame_count % 20 == 0:
        #     annotated = results[0].plot()
        #     filename  = os.path.join(self.save_dir, f"yolo_detection_{self.frame_count:05d}.jpg")
        #     cv2.imwrite(filename, annotated)
        #     self.get_logger().info(f"Saved annotated image: {filename}")

        self.frame_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()