import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
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
            Bool,
            '/aeac/external/describe_scene',
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
        
        self.ui_message_pub = self.create_publisher(
            UiMessage,
            '/aeac/external/send_to_ui',
            qos_reliable
        )

        # --- Compressed Image Publisher ---
        self.overlay_pub = self.create_publisher(
            CompressedImage,
            '/aeac/external/detection_overlay',
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

    def draw_overlay(self, img, detections):
        """
        Draw bounding boxes, center crosshairs, labels, and 3D distance on the image.
        detections: list of dicts with keys: x1,y1,x2,y2, class_name, conf, pos_3d (tuple or None)
        """
        overlay = img.copy()

        # Color palette — cycles by detection index
        COLORS = [
            (0, 255, 0),    # green
            (0, 180, 255),  # orange
            (255, 80, 80),  # blue-ish
            (255, 0, 200),  # magenta
            (0, 255, 220),  # cyan
        ]

        for i, det in enumerate(detections):
            color = COLORS[i % len(COLORS)]
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pos_3d = det.get('pos_3d')
            class_name = det.get('class_name', '?')
            conf = det.get('conf', 0.0)

            # Bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Center crosshair
            ch_size = 10
            cv2.line(overlay, (cx - ch_size, cy), (cx + ch_size, cy), color, 2)
            cv2.line(overlay, (cx, cy - ch_size), (cx, cy + ch_size), color, 2)
            cv2.circle(overlay, (cx, cy), ch_size + 2, color, 1)

            # Label background + text
            if pos_3d:
                x3d, y3d, z3d = pos_3d
                label = f"{class_name} {conf:.2f} | {z3d:.2f}m"
                sub_label = f"X:{x3d:.2f} Y:{y3d:.2f}"
            else:
                label = f"{class_name} {conf:.2f} | no depth"
                sub_label = ""

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw pill-shaped background for label
            pad = 4
            label_y = max(y1 - th - pad * 2, 0)
            cv2.rectangle(overlay, (x1, label_y), (x1 + tw + pad * 2, label_y + th + pad * 2), color, -1)
            cv2.putText(overlay, label, (x1 + pad, label_y + th + pad), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Sub-label (X/Y coords) just below main label
            if sub_label:
                (sw, sh), _ = cv2.getTextSize(sub_label, font, font_scale * 0.85, thickness)
                sub_y = label_y + th + pad * 2
                cv2.rectangle(overlay, (x1, sub_y), (x1 + sw + pad * 2, sub_y + sh + pad * 2), color, -1)
                cv2.putText(overlay, sub_label, (x1 + pad, sub_y + sh + pad), font, font_scale * 0.85, (0, 0, 0), thickness, cv2.LINE_AA)

            # Line from label to center
            cv2.line(overlay, (x1, y1), (cx, cy), color, 1, cv2.LINE_AA)

        # Corner watermark
        cv2.putText(overlay, "AEAC Vision", (10, overlay.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return overlay

    def publish_compressed(self, img_bgr, stamp):
        """Encode image as JPEG and publish as CompressedImage."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buffer = cv2.imencode('.jpg', img_bgr, encode_params)
        if not success:
            self.get_logger().warn("Failed to encode overlay image.")
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = 'zed_left_camera_optical_frame'
        msg.format = 'jpeg'
        msg.data = buffer.tobytes()
        self.overlay_pub.publish(msg)

    def sync_callback(self, left_msg, right_msg):
        if not self.trigger_requested:
            return

        if self.f_pixel is None:
            self.get_logger().warn("Triggered, but waiting for CameraInfo...")
            return

        objs_stamped_msg = ObjectsStamped()
        objs_stamped_msg.objects = []

        try:
            self.trigger_requested = False

            img_L = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            img_R = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            results_L = self.model(img_L, verbose=False)[0]
            results_R = self.model(img_R, verbose=False)[0]

            boxes_L = results_L.boxes.data.cpu().numpy() if len(results_L.boxes) > 0 else []
            boxes_R = results_R.boxes.data.cpu().numpy() if len(results_R.boxes) > 0 else []

            detection_saved = False
            overlay_detections = []  # accumulate for drawing

            for box_L in boxes_L:
                x1_L, y1_L, x2_L, y2_L, conf_L, cls_L = box_L
                center_L = (int((x1_L + x2_L) / 2), int((y1_L + y2_L) / 2))

                best_match = None
                min_y_diff = 1000

                for box_R in boxes_R:
                    x1_R, y1_R, x2_R, y2_R, conf_R, cls_R = box_R
                    center_R = (int((x1_R + x2_R) / 2), int((y1_R + y2_R) / 2))

                    if int(cls_L) != int(cls_R):
                        continue

                    y_diff = abs(center_L[1] - center_R[1])
                    if y_diff < 20 and center_L[0] > center_R[0]:
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_match = center_R

                pos_3d = None
                if best_match:
                    pos_3d = self.calculate_3d_position(
                        center_L[0], center_L[1], best_match[0], best_match[1]
                    )

                class_name = self.model.names[int(cls_L)]

                # Always draw the detection (even without depth)
                overlay_detections.append({
                    'x1': x1_L, 'y1': y1_L, 'x2': x2_L, 'y2': y2_L,
                    'class_name': class_name,
                    'conf': float(conf_L),
                    'pos_3d': pos_3d,
                })

                if pos_3d:
                    x, y, z = pos_3d
                    objs_stamped_msg.header.stamp = left_msg.header.stamp
                    objs_stamped_msg.header.frame_id = left_msg.header.frame_id

                    obj_msg = Object()
                    obj_msg.label = class_name
                    obj_msg.confidence = float(conf_L)
                    obj_msg.position = [float(x), float(y), float(z)]
                    objs_stamped_msg.objects.append(obj_msg)
                    detection_saved = True

            # --- Draw and publish overlay ---
            annotated = self.draw_overlay(img_L, overlay_detections)
            self.publish_compressed(annotated, left_msg.header.stamp)

            # --- Publish object list ---
            self.objects_stamped_pub.publish(objs_stamped_msg)

            # --- UI feedback ---
            ui_msg = UiMessage()
            if detection_saved:
                ui_msg.message = f"Objects Detected. First one is {objs_stamped_msg.objects[0].label}"
                ui_msg.is_success = True
            else:
                self.get_logger().info("Triggered, but no stereo-matched objects found.")
                ui_msg.message = "No object detected. Canceling auto approach"
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