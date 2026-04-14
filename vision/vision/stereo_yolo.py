import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Header
from zed_msgs.msg import Object, ObjectsStamped, BoundingBox2Di

import cv2
import numpy as np
import message_filters
from ultralytics import YOLO, YOLOWorld
import os


class StereoYOLONode(Node):

    def __init__(self):
        super().__init__('stereo_yolo_node')
        self.bridge = CvBridge()

        # --- Configuration via Launch File ---
        self.declare_parameter('save_detections', False)
        self.declare_parameter('save_dir', '/tmp/stereo_yolo')

        self.declare_parameter('world_conf_threshold', 0.05)
        self.declare_parameter('world_classes', ["window", "building","house door","windows","door","garage door","glass","glass door","reflecting window","door","blue door","white window","small white window"])


        self.declare_parameter('world_trigger_interval', 4.0)

        self.do_save     = self.get_parameter('save_detections').value
        self.save_dir    = self.get_parameter('save_dir').value
        self.world_conf  = self.get_parameter('world_conf_threshold').value
        self.world_classes = self.get_parameter('world_classes').value

        if self.do_save:
            os.makedirs(self.save_dir, exist_ok=True)
            self.get_logger().info(f"Sauvegarde activée dans : {self.save_dir}")

        self.frame_count     = 0
        self.triggered_save  = False
        self.triggered_world = False  # Independent trigger for YOLOWorld

        # Main model timer (2s → same as before)
        self.trigger_timer = self.create_timer(4.0, self.trigger_timer_callback)

        # YOLOWorld timer — independent, slower cadence to spare the Jetson
        world_interval = self.get_parameter('world_trigger_interval').value
        self.world_timer = self.create_timer(
            world_interval, self.world_trigger_callback)
        self.get_logger().info(
            f"YOLOWorld trigger interval: {world_interval}s "
            f"(~{1.0/world_interval:.2f} Hz)"
        )

        # Camera parameters
        self.BASELINE = 0.063
        self.f_pixel  = None
        self.cx       = None
        self.cy       = None

        # --- Subscribers ---
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info', #/zed/zed_node/rgb/color/rect/image/camera_info
            self.info_callback,
            10
        )

        self.objects_stamped_pub = self.create_publisher(
            ObjectsStamped,
            '/aeac/internal/target_detected',
            10
        )

        # ── Load primary YOLO model ───────────────────────────────────────
        pkg_share  = get_package_share_directory('vision')
        model_path = os.path.join(pkg_share, 'models', 'yolo_m_100_epoch.pt')
        self.get_logger().info(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.get_logger().info(f"Classes YOLO: {self.model.names}")

        # ── Load YOLOWorld model ──────────────────────────────────────────
        world_model_path = os.path.join(pkg_share, 'models', 'world_model.pt')
        self.get_logger().info(f"Loading YOLOWorld model: {world_model_path}...")
        self.world_model = YOLOWorld(world_model_path)
        self.world_model.set_classes(self.world_classes)
        self.get_logger().info(
            f"YOLOWorld classes: {self.world_classes}"
        )

        # ── Stereo subscribers ────────────────────────────────────────────
        self.left_sub = message_filters.Subscriber(
            self, Image, '/zed/zed_node/left/color/rect/image'
        )
        self.right_sub = message_filters.Subscriber(
            self, Image, '/zed/zed_node/right/color/rect/image'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=80,
            slop=3000000000.0,
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(
            "Stereo YOLO Node Initialisé — En attente d'images...")

    # ── Timers ────────────────────────────────────────────────────────────────

    def trigger_timer_callback(self):
        """Allows the primary YOLO to process the next frame."""
        self.triggered_save = True

    def world_trigger_callback(self):
        """Allows YOLOWorld to process the next frame that arrives."""
        self.triggered_world = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def info_callback(self, msg):
        if self.f_pixel is None:
            self.f_pixel = msg.k[0]
            self.cx      = msg.k[2]
            self.cy      = msg.k[5]
            self.get_logger().info(
                f"Camera Info Reçu: f={self.f_pixel:.2f}")

    def calculate_3d_position(self, u_left, v_left, u_right, v_right):
        disparity = u_left - u_right
        if disparity <= 0:
            return None
        z_depth = (self.f_pixel * self.BASELINE) / disparity
        x_pos   = (u_left - self.cx) * z_depth / self.f_pixel
        y_pos   = (v_left - self.cy) * z_depth / self.f_pixel
        return (x_pos, y_pos, z_depth)

    # ── Stereo matching helper (shared by both models) ────────────────────────

    def _stereo_match_boxes(self, boxes_L, boxes_R, class_names):
        """
        Match left/right detections using epipolar constraint.

        Returns a list of dicts:
            {
                'box_L':    (x1, y1, x2, y2),
                'conf':     float,
                'cls':      int,
                'class_name': str,
                'center_L': (u, v),
                'center_R': (u, v),
                'pos_3d':   (x, y, z) or None,
            }
        """
        matched = []

        for box_L in boxes_L:
            x1_L, y1_L, x2_L, y2_L, conf_L, cls_L = box_L
            center_L = self.get_center([x1_L, y1_L, x2_L, y2_L])

            best_match  = None
            min_y_diff  = 1000

            for box_R in boxes_R:
                x1_R, y1_R, x2_R, y2_R, conf_R, cls_R = box_R
                center_R = self.get_center([x1_R, y1_R, x2_R, y2_R])

                if int(cls_L) != int(cls_R):
                    continue

                y_diff = abs(center_L[1] - center_R[1])


                self.get_logger().info(
                    f'MATCH DEBUG [{class_names[int(cls_L)]}] '
                    f'center_L={center_L} center_R={center_R} '
                    f'y_diff={y_diff} x_ok={center_L[0] > center_R[0]}'
                )
                
                if y_diff < 30:
                    if center_L[0] > center_R[0]:   # Epipolar stereo constraint
                        if y_diff < min_y_diff:
                            min_y_diff  = y_diff
                            best_match  = center_R
                    else:
                        self.get_logger().debug(
                            f"Match rejeté: x_L ({center_L[0]}) < x_R ({center_R[0]})")
                else:
                    self.get_logger().debug(
                        f"Match rejeté: y_diff trop grand ({y_diff})")

            pos_3d = None
            if best_match:
                pos_3d = self.calculate_3d_position(
                    center_L[0], center_L[1],
                    best_match[0], best_match[1]
                )

            matched.append({
                'box_L':      (x1_L, y1_L, x2_L, y2_L),
                'conf':       float(conf_L),
                'cls':        int(cls_L),
                'class_name': class_names[int(cls_L)],
                'center_L':   center_L,
                'center_R':   best_match,
                'pos_3d':     pos_3d,
            })

        return matched

    # ── Build Object message ──────────────────────────────────────────────────

    def _build_object_msg(self, det: dict) -> Object:
        """Convert a matched detection dict into a zed_msgs/Object."""
        x1_L, y1_L, x2_L, y2_L = det['box_L']
        obj_msg           = Object()
        obj_msg.label     = det['class_name']
        obj_msg.confidence = det['conf'] * 100.0

        if det['pos_3d'] is not None:
            x, y, z = det['pos_3d']
            obj_msg.position = [float(x), float(y), float(z)]
        else:
            obj_msg.position = [0.0, 0.0, 0.0]

        for i, (px, py) in enumerate([
            (x1_L, y1_L), (x2_L, y1_L), (x2_L, y2_L), (x1_L, y2_L)
        ]):
            obj_msg.bounding_box_2d.corners[i].kp = [int(px), int(py)]

        return obj_msg

    # ── Main sync callback ────────────────────────────────────────────────────

    def sync_callback(self, left_msg, right_msg):
        run_main  = self.triggered_save
        run_world = self.triggered_world

        if not run_main and not run_world:
            return

        # Consume triggers immediately so they don't double-fire
        if run_main:
            self.triggered_save  = False
        if run_world:
            self.triggered_world = False

        self.get_logger().debug("Sync Callback reçu !")

        if self.f_pixel is None:
            self.get_logger().warn(
                "Waiting for CameraInfo...", throttle_duration_sec=2.0)
            return

        try:
            img_L = self.bridge.imgmsg_to_cv2(
                left_msg,  desired_encoding='bgr8')
            img_R = self.bridge.imgmsg_to_cv2(
                right_msg, desired_encoding='bgr8')

            objs_stamped_msg            = ObjectsStamped()
            objs_stamped_msg.header.stamp    = left_msg.header.stamp
            objs_stamped_msg.header.frame_id = left_msg.header.frame_id

            annotated_L = img_L.copy()

            # ══════════════════════════════════════════════════════════════
            # PRIMARY YOLO — target detection (unchanged logic)
            # ══════════════════════════════════════════════════════════════
            if run_main:
                results_L = self.model(img_L, verbose=False)[0]
                results_R = self.model(img_R, verbose=False)[0]

                boxes_L = (results_L.boxes.data.cpu().numpy()
                           if len(results_L.boxes) > 0 else [])
                boxes_R = (results_R.boxes.data.cpu().numpy()
                           if len(results_R.boxes) > 0 else [])

                # Log raw detections (kept from original)
                for box in boxes_L:
                    x1, y1, x2, y2, conf, cls = box
                    self.get_logger().info(
                        f'Détection brute: {self.model.names[int(cls)]} '
                        f'conf={conf:.2f} '
                        f'bbox=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]'
                    )

                if len(boxes_L) > 0 or len(boxes_R) > 0:
                    self.get_logger().info(
                        f"YOLO detecte: Gauche={len(boxes_L)} "
                        f"Droite={len(boxes_R)}"
                    )

                matched_targets = self._stereo_match_boxes(
                    boxes_L, boxes_R, self.model.names)

                for det in matched_targets:
                    if det['pos_3d'] is None:
                        continue        # no valid stereo match → skip

                    obj_msg = self._build_object_msg(det)
                    objs_stamped_msg.objects.append(obj_msg)

                    x1_L, y1_L, x2_L, y2_L = det['box_L']
                    x, y, z = det['pos_3d']
                    cv2.rectangle(
                        annotated_L,
                        (int(x1_L), int(y1_L)), (int(x2_L), int(y2_L)),
                        (0, 255, 0), 2)
                    cv2.putText(
                        annotated_L,
                        f"{det['class_name']} Z:{z:.2f}m",
                        (int(x1_L), int(y1_L) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if run_main and len(matched_targets) == 0 and len(boxes_L) > 0:
                    self.get_logger().info(
                        "Objets vus mais aucun matching stéréo valide.")


            if run_world:
                self.get_logger().info(
                    "YOLOWorld: lancement détection références...")

                world_results_L = self.world_model.predict(
                    img_L, conf=self.world_conf, verbose=False)[0]
                world_results_R = self.world_model.predict(
                    img_R, conf=self.world_conf, verbose=False)[0]

                world_boxes_L = (world_results_L.boxes.data.cpu().numpy()
                                 if len(world_results_L.boxes) > 0 else [])
                world_boxes_R = (world_results_R.boxes.data.cpu().numpy()
                                 if len(world_results_R.boxes) > 0 else [])

                self.get_logger().info(
                    f"YOLOWorld détecte: Gauche={len(world_boxes_L)} "
                    f"Droite={len(world_boxes_R)}"
                )

                # Build a name-map index → class string for YOLOWorld
                # (world_model.names may be a dict {0: 'window', …})
                world_names = self.world_model.names  # dict int→str

                matched_refs = self._stereo_match_boxes(
                    world_boxes_L, world_boxes_R, world_names)

                ref_count = 0
                for det in matched_refs:
                    if det['pos_3d'] is None:
                        self.get_logger().debug(
                            f"YOLOWorld [{det['class_name']}] : "
                            f"pas de match stéréo valide, ignoré")
                        continue

                    obj_msg = self._build_object_msg(det)
                    objs_stamped_msg.objects.append(obj_msg)
                    ref_count += 1

                    x1_L, y1_L, x2_L, y2_L = det['box_L']
                    x, y, z = det['pos_3d']
                    # Draw in blue so references are visually distinct
                    cv2.rectangle(
                        annotated_L,
                        (int(x1_L), int(y1_L)), (int(x2_L), int(y2_L)),
                        (255, 100, 0), 2)
                    cv2.putText(
                        annotated_L,
                        f"[REF] {det['class_name']} Z:{z:.2f}m",
                        (int(x1_L), int(y1_L) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

                self.get_logger().info(
                    f"YOLOWorld: {ref_count} référence(s) avec position 3D")

            # ══════════════════════════════════════════════════════════════
            # Publish & save
            # ══════════════════════════════════════════════════════════════
            if len(objs_stamped_msg.objects) > 0:
                self.objects_stamped_pub.publish(objs_stamped_msg)
                self.get_logger().info(
                    f"Publishing {len(objs_stamped_msg.objects)} objects "
                    f"(targets + refs)"
                )

                if self.do_save:
                    filename = os.path.join(
                        self.save_dir,
                        f"stereo_det_{self.frame_count:05d}.jpg")
                    cv2.imwrite(filename, annotated_L)
                    self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StereoYOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()