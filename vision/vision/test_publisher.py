#!/usr/bin/env python3
"""
test_publisher.py — Injecteur de fausses détections pour tester wall_detector

PRINCIPE
────────
Publie sur /aeac/test/objects avec le timestamp copié du point cloud du rosbag.
C'est la garantie que wall_detector trouvera toujours le bon cloud dans son
buffer (décalage ~0ms), sans dépendre de use_sim_time ou --clock.

UTILISATION
───────────
  1. Lance le rosbag normalement :
       ros2 bag play /aeac/data/rosbag2_2026_02_03-22_19_30 --loop

  2. Dans wall_detector.py, la subscription ObjectsStamped doit pointer vers :
       '/aeac/test/objects'

  3. Lance wall_detector, puis ce noeud (pas besoin de use_sim_time) :
       ros2 run <package> wall_detector
       ros2 run <package> test_publisher

CONFIGURATION
─────────────
Ajuste les bbox en pixels avec rqt_image_view :
  ros2 run rqt_image_view rqt_image_view
  → topic : /zed/zed_node/rgb/color/rect/image
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2
from zed_msgs.msg import ObjectsStamped, Object, Keypoint2Di


OUTPUT_TOPIC = '/aeac/test/objects'

FAKE_OBJECTS = [
    {'label': 'window', 'id': 1, 'bbox': (80, 46,  135, 112)},  # 380*0.35, 130*0.355...
    {'label': 'target', 'id': 2, 'bbox': (326, 20,  396, 80)},  # 932*0.35, 120*0.355...
]
# ──────────────────────────────────────────────────────────────────────────────


def make_bbox(u_min, v_min, u_max, v_max):
    corners = []
    for u, v in [(u_min, v_min), (u_max, v_min), (u_max, v_max), (u_min, v_max)]:
        kp = Keypoint2Di()
        kp.kp = [u, v]
        corners.append(kp)
    return corners


class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub          = self.create_publisher(ObjectsStamped, OUTPUT_TOPIC, qos)
        self.latest_stamp = None

        # On s'abonne au cloud UNIQUEMENT pour copier son timestamp.
        # Dès qu'un cloud arrive, on publie immédiatement les fausses détections
        # avec le même stamp → décalage garanti ~0ms dans wall_detector.
        self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.cloud_cb,
            qos
        )

        self.get_logger().info(
            f'Test publisher démarré → {OUTPUT_TOPIC}\n'
            + '\n'.join(f'  [{o["id"]}] {o["label"]} bbox={o["bbox"]}' for o in FAKE_OBJECTS)
        )

    def cloud_cb(self, msg):
        """
        Appelé à chaque nouveau cloud. On publie immédiatement avec le même
        timestamp → la détection et le cloud sont parfaitement synchronisés.
        On publie ici plutôt que dans un timer pour coller exactement au rythme
        du rosbag sans timer drift.
        """
        out = ObjectsStamped()
        out.header.stamp    = msg.header.stamp   # même stamp que le cloud
        out.header.frame_id = 'zed_left_camera_frame'

        for cfg in FAKE_OBJECTS:
            obj            = Object()
            obj.label_id   = cfg['id']
            obj.label      = cfg['label']
            obj.confidence = 99.0
            obj.position   = [0.0, 0.0, 1.0]

            u_min, v_min, u_max, v_max  = cfg['bbox']
            obj.bounding_box_2d.corners = make_bbox(u_min, v_min, u_max, v_max)
            out.objects.append(obj)

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(TestPublisher())
    rclpy.shutdown()


if __name__ == '__main__':
    main()