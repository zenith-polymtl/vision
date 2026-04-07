import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped


class FakeMavrosNode(Node):

    def __init__(self):
        super().__init__('fake_mavros_node')

        self.declare_parameter('heading_deg', 0.0)   # cap en degrés
        self.declare_parameter('altitude_m',  6.0)   # hauteur en mètres
        self.declare_parameter('pub_rate_hz', 10.0)  # fréquence de publication

        self.heading  = self.get_parameter('heading_deg').value
        self.altitude = self.get_parameter('altitude_m').value
        rate          = self.get_parameter('pub_rate_hz').value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.hdg_pub  = self.create_publisher(
            Float64, '/mavros/global_position/compass_hdg', qos)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/mavros/local_position/pose', qos)

        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f'FakeMavros démarré — '
            f'heading={self.heading}°  altitude={self.altitude}m  '
            f'rate={rate}Hz'
        )

    def _publish(self):
        now = self.get_clock().now().to_msg()

        hdg_msg      = Float64()
        hdg_msg.data = self.heading
        self.hdg_pub.publish(hdg_msg)

        pose_msg                    = PoseStamped()
        pose_msg.header.stamp       = now
        pose_msg.header.frame_id    = 'map'
        pose_msg.pose.position.z    = self.altitude
        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FakeMavrosNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()