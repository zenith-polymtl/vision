import rclpy
from rclpy.node import Node
from zed_msgs.msg import ObjectsStamped  # Import ZED Object message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ZEDObjectSubscriber(Node):
    def __init__(self):
        super().__init__('zed_object_subscriber')
        self.bridge = CvBridge()

        # 1. Subscribe to the Object Detection data
        self.obj_sub = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.object_callback,
            10
        )

        # 2. Subscribe to the image (only if you want to save annotated pictures)
        self.img_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/color/rect/image',
            self.image_callback,
            10
        )

        self.latest_image = None
        self.save_dir = '/water_ws/Pictures/zed_sdk_detections'
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0

    def image_callback(self, msg):
        # Just store the latest image to use when a detection arrives
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def object_callback(self, msg):
        if self.latest_image is None:
            return

        self.get_logger().info(f"Detected {len(msg.objects)} objects")

        # Create a copy of the image to draw on
        annotated_image = self.latest_image.copy()

        for obj in msg.objects:
            # Get label and confidence
            label = obj.label
            conf = obj.confidence
            
            # 3D Position (This is the magic of ZED SDK!)
            pos = obj.position
            dist = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5

            # 2D Bounding Box (top-left, top-right, bottom-right, bottom-left)
            # We use the top-left (0) and bottom-right (2)
            top_left = (int(obj.bounding_box_2d.corners[0].kp[0]), int(obj.bounding_box_2d.corners[0].kp[1]))
            bottom_right = (int(obj.bounding_box_2d.corners[2].kp[0]), int(obj.bounding_box_2d.corners[2].kp[1]))

            # Draw on image
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
            text = f"{label} ({conf:.1f}%) {dist:.2f}m"
            cv2.putText(annotated_image, text, (top_left[0], top_left[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save result
        filename = os.path.join(self.save_dir, f"zed_det_{self.frame_count:05d}.jpg")
        cv2.imwrite(filename, annotated_image)
        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = ZEDObjectSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()