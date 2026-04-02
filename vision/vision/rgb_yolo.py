import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import os
from ultralytics import YOLO  # make sure ultralytics is installed on your system

from custom_interfaces.msg import AimError

def compute_error(yolo_results, img_width, img_height, offset_x=0, offset_y=0):
    best_distance = None
    error_yaw = 0.0
    error_pitch = 0.0

    img_center_x = img_width / 2  + offset_x
    img_center_y = img_height / 2 - offset_y
    
    for result in yolo_results:
        for box in result.boxes:
            # confidence = float(box.conf)
            # if confidence < 0.8:
            #     continue

            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = coords

            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            distance = ((box_center_x - img_center_x)**2 + (box_center_y - img_center_y)**2)**0.5
            if best_distance is None or distance < best_distance:
                best_distance = distance

                error_yaw = box_center_x - img_center_x
                error_pitch = -(box_center_y - img_center_y)

    if best_distance is not None:
        return error_pitch, error_yaw
    else:
        return 0.0, 0.0

class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')
        self.bridge = CvBridge()

        # Subscribe to ZED RGB feed
        self.subscription = self.create_subscription(
            Image,
            'zed/zed_node/rgb/color/rect/image',
            self.image_callback,
            10
        )

        self.error_publisher = self.create_publisher(AimError, '/aeac/internal/gimbal/target_error', 10)

        # Set up a directory to save the output images
        self.save_dir = '/water_ws/Pictures/yolo_without_distances'
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0

        # Load YOLO model
        pkg_share = get_package_share_directory('vision')  # ← your package name
        model_path = os.path.join(pkg_share, 'models', 'best-medium.pt')
        self.model = YOLO(model_path)

        self.get_logger().info("YOLO ROS2 Node Started")
        self.get_logger().info(f"Images will be saved to: {self.save_dir}")

    def image_callback(self, msg):
        try:
            self.get_logger().info("Received image, running YOLO inference...")
            # Convert ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run YOLO inference
            results = self.model(cv_image)  # returns a Results object

            height, width = cv_image.shape[:2]
            error_pitch, error_yaw = compute_error(results, width, height)

            target_error = AimError()
            target_error.pitch_error = error_pitch
            target_error.yaw_error = error_yaw
            self.error_publisher.publish(target_error)

            # Draw results on image
            annotated_frame = results[0].plot()  # returns numpy image with boxes and labels

            # Save the annotated frame to disk instead of showing it
            filename = os.path.join(self.save_dir, f"yolo_detection_{self.frame_count:05d}.jpg")
            cv2.imwrite(filename, annotated_frame)
            
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()