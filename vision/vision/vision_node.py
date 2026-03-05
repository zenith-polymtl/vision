#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os



class YoloCameraNode(Node):

    def __init__(self):
        super().__init__('yolo_camera_node')

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        # Load YOLO model
        package_share_dir = get_package_share_directory('uav_vision')
        model_path = os.path.join(package_share_dir, 'models', 'best-medium.pt')

        self.model = YOLO(model_path)

        #self.model = YOLO("/home/haipy/your_model_path/my_model_v8n.pt")

        self.get_logger().info("YOLO Camera Node Started")

    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        # Run YOLO inference
        results = self.model(frame, verbose=False)

        # Draw detections
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                class_id = int(box.cls)
                label = self.model.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw label + confidence
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

        # Show frame
        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

