import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
import os


class YOLO3DSubscriber(Node):

    def __init__(self):
        super().__init__('yolo_3d_subscriber')
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/color/rect/image',
            self.image_callback,
            10
        )

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pc_callback,
            10
        )

        self.latest_pc_np = None  # stored organized point cloud


        # Load YOLO model

        self.model = YOLO('/home/haipy/Desktop/UAV_vision/uav_vision/uav_vision/models/best-medium.pt')

        self.get_logger().info("YOLO ROS2 Node Started")

    # -----------------------------------------------------
    # Point Cloud Callback (convert once, not per frame)
    # -----------------------------------------------------
    def pc_callback(self, msg):

        # Fast conversion using buffer (organized cloud)
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('padding', np.float32)
        ])

        cloud_arr = np.frombuffer(msg.data, dtype=dtype)
        cloud_arr = cloud_arr.reshape(msg.height, msg.width)

        xyz = np.stack(
            (cloud_arr['x'],
             cloud_arr['y'],
             cloud_arr['z']),
            axis=-1
        )

        self.latest_pc_np = xyz

    # -----------------------------------------------------
    # Image Callback
    # -----------------------------------------------------
    def image_callback(self, msg):

        if self.latest_pc_np is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.model(cv_image, verbose=False)

            annotated = cv_image.copy()

            for r in results:

                if r.masks is None:
                    continue

                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)

                for i, mask in enumerate(masks):

                    class_id = class_ids[i]
                    class_name = self.model.names[class_id]

                    # Resize mask to image size (important!)
                    mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]))
                    mask = mask > 0.5

                    # Get pixel indices inside mask
                    y_idxs, x_idxs = np.where(mask)

                    if len(y_idxs) == 0:
                        continue

                    # Get corresponding 3D points
                    points3d = self.latest_pc_np[y_idxs, x_idxs, :]
                    points3d = points3d[np.isfinite(points3d).all(axis=1)]

                    if points3d.shape[0] == 0:
                        label = f"{class_name} [NO DEPTH]"
                    else:
                        centroid = np.median(points3d, axis=0)
                        x3d, y3d, z3d = centroid
                        label = f"{class_name} [{x3d:.2f}, {y3d:.2f}, {z3d:.2f}]"

                    # Compute centroid
                    centroid = points3d.mean(axis=0)

                    x3d, y3d, z3d = centroid

                    # Draw segmentation mask overlay
                    color = (0, 255, 0)
                    annotated[mask] = annotated[mask] * 0.5 + np.array(color) * 0.5

                    # Compute 2D centroid for label placement
                    u = int(np.mean(x_idxs))
                    v = int(np.mean(y_idxs))

                    label = f"{class_name}  [{x3d:.2f}, {y3d:.2f}, {z3d:.2f}]"

                    cv2.putText(
                        annotated,
                        label,
                        (u, v),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            cv2.imshow("YOLO 3D Segmentation", annotated)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YOLO3DSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
