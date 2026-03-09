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

        # Save Configuration
        # os.path.expanduser handles the '~' symbol automatically
        self.save_dir = os.path.expanduser('/water_ws/Pictures/yolo_3D')
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0

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
        pkg_share = get_package_share_directory('vision')  # ← your package name
        model_path = os.path.join(pkg_share, 'models', 'best-medium.pt')
        self.model = YOLO(model_path)

        self.get_logger().info("YOLO 3D ROS2 Node Started")
        self.get_logger().info(f"Saving detections to: {self.save_dir}")

    def pc_callback(self, msg):
        # Log that we are actually receiving point clouds!
        self.get_logger().debug("Received PointCloud2 message") 
        
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
        self.get_logger().info("Successfully parsed point cloud.", once=True)

    def image_callback(self, msg):
        self.get_logger().info("Received image frame...", once=True)

        if self.latest_pc_np is None:
            self.get_logger().warning("Waiting for point cloud data... skipping image.", throttle_duration_sec=2.0)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_h, img_w = cv_image.shape[:2]
            pc_h, pc_w = self.latest_pc_np.shape[:2]
            
            results = self.model(cv_image, verbose=False)            
            annotated = cv_image.copy()
            detection_found = False

            for r in results:
                if len(r.boxes) == 0:
                    continue  # Nothing detected at all

                if r.masks is None:
                    self.get_logger().warning("Detections found, but NO MASKS. Is this a segmentation model?", throttle_duration_sec=5.0)
                    continue

                detection_found = True
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)

                for i, mask in enumerate(masks):
                    class_id = class_ids[i]
                    class_name = self.model.names[class_id]

                    mask = cv2.resize(mask, (img_w, img_h))
                    mask = mask > 0.5

                    y_idxs, x_idxs = np.where(mask)
                    if len(y_idxs) == 0:
                        continue
                    
                    # Scale indices if resolutions don't match
                    if img_h != pc_h or img_w != pc_w:
                        scale_y = pc_h / img_h
                        scale_x = pc_w / img_w
                        
                        pc_y_idxs = (y_idxs * scale_y).astype(int)
                        pc_x_idxs = (x_idxs * scale_x).astype(int)
                        
                        # Clip safely to prevent out-of-bounds due to rounding
                        pc_y_idxs = np.clip(pc_y_idxs, 0, pc_h - 1)
                        pc_x_idxs = np.clip(pc_x_idxs, 0, pc_w - 1)
                        
                        points3d = self.latest_pc_np[pc_y_idxs, pc_x_idxs, :]
                    else:
                        points3d = self.latest_pc_np[y_idxs, x_idxs, :]

                    points3d = points3d[np.isfinite(points3d).all(axis=1)]
                    
                    if points3d.shape[0] == 0:
                        x3d, y3d, z3d = 0.0, 0.0, 0.0
                        label = f"{class_name} [NO DEPTH]"
                    else:
                        centroid = np.median(points3d, axis=0)
                        x3d, y3d, z3d = centroid
                        label = f"{class_name} [{x3d:.2f}, {y3d:.2f}, {z3d:.2f}]"

                    color = (0, 255, 0)
                    mask_indices = np.where(mask)
                    annotated[mask_indices[0], mask_indices[1]] = \
                        annotated[mask_indices[0], mask_indices[1]] * 0.5 + np.array(color) * 0.5

                    u = int(np.mean(x_idxs))
                    v = int(np.mean(y_idxs))

                    cv2.putText(
                        annotated, label, (u, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

            if detection_found:
                filename = os.path.join(self.save_dir, f"det_3d_{self.frame_count:05d}.jpg")
                success = cv2.imwrite(filename, annotated)
                
                if success:
                    self.get_logger().info(f"Successfully saved image: {filename}")
                else:
                    self.get_logger().error(f"Failed to write image to {filename}. Check permissions!")
                
                self.frame_count += 1

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