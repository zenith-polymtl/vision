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
from ultralytics import YOLO
import os

class StereoYOLONode(Node):

    def __init__(self):
        super().__init__('stereo_yolo_node')
        self.bridge = CvBridge()

        # Save Configuration
        self.save_dir = os.path.expanduser('/water_ws/Pictures/Stereo')
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0

        # In meters
        self.BASELINE = 0.12
        self.f_pixel = None
        self.cx = None
        self.cy = None
        
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/left/color/rect/camera_info',
            self.info_callback,
            10
        )

        self.objects_stamped_pub = self.create_publisher(ObjectsStamped, '/aeac/internal/target_detected', 10)

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

        self.get_logger().info("Stereo YOLO ROS 2 Node Started")
        self.get_logger().info(f"Saving detections to: {self.save_dir}")

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def info_callback(self, msg):
        if self.f_pixel is None:
            self.f_pixel = msg.k[0]  
            self.cx = msg.k[2]       
            self.cy = msg.k[5]       
            
            self.get_logger().info(
                f"Camera Params Loaded: f={self.f_pixel:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
    
    def calculate_3d_position(self, u_left, v_left, u_right, v_right):
        disparity = u_left - u_right

        if disparity <= 0:
            return None  

        z_depth = (self.f_pixel * self.BASELINE) / disparity
        x_pos = (u_left - self.cx) * z_depth / self.f_pixel
        y_pos = (v_left - self.cy) * z_depth / self.f_pixel

        return (x_pos, y_pos, z_depth)

    def sync_callback(self, left_msg, right_msg):
        if self.f_pixel is None:
            self.get_logger().debug("Waiting for CameraInfo...")
            return
        try:
            img_L = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            img_R = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            results_L = self.model(img_L, verbose=False)[0]
            results_R = self.model(img_R, verbose=False)[0]

            boxes_L = results_L.boxes.data.cpu().numpy() if len(results_L.boxes) > 0 else []
            boxes_R = results_R.boxes.data.cpu().numpy() if len(results_R.boxes) > 0 else []

            annotated_L = img_L.copy()
            detection_saved = False
            objs_stamped_msg = ObjectsStamped()
            objs_stamped_msg.objects = []
            
            for box_L in boxes_L:
                x1_L, y1_L, x2_L, y2_L, conf_L, cls_L = box_L
                center_L = self.get_center([x1_L, y1_L, x2_L, y2_L])

                best_match = None
                min_y_diff = 1000 
                
                for box_R in boxes_R:
                    x1_R, y1_R, x2_R, y2_R, conf_R, cls_R = box_R
                    center_R = self.get_center([x1_R, y1_R, x2_R, y2_R])
                    
                    if int(cls_L) != int(cls_R):
                        continue
                        
                    y_diff = abs(center_L[1] - center_R[1])
                    
                    if y_diff < 20: 
                        if center_L[0] > center_R[0]:
                            if y_diff < min_y_diff:
                                min_y_diff = y_diff
                                best_match = center_R

                if best_match:
                    pos_3d = self.calculate_3d_position(center_L[0], center_L[1], best_match[0], best_match[1])
                    
                    if pos_3d:
                        x, y, z = pos_3d
                        class_name = self.model.names[int(cls_L)]
                        
                        objs_stamped_msg.header.stamp = left_msg.header.stamp
                        objs_stamped_msg.header.frame_id = left_msg.header.frame_id

                        obj_msg = Object()

                        obj_msg.label = class_name
                        obj_msg.confidence = float(conf_L)
                        obj_msg.position = [float(x), float(y), float(z)]
                                                                        
                        obj_msg.bounding_box_2d.corners[0].kp = [int(x1_L), int(y1_L)]
                        obj_msg.bounding_box_2d.corners[1].kp = [int(x2_L), int(y1_L)]
                        obj_msg.bounding_box_2d.corners[2].kp = [int(x2_L), int(y2_L)]
                        obj_msg.bounding_box_2d.corners[3].kp = [int(x1_L), int(y2_L)]
                        
                        objs_stamped_msg.objects.append(obj_msg)
                                            
                        cv2.rectangle(annotated_L, (int(x1_L), int(y1_L)), (int(x2_L), int(y2_L)), (0, 255, 0), 2)
                        label = f"{class_name} X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m"
                        cv2.putText(annotated_L, label, (int(x1_L), int(y1_L)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detection_saved = True
            
            if detection_saved:
                self.get_logger().info("Saving to file and publishing")
                filename = os.path.join(self.save_dir, f"stereo_det_{self.frame_count:05d}.jpg")
                success = cv2.imwrite(filename, annotated_L)
                self.frame_count += 1
                
                self.objects_stamped_pub.publish(objs_stamped_msg)
            else:
                self.get_logger().info("No object detected...")


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