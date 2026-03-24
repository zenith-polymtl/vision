import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import os
from ultralytics import YOLO  # make sure ultralytics is installed on your system

from std_msgs.msg import Bool
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
        super().__init__('rgb_yolo')
        
        self.intialize_parameters()
        self.intialize_attributes()
        self.intialize_topics()
        
        self.get_logger().info("YOLO Node Started")
        
    def intialize_attributes(self):
        self.intialize_topics()
        self.bridge = CvBridge()        
        self.is_activated = False
        
        # Load YOLO model
        pkg_share = get_package_share_directory('vision')
        model_path = os.path.join(pkg_share, 'models', self.model_named)
        self.model = YOLO(model_path)
                
        # Variables for saving picture localy
        # os.makedirs(self.save_dir, exist_ok=True)
        # self.frame_count = 0

    
    def intialize_parameters(self):
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('activation_topic', '/aeac/internal/auto_shoot/start_hr_aiming')
        self.declare_parameter('gimbal_error_topic', '/aeac/internal/gimbal/target_error')
        self.declare_parameter('image_save_dir', '/water_ws/Pictures/yolo_without_distances')
        self.declare_parameter('model_name', 'best-medium.pt')
        self.declare_parameter('initial_offset_x', 0.0)
        self.declare_parameter('initial_offset_y', 0.0)

        gp = self.get_parameter

        self.image_topic = gp('image_topic').value
        self.activation_topic = gp('activation_topic').value
        self.gimbal_error_topic = gp('gimbal_error_topic').value
        self.model_named = gp('model_name').value
        self.save_dir = gp('image_save_dir').value
        self.offset_x = gp('initial_offset_x').value
        self.offset_y = gp('initial_offset_y').value


    
    def intialize_topics(self):
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(Bool, self.activation_topic, self.activation_callback, 10)

        self.error_publisher = self.create_publisher(AimError, self.gimbal_error_topic, 10)
            
    def activation_callback(self, msg):
        self.is_activated = msg.data

    def image_callback(self, msg):
        if not self.is_activated:
            return
        try:
            # Convert ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run YOLO inference
            results = self.model(cv_image)  # returns a Results object

            height, width = cv_image.shape[:2]
            error_pitch, error_yaw = compute_error(results, width, height, self.offset_x, self.offset_y)

            target_error = AimError()
            target_error.pitch_error = error_pitch
            target_error.yaw_error = error_yaw
            self.error_publisher.publish(target_error)

            # Draw results on image
            # annotated_frame = results[0].plot()  # returns numpy image with boxes and labels

            # Save the annotated frame to disk instead of showing it
            # filename = os.path.join(self.save_dir, f"yolo_detection_{self.frame_count:05d}.jpg")
            # cv2.imwrite(filename, annotated_frame)
            
            # self.frame_count += 1

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