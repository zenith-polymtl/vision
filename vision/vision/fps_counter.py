import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import time
from custom_interfaces.msg import AimError

class FpsCounterNode(Node):
    def __init__(self):
        super().__init__('fps_counter_node')

        # Declare parameters for topic names
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('target_error_topic', '/aeac/internal/gimbal/target_error')
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').value
        self.target_error_topic = self.get_parameter('target_error_topic').value

        # Image subscription
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # Target error subscription
        self.target_error_subscription = self.create_subscription(
            AimError,
            self.target_error_topic,
            self.target_error_callback,
            10
        )

        # Image FPS tracking
        self.image_frame_count = 0
        self.image_start_time = time.time()
        self.image_received = False
        
        # Target error FPS tracking
        self.target_error_frame_count = 0
        self.target_error_start_time = time.time()
        self.target_error_received = False
        
        # Create a timer to log FPS every 1 second
        self.create_timer(1.0, self.log_fps)
        
        self.get_logger().info(f"FPS Counter started. Monitoring:")
        self.get_logger().info(f"  - Images on: {self.image_topic}")
        self.get_logger().info(f"  - Target errors on: {self.target_error_topic}")

    def image_callback(self, msg):
        if not self.image_received:
            self.get_logger().info(f"✓ Receiving images on {self.image_topic}")
            self.image_received = True
            
        self.image_frame_count += 1

    def target_error_callback(self, msg):
        if not self.target_error_received:
            self.get_logger().info(f"✓ Receiving target errors on {self.target_error_topic}")
            self.target_error_received = True
            
        self.target_error_frame_count += 1

    def log_fps(self):
        """Log FPS for both feeds on a single line every 1 second"""
        current_time = time.time()
        
        # Calculate image FPS
        image_elapsed = current_time - self.image_start_time
        if image_elapsed > 0:
            image_fps = self.image_frame_count / image_elapsed
        else:
            image_fps = 0.0
        
        # Calculate target error FPS
        target_elapsed = current_time - self.target_error_start_time
        if target_elapsed > 0:
            target_fps = self.target_error_frame_count / target_elapsed
        else:
            target_fps = 0.0
        
        # Log both on one line
        self.get_logger().info(f"FPS | Image: {image_fps:.1f}  Target Error: {target_fps:.1f}")
        
        # Reset counters
        self.image_frame_count = 0
        self.image_start_time = current_time
        self.target_error_frame_count = 0
        self.target_error_start_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = FpsCounterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down FPS Counter...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
