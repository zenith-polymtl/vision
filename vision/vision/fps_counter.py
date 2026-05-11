import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
from custom_interfaces.msg import AimError

class FpsCounterNode(Node):
    def __init__(self):
        super().__init__('fps_counter_node')


        
        image = False
        
        if image:
            self.topic_name = '/zed/zed_node/left/color/rect/image' 
            
            # QoS profile of 10 keeps a small queue to prevent latency buildup
            self.subscription = self.create_subscription(
                Image,
                self.topic_name,
                self.callback,
                10
            )
        else:
            self.topic_name = '/aeac/internal/gimbal/target_error'

            self.subscription = self.create_subscription(
                AimError,
                self.topic_name,
                self.callback,
                10
            )

        
        self.frame_count = 0
        self.start_time = time.time()
        
        self.get_logger().info(f"FPS Counter started. Waiting for images on: {self.topic_name}")

    def callback(self, msg):
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate and print FPS every 1.0 seconds
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.get_logger().info(f"Incoming Feed: {fps:.1f} FPS")
            
            # Reset counters for the next second
            self.frame_count = 0
            self.start_time = current_time

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
