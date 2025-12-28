#!/usr/bin/env python3  
import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import PointCloud2  
from geometry_msgs.msg import PointStamped  
import numpy as np  
import struct  
from custom_interfaces.msg import Pixels  
  
class PixelToXYZFromPointCloud(Node):  
    def __init__(self):  
        super().__init__('pixel_to_xyz_pointcloud')  
          
        # Subscribe to custom Pixels message  
        self.pixels_sub = self.create_subscription(  
            Pixels,  
            '/vision/pixels',  
            self.pixels_callback,  
            10  
        )  
          
        # Subscribe to ZED point cloud  
        self.pc_sub = self.create_subscription(  
            PointCloud2,  
            '/zed/zed_node/point_cloud/cloud_registered',  
            self.pointcloud_callback,  
            10  
        )  
          
        # Publisher for XYZ results  
        self.xyz_pub = self.create_publisher(  
            PointStamped,  
            '/vision/xyz_result',  
            10  
        )  
          
        self.last_pointcloud = None  
        self.get_logger().info('Pixel to XYZ (Point Cloud) converter started')  
  
    def pixels_callback(self, msg):  
        """Process incoming pixel coordinates"""  
        if self.last_pointcloud is None:  
            self.get_logger().warn('No point cloud data available')  
            return  
              
        
        x, y = msg.x, msg.y  
        xyz = self.extract_xyz_from_pointcloud(x, y, self.last_pointcloud)  
        #xyz is in FLU meters (FORWARD, LEFT, UP)
            
        if xyz is not None:  
            # Create and publish PointStamped message  
            point_msg = PointStamped()  
            point_msg.header.stamp = self.get_clock().now().to_msg()  
            point_msg.header.frame_id = 'zed_camera_left_optical_frame'  
            point_msg.point.x = xyz[0]  
            point_msg.point.y = xyz[1]  
            point_msg.point.z = xyz[2]  
                
            self.xyz_pub.publish(point_msg)  
                
            self.get_logger().info(  
                f'Pixel ({x}, {y}) -> XYZ: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] meters'  
            )  
  
    def pointcloud_callback(self, msg):  
        """Store latest point cloud data"""  
        self.last_pointcloud = msg  
  
    def extract_xyz_from_pointcloud(self, pixel_x, pixel_y, pc_msg):  
        """Extract XYZ coordinates from point cloud using pixel coordinates"""  
        width = pc_msg.width  
        height = pc_msg.height  
        point_step = pc_msg.point_step  
          
        # Check if pixel coordinates are valid  
        if pixel_x >= width or pixel_y >= height or pixel_x < 0 or pixel_y < 0:  
            self.get_logger().warn(f'Invalid pixel coordinates: ({pixel_x}, {pixel_y})')  
            return None  
              
        # Calculate index in point cloud (organized point cloud)  
        index = pixel_y * width + pixel_x  
        offset = index * point_step  
          
        # Check if we have enough data  
        if offset + 12 > len(pc_msg.data):  
            self.get_logger().warn(f'Point cloud data insufficient for pixel ({pixel_x}, {pixel_y})')  
            return None  
              
        try:  
            # Extract XYZ from point cloud data (first 12 bytes: 3 floats)  
            x, y, z = struct.unpack('fff', pc_msg.data[offset:offset+12])  
              
            # Check if point is valid (not NaN)  
            if not np.isnan(z) and not np.isnan(x) and not np.isnan(y):  
                return [x, y, z]  
            else:  
                self.get_logger().debug(f'Invalid point at pixel ({pixel_x}, {pixel_y}): NaN values')  
                return None  
                  
        except struct.error as e:  
            self.get_logger().error(f'Error unpacking point cloud data: {e}')  
            return None  
  
def main(args=None):  
    rclpy.init(args=args)  
    node = PixelToXYZFromPointCloud()  
      
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        pass  
    finally:  
        node.destroy_node()  
        rclpy.shutdown()  
  
if __name__ == '__main__':  
    main()