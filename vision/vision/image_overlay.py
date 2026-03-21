#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from zed_msgs.msg import ObjectsStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from cv_bridge import CvBridge
import cv2


class ZedOverlayNode(Node):
    def __init__(self):
        super().__init__('overlay_node')
        self.bridge = CvBridge()
        
        # Stocker les N derniers objets avec leurs timestamps
        self.objects_history = []  # Liste de tuples (timestamp, ObjectsStamped)
        self.max_history = 100  # Garder les 100 dernières détections
        
        # Compteurs
        self.image_count = 0
        self.objects_count = 0
        self.matched_count = 0
        self.unmatched_count = 0

        # QoS RELIABLE pour correspondre au rosbag
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.get_logger().info('Creating subscriptions')

        # Subscriber pour les images
        self.image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/color/rect/image',
            self.image_callback,
            qos
        )

        # Subscriber pour les objets
        self.objects_sub = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.objects_callback,
            qos
        )

        # Publisher BEST_EFFORT
        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.overlay_pub = self.create_publisher(
            Image,
            'aeac/internal/vision/overlay_image',
            pub_qos
        )

        # Timer pour les stats
        self.create_timer(10.0, self.print_stats)

        self.get_logger().info('=== Overlay Node started (timestamp matching) ===')

    def objects_callback(self, msg):
        """Stocke les objets avec leur timestamp"""
        self.objects_count += 1
        
        # Ajouter à l'historique
        timestamp = msg.header.stamp
        self.objects_history.append((timestamp, msg))
        
        # Limiter la taille de l'historique
        if len(self.objects_history) > self.max_history:
            self.objects_history.pop(0)
        
        if self.objects_count % 50 == 0:
            self.get_logger().info(
                f'Objects buffer: {len(self.objects_history)} entries, '
                f'{len(msg.objects)} objects in latest'
            )

    def find_matching_objects(self, image_timestamp, tolerance_sec=0.1):
        """
        Trouve les objets avec le timestamp le plus proche de l'image
        
        Args:
            image_timestamp: timestamp de l'image
            tolerance_sec: tolérance maximale en secondes
        
        Returns:
            ObjectsStamped ou None si aucun match trouvé
        """
        if not self.objects_history:
            return None
        
        # Convertir le timestamp en secondes
        img_time = image_timestamp.sec + image_timestamp.nanosec / 1e9
        
        best_match = None
        best_diff = float('inf')
        
        for obj_timestamp, obj_msg in self.objects_history:
            obj_time = obj_timestamp.sec + obj_timestamp.nanosec / 1e9
            time_diff = abs(img_time - obj_time)
            
            if time_diff < best_diff:
                best_diff = time_diff
                best_match = obj_msg
        
        # Vérifier si le match est dans la tolérance
        if best_diff <= tolerance_sec:
            return best_match, best_diff
        else:
            return None, best_diff

    def image_callback(self, image_msg):
        """Traite chaque image avec les objets synchronisés"""
        self.image_count += 1
        
        if self.image_count == 1:
            self.get_logger().info('First image received!')
        
        try:
            image_cv = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Chercher les objets correspondant à cette image
        result = self.find_matching_objects(image_msg.header.stamp, tolerance_sec=0.15)
        
        if result[0] is not None:
            matched_objects, time_diff = result
            self.matched_count += 1
            
            # Log occasionnel
            if self.matched_count % 50 == 0:
                self.get_logger().info(
                    f'Matched frame #{self.image_count} with objects '
                    f'(dt={time_diff*1000:.1f}ms)'
                )
            
            self.draw_objects(image_cv, matched_objects, time_diff)
        else:
            self.unmatched_count += 1
            time_diff = result[1]
            
            # Afficher un avertissement sur l'image
            cv2.putText(
                image_cv,
                f"No matching objects (nearest: {time_diff*1000:.0f}ms)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )
            
            if self.unmatched_count % 50 == 0:
                self.get_logger().warn(
                    f'No match for frame #{self.image_count} '
                    f'(nearest: {time_diff*1000:.0f}ms)'
                )

        # Publier l'image
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(image_cv, encoding='bgr8')
            overlay_msg.header = image_msg.header
            self.overlay_pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish: {e}")

    def draw_objects(self, image_cv, objects_msg, time_diff):
        """Dessine les bounding boxes"""
        num_objects = len(objects_msg.objects)
        
        for obj in objects_msg.objects:
            if not hasattr(obj, 'bounding_box_2d') or obj.bounding_box_2d is None:
                continue
            
            if not hasattr(obj.bounding_box_2d, 'corners') or len(obj.bounding_box_2d.corners) == 0:
                continue
            
            try:
                x_coords = [int(p.kp[0]) for p in obj.bounding_box_2d.corners]
                y_coords = [int(p.kp[1]) for p in obj.bounding_box_2d.corners]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Rectangle vert
                cv2.rectangle(
                    image_cv,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

                label = obj.label if hasattr(obj, 'label') else 'Unknown'
                confidence = obj.confidence if hasattr(obj, 'confidence') else 0.0
                
                text = f"{label} ({confidence:.2f})"
                
                # Position 3D
                if hasattr(obj, 'position') and len(obj.position) >= 3:
                    pos = obj.position
                    pos_text = f"x={pos[0]:.2f}m y={pos[1]:.2f}m z={pos[2]:.2f}m"
                else:
                    pos_text = "No 3D pos"

                # Afficher label
                cv2.putText(
                    image_cv,
                    text,
                    (x_min, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Afficher position
                cv2.putText(
                    image_cv,
                    pos_text,
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                
            except Exception as e:
                self.get_logger().warn(f"Error drawing object: {e}")
                continue
        
        # Afficher info de sync en haut à droite
        height, width = image_cv.shape[:2]
        sync_text = f"Sync: {time_diff*1000:.1f}ms | Objects: {num_objects}"
        cv2.putText(
            image_cv,
            sync_text,
            (width - 400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    def print_stats(self):
        """Statistiques détaillées"""
        match_rate = (self.matched_count / self.image_count * 100) if self.image_count > 0 else 0
        
        self.get_logger().info(
            f'=== Stats: {self.image_count} images, {self.objects_count} obj updates | '
            f'Matched: {self.matched_count} ({match_rate:.1f}%), '
            f'Unmatched: {self.unmatched_count} ==='
        )


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ZedOverlayNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutdown requested...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
