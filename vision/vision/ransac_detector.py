#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
import open3d as o3d
import gc
import time

from sensor_msgs.msg import PointCloud2
from zed_msgs.msg import ObjectsStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

class ZEDWallAdvancedDetector(Node):
    def __init__(self):
        super().__init__('wall_detector_node')
        
        # --- PARAMÈTRES ---
        self.declare_parameter('ransac_dist', 0.03)
        self.declare_parameter('padding_ratio', 0.2) # On agrandit la zone de 30%
        
        self.rans_dist = self.get_parameter('ransac_dist').value
        self.padding = self.get_parameter('padding_ratio').value
        
        # Mémoire des plans { unique_key: {data} }
        self.plane_memory = {}
        self.latest_cloud = None
        self.is_processing = False
        
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )


        self.cloud_sub = self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.cloud_cb, qos)
        self.obj_sub = self.create_subscription(ObjectsStamped, '/zed/zed_node/obj_det/objects', self.obj_cb, qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_walls', 10)
        
        self.get_logger().info('Détecteur de Murs : Mode VOISINAGE ÉTENDU activé')

    def cloud_cb(self, msg):
        self.latest_cloud = msg

    def obj_cb(self, msg):
        if self.latest_cloud is None or self.is_processing:
            return
        
        self.is_processing = True
        try:
            cloud_msg = self.latest_cloud
            current_time = time.time()
            h, w = cloud_msg.height, cloud_msg.width
            
            # Conversion rapide
            cloud_array = np.frombuffer(cloud_msg.data, dtype=np.uint8)
            dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)])
            cloud_2d = np.ndarray(shape=(h, w), dtype=dtype, buffer=cloud_array)

            marker_array = MarkerArray()
            num_detected_this_frame = 0

            for idx, obj in enumerate(msg.objects):
                # 1. CRÉATION D'UN ID UNIQUE (Même si ZED renvoie ID 0 pour tous)
                # On combine l'ID ZED et l'index dans le tableau pour éviter l'écrasement
                unique_id = int(getattr(obj, 'id', 0) + (idx * 100))
                
                # 2. VOISINAGE ÉTENDU (Padding de la BBox)
                bbox = obj.bounding_box_2d.corners
                u = [c.kp[0] for c in bbox]; v = [c.kp[1] for c in bbox]
                u_min, u_max = min(u), max(u)
                v_min, v_max = min(v), max(v)
                
                # Calcul des dimensions pour le padding
                bw = u_max - u_min
                bh = v_max - v_min
                
                # On gonfle le rectangle (voisinage)
                u_min_ext = max(0, int(u_min - bw * self.padding))
                u_max_ext = min(w - 1, int(u_max + bw * self.padding))
                v_min_ext = max(0, int(v_min - bh * self.padding))
                v_max_ext = min(h - 1, int(v_max + bh * self.padding))

                # 3. EXTRACTION DES POINTS (Plus dense: step=1 ou 2)
                roi = cloud_2d[v_min_ext:v_max_ext:3, u_min_ext:u_max_ext:3]
                pts = np.stack([roi['x'].flatten(), roi['y'].flatten(), roi['z'].flatten()], axis=1)
                pts = pts[~np.isnan(pts).any(axis=1)]

                # 4. RANSAC
                if len(pts) > 100:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    model, inliers = pcd.segment_plane(self.rans_dist, 3, 150)
                    
                    # Normalisation
                    a, b, c, d = model
                    n = np.linalg.norm([a, b, c])
                    norm_model = np.array([a/n, b/n, c/n, d/n])
                    centroid = np.mean(pts[inliers], axis=0)

                    # Stockage en mémoire
                    self.plane_memory[unique_id] = {
                        'model': norm_model, 
                        'time': current_time, 
                        'centroid': centroid,
                        'label': obj.label
                    }
                    num_detected_this_frame += 1

            # --- NETTOYAGE ET ANALYSE ---
            # On oublie les objets disparus depuis plus de 3s
            self.plane_memory = {k: v for k, v in self.plane_memory.items() if current_time - v['time'] < 3.0}
            active_ids = list(self.plane_memory.keys())

            if len(active_ids) >= 2:
                self.get_logger().info(f"Analyse : {len(active_ids)} surfaces actives...")
                for i in range(len(active_ids)):
                    for j in range(i + 1, len(active_ids)):
                        id1, id2 = active_ids[i], active_ids[j]
                        p1, p2 = self.plane_memory[id1]['model'], self.plane_memory[id2]['model']
                        
                        sim = np.abs(np.dot(p1[:3], p2[:3]))
                        dist = np.abs(p1[3] - p2[3])

                        # CRITÈRE : MÊME MUR
                        if sim > 0.98 and dist < 0.15:
                            status = "MÊME PLAN"
                            color = (0.0, 1.0, 0.0) # VERT
                        else:
                            status = "PLANS DIFF"
                            color = (1.0, 1.0, 0.0) # JAUNE
                        
                        self.get_logger().info(f"[{id1}] vs [{id2}] : {status} (Sim:{sim:.3f}, Dist:{dist:.2f}m)")
                        
                        marker_array.markers.append(self.create_arrow(id1, self.plane_memory[id1], color, cloud_msg.header))
                        marker_array.markers.append(self.create_arrow(id2, self.plane_memory[id2], color, cloud_msg.header))
            
            elif len(active_ids) == 1:
                oid = active_ids[0]
                marker_array.markers.append(self.create_arrow(oid, self.plane_memory[oid], (1.0, 0.0, 0.0), cloud_msg.header))

            self.marker_pub.publish(marker_array)
            gc.collect()

        except Exception as e:
            self.get_logger().error(f"Erreur: {e}")
        
        self.is_processing = False

    def create_arrow(self, unique_id, data, color, header):
        m = Marker()
        m.header = header; m.ns = "walls"; m.id = unique_id; m.type = Marker.ARROW; m.action = Marker.ADD
        c = data['centroid']; n = data['model']
        m.points = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2])),
                    Point(x=float(c[0]+n[0]*0.5), y=float(c[1]+n[1]*0.5), z=float(c[2]+n[2]*0.5))]
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.1, 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 1.0
        m.lifetime = Duration(sec=1, nanosec=0)
        return m

def main():
    rclpy.init()
    rclpy.spin(ZEDWallAdvancedDetector())
    rclpy.shutdown()

if __name__ == '__main__':
    main()