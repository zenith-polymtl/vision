#!/usr/bin/env python3
"""
wall_detector.py — Détection de murs + construction de référentiels locaux

ARCHITECTURE
────────────
Chaque SceneFrame est traité de façon complètement indépendante.
frame_planes contient UNIQUEMENT les plans détectés dans cette frame.
self.plane_memory est utilisé UNIQUEMENT pour la visualisation RViz.

REPÈRE CAMÉRA (ZED ROS2 wrapper, REP-103)
──────────────────────────────────────────
  X = Forward  (devant la caméra)
  Y = Left     (gauche)
  Z = Up       (haut)

VECTEUR UP
──────────
  Caméra à plat (rosbag/test) : camera_pitch_deg=0  → V_UP = [0, 0, 1]
  Caméra à 45° vers le bas    : camera_pitch_deg=45 → V_UP = [-0.707, 0, 0.707]

FORMAT JSON PUBLIÉ sur /scene_description
─────────────────────────────────────────
{
  "timestamp": float,
  "cloud_dt":  float (ms),
  "image_dt":  float (ms) | null,
  "targets": [
    {
      "id":           int,
      "label":        str,
      "wall_normal":  [nx, ny, nz],      ← normale du plan de la cible
      "reference":    {"id": int, "label": str} | null,
      "local_coords": {"x": float,       ← droite sur le mur
                       "y": float,       ← haut sur le mur
                       "z": float}       ← profondeur (bruit, non utilisé)
                    | null,
      "h_rel_cam":    float              ← Z du centroïde (Z=Up)
    }
  ]
}
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
import open3d as o3d
import gc
import json
import math
from collections import deque

from sensor_msgs.msg import PointCloud2, Image
from zed_msgs.msg import ObjectsStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

from std_msgs.msg import Float64
from std_msgs.msg import String


REFERENCE_LABELS = {'window', 'door', 'fenetre', 'porte'}
TARGET_LABELS    = {'person', 'target', 'cible',
                    'red_target', 'blue_target', 'green_target',
                    'yellow_target', 'orange_target', 'red_circle', 'blue_circle', 'green_circle', 'yellow_circle', 'circle_red', 'circle_blue', 'circle_green', 'circle_yellow'}


# ── FONCTIONS PURES ────────────────────────────────────────────────────────────

def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def find_closest(buffer: deque, target_time: float):
    if not buffer:
        return None, float('inf')
    best_msg, best_diff = None, float('inf')
    for ts, msg in buffer:
        diff = abs(ts - target_time)
        if diff < best_diff:
            best_diff, best_msg = diff, msg
    return best_msg, best_diff


def build_local_frame(normal: np.ndarray, v_up: np.ndarray) -> np.ndarray:
    """
    Construit R (3×3) dont les colonnes sont [X_local, Y_local, Z_local] :
      Z_local = normale (vers la caméra)
      Y_local = projection de v_up sur le plan ("haut" sur le mur)
      X_local = Y_local × Z_local ("droite" sur le mur)
    """
    z_loc  = normal
    y_loc  = v_up - np.dot(v_up, z_loc) * z_loc
    norm_y = np.linalg.norm(y_loc)
    if norm_y < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0])
        y_loc    = fallback - np.dot(fallback, z_loc) * z_loc
        y_loc   /= np.linalg.norm(y_loc)
    else:
        y_loc /= norm_y
    x_loc = np.cross(y_loc, z_loc)
    return np.column_stack([x_loc, y_loc, z_loc])


def to_local_coords(R: np.ndarray, origin: np.ndarray, point: np.ndarray) -> np.ndarray:
    return R.T @ (point - origin)


def is_same_plane(m1: np.ndarray, m2: np.ndarray,
                  sim_thresh: float, dist_thresh: float) -> bool:
    sim  = float(np.abs(np.dot(m1[:3], m2[:3])))
    dist = float(np.abs(m1[3] - m2[3]))
    return sim > sim_thresh and dist < dist_thresh


# ── SCENEFRAME ─────────────────────────────────────────────────────────────────

class SceneFrame:
    __slots__ = ('det_time', 'objects_msg', 'cloud_msg', 'image_msg', 'cloud_dt', 'image_dt')

    def __init__(self, objects_msg, cloud_msg, image_msg, cloud_dt, image_dt):
        self.det_time    = stamp_to_sec(objects_msg.header.stamp)
        self.objects_msg = objects_msg
        self.cloud_msg   = cloud_msg
        self.image_msg   = image_msg
        self.cloud_dt    = cloud_dt
        self.image_dt    = image_dt


# ── NODE ───────────────────────────────────────────────────────────────────────

class ZEDWallDetector(Node):
    def __init__(self):
        super().__init__('wall_detector_node')

        # ── PARAMÈTRES ────────────────────────────────────────────────────────
        self.declare_parameter('ransac_dist',       0.03)
        self.declare_parameter('ransac_iterations', 150)
        self.declare_parameter('min_inlier_ratio',  0.50)
        self.declare_parameter('padding_ratio',     0.20)
        self.declare_parameter('sync_tolerance',    1.50)
        self.declare_parameter('image_tolerance',   5.00)
        self.declare_parameter('buffer_size',       60)
        self.declare_parameter('camera_pitch_deg',  0.0)
        self.declare_parameter('plane_sim_thresh',  0.85)
        self.declare_parameter('plane_dist_thresh', 0.35)
        self.declare_parameter('image_width',  1280)
        self.declare_parameter('image_height', 720)


        self.rans_dist   = self.get_parameter('ransac_dist').value
        self.rans_iter   = self.get_parameter('ransac_iterations').value
        self.min_inliers = self.get_parameter('min_inlier_ratio').value
        self.padding     = self.get_parameter('padding_ratio').value
        self.sync_tol    = self.get_parameter('sync_tolerance').value
        self.img_tol     = self.get_parameter('image_tolerance').value
        self.buf_size    = self.get_parameter('buffer_size').value
        self.sim_thresh  = self.get_parameter('plane_sim_thresh').value
        self.dist_thresh = self.get_parameter('plane_dist_thresh').value
        self.altitude_buffer = deque(maxlen=self.buf_size)
        self.heading_buffer  = deque(maxlen=self.buf_size)
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value

        pitch    = math.radians(self.get_parameter('camera_pitch_deg').value)
        self.v_up = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])

        # ── ÉTAT ──────────────────────────────────────────────────────────────
        self.cloud_buffer  = deque(maxlen=self.buf_size)
        self.image_buffer  = deque(maxlen=self.buf_size)
        self.plane_memory  = {}   # RViz uniquement
        self.is_processing = False
        self.cnt_cloud = self.cnt_image = self.cnt_obj = self.cnt_frames = 0

        # ── QoS ───────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        mavros_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )


        self.create_subscription(
            Float64,'/mavros/global_position/compass_hdg',self._heading_cb, mavros_qos)
        self.create_subscription(
            PoseStamped,'/mavros/local_position/pose',self._altitude_cb, mavros_qos)
        self.create_subscription(
            PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.cloud_cb, qos)
        self.create_subscription(
            Image, '/zed/zed_node/rgb/color/rect/image', self.image_cb, qos)
        self.create_subscription(
            ObjectsStamped, '/zed/zed_node/obj_det/objects', self.obj_cb, qos)

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_walls',    10)
        self.scene_pub  = self.create_publisher(String,      '/scene_description', 10)

        self.create_timer(5.0, self._log_stats)

        self.get_logger().info(
            f'WallDetector démarré\n'
            f'  v_up={self.v_up.round(3).tolist()} '
            f'(pitch={self.get_parameter("camera_pitch_deg").value}°)\n'
            f'  sim_thresh={self.sim_thresh}  dist_thresh={self.dist_thresh}m'
        )

    # ── BUFFERS ───────────────────────────────────────────────────────────────

    def cloud_cb(self, msg):
        self.cloud_buffer.append((stamp_to_sec(msg.header.stamp), msg))
        self.cnt_cloud += 1

    def _heading_cb(self, msg):
    # compass_hdg n'a pas de header — on utilise le temps ROS courant
        t = self.get_clock().now().nanoseconds * 1e-9
        self.heading_buffer.append((t, msg.data))

    def _altitude_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self.altitude_buffer.append((t, msg.pose.position.z))

    def image_cb(self, msg):
        self.image_buffer.append((stamp_to_sec(msg.header.stamp), msg))
        self.cnt_image += 1

    def _log_stats(self):
        self.get_logger().info(
            f'[STATS 5s] cloud:{self.cnt_cloud} img:{self.cnt_image} '
            f'obj:{self.cnt_obj} frames:{self.cnt_frames}'
        )
        self.cnt_cloud = self.cnt_image = self.cnt_obj = self.cnt_frames = 0

    # ── DÉCLENCHEUR ───────────────────────────────────────────────────────────

    def obj_cb(self, msg):
        self.cnt_obj += 1
        if self.is_processing:
            return

        det_time = stamp_to_sec(msg.header.stamp)

        cloud_msg, cloud_dt = find_closest(self.cloud_buffer, det_time)
        if cloud_msg is None or cloud_dt > self.sync_tol:
            self.get_logger().warn(
                f'Cloud absent ou trop loin (dt={cloud_dt*1000:.0f}ms)')
            return

        image_msg, image_dt = find_closest(self.image_buffer, det_time)
        if image_msg is None or image_dt > self.img_tol:
            image_msg, image_dt = None, None

        frame = SceneFrame(msg, cloud_msg, image_msg, cloud_dt, image_dt)
        self.is_processing = True
        try:
            self._process_frame(frame)
            self.cnt_frames += 1
        except Exception as e:
            self.get_logger().error(f'Erreur: {e}')
        finally:
            self.is_processing = False

    # ── TRAITEMENT ────────────────────────────────────────────────────────────

    def _process_frame(self, frame: SceneFrame):
        cloud_msg    = frame.cloud_msg
        current_time = frame.det_time
        h, w         = cloud_msg.height, cloud_msg.width

        alt_val, alt_dt = find_closest(self.altitude_buffer, current_time)
        hdg_val, hdg_dt = find_closest(self.heading_buffer,  current_time)

        scale_u = w / self.img_w
        scale_v = h / self.img_h
        drone_altitude = alt_val
        drone_heading  = hdg_val if hdg_val is not None else 0.0

        if alt_dt > 0.5:
            self.get_logger().warn(f'Altitude MAVROS désynchronisée (dt={alt_dt*1000:.0f}ms)')
        if hdg_dt > 0.5:
            self.get_logger().warn(f'Heading(N,S,E,W) MAVROS désynchronisé (dt={hdg_dt*1000:.0f}ms)')

        cloud_bytes = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        dtype       = np.dtype([('x', np.float32), ('y', np.float32),
                                 ('z', np.float32), ('rgb', np.float32)])
        cloud_2d    = np.ndarray(shape=(h, w), dtype=dtype, buffer=cloud_bytes)

        # ── ÉTAPE 1 : RANSAC — un plan par objet dans cette frame ─────────────
        frame_planes = {}

        for idx, obj in enumerate(frame.objects_msg.objects):
            uid   = int(getattr(obj, 'label_id', 0) + idx * 100)
            label = obj.label

            corners = obj.bounding_box_2d.corners
            us = [c.kp[0] * scale_u for c in corners]
            vs = [c.kp[1] * scale_v for c in corners]
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)
            bw, bh = u_max - u_min, v_max - v_min

            u0 = max(0,     int(u_min - bw * self.padding))
            u1 = min(w - 1, int(u_max + bw * self.padding))
            v0 = max(0,     int(v_min - bh * self.padding))
            v1 = min(h - 1, int(v_max + bh * self.padding))

            if u1 <= u0 or v1 <= v0:
                self.get_logger().warn(
                    f'[{label}] ROI invalide — bbox hors du cloud {w}×{h}')
                continue

            roi = cloud_2d[v0:v1:3, u0:u1:3]
            pts = np.stack(
                [roi['x'].flatten(), roi['y'].flatten(), roi['z'].flatten()], axis=1)
            pts = pts[~np.isnan(pts).any(axis=1)]

            if len(pts) < 100:
                self.get_logger().warn(
                    f'[{label}] SKIP — {len(pts)} points valides (< 100)')
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            model, inliers = pcd.segment_plane(self.rans_dist, 3, self.rans_iter)

            ratio = len(inliers) / len(pts)
            if ratio < self.min_inliers:
                self.get_logger().warn(
                    f'[{label}] SKIP — plan peu fiable ({ratio:.0%} inliers)')
                continue

            a, b, c, d = model
            norm_n   = np.linalg.norm([a, b, c])
            normal   = np.array([a, b, c]) / norm_n
            d_norm   = d / norm_n
            centroid = np.mean(pts[inliers], axis=0)

            if np.dot(normal, centroid) > 0:
                normal = -normal
                d_norm = -d_norm

            frame_planes[uid] = {
                'model':    np.array([*normal, d_norm]),
                'R':        build_local_frame(normal, self.v_up),
                'centroid': centroid,
                'label':    label,
            }

            self.get_logger().info(
                f'[{label}|{uid}] plan OK — '
                f'n={normal.round(3).tolist()} '
                f'centroïde={centroid.round(3).tolist()} '
                f'inliers={len(inliers)}/{len(pts)} ({ratio:.0%})'
            )

        if not frame_planes:
            return

        # ── ÉTAPE 2 : COMPARAISON PAR PAIRES dans cette frame ─────────────────
        ids    = list(frame_planes.keys())
        colors = {uid: (1.0, 0.0, 0.0) for uid in ids}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                p1, p2   = frame_planes[id1], frame_planes[id2]
                same     = is_same_plane(
                    p1['model'], p2['model'], self.sim_thresh, self.dist_thresh)
                color = (0.0, 1.0, 0.0) if same else (1.0, 1.0, 0.0)
                colors[id1] = color
                colors[id2] = color
                sim  = float(np.abs(np.dot(p1['model'][:3], p2['model'][:3])))
                dist = float(np.abs(p1['model'][3] - p2['model'][3]))
                self.get_logger().info(
                    f'[{p1["label"]}] vs [{p2["label"]}] : '
                    f'{"MÊME PLAN" if same else "PLANS DIFF"} '
                    f'(sim={sim:.3f}, dist={dist:.3f}m)'
                )

        # ── ÉTAPE 3 : DESCRIPTION PAR CIBLE ───────────────────────────────────
        scene_targets = []
        marker_array  = MarkerArray()

        for uid in ids:
            data     = frame_planes[uid]
            label    = data['label']
            centroid = data['centroid']
            normal   = data['model'][:3]
            # Hauteur absolue = altitude drone + hauteur relative de la cible dans le repère caméra
            if drone_altitude is not None:
                h_abs = drone_altitude + float(centroid[2])
                height_m = round(h_abs, 3)
                height_source = 'absolute'
            else:
                height_m = round(float(centroid[2]), 3)
                height_source = 'relative_cam'

            marker_array.markers.append(
                self._make_arrow(uid, data, colors[uid], cloud_msg.header))

            if label.lower() not in TARGET_LABELS:
                continue

            # Référence sur le MÊME plan dans CETTE frame uniquement
            ref_id, ref_data = self._find_reference_same_plane(
                uid, data, frame_planes)

            wall_normal = [round(float(n), 4) for n in normal]

            if ref_id is not None:
                P = to_local_coords(
                    ref_data['R'], ref_data['centroid'], centroid)
                self.get_logger().info(
                    f'[{label}|{uid}] → réf [{ref_data["label"]}|{ref_id}] | '
                    f'x={P[0]:+.3f}m droite, y={P[1]:+.3f}m haut | '
                    f'height_m={height_m:+.3f}m'
                )
                scene_targets.append({
                    'id':           uid,
                    'label':        label,
                    'wall_normal':  wall_normal,
                    'reference':    {'id': ref_id, 'label': ref_data['label']},
                    'local_coords': {
                        'x': round(float(P[0]), 3),
                        'y': round(float(P[1]), 3),
                        'z': round(float(P[2]), 3),
                    },
                    'height_m': round(height_m, 3),
                    'height_source': height_source,
                })
            else:
                self.get_logger().info(
                    f'[{label}|{uid}] sans référence sur ce plan | '
                    f'height_m={height_m:+.3f}m'
                )
                scene_targets.append({
                    'id':           uid,
                    'label':        label,
                    'wall_normal':  wall_normal,
                    'reference':    None,
                    'local_coords': None,
                    'height_m':    round(height_m, 3),
                    'height_source': height_source,
                })

        # Publication
        self.marker_pub.publish(marker_array)

        # Mise à jour mémoire RViz (persistance 3s, pas utilisée pour la description)
        for uid, data in frame_planes.items():
            self.plane_memory[uid] = {**data, 'time': current_time}
        self.plane_memory = {
            k: v for k, v in self.plane_memory.items()
            if current_time - v['time'] < 3.0}

        if scene_targets:
            out      = String()
            out.data = json.dumps({
                'timestamp': round(current_time, 4),
                'cloud_dt':  round(frame.cloud_dt * 1000, 1),
                'image_dt':  round(frame.image_dt * 1000, 1) if frame.image_dt else None,
                'drone_heading':  round(drone_heading, 1),
                'targets':   scene_targets,
            })
            self.scene_pub.publish(out)
            self.get_logger().info(
                f'/scene_description → {len(scene_targets)} cible(s)')

        gc.collect()

    # ── UTILITAIRES ───────────────────────────────────────────────────────────

    def _find_reference_same_plane(self, target_id, target_data, frame_planes):
        best_id, best_data, best_dist = None, None, float('inf')
        for oid, data in frame_planes.items():
            if oid == target_id:
                continue
            if data['label'].lower() not in REFERENCE_LABELS:
                continue
            if not is_same_plane(target_data['model'], data['model'],
                                  self.sim_thresh, self.dist_thresh):
                continue
            d = float(np.linalg.norm(data['centroid'] - target_data['centroid']))
            if d < best_dist:
                best_dist, best_id, best_data = d, oid, data
        return best_id, best_data

    def _make_arrow(self, uid, data, color, header):
        m = Marker()
        m.header = header
        m.ns, m.id = 'walls', uid
        m.type, m.action = Marker.ARROW, Marker.ADD
        c = data['centroid']
        n = data['model'][:3]
        m.points = [
            Point(x=float(c[0]),           y=float(c[1]),           z=float(c[2])),
            Point(x=float(c[0]+n[0]*0.5), y=float(c[1]+n[1]*0.5), z=float(c[2]+n[2]*0.5)),
        ]
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.1, 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 1.0
        m.lifetime = Duration(sec=1, nanosec=0)
        return m


def main():
    rclpy.init()
    rclpy.spin(ZEDWallDetector())
    rclpy.shutdown()


if __name__ == '__main__':
    main()