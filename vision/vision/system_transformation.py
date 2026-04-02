#!/usr/bin/env python3
"""
system_transformation.py — Détection de murs + construction de référentiels locaux
VERSION HARDWARE JETSON — optimisée pour latence < 200ms

CHANGEMENTS vs version rosbag
──────────────────────────────
  1. MultiThreadedExecutor + ReentrantCallbackGroup pour les buffers
     → Les callbacks cloud/image/objects tournent en parallèle du RANSAC
  2. Buffers cloud/image réduits à maxlen=3 (on veut le PRÉSENT, pas l'histoire)
     → Élimine cloud_dt de 3-4s causé par des données périmées dans un grand buffer
  3. Rate limiter dans obj_cb : min 0.4s entre deux traitements (≈2.5 FPS)
     → Évite la saturation CPU, le RANSAC finit avant la prochaine détection
  4. Stride adaptatif agressif : vise max 500 points par objet
     → Sur Jetson, 500 pts RANSAC ≈ 30ms vs 5000 pts ≈ 300ms
  5. ransac_iterations défaut abaissé à 80 (suffisant pour des murs plats)
  6. sync_tolerance élargie à 3.0s par défaut pour le hardware

REPÈRE CAMÉRA (ZED ROS2 wrapper, REP-103)
──────────────────────────────────────────
  X = Forward  (devant la caméra)
  Y = Left     (gauche)
  Z = Up       (haut)
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import PoseStamped
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
import numpy as np
import pyransac3d as pyrsc
import gc
import json
import math
from collections import deque

from sensor_msgs.msg import PointCloud2, Image
from zed_msgs.msg import ObjectsStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64, String


# ── LABELS ────────────────────────────────────────────────────────────────────

REFERENCE_LABELS = {'window', 'door', 'fenetre', 'porte'}

TARGET_LABELS = {
    'person', 'target', 'cible',
    'red_target', 'blue_target', 'green_target',
    'yellow_target', 'orange_target',
    'red target', 'blue target', 'green target',
    'yellow target', 'orange target',
    'red_circle', 'blue_circle', 'green_circle', 'yellow_circle',
    'red circle', 'blue circle', 'green circle', 'yellow circle',
    'circle_red', 'circle_blue', 'circle_green', 'circle_yellow',
}

FLOOR_NZ_THRESH = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS PURES
# ══════════════════════════════════════════════════════════════════════════════

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


def to_local_coords(R: np.ndarray, origin: np.ndarray,
                    point: np.ndarray) -> np.ndarray:
    return R.T @ (point - origin)


def is_same_plane(m1: np.ndarray, m2: np.ndarray,
                  sim_thresh: float, dist_thresh: float) -> bool:
    sim  = float(np.abs(np.dot(m1[:3], m2[:3])))
    dist = float(np.abs(m1[3] - m2[3]))
    return sim > sim_thresh and dist < dist_thresh


def normalize_label(label: str) -> str:
    return label.lower().replace('-', '_')


def is_target_label(label: str) -> bool:
    lo = label.lower()
    return lo in TARGET_LABELS or normalize_label(lo) in TARGET_LABELS


def is_reference_label(label: str) -> bool:
    return label.lower() in REFERENCE_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# SCENEFRAME
# ══════════════════════════════════════════════════════════════════════════════

class SceneFrame:
    __slots__ = ('det_time', 'objects_msg', 'cloud_msg',
                 'image_msg', 'cloud_dt', 'image_dt')

    def __init__(self, objects_msg, cloud_msg, image_msg, cloud_dt, image_dt):
        self.det_time    = stamp_to_sec(objects_msg.header.stamp)
        self.objects_msg = objects_msg
        self.cloud_msg   = cloud_msg
        self.image_msg   = image_msg
        self.cloud_dt    = cloud_dt
        self.image_dt    = image_dt


# ══════════════════════════════════════════════════════════════════════════════
# NODE
# ══════════════════════════════════════════════════════════════════════════════

class ZEDWallDetector(Node):

    def __init__(self):
        super().__init__('system_transformation_node')

        # ── Paramètres ────────────────────────────────────────────────────
        self.declare_parameter('ransac_dist',        0.03)
        self.declare_parameter('ransac_iterations',  80)    # ← 150→80 pour Jetson
        self.declare_parameter('min_inlier_ratio',   0.40)  # ← 0.50→0.40 plus souple
        self.declare_parameter('padding_ratio',      0.10)  # ← 0.20→0.10 ROI plus petite
        self.declare_parameter('sync_tolerance',     3.00)  # ← 1.5→3.0 pour hardware
        self.declare_parameter('image_tolerance',    5.00)
        self.declare_parameter('buffer_size',        60)    # buffer overlay inchangé
        self.declare_parameter('camera_pitch_deg',   0.0)
        self.declare_parameter('plane_sim_thresh',   0.85)
        self.declare_parameter('plane_dist_thresh',  0.35)
        self.declare_parameter('image_width',        640)   # ← défaut hardware 640×360
        self.declare_parameter('image_height',       360)
        self.declare_parameter('objects_topic',      '/aeac/test/objects')
        self.declare_parameter('min_proc_interval',  0.4)   # ← NOUVEAU : rate limiter

        self.rans_dist        = self.get_parameter('ransac_dist').value
        self.rans_iter        = self.get_parameter('ransac_iterations').value
        self.min_inliers      = self.get_parameter('min_inlier_ratio').value
        self.padding          = self.get_parameter('padding_ratio').value
        self.sync_tol         = self.get_parameter('sync_tolerance').value
        self.img_tol          = self.get_parameter('image_tolerance').value
        self.buf_size         = self.get_parameter('buffer_size').value
        self.sim_thresh       = self.get_parameter('plane_sim_thresh').value
        self.dist_thresh      = self.get_parameter('plane_dist_thresh').value
        self.img_w            = self.get_parameter('image_width').value
        self.img_h            = self.get_parameter('image_height').value
        objects_topic         = self.get_parameter('objects_topic').value
        self.min_proc_interval= self.get_parameter('min_proc_interval').value

        pitch     = math.radians(self.get_parameter('camera_pitch_deg').value)
        self.v_up = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])

        # ── État ──────────────────────────────────────────────────────────
        # CRITIQUE : buffers cloud/image = 3 seulement → on garde uniquement le PRÉSENT
        # Un grand buffer (60+) garantit des cloud_dt élevés car le RANSAC
        # ne vide pas le buffer, il choisit le message le plus proche qui date
        # déjà de plusieurs secondes quand le CPU est surchargé.
        self.cloud_buffer    = deque(maxlen=3)   # ← WAS self.buf_size
        self.image_buffer    = deque(maxlen=3)   # ← WAS self.buf_size
        self.altitude_buffer = deque(maxlen=10)
        self.heading_buffer  = deque(maxlen=10)
        self.plane_memory    = {}
        self.is_processing   = False
        self.last_proc_time  = 0.0               # ← NOUVEAU : rate limiter
        self.cnt_cloud = self.cnt_image = self.cnt_obj = self.cnt_frames = 0

        # ── Callback groups ───────────────────────────────────────────────
        # Les callbacks de buffer tournent en parallèle du RANSAC (Reentrant)
        # Le callback de traitement est exclusif (MutuallyExclusive)
        self.buf_group  = ReentrantCallbackGroup()
        self.proc_group = MutuallyExclusiveCallbackGroup()

        # ── QoS ───────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5   # ← réduit de 10→5, on ne veut pas de file d'attente
        )
        mavros_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        # Buffers → ReentrantCallbackGroup : tournent même pendant RANSAC
        self.create_subscription(
            Float64, '/mavros/global_position/compass_hdg',
            self._heading_cb, mavros_qos,
            callback_group=self.buf_group)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self._altitude_cb, mavros_qos,
            callback_group=self.buf_group)
        self.create_subscription(
            PointCloud2, '/zed/zed_node/point_cloud/cloud_registered',
            self.cloud_cb, qos,
            callback_group=self.buf_group)
        self.create_subscription(
            Image, '/zed/zed_node/rgb/color/rect/image',
            self.image_cb, qos,
            callback_group=self.buf_group)

        # Traitement → MutuallyExclusiveCallbackGroup : le RANSAC est exclusif
        self.create_subscription(
            ObjectsStamped, objects_topic,
            self.obj_cb, qos,
            callback_group=self.proc_group)

        # ── Publishers ────────────────────────────────────────────────────
        self.marker_pub = self.create_publisher(
            MarkerArray, 'aeac/internal/detected_walls', 10)
        self.scene_pub = self.create_publisher(
            String, 'aeac/internal/scene_description', 10)

        self.create_timer(5.0, self._log_stats,
                          callback_group=self.buf_group)

        self.get_logger().info(
            f'SystemTransformation démarré (mode HARDWARE)\n'
            f'  objects_topic={objects_topic}\n'
            f'  v_up={self.v_up.round(3).tolist()} '
            f'(pitch={self.get_parameter("camera_pitch_deg").value}°)\n'
            f'  bbox_ref={self.img_w}×{self.img_h}\n'
            f'  ransac_iter={self.rans_iter}  min_inliers={self.min_inliers}\n'
            f'  sync_tol={self.sync_tol}s  min_proc_interval={self.min_proc_interval}s\n'
            f'  cloud_buffer=3  image_buffer=3  (mode présent)\n'
            f'  sim_thresh={self.sim_thresh}  '
            f'dist_thresh={self.dist_thresh}m  '
            f'floor_nz_thresh={FLOOR_NZ_THRESH}'
        )

    # ── Buffers (ReentrantCallbackGroup — tournent pendant RANSAC) ────────────

    def cloud_cb(self, msg):
        self.cloud_buffer.append((stamp_to_sec(msg.header.stamp), msg))
        self.cnt_cloud += 1

    def image_cb(self, msg):
        self.image_buffer.append((stamp_to_sec(msg.header.stamp), msg))
        self.cnt_image += 1

    def _heading_cb(self, msg):
        t = self.get_clock().now().nanoseconds * 1e-9
        self.heading_buffer.append((t, msg.data))

    def _altitude_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self.altitude_buffer.append((t, msg.pose.position.z))

    def _log_stats(self):
        self.get_logger().info(
            f'[STATS 5s] cloud:{self.cnt_cloud} img:{self.cnt_image} '
            f'obj:{self.cnt_obj} frames:{self.cnt_frames} '
            f'cloud_buf:{len(self.cloud_buffer)} img_buf:{len(self.image_buffer)}'
        )
        self.cnt_cloud = self.cnt_image = self.cnt_obj = self.cnt_frames = 0

    # ── Déclencheur (MutuallyExclusiveCallbackGroup) ──────────────────────────

    def obj_cb(self, msg):
        self.cnt_obj += 1

        # ── Rate limiter ──────────────────────────────────────────────────
        # Sur Jetson, on limite le traitement à ~2.5 FPS pour que le RANSAC
        # finisse avant la prochaine détection et que les buffers restent frais.
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self.last_proc_time) < self.min_proc_interval:
            return

        if self.is_processing:
            return

        det_time = stamp_to_sec(msg.header.stamp)

        cloud_msg, cloud_dt = find_closest(self.cloud_buffer, det_time)
        if cloud_msg is None or cloud_dt > self.sync_tol:
            self.get_logger().warn(
                f'Cloud absent ou trop loin (dt={cloud_dt * 1000:.0f}ms) '
                f'— buf_size={len(self.cloud_buffer)}')
            return

        image_msg, image_dt = find_closest(self.image_buffer, det_time)
        if image_msg is None or image_dt > self.img_tol:
            image_msg, image_dt = None, None

        frame = SceneFrame(msg, cloud_msg, image_msg, cloud_dt, image_dt)
        self.is_processing = True
        try:
            self._process_frame(frame)
            self.cnt_frames += 1
            self.last_proc_time = now  # ← marquer seulement si succès
        except Exception as e:
            self.get_logger().error(f'Erreur _process_frame: {e}')
        finally:
            self.is_processing = False

    # ── Traitement principal ──────────────────────────────────────────────────

    def _process_frame(self, frame: SceneFrame):
        cloud_msg    = frame.cloud_msg
        current_time = frame.det_time
        h, w         = cloud_msg.height, cloud_msg.width

        alt_val, alt_dt = find_closest(self.altitude_buffer, current_time)
        hdg_val, hdg_dt = find_closest(self.heading_buffer,  current_time)

        scale_u        = w / self.img_w
        scale_v        = h / self.img_h
        drone_altitude = alt_val
        drone_heading  = hdg_val if hdg_val is not None else 0.0

        if alt_dt > 0.5:
            self.get_logger().warn(
                f'Altitude MAVROS désynchronisée (dt={alt_dt * 1000:.0f}ms)')
        if hdg_dt > 0.5:
            self.get_logger().warn(
                f'Heading MAVROS désynchronisé (dt={hdg_dt * 1000:.0f}ms)')

        image_stamp_sec = (
            stamp_to_sec(frame.image_msg.header.stamp)
            if frame.image_msg is not None else None
        )

        # Décodage du nuage de points
        cloud_bytes = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        dtype       = np.dtype([('x', np.float32), ('y', np.float32),
                                 ('z', np.float32), ('rgb', np.float32)])
        cloud_2d    = np.ndarray(shape=(h, w), dtype=dtype, buffer=cloud_bytes)

        # ══════════════════════════════════════════════════════════════════
        # ÉTAPE 1 — RANSAC : un plan par objet
        # ══════════════════════════════════════════════════════════════════
        frame_planes = {}

        for idx, obj in enumerate(frame.objects_msg.objects):
            uid   = int(getattr(obj, 'label_id', 0) + idx * 100)
            label = obj.label

            corners = obj.bounding_box_2d.corners
            us = [c.kp[0] * scale_u for c in corners]
            vs = [c.kp[1] * scale_v for c in corners]
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)
            bw = u_max - u_min
            bh = v_max - v_min

            u0 = max(0,     int(u_min - bw * self.padding))
            u1 = min(w - 1, int(u_max + bw * self.padding))
            v0 = max(0,     int(v_min - bh * self.padding))
            v1 = min(h - 1, int(v_max + bh * self.padding))

            if u1 <= u0 or v1 <= v0:
                self.get_logger().warn(
                    f'[{label}] ROI invalide — bbox hors du cloud {w}×{h}')
                continue

            # ── Stride agressif : vise max ~500 points ────────────────────
            # Sur Jetson, chaque point supplémentaire coûte cher au RANSAC Python.
            # 500 pts bien répartis suffisent pour détecter un mur plat.
            area   = (v1 - v0) * (u1 - u0)
            stride = max(1, int(math.sqrt(area / 500)))

            roi = cloud_2d[v0:v1:stride, u0:u1:stride]
            pts = np.stack(
                [roi['x'].flatten(), roi['y'].flatten(), roi['z'].flatten()],
                axis=1)
            pts = pts[~np.isnan(pts).any(axis=1)]

            if len(pts) < 50:  # ← seuil abaissé de 100→50 pour petites bbox
                self.get_logger().warn(
                    f'[{label}] SKIP — {len(pts)} points valides (< 50)')
                continue

            plane          = pyrsc.Plane()
            model, inliers = plane.fit(
                pts, thresh=self.rans_dist, maxIteration=self.rans_iter)

            ratio = len(inliers) / len(pts)
            if ratio < self.min_inliers:
                self.get_logger().warn(
                    f'[{label}] SKIP — plan peu fiable ({ratio:.0%} inliers)')
                continue

            a, b, c, d = model
            norm_n = np.linalg.norm([a, b, c])
            if norm_n < 1e-9:
                continue

            normal = np.array([a, b, c]) / norm_n
            d_norm = d / norm_n
            centroid = np.mean(pts[inliers], axis=0)

            if np.dot(normal, centroid) > 0:
                normal = -normal
                d_norm = -d_norm

            surface = 'floor' if abs(float(normal[2])) > FLOOR_NZ_THRESH else 'wall'

            frame_planes[uid] = {
                'model':    np.array([*normal, d_norm]),
                'R':        build_local_frame(normal, self.v_up),
                'centroid': centroid,
                'label':    label,
                'surface':  surface,
            }

            self.get_logger().info(
                f'[{label}|{uid}] plan OK — surface={surface} '
                f'n={normal.round(3).tolist()} '
                f'centroïde={centroid.round(3).tolist()} '
                f'pts={len(pts)} inliers={len(inliers)} ({ratio:.0%})'
            )

        if not frame_planes:
            return

        # ══════════════════════════════════════════════════════════════════
        # ÉTAPE 2 — Comparaison par paires (visualisation RViz)
        # ══════════════════════════════════════════════════════════════════
        ids    = list(frame_planes.keys())
        colors = {uid: (1.0, 0.0, 0.0) for uid in ids}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                p1, p2   = frame_planes[id1], frame_planes[id2]
                same     = is_same_plane(
                    p1['model'], p2['model'],
                    self.sim_thresh, self.dist_thresh)
                color         = (0.0, 1.0, 0.0) if same else (1.0, 1.0, 0.0)
                colors[id1]   = color
                colors[id2]   = color
                sim  = float(np.abs(np.dot(p1['model'][:3], p2['model'][:3])))
                dist = float(np.abs(p1['model'][3] - p2['model'][3]))
                self.get_logger().info(
                    f'[{p1["label"]}] vs [{p2["label"]}] : '
                    f'{"MÊME PLAN" if same else "PLANS DIFF"} '
                    f'(sim={sim:.3f}, dist={dist:.3f}m)'
                )

        # ══════════════════════════════════════════════════════════════════
        # ÉTAPE 3 — Description par cible
        # ══════════════════════════════════════════════════════════════════
        scene_targets = []
        marker_array  = MarkerArray()

        for uid in ids:
            data     = frame_planes[uid]
            label    = data['label']
            centroid = data['centroid']
            normal   = data['model'][:3]
            surface  = data['surface']

            if drone_altitude is not None:
                height_m      = round(drone_altitude + float(centroid[2]), 3)
                height_source = 'absolute'
            else:
                height_m      = round(float(centroid[2]), 3)
                height_source = 'relative_cam'

            marker_array.markers.append(
                self._make_arrow(uid, data, colors[uid], cloud_msg.header))

            if not is_target_label(label):
                continue

            wall_normal = [round(float(n), 4) for n in normal]
            ref_id, ref_data = self._find_reference(uid, data, frame_planes)

            if ref_id is not None:
                local_coords = self._compute_local_coords(data, ref_data, centroid)
                self.get_logger().info(
                    f'[{label}|{uid}] surface={surface} '
                    f'→ réf [{ref_data["label"]}|{ref_id}] | '
                    f'coords={local_coords} | height_m={height_m:+.3f}m'
                )
                scene_targets.append({
                    'id':            uid,
                    'label':         label,
                    'surface':       surface,
                    'wall_normal':   wall_normal,
                    'reference':     {'id': ref_id, 'label': ref_data['label']},
                    'local_coords':  local_coords,
                    'height_m':      height_m,
                    'height_source': height_source,
                })
            else:
                self.get_logger().info(
                    f'[{label}|{uid}] surface={surface} '
                    f'sans référence | height_m={height_m:+.3f}m'
                )
                scene_targets.append({
                    'id':            uid,
                    'label':         label,
                    'surface':       surface,
                    'wall_normal':   wall_normal,
                    'reference':     None,
                    'local_coords':  None,
                    'height_m':      height_m,
                    'height_source': height_source,
                })

        self.marker_pub.publish(marker_array)

        for uid, data in frame_planes.items():
            self.plane_memory[uid] = {**data, 'time': current_time}
        self.plane_memory = {
            k: v for k, v in self.plane_memory.items()
            if current_time - v['time'] < 3.0
        }

        if scene_targets:
            out      = String()
            out.data = json.dumps({
                'timestamp':     round(current_time, 4),
                'image_stamp':   round(image_stamp_sec, 6)
                                 if image_stamp_sec else None,
                'cloud_dt':      round(frame.cloud_dt * 1000, 1),
                'image_dt':      round(frame.image_dt * 1000, 1)
                                 if frame.image_dt else None,
                'drone_heading': round(drone_heading, 1),
                'targets':       scene_targets,
            })
            self.scene_pub.publish(out)
            self.get_logger().info(
                f'/aeac/internal/scene_description → '
                f'{len(scene_targets)} cible(s) | '
                f'cloud_dt={frame.cloud_dt*1000:.0f}ms')

        gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITAIRES
    # ══════════════════════════════════════════════════════════════════════════

    def _find_reference(self, target_id: int, target_data: dict,
                        frame_planes: dict):
        surface    = target_data.get('surface', 'wall')
        best_id    = None
        best_data  = None
        best_dist  = float('inf')

        for oid, data in frame_planes.items():
            if oid == target_id:
                continue
            if not is_reference_label(data['label']):
                continue
            if surface == 'wall':
                if not is_same_plane(
                        target_data['model'], data['model'],
                        self.sim_thresh, self.dist_thresh):
                    continue
            d = float(np.linalg.norm(
                data['centroid'] - target_data['centroid']))
            if d < best_dist:
                best_dist = d
                best_id   = oid
                best_data = data

        return best_id, best_data

    def _compute_local_coords(self, target_data: dict, ref_data: dict,
                               centroid: np.ndarray) -> dict:
        surface = target_data.get('surface', 'wall')
        delta   = centroid - ref_data['centroid']

        if surface == 'wall':
            P = to_local_coords(ref_data['R'], ref_data['centroid'], centroid)
            return {
                'x': round(float(P[0]), 3),
                'y': round(float(P[1]), 3),
                'z': round(float(P[2]), 3),
            }
        else:
            R_ref     = ref_data['R']
            x_local   = R_ref[:, 0]
            x_lat     = float(np.dot(delta, x_local))
            n_wall    = ref_data['model'][:3]
            d_wall    = ref_data['model'][3]
            dist_wall = abs(float(np.dot(n_wall, centroid) + d_wall))
            delta_z   = float(delta[2])
            return {
                'x':         round(x_lat,    3),
                'y':         round(delta_z,  3),
                'z':         round(dist_wall, 3),
                'dist_wall': round(dist_wall, 3),
            }

    def _make_arrow(self, uid: int, data: dict, color: tuple, header) -> Marker:
        m = Marker()
        m.header     = header
        m.ns, m.id   = 'walls', uid
        m.type       = Marker.ARROW
        m.action     = Marker.ADD
        c = data['centroid']
        n = data['model'][:3]
        m.points = [
            Point(x=float(c[0]),           y=float(c[1]),           z=float(c[2])),
            Point(x=float(c[0]+n[0]*0.5), y=float(c[1]+n[1]*0.5), z=float(c[2]+n[2]*0.5)),
        ]
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.1, 0.1
        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]
        m.color.a = 1.0
        m.lifetime = Duration(sec=1, nanosec=0)
        return m


def main():
    rclpy.init()
    node = ZEDWallDetector()

    # MultiThreadedExecutor — les callbacks buffer tournent en parallèle du RANSAC
    # Sans ça, is_processing=True bloque cloud_cb/image_cb → cloud_dt explose
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()