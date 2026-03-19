#!/usr/bin/env python3
"""
overlay_node.py — Overlay visuel synchronisé avec le pipeline AEAC

FIXES v2
────────
  - Topic objects : paramètre 'objects_topic' (défaut /aeac/test/objects pour test,
    /zed/zed_node/obj_det/objects pour hardware)
  - Topic scene_text : paramètre 'scene_text_topic' (défaut /aeac/internal/scene_text)
  - Fonts et épaisseurs adaptatives selon la résolution de l'image
  - Bboxes visibles : épaisseur et couleurs revues
  - Description complète dans le .txt même si scene_text arrive en retard
  - Sauvegarde du JSON brut en plus du résumé

TOPICS (configurables via paramètres ROS2)
──────────────────────────────────────────
  Subscribe:
    <image_topic>        /zed/zed_node/rgb/color/rect/image
    <objects_topic>      /aeac/test/objects            (test)
                         /zed/zed_node/obj_det/objects (hardware)
    <scene_desc_topic>   /aeac/internal/scene_description
    <scene_text_topic>   /aeac/internal/scene_text
  Publish:
    /aeac/external/overlay_image   sensor_msgs/Image
    /aeac/external/scene_frame     custom_interfaces/SceneFrame (si compilé)

PARAMÈTRES ROS2
───────────────
  image_topic        str   /zed/zed_node/rgb/color/rect/image
  objects_topic      str   /aeac/test/objects
  scene_desc_topic   str   /aeac/internal/scene_description
  scene_text_topic   str   /aeac/internal/scene_text
  image_width        int   448    référence bbox (= résolution cloud pour test)
  image_height       int   256
  buffer_size        int   60
  save_images        bool  true
  save_dir           str   /tmp/aeac_overlay
  max_saved          int   20
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

import cv2
import json
from collections import deque
from pathlib import Path

import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import String
from sensor_msgs.msg import Image

# ── Message custom ────────────────────────────────────────────────────────────
CUSTOM_MSG_AVAILABLE = False
try:
    from custom_interfaces.msg import SceneFrame
    CUSTOM_MSG_AVAILABLE = True
except ImportError:
    pass

# ── zed_msgs ──────────────────────────────────────────────────────────────────
ZED_MSGS_AVAILABLE = False
try:
    from zed_msgs.msg import ObjectsStamped
    ZED_MSGS_AVAILABLE = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES VISUELLES — valeurs de base pour 1280×720
# Les valeurs réelles sont scalées selon la résolution dans draw_overlay()
# ══════════════════════════════════════════════════════════════════════════════

# Couleurs BGR
C_TARGET    = (0,   200, 255)   # Orange vif   → cibles
C_REFERENCE = (50,  230,  80)   # Vert vif     → références (fenêtre/porte)
C_OTHER     = (160, 160, 160)   # Gris clair   → autres objets
C_NORMAL    = (0,   220, 255)   # Cyan         → flèche normale du plan
C_COORDS    = (100, 255, 100)   # Vert vif     → coords locales
C_RAW_POS   = (220, 180, 255)   # Mauve        → position brute ZED
C_HUD_BG    = (10,   10,  10)   # Fond HUD
C_HUD_TXT   = (230, 230, 230)   # Texte HUD
C_WHITE     = (255, 255, 255)
C_BLACK     = (  0,   0,   0)
C_YELLOW    = (0,   255, 255)   # Jaune

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Labels
REFERENCE_LABELS = {'window', 'door', 'fenetre', 'porte'}
TARGET_LABELS = {
    'person', 'target', 'cible',
    'red_target', 'blue_target', 'green_target', 'yellow_target', 'orange_target',
    'yellow target', 'orange target',
    'red circle', 'blue circle', 'green circle', 'yellow circle',
    'circle_red', 'circle_blue', 'circle_green', 'circle_yellow',
}

IMG_MATCH_TOL = 0.15   # secondes — tolérance sync image↔JSON


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def find_closest(buffer: deque, target_sec: float):
    if not buffer:
        return None, float('inf')
    best_msg, best_diff = None, float('inf')
    for ts, msg in buffer:
        diff = abs(ts - target_sec)
        if diff < best_diff:
            best_diff, best_msg = diff, msg
    return best_msg, best_diff


def _label_color(label: str):
    lo = label.lower().replace('-', '_').replace(' ', '_')
    if lo in {l.replace(' ', '_') for l in TARGET_LABELS}:
        return C_TARGET
    if lo in REFERENCE_LABELS:
        return C_REFERENCE
    return C_OTHER


def _scale_factors(img_w: int, img_h: int):
    """
    Retourne (font_scale_main, font_scale_small, thickness_box, thickness_text, line_h)
    adaptés à la résolution de l'image.
    Calibrés sur 1280×720 comme référence.
    """
    ref_diag  = (1280**2 + 720**2) ** 0.5
    img_diag  = (img_w**2 + img_h**2) ** 0.5
    s         = img_diag / ref_diag          # facteur d'échelle

    fs_main   = max(0.5,  round(0.7  * s, 2))
    fs_small  = max(0.4,  round(0.55 * s, 2))
    th_box    = max(2,    int(3   * s))
    th_txt    = max(1,    int(1.5 * s))
    line_h    = max(20,   int(26  * s))

    return fs_main, fs_small, th_box, th_txt, line_h


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVES DESSIN
# ══════════════════════════════════════════════════════════════════════════════

def _draw_bbox(img, x1, y1, x2, y2, color, th):
    """Rectangle plein semi-transparent + coins en L bien visibles."""
    # Fond semi-transparent
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # Contour plein
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)

    # Coins en L épais
    arm = max(12, min(30, (x2 - x1) // 4, (y2 - y1) // 4))
    corners = [
        ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),
        ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
        ((x2, y2 - arm), (x2, y2), (x2 - arm, y2)),
        ((x1 + arm, y2), (x1, y2), (x1, y2 - arm)),
    ]
    for p1, mid, p2 in corners:
        cv2.line(img, p1, mid, C_WHITE, th + 2, cv2.LINE_AA)
        cv2.line(img, mid, p2, C_WHITE, th + 2, cv2.LINE_AA)
        cv2.line(img, p1, mid, color, th, cv2.LINE_AA)
        cv2.line(img, mid, p2, color, th, cv2.LINE_AA)


def _text_badge(img, x, y, text, color, fs, th_txt) -> int:
    """
    Texte avec fond opaque foncé.
    Retourne y_bottom pour empiler verticalement.
    """
    (tw, text_h), baseline = cv2.getTextSize(text, FONT, fs, th_txt)
    pad  = max(4, int(6 * fs))
    rx1  = max(0, x - pad)
    ry1  = max(0, y - pad)
    rx2  = min(img.shape[1] - 1, x + tw + pad)
    ry2  = min(img.shape[0] - 1, y + text_h + baseline + pad)

    # Fond opaque
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), C_HUD_BG, -1)
    # Bordure couleur fine
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 1)
    # Texte
    cv2.putText(img, text, (x, y + text_h), FONT, fs, color, th_txt, cv2.LINE_AA)

    return ry2


def _draw_normal_arrow(img, cx, cy, normal_cam, scale, color, th):
    """Flèche projetée depuis le centroïde bbox."""
    _, ny, nz = normal_cam
    dx  = int(-ny * scale)
    dy  = int(-nz * scale)
    tip = (cx + dx, cy + dy)
    cv2.arrowedLine(img, (cx + 1, cy + 1), (tip[0] + 1, tip[1] + 1),
                    C_BLACK, th + 2, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), tip,
                    color, th, tipLength=0.25, line_type=cv2.LINE_AA)


def _draw_legend(img, fs_sm, th_txt):
    """Légende couleurs — coin haut droit."""
    entries = [
        (C_TARGET,    'target'),
        (C_REFERENCE, 'reference (door/window)'),
        (C_NORMAL,    'wall normal'),
        (C_COORDS,    'local coords'),
        (C_RAW_POS,   'ZED 3D position'),
    ]
    margin   = 12
    pad      = 6
    box_w    = max(14, int(18 * fs_sm))
    line_h   = max(20, int(cv2.getTextSize('A', FONT, fs_sm, th_txt)[0][1] * 2.2))
    max_tw   = max(cv2.getTextSize(lbl, FONT, fs_sm, th_txt)[0][0] for _, lbl in entries)
    total_w  = pad + box_w + 10 + max_tw + pad
    total_h  = len(entries) * line_h + pad * 2

    x0 = img.shape[1] - total_w - margin
    y0 = margin

    # Fond
    cv2.rectangle(img, (x0, y0), (x0 + total_w, y0 + total_h), C_HUD_BG, -1)
    cv2.rectangle(img, (x0, y0), (x0 + total_w, y0 + total_h), (100, 100, 100), 1)

    for i, (color, label) in enumerate(entries):
        yc = y0 + pad + i * line_h + line_h // 2
        cv2.rectangle(img,
                      (x0 + pad, yc - box_w // 2),
                      (x0 + pad + box_w, yc + box_w // 2),
                      color, -1)
        cv2.putText(img, label,
                    (x0 + pad + box_w + 8, yc + int(box_w * 0.4)),
                    FONT, fs_sm, C_HUD_TXT, th_txt, cv2.LINE_AA)


def _draw_hud(img, scene_json: dict, fs_sm, th_txt):
    """Bande d'info opaque en bas de l'image."""
    ts    = scene_json.get('timestamp', 0.0)
    c_dt  = scene_json.get('cloud_dt', 0.0)
    i_dt  = scene_json.get('image_dt')
    hdg   = scene_json.get('drone_heading', 0.0)
    n_t   = len(scene_json.get('targets', []))
    i_str = f'  img_dt={i_dt:.0f}ms' if i_dt else ''

    line1 = f't={ts:.3f}s   cloud_dt={c_dt:.0f}ms{i_str}'
    line2 = f'heading={hdg:.1f}deg   targets={n_t}'

    h, w   = img.shape[:2]
    _, lh  = cv2.getTextSize('A', FONT, fs_sm, th_txt)[0]
    band_h = lh * 4 + 10

    cv2.rectangle(img, (0, h - band_h), (w, h), C_HUD_BG, -1)
    cv2.line(img, (0, h - band_h), (w, h - band_h), (80, 80, 80), 1)
    cv2.putText(img, line1, (10, h - band_h + lh + 6),
                FONT, fs_sm, C_HUD_TXT, th_txt, cv2.LINE_AA)
    cv2.putText(img, line2, (10, h - band_h + lh * 2 + 14),
                FONT, fs_sm, C_HUD_TXT, th_txt, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(image_cv: np.ndarray,
                 scene_json: dict,
                 objects_msg,
                 ref_img_w: int,
                 ref_img_h: int) -> np.ndarray:
    """
    Dessine l'overlay complet et retourne l'image annotée.

    ref_img_w/h : résolution dans laquelle les bbox sont exprimées
                  (= résolution du cloud pour test, = résolution image RGB pour hardware)
    """
    out    = image_cv.copy()
    h_img, w_img = out.shape[:2]

    # Scale bbox → image reçue
    scale_u = w_img / ref_img_w
    scale_v = h_img / ref_img_h

    # Facteurs visuels adaptés à la résolution
    fs_main, fs_sm, th_box, th_txt, line_h = _scale_factors(w_img, h_img)

    # Flèche normale : scale proportionnel à la diagonale de l'image
    arrow_scale = int(80 * ((w_img**2 + h_img**2)**0.5) / ((1280**2 + 720**2)**0.5))

    # Index JSON par label
    targets_by_label: dict[str, list] = {}
    for t in scene_json.get('targets', []):
        key = t['label'].lower()
        targets_by_label.setdefault(key, []).append(t)

    # ── Dessin par objet ───────────────────────────────────────────────────
    if objects_msg is not None:
        for obj in objects_msg.objects:

            if not hasattr(obj, 'bounding_box_2d') or not obj.bounding_box_2d.corners:
                continue

            us = [c.kp[0] * scale_u for c in obj.bounding_box_2d.corners]
            vs = [c.kp[1] * scale_v for c in obj.bounding_box_2d.corners]
            x1 = max(0,         int(min(us)))
            x2 = min(w_img - 1, int(max(us)))
            y1 = max(0,         int(min(vs)))
            y2 = min(h_img - 1, int(max(vs)))
            if x2 <= x1 or y2 <= y1:
                continue

            label = getattr(obj, 'label', 'unknown')
            color = _label_color(label)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ── Bbox ──────────────────────────────────────────────────────
            _draw_bbox(out, x1, y1, x2, y2, color, th_box)
            cv2.circle(out, (cx, cy), max(4, th_box + 1), color, -1, cv2.LINE_AA)

            # ── Badges AU-DESSUS ───────────────────────────────────────────
            # Ligne 1 : label + confiance
            conf  = getattr(obj, 'confidence', 0.0)
            badge = f'{label}  {conf:.0f}%'
            (_, bh), _ = cv2.getTextSize(badge, FONT, fs_main, th_txt)
            y_lbl = max(bh + 6, y1 - line_h - 2)
            _text_badge(out, x1, y_lbl, badge, color, fs_main, th_txt)

            # Ligne 2 : position brute ZED 3D
            if hasattr(obj, 'position') and len(obj.position) >= 3:
                px, py, pz = obj.position[0], obj.position[1], obj.position[2]
                raw = f'ZED  x={px:+.2f}  y={py:+.2f}  z={pz:+.2f} m'
                y_raw = max(bh * 2 + 8, y1 - line_h * 2 - 4)
                _text_badge(out, x1, y_raw, raw, C_RAW_POS, fs_sm, th_txt)

            # ── Badges EN-DESSOUS ──────────────────────────────────────────
            y_below = y2 + 6

            lo     = label.lower()
            jt_list = targets_by_label.get(lo, [])
            best_t  = jt_list[0] if jt_list else None

            if best_t is not None:
                # Normale du plan
                wn = best_t.get('wall_normal')
                if wn and len(wn) == 3:
                    _draw_normal_arrow(out, cx, cy, wn, arrow_scale, C_NORMAL, th_txt)
                    n_text = f'n=[{wn[0]:+.2f} {wn[1]:+.2f} {wn[2]:+.2f}]'
                    y_below = _text_badge(out, x1, y_below, n_text,
                                          C_NORMAL, fs_sm, th_txt) + 4

                # Coordonnées locales
                coords = best_t.get('local_coords')
                ref    = best_t.get('reference')
                if coords and ref:
                    x_m, y_m = coords['x'], coords['y']
                    x_dir = 'R' if x_m >= 0 else 'L'
                    y_dir = '↑' if y_m >= 0 else '↓'
                    c_text = (f'{abs(x_m):.1f}m {x_dir}  '
                              f'{abs(y_m):.1f}m {y_dir}  '
                              f'[{ref["label"]}]')
                    y_below = _text_badge(out, x1, y_below, c_text,
                                          C_COORDS, fs_sm, th_txt) + 4
                elif lo in {l.replace(' ', '_') for l in TARGET_LABELS}:
                    y_below = _text_badge(out, x1, y_below,
                                          'no ref on this wall',
                                          C_OTHER, fs_sm, th_txt) + 4

                # Hauteur
                h_m = best_t.get('height_m')
                if h_m is not None:
                    src   = best_t.get('height_source', 'relative_cam')
                    h_tag = 'ABS' if src == 'absolute' else 'REL'
                    _text_badge(out, x1, y_below,
                                f'H={h_m:+.2f}m ({h_tag})',
                                C_YELLOW, fs_sm, th_txt)

    # ── HUD et légende ─────────────────────────────────────────────────────
    _draw_hud(out, scene_json, fs_sm, th_txt)
    _draw_legend(out, fs_sm, th_txt)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# NODE ROS2
# ══════════════════════════════════════════════════════════════════════════════

class OverlayNode(Node):

    def __init__(self):
        super().__init__('overlay_node')

        # ── Paramètres ────────────────────────────────────────────────────
        self.declare_parameter('image_topic',
                               '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('objects_topic',
                               '/aeac/test/objects')
        self.declare_parameter('scene_desc_topic',
                               '/aeac/internal/scene_description')
        self.declare_parameter('scene_text_topic',
                               '/aeac/internal/scene_text')
        self.declare_parameter('image_width',  448)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('buffer_size',   60)
        self.declare_parameter('save_images',  True)
        self.declare_parameter('save_dir',     '/tmp/aeac_overlay')
        self.declare_parameter('max_saved',      20)

        image_topic      = self.get_parameter('image_topic').value
        objects_topic    = self.get_parameter('objects_topic').value
        scene_desc_topic = self.get_parameter('scene_desc_topic').value
        scene_text_topic = self.get_parameter('scene_text_topic').value
        self.ref_w       = self.get_parameter('image_width').value
        self.ref_h       = self.get_parameter('image_height').value
        self.buf_size    = self.get_parameter('buffer_size').value
        self.do_save     = self.get_parameter('save_images').value
        self.save_dir    = Path(self.get_parameter('save_dir').value)
        self.max_saved   = self.get_parameter('max_saved').value

        # ── État ──────────────────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.image_buffer    = deque(maxlen=self.buf_size)
        self.objects_buffer  = deque(maxlen=self.buf_size)
        self.last_scene_text = ''
        self.cnt_frames      = 0
        self.cnt_no_img      = 0

        if self.do_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # ── QoS ───────────────────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(Image, image_topic,
                                 self._image_cb, qos_reliable)

        if ZED_MSGS_AVAILABLE:
            self.create_subscription(ObjectsStamped, objects_topic,
                                     self._objects_cb, qos_reliable)
            self.get_logger().info(f'Objects → {objects_topic}')
        else:
            self.get_logger().warn('zed_msgs non disponible — bbox désactivées')

        self.create_subscription(String, scene_desc_topic,
                                 self._scene_cb, qos_reliable)

        self.create_subscription(String, scene_text_topic,
                                 self._text_cb, qos_reliable)

        # ── Publishers ────────────────────────────────────────────────────
        self.overlay_pub = self.create_publisher(
            Image, '/aeac/external/overlay_image', 5)

        if CUSTOM_MSG_AVAILABLE:
            self.scene_frame_pub = self.create_publisher(
                SceneFrame, '/aeac/external/scene_frame', 5)
        else:
            self.scene_frame_pub = None
            self.get_logger().warn(
                'custom_interfaces/SceneFrame non compilé → '
                '/aeac/external/scene_frame désactivé')

        self.create_timer(10.0, self._log_stats)

        self.get_logger().info(
            f'OverlayNode démarré\n'
            f'  image    : {image_topic}\n'
            f'  objects  : {objects_topic}\n'
            f'  scene_desc: {scene_desc_topic}\n'
            f'  scene_text: {scene_text_topic}\n'
            f'  bbox_ref : {self.ref_w}×{self.ref_h}\n'
            f'  SceneFrame: {"OK" if CUSTOM_MSG_AVAILABLE else "DÉSACTIVÉ"}\n'
            f'  save     : {self.do_save}  dir={self.save_dir}  max={self.max_saved}'
        )

    # ── BUFFERS ───────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        self.image_buffer.append((stamp_to_sec(msg.header.stamp), msg))

    def _objects_cb(self, msg):
        self.objects_buffer.append((stamp_to_sec(msg.header.stamp), msg))

    def _text_cb(self, msg: String):
        self.last_scene_text = msg.data

    # ── DÉCLENCHEUR PRINCIPAL ─────────────────────────────────────────────────

    def _scene_cb(self, msg: String):
        try:
            scene_json = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON invalide: {e}')
            return

        ts_scene    = scene_json.get('timestamp', 0.0)
        image_stamp = scene_json.get('image_stamp')
        sync_target = image_stamp if image_stamp is not None else ts_scene
        sync_mode   = 'image_stamp' if image_stamp is not None else 'timestamp(fallback)'

        # Image synchronisée
        image_msg, img_diff = find_closest(self.image_buffer, sync_target)
        if image_msg is None or img_diff > IMG_MATCH_TOL:
            self.cnt_no_img += 1
            self.get_logger().warn(
                f'Image non trouvée (sync={sync_mode}, '
                f'diff={img_diff * 1000:.0f}ms > tol={IMG_MATCH_TOL * 1000:.0f}ms)')
            return

        # Objets synchronisés
        objects_msg, obj_diff = find_closest(self.objects_buffer, ts_scene)
        if obj_diff > 0.5:
            objects_msg = None
            self.get_logger().warn(f'Objets désynchronisés ({obj_diff*1000:.0f}ms)')

        # Conversion image
        try:
            image_cv = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge: {e}')
            return

        # Overlay
        annotated = draw_overlay(image_cv, scene_json, objects_msg,
                                 self.ref_w, self.ref_h)

        # ROS Image
        try:
            overlay_ros        = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            overlay_ros.header = image_msg.header
        except Exception as e:
            self.get_logger().error(f'cv2_to_imgmsg: {e}')
            return

        # Publier overlay image
        self.overlay_pub.publish(overlay_ros)

        # Publier SceneFrame
        if CUSTOM_MSG_AVAILABLE and self.scene_frame_pub is not None:
            try:
                sf               = SceneFrame()
                sf.header        = image_msg.header
                sf.scene_json    = msg.data
                sf.scene_text    = self.last_scene_text
                sf.overlay_image = overlay_ros
                self.scene_frame_pub.publish(sf)
            except Exception as e:
                self.get_logger().error(f'Publish SceneFrame: {e}')

        # Sauvegarde locale
        if self.do_save:
            
            self._save(annotated, ts_scene, scene_json, msg.data)

        self.cnt_frames += 1
        self.get_logger().info(
            f'Overlay #{self.cnt_frames} | {sync_mode} | '
            f'img_diff={img_diff * 1000:.1f}ms | '
            f'{"avec bbox" if objects_msg else "sans bbox"}')

    # ── SAUVEGARDE LOCALE ─────────────────────────────────────────────────────

    def _save(self, img: np.ndarray, ts: float, scene_json: dict, raw_json: str):
        import time
        wall_ts = time.time()
        ts_ms   = int(wall_ts * 1000)
        stem    = f'aeac_{ts_ms:015d}'
        img_path = self.save_dir / f'{stem}.jpg'
        txt_path = self.save_dir / f'{stem}.txt'

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        targets = scene_json.get('targets', [])
        lines = [
            '═' * 60,
            f'FRAME  t={ts:.4f}s',
            '═' * 60,
            f'heading     : {scene_json.get("drone_heading", 0.0):.1f} deg',
            f'cloud_dt    : {scene_json.get("cloud_dt", 0.0):.0f} ms',
            f'image_dt    : {scene_json.get("image_dt") or "N/A"} ms',
            f'targets     : {len(targets)}',
            '',
        ]

        for i, t in enumerate(targets):
            ref = t.get('reference')
            lines.append(f'── Target [{i+1}] ──────────────────────────')
            lines.append(f'  label    : {t.get("label", "?")}')
            lines.append(f'  height   : {t.get("height_m", 0.0):+.3f}m '
                         f'({"ABS" if t.get("height_source") == "absolute" else "REL"})')
            lines.append(f'  reference: {ref["label"] if ref else "none"}')
            wn = t.get('wall_normal')
            if wn:
                lines.append(f'  normal   : [{wn[0]:+.3f} {wn[1]:+.3f} {wn[2]:+.3f}]')
            if t.get('local_coords') and ref:
                c = t['local_coords']
                lines.append(f'  local_x  : {c["x"]:+.3f}m  (droite+)')
                lines.append(f'  local_y  : {c["y"]:+.3f}m  (haut+)')
            lines.append('')

        # Description textuelle
        if self.last_scene_text:
            lines += ['═' * 60, 'SCENE DESCRIPTION', '═' * 60,
                      self.last_scene_text, '']
        else:
            lines += ['═' * 60,
                      'SCENE DESCRIPTION : (pas encore reçue — '
                      'vérifier topic scene_text_topic)',
                      '═' * 60, '']

        txt_path.write_text('\n'.join(lines), encoding='utf-8')

        # Rotation
        existing = sorted(self.save_dir.glob('aeac_*.jpg'))
        while len(existing) > self.max_saved:
            old = existing.pop(0)
            old.unlink(missing_ok=True)
            old.with_suffix('.txt').unlink(missing_ok=True)

    # ── STATS ─────────────────────────────────────────────────────────────────

    def _log_stats(self):
        self.get_logger().info(
            f'[STATS 10s] overlay={self.cnt_frames}  no_img={self.cnt_no_img}  '
            f'img_buf={len(self.image_buffer)}  obj_buf={len(self.objects_buffer)}'
        )
        self.cnt_frames = self.cnt_no_img = 0


def main():
    rclpy.init()
    rclpy.spin(OverlayNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()