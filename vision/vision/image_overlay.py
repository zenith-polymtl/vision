#!/usr/bin/env python3
"""
overlay_node.py — Overlay visuel synchronisé avec le pipeline AEAC

FIXES v3
────────
  1. Leader line toujours tracée (ligne brisée en L + boule d'ancrage).
     Plus de confusion entre quel bloc appartient à quelle bbox.

  2. Label + conf inclus en première ligne du bloc d'info.
     Le badge flottant au-dessus de la bbox est supprimé.
     → un seul élément par détection, toujours relié visuellement.

  3. BlockPlacer inchangé — 12 candidats + scan grille fallback.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

import cv2
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import String
from sensor_msgs.msg import Image

CUSTOM_MSG_AVAILABLE = False
try:
    from custom_interfaces.msg import SceneFrame
    CUSTOM_MSG_AVAILABLE = True
except ImportError:
    pass

ZED_MSGS_AVAILABLE = False
try:
    from zed_msgs.msg import ObjectsStamped
    ZED_MSGS_AVAILABLE = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PALETTE & CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

C_TARGET    = (0,   200, 255)
C_REFERENCE = (50,  230,  80)
C_OTHER     = (150, 150, 150)
C_COORDS    = (120, 255, 120)
C_RAW       = (210, 170, 255)
C_HUD_BG    = (10,   10,  10)
C_HUD_TXT   = (220, 220, 220)
C_WHITE     = (255, 255, 255)
C_BLACK     = (  0,   0,   0)
C_HEIGHT    = (0,   240, 200)
C_LINE      = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX

REFERENCE_LABELS = {'window', 'door', 'fenetre', 'porte'}
TARGET_LABELS = {
    'person', 'target', 'cible',
    'red_target',    'green', 'yellow', 'red', 'blue',  'blue_target',    'green_target',
    'yellow_target', 'orange_target',  'white_target',  'black_target',
    'red target',    'blue target',    'green target',
    'yellow target', 'orange target',  'white target',  'black target',
    'red_circle',    'blue_circle',    'green_circle',
    'yellow_circle', 'white_circle',   'black_circle',
    'red circle',    'blue circle',    'green circle',
    'yellow circle', 'white circle',   'black circle',
    'circle_red',    'circle_blue',    'circle_green',
    'circle_yellow', 'circle_white',   'circle_black',
}

IMG_MATCH_TOL = 50


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES ROS
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DESSIN
# ══════════════════════════════════════════════════════════════════════════════

def _label_color(label: str):
    lo = label.lower().replace('-', '_').replace(' ', '_')
    tl = {l.lower().replace(' ', '_').replace('-', '_') for l in TARGET_LABELS}
    if lo in tl:
        return C_TARGET
    if lo in REFERENCE_LABELS:
        return C_REFERENCE
    return C_OTHER


def _scale_factors(img_w: int, img_h: int):
    ref_diag = (1280**2 + 720**2) ** 0.5
    img_diag = (img_w**2  + img_h**2)  ** 0.5
    s        = img_diag / ref_diag
    return {
        'fs_lg':  max(0.38, round(0.48 * s, 2)),
        'fs_md':  max(0.30, round(0.38 * s, 2)),
        'fs_sm':  max(0.28, round(0.32 * s, 2)),
        'th_box': max(1,    int(1.5    * s)),
        'th_txt': max(1,    int(1.0  * s)),
        'pad':    max(2,    int(4    * s)),
    }


def _draw_bbox(img, x1, y1, x2, y2, color, th):

    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, 0.10, img, 0.90, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)
    arm = max(10, min(24, (x2 - x1) // 4, (y2 - y1) // 4))
    for p1, mid, p2 in [
        ((x1, y1 + arm), (x1, y1),   (x1 + arm, y1)),
        ((x2 - arm, y1), (x2, y1),   (x2, y1 + arm)),
        ((x2, y2 - arm), (x2, y2),   (x2 - arm, y2)),
        ((x1 + arm, y2), (x1, y2),   (x1, y2 - arm)),
    ]:
        cv2.line(img, p1, mid, C_WHITE, th + 2, cv2.LINE_AA)
        cv2.line(img, mid, p2, C_WHITE, th + 2, cv2.LINE_AA)
        cv2.line(img, p1, mid, color,   th,     cv2.LINE_AA)
        cv2.line(img, mid, p2, color,   th,     cv2.LINE_AA)


def _badge(img, x, y, text: str, color, fs: float, th: int,
           bg_alpha: float = 0.85) -> int:
    (tw, th_), base = cv2.getTextSize(text, FONT, fs, th)
    pad = max(3, int(5 * fs))
    rx1 = max(0, x - pad)
    ry1 = max(0, y - pad)
    rx2 = min(img.shape[1] - 1, x + tw + pad)
    ry2 = min(img.shape[0] - 1, y + th_ + base + pad)
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size > 0:
        bg = np.full_like(roi, C_HUD_BG)
        cv2.addWeighted(bg, bg_alpha, roi, 1 - bg_alpha, 0, roi)
        img[ry1:ry2, rx1:rx2] = roi
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 1)
    cv2.putText(img, text, (x, y + th_), FONT, fs, color, th, cv2.LINE_AA)
    return ry2


def _measure_block(lines: list, sc: dict) -> tuple:
    """Retourne (width, height) du bloc sans le dessiner."""
    pad   = max(3, int(5 * sc['fs_md']))
    fs    = sc['fs_md']
    th    = sc['th_txt']
    max_w = 0
    total_h = 0
    for text, _ in lines:
        (tw, th_), base = cv2.getTextSize(text, FONT, fs, th)
        max_w    = max(max_w, tw + pad * 2)
        total_h += th_ + base + pad * 2 + 2
    return max_w, total_h


def _draw_info_block(img, x, y, lines: list, sc: dict) -> int:
    """Dessine le bloc et retourne y_bottom."""
    y_cur = y
    for text, color in lines:
        y_cur = _badge(img, x, y_cur, text, color,
                       sc['fs_md'], sc['th_txt']) + 2
    return y_cur


def _draw_legend(img, sc: dict):
    entries = [
        (C_TARGET,    'target'),
        (C_REFERENCE, 'reference (door/window)'),
        (C_COORDS,    'local coords / dist wall'),
        (C_RAW,       'ZED 3D position'),
        (C_HEIGHT,    'height'),
    ]
    fs    = sc['fs_sm']
    th    = sc['th_txt']
    pad   = 6
    box_w = max(12, int(16 * (sc['fs_sm'] / 0.4)))
    lh    = max(18, int(cv2.getTextSize('A', FONT, fs, th)[0][1] * 2.3))
    max_tw = max(cv2.getTextSize(lbl, FONT, fs, th)[0][0] for _, lbl in entries)
    tw_tot = pad + box_w + 8 + max_tw + pad
    th_tot = len(entries) * lh + pad * 2
    margin = 10
    x0 = img.shape[1] - tw_tot - margin
    y0 = margin
    cv2.rectangle(img, (x0, y0), (x0 + tw_tot, y0 + th_tot), C_HUD_BG, -1)
    cv2.rectangle(img, (x0, y0), (x0 + tw_tot, y0 + th_tot), (90, 90, 90), 1)
    for i, (col, lbl) in enumerate(entries):
        yc = y0 + pad + i * lh + lh // 2
        cv2.rectangle(img,
                      (x0 + pad,         yc - box_w // 2),
                      (x0 + pad + box_w, yc + box_w // 2),
                      col, -1)
        cv2.putText(img, lbl,
                    (x0 + pad + box_w + 6, yc + int(box_w * 0.4)),
                    FONT, fs, C_HUD_TXT, th, cv2.LINE_AA)
    return x0, y0, x0 + tw_tot, y0 + th_tot


def _draw_hud(img, scene_json: dict, sc: dict):
    ts    = scene_json.get('timestamp', 0.0)
    c_dt  = scene_json.get('cloud_dt', 0.0)
    i_dt  = scene_json.get('image_dt')
    hdg   = scene_json.get('drone_heading', 0.0)
    n_t   = len(scene_json.get('targets', []))
    idt_s = f'  img_dt={i_dt:.0f}ms' if i_dt else ''

    lines = [
        f't={ts:.3f}s   cloud_dt={c_dt:.0f}ms{idt_s}',
        f'heading={hdg:.1f}deg   targets={n_t}',
    ]
    h, w  = img.shape[:2]
    fs    = sc['fs_sm']
    th    = sc['th_txt']
    _, lh = cv2.getTextSize('A', FONT, fs, th)[0]
    band  = lh * 4 + 10
    cv2.rectangle(img, (0, h - band), (w, h), C_HUD_BG, -1)
    cv2.line(img, (0, h - band), (w, h - band), (70, 70, 70), 1)
    for i, line in enumerate(lines):
        cv2.putText(img, line,
                    (10, h - band + lh * (i + 1) + 4 + i * 6),
                    FONT, fs, C_HUD_TXT, th, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT MANAGER ANTI-COLLISION
# ══════════════════════════════════════════════════════════════════════════════

class BlockPlacer:
    MARGIN = 8
    GAP    = 4

    def __init__(self, img_w: int, img_h: int, hud_band: int):
        self.img_w    = img_w
        self.img_h    = img_h
        self.hud_band = hud_band
        self._placed: list = []

    def _overlaps_any(self, rx1, ry1, rx2, ry2) -> bool:
        for (ox1, oy1, ox2, oy2) in self._placed:
            if (rx1 < ox2 + self.GAP and rx2 > ox1 - self.GAP and
                    ry1 < oy2 + self.GAP and ry2 > oy1 - self.GAP):
                return True
        return False

    def _in_bounds(self, rx1, ry1, rx2, ry2) -> bool:
        usable_h = self.img_h - self.hud_band
        return rx1 >= 0 and ry1 >= 0 and rx2 <= self.img_w and ry2 <= usable_h

    def _try(self, x, y, w, h) -> bool:
        rx1, ry1, rx2, ry2 = x, y, x + w, y + h
        return (self._in_bounds(rx1, ry1, rx2, ry2) and
                not self._overlaps_any(rx1, ry1, rx2, ry2))

    def place(self, bx1, by1, bx2, by2, blk_w, blk_h) -> tuple:
        M  = self.MARGIN
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        candidates = [
            (bx2 + M,           by1),
            (bx2 + M,           cy - blk_h // 2),
            (bx2 + M,           by2 - blk_h),
            (bx1 - blk_w - M,   by1),
            (bx1 - blk_w - M,   cy - blk_h // 2),
            (bx1 - blk_w - M,   by2 - blk_h),
            (bx1,               by1 - blk_h - M),
            (cx  - blk_w // 2,  by1 - blk_h - M),
            (bx2 - blk_w,       by1 - blk_h - M),
            (bx1,               by2 + M),
            (cx  - blk_w // 2,  by2 + M),
            (bx2 - blk_w,       by2 + M),
        ]

        usable_h = self.img_h - self.hud_band
        for px, py in candidates:
            px = int(max(0, min(px, self.img_w - blk_w)))
            py = int(max(0, min(py, usable_h - blk_h)))
            if self._try(px, py, blk_w, blk_h):
                self._placed.append((px, py, px + blk_w, py + blk_h))
                return px, py

        # Fallback scan de grille
        step = max(blk_w, blk_h) // 2
        best_px, best_py, best_dist = 0, 0, float('inf')
        for py in range(0, usable_h - blk_h, step):
            for px in range(0, self.img_w - blk_w, step):
                if self._try(px, py, blk_w, blk_h):
                    dist = ((px + blk_w // 2 - cx) ** 2 +
                            (py + blk_h // 2 - cy) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_px, best_py = px, py

        self._placed.append((best_px, best_py,
                             best_px + blk_w, best_py + blk_h))
        return best_px, best_py

    def reserve(self, x1, y1, x2, y2):
        self._placed.append((x1, y1, x2, y2))


def _draw_leader(img, bx1, by1, bx2, by2, px, py, pw, ph, color):
    """
    Toujours tracé. Ligne brisée en L entre la bbox et le bloc.
    Boule colorée sur le point d'ancrage de la bbox.
    """
    blk_cx  = px + pw // 2
    blk_cy  = py + ph // 2
    bbox_cx = (bx1 + bx2) // 2
    bbox_cy = (by1 + by2) // 2

    # Point d'ancrage sur la bbox (coin le plus proche du bloc)
    ax = bx2 if blk_cx >= bbox_cx else bx1
    ay = by1 if blk_cy <= bbox_cy else by2

    # Point d'arrivée sur le bloc (coin le plus proche de la bbox)
    tx = px      if blk_cx >= bbox_cx else px + pw
    ty = py + ph if blk_cy >= bbox_cy else py

    # Ligne brisée en L
    mid_x = int((int(ax) + int(tx)) // 2)
    cv2.line(img, (int(ax), int(ay)), (mid_x,   int(ay)), C_LINE, 1, cv2.LINE_AA)
    cv2.line(img, (mid_x,   int(ay)), (mid_x,   int(ty)), C_LINE, 1, cv2.LINE_AA)
    cv2.line(img, (mid_x,   int(ty)), (int(tx), int(ty)), C_LINE, 1, cv2.LINE_AA)

    # Boule d'ancrage sur la bbox
    cv2.circle(img, (int(ax), int(ay)), 4, color,  -1, cv2.LINE_AA)
    cv2.circle(img, (int(ax), int(ay)), 4, C_WHITE,  1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(image_cv: np.ndarray,
                 scene_json: dict,
                 objects_msg,
                 ref_img_w: int,
                 ref_img_h: int) -> np.ndarray:
    out          = image_cv.copy()
    h_img, w_img = out.shape[:2]
    scale_u      = w_img / ref_img_w
    scale_v      = h_img / ref_img_h
    sc           = _scale_factors(w_img, h_img)

    # Hauteur HUD
    _, lh    = cv2.getTextSize('A', FONT, sc['fs_sm'], sc['th_txt'])[0]
    hud_band = lh * 4 + 10

    # Légende — dessinée en premier, zone réservée pour le placer
    leg_x0, leg_y0, leg_x1, leg_y1 = _draw_legend(out, sc)

    placer = BlockPlacer(w_img, h_img, hud_band)
    placer.reserve(leg_x0, leg_y0, leg_x1, leg_y1)

    targets_by_label: dict = {}
    for t in scene_json.get('targets', []):
        targets_by_label.setdefault(t['label'].lower(), []).append(t)

    if objects_msg is None:
        _draw_hud(out, scene_json, sc)
        return out

    # ── Passe 1 : collecte ───────────────────────────────────────────────────
    detections = []

    for obj in objects_msg.objects:
        if not hasattr(obj, 'bounding_box_2d') or \
           not obj.bounding_box_2d.corners:
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
        lo    = label.lower()

        if lo not in targets_by_label:
            best_t = None
        elif len(targets_by_label[lo]) == 1:
            best_t = targets_by_label[lo][0]
        else:
            best_t = targets_by_label[lo].pop(0)

        conf = getattr(obj, 'confidence', 0.0)

        # Label + conf = première ligne du bloc (plus de badge séparé)
        info_lines = [
            (f'{label}  {conf:.0f}%', color),
        ]

        if hasattr(obj, 'position') and len(obj.position) >= 3:
            px_pos, py_pos, pz = (obj.position[0],
                                  obj.position[1], obj.position[2])
            info_lines.append(
                (f'ZED {px_pos:+.1f} {py_pos:+.1f} {pz:+.1f}m', C_RAW))

        if best_t is not None:
            surface = best_t.get('surface', 'wall')
            coords  = best_t.get('local_coords')
            ref     = best_t.get('reference')
            h_m     = best_t.get('height_m')
            h_src   = best_t.get('height_source', 'relative_cam')

            if coords and ref:
                ref_lbl = ref['label']
                if surface == 'floor':
                    dist_w = coords.get('dist_wall', coords.get('z', 0.0))
                    x_lat  = coords.get('x', 0.0)
                    x_dir  = 'R' if x_lat >= 0 else 'L'
                    info_lines.append((
                        f'{abs(dist_w):.1f}m wall  '
                        f'{abs(x_lat):.1f}m {x_dir}  [{ref_lbl}]',
                        C_COORDS
                    ))
                else:
                    xm = coords.get('x', 0.0)
                    ym = coords.get('y', 0.0)
                    xd = 'R' if xm >= 0 else 'L'
                    yd = 'U' if ym >= 0 else 'D'
                    info_lines.append((
                        f'{abs(xm):.1f}m {xd}  '
                        f'{abs(ym):.1f}m {yd}  [{ref_lbl}]',
                        C_COORDS
                    ))
            elif lo in {l.lower().replace(' ', '_')
                        for l in TARGET_LABELS}:
                info_lines.append(('no ref', C_OTHER))

            if h_m is not None:
                tag = 'ABS' if h_src == 'absolute' else 'REL'
                info_lines.append((f'H {h_m:+.2f}m {tag}', C_HEIGHT))

        detections.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'color':      color,
            'info_lines': info_lines,
        })

    # ── Tri du haut vers le bas ───────────────────────────────────────────────
    detections.sort(key=lambda d: d['y1'])

    for det in detections:
        placer.reserve(det['x1'], det['y1'], det['x2'], det['y2'])

    # ── Passe 2 : dessin ─────────────────────────────────────────────────────
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        color      = det['color']
        info_lines = det['info_lines']

        # Bbox
        _draw_bbox(out, x1, y1, x2, y2, color, sc['th_box'])
         
        # Placement anti-collision
        blk_w, blk_h = _measure_block(info_lines, sc)
        px, py = placer.place(x1, y1, x2, y2, blk_w, blk_h)
        placer.reserve(px, py, px + blk_w, py + blk_h)


        # Fil de liaison (toujours tracé)
        _draw_leader(out, x1, y1, x2, y2, px, py, blk_w, blk_h, color)

        # Bloc d'info
        _draw_info_block(out, px, py, info_lines, sc)

    # HUD bas de page
    _draw_hud(out, scene_json, sc)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# NODE ROS2
# ══════════════════════════════════════════════════════════════════════════════

class OverlayNode(Node):

    def __init__(self):
        super().__init__('overlay_node')

        self.declare_parameter('image_topic',      '/zed/zed_node/left/color/rect/image')
        self.declare_parameter('objects_topic',    '/aeac/internal/target_detected')
        self.declare_parameter('scene_desc_topic', '/aeac/internal/scene_description')
        self.declare_parameter('scene_text_topic', '/aeac/internal/scene_text')
        self.declare_parameter('image_width',      1280)
        self.declare_parameter('image_height',     720)
        self.declare_parameter('buffer_size',       10)
        self.declare_parameter('save_images',      True)
        self.declare_parameter('save_dir',         '/tmp/aeac_overlay')
        self.declare_parameter('max_saved',         20)

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

        self.bridge         = CvBridge()
        self.image_buffer   = deque(maxlen=50)
        self.objects_buffer = deque(maxlen=10)
        self.last_scene_text = ''
        self.cnt_frames = self.cnt_no_img = 0

        if self.do_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_subscription(
            Image, image_topic, self._image_cb, reliable_qos)

        if ZED_MSGS_AVAILABLE:
            self.create_subscription(
                ObjectsStamped, objects_topic,
                self._objects_cb, reliable_qos)
        else:
            self.get_logger().warn('zed_msgs non disponible — bbox désactivées')

        self.create_subscription(
            String, scene_desc_topic, self._scene_cb, reliable_qos)
        self.create_subscription(
            String, scene_text_topic, self._text_cb,  reliable_qos)

        self.overlay_pub = self.create_publisher(
            Image, '/aeac/external/overlay_image', 5)

        if CUSTOM_MSG_AVAILABLE:
            self.scene_frame_pub = self.create_publisher(
                SceneFrame, '/aeac/external/scene_frame', 5)
        else:
            self.scene_frame_pub = None

        self.create_timer(10.0, self._log_stats)

        self.get_logger().info(
            f'OverlayNode démarré\n'
            f'  image      : {image_topic}\n'
            f'  objects    : {objects_topic}\n'
            f'  bbox_ref   : {self.ref_w}×{self.ref_h}\n'
            f'  IMG_MATCH_TOL : {IMG_MATCH_TOL*1000:.0f}ms\n'
            f'  SceneFrame : {"OK" if CUSTOM_MSG_AVAILABLE else "DÉSACTIVÉ"}'
        )

    def _image_cb(self, msg: Image):
        self.image_buffer.append((stamp_to_sec(msg.header.stamp), msg))

    def _objects_cb(self, msg):
        self.objects_buffer.append((stamp_to_sec(msg.header.stamp), msg))

    def _text_cb(self, msg: String):
        self.last_scene_text = msg.data

    def _scene_cb(self, msg: String):
        try:
            scene_json = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON invalide: {e}')
            return

        ts_scene    = scene_json.get('timestamp', 0.0)
        image_stamp = scene_json.get('image_stamp')

        sync_target = image_stamp if image_stamp is not None else ts_scene
        sync_mode   = 'image_stamp' if image_stamp is not None else 'ts_scene'

        image_msg, img_diff = find_closest(self.image_buffer, sync_target)
        if image_msg is None or img_diff > IMG_MATCH_TOL:
            self.cnt_no_img += 1
            self.get_logger().warn(
                f'Image non trouvée (mode={sync_mode} '
                f'target={sync_target:.3f} '
                f'diff={img_diff*1000:.0f}ms > {IMG_MATCH_TOL*1000:.0f}ms  '
                f'buf={len(self.image_buffer)})',
                throttle_duration_sec=2.0)
            return

        objects_msg, obj_diff = find_closest(self.objects_buffer, ts_scene)
        if obj_diff > 0.5:
            objects_msg = None

        try:
            image_cv = self.bridge.imgmsg_to_cv2(
                image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge: {e}')
            return

        annotated = draw_overlay(
            image_cv, scene_json, objects_msg, self.ref_w, self.ref_h)

        try:
            overlay_ros        = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            overlay_ros.header = image_msg.header
        except Exception as e:
            self.get_logger().error(f'cv2_to_imgmsg: {e}')
            return

        self.overlay_pub.publish(overlay_ros)

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

        if self.do_save:
            self._save(annotated, ts_scene, scene_json)

        self.cnt_frames += 1
        self.get_logger().info(
            f'Overlay #{self.cnt_frames} | diff={img_diff*1000:.1f}ms | '
            f'{"avec bbox" if objects_msg else "sans bbox"}',
            throttle_duration_sec=1.0)

    def _save(self, img: np.ndarray, ts: float, scene_json: dict):
        ts_ms    = int(time.time() * 1000)
        stem     = f'aeac_{ts_ms:015d}'
        img_path = self.save_dir / f'{stem}.jpg'
        txt_path = self.save_dir / f'{stem}.txt'

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        targets = scene_json.get('targets', [])
        lines   = [
            '═' * 60,
            f'FRAME  t={ts:.4f}s',
            '═' * 60,
            f'heading  : {scene_json.get("drone_heading", 0.0):.1f} deg',
            f'cloud_dt : {scene_json.get("cloud_dt", 0.0):.0f} ms',
            f'image_dt : {scene_json.get("image_dt") or "N/A"} ms',
            f'targets  : {len(targets)}',
            '',
        ]

        for i, t in enumerate(targets):
            ref     = t.get('reference')
            surface = t.get('surface', 'wall')
            coords  = t.get('local_coords')
            lines.append(f'── Target [{i+1}]  ({surface}) ' + '─' * 30)
            lines.append(f'  label     : {t.get("label", "?")}')
            lines.append(f'  height    : {t.get("height_m", 0.0):+.3f}m '
                         f'({"ABS" if t.get("height_source") == "absolute" else "REL"})')
            lines.append(f'  reference : {ref["label"] if ref else "none"}')
            wn = t.get('wall_normal')
            if wn:
                lines.append(f'  normal    : [{wn[0]:+.3f} {wn[1]:+.3f} {wn[2]:+.3f}]')
            if coords and ref:
                if surface == 'floor':
                    dw = coords.get('dist_wall', coords.get('z', 0.0))
                    lines.append(f'  dist_wall : {dw:+.3f}m')
                    lines.append(f'  lateral   : {coords.get("x", 0.0):+.3f}m')
                else:
                    lines.append(f'  local_x   : {coords.get("x", 0.0):+.3f}m  (droite+)')
                    lines.append(f'  local_y   : {coords.get("y", 0.0):+.3f}m  (haut+)')
            lines.append('')

        if self.last_scene_text:
            lines += ['═' * 60, 'SCENE DESCRIPTION', '═' * 60,
                      self.last_scene_text, '']

        txt_path.write_text('\n'.join(lines), encoding='utf-8')

        existing = sorted(self.save_dir.glob('aeac_*.jpg'))
        while len(existing) > self.max_saved:
            old = existing.pop(0)
            old.unlink(missing_ok=True)
            old.with_suffix('.txt').unlink(missing_ok=True)

    def _log_stats(self):
        self.get_logger().info(
            f'[STATS 10s] overlay={self.cnt_frames}  '
            f'no_img={self.cnt_no_img}  '
            f'img_buf={len(self.image_buffer)}  '
            f'obj_buf={len(self.objects_buffer)}'
        )
        self.cnt_frames = self.cnt_no_img = 0


def main():
    rclpy.init()
    rclpy.spin(OverlayNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()