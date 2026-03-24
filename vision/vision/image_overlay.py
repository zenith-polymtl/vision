#!/usr/bin/env python3
"""
overlay_node.py — Overlay visuel synchronisé avec le pipeline AEAC

CHANGEMENTS v3
──────────────
  - Suppression des flèches normales (peu lisibles, peu utiles visuellement)
  - Layout compact : toutes les infos d'une cible dans un seul bloc à droite
    de la bbox, plutôt qu'empilées au-dessus et en-dessous
  - Badge label principal collé en haut de la bbox (plus petit, plus propre)
  - Infos ZED compactées sur une ligne : "ZED 2.6 -1.3 0.8m"
  - Normale affichée comme texte court "n[-0.92 +0.39]" (x,y seulement, z omis)
    sous le bloc principal si place disponible
  - Indicateur surface sol/mur : icône 🟥 (mur) ou 🔲 (sol) dans le badge
  - HUD bas de frame inchangé
  - Légende simplifiée (sans entrée normale)

TOPICS
──────
  Subscribe:
    <image_topic>        /zed/zed_node/rgb/color/rect/image
    <objects_topic>      /aeac/test/objects
    <scene_desc_topic>   /aeac/internal/scene_description
    <scene_text_topic>   /aeac/internal/scene_text
  Publish:
    /aeac/external/overlay_image   sensor_msgs/Image
    /aeac/external/scene_frame     custom_interfaces/SceneFrame

PARAMÈTRES ROS2
───────────────
  image_topic        str   /zed/zed_node/rgb/color/rect/image
  objects_topic      str   /aeac/test/objects
  scene_desc_topic   str   /aeac/internal/scene_description
  scene_text_topic   str   /aeac/internal/scene_text
  image_width        int   448
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
import time
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
# PALETTE & CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# BGR
C_TARGET    = (0,   200, 255)   # Orange vif   → cibles
C_REFERENCE = (50,  230,  80)   # Vert vif     → références (porte/fenêtre)
C_OTHER     = (150, 150, 150)   # Gris         → autres objets
C_COORDS    = (120, 255, 120)   # Vert pâle    → coords locales / dist mur
C_RAW       = (210, 170, 255)   # Mauve pâle   → position brute ZED
C_HUD_BG    = (10,   10,  10)
C_HUD_TXT   = (220, 220, 220)
C_WHITE     = (255, 255, 255)
C_BLACK     = (  0,   0,   0)
C_HEIGHT    = (0,   240, 200)   # Turquoise    → hauteur

FONT = cv2.FONT_HERSHEY_SIMPLEX

REFERENCE_LABELS = {'window', 'door', 'fenetre', 'porte'}
TARGET_LABELS = {
    'person', 'target', 'cible',
    'red_target', 'blue_target', 'green_target', 'yellow_target', 'orange_target',
    'red target', 'blue target', 'green target', 'yellow target', 'orange target',
    'red_circle', 'blue_circle', 'green_circle', 'yellow_circle',
    'red circle', 'blue circle', 'green circle', 'yellow circle',
    'circle_red', 'circle_blue', 'circle_green', 'circle_yellow',
}

IMG_MATCH_TOL = 0.15   # secondes


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS GÉNÉRAUX
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
    if lo in {l.lower().replace(' ', '_').replace('-', '_') for l in TARGET_LABELS}:
        return C_TARGET
    if lo in REFERENCE_LABELS:
        return C_REFERENCE
    return C_OTHER


def _scale_factors(img_w: int, img_h: int):
    """Facteurs visuels calibrés sur 1280×720."""
    ref_diag = (1280**2 + 720**2) ** 0.5
    img_diag = (img_w**2 + img_h**2) ** 0.5
    s        = img_diag / ref_diag

    return {
        'fs_lg':  max(0.50, round(0.68 * s, 2)),   # label principal
        'fs_md':  max(0.40, round(0.52 * s, 2)),   # infos secondaires
        'fs_sm':  max(0.35, round(0.42 * s, 2)),   # HUD / légende
        'th_box': max(2,    int(3    * s)),
        'th_txt': max(1,    int(1.5  * s)),
        'pad':    max(4,    int(6    * s)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVES DESSIN
# ══════════════════════════════════════════════════════════════════════════════

def _draw_bbox(img, x1, y1, x2, y2, color, th):
    """Rectangle + coins en L. Fond très léger."""
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
    """
    Texte sur fond opaque.
    (x, y) = coin haut-gauche.
    Retourne y_bottom.
    """
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


def _info_block(img, x, y, lines: list, sc: dict) -> int:
    """
    Affiche une liste de (text, color) empilés verticalement.
    x, y = coin haut-gauche du bloc.
    Retourne y_bottom du bloc.
    """
    y_cur = y
    for text, color in lines:
        y_cur = _badge(img, x, y_cur, text, color,
                       sc['fs_md'], sc['th_txt']) + 2
    return y_cur


def _draw_legend(img, sc: dict):
    """Légende simplifiée — coin haut droit."""
    entries = [
        (C_TARGET,    'target'),
        (C_REFERENCE, 'reference (door/window)'),
        (C_COORDS,    'local coords / dist wall'),
        (C_RAW,       'ZED 3D position'),
        (C_HEIGHT,    'height'),
    ]
    fs     = sc['fs_sm']
    th     = sc['th_txt']
    pad    = 6
    box_w  = max(12, int(16 * (sc['fs_sm'] / 0.4)))
    lh     = max(18, int(cv2.getTextSize('A', FONT, fs, th)[0][1] * 2.3))
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
                      (x0 + pad,          yc - box_w // 2),
                      (x0 + pad + box_w,  yc + box_w // 2),
                      col, -1)
        cv2.putText(img, lbl,
                    (x0 + pad + box_w + 6, yc + int(box_w * 0.4)),
                    FONT, fs, C_HUD_TXT, th, cv2.LINE_AA)


def _draw_hud(img, scene_json: dict, sc: dict):
    """Bande d'info opaque en bas."""
    ts   = scene_json.get('timestamp', 0.0)
    c_dt = scene_json.get('cloud_dt', 0.0)
    i_dt = scene_json.get('image_dt')
    hdg  = scene_json.get('drone_heading', 0.0)
    n_t  = len(scene_json.get('targets', []))
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
# OVERLAY PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(image_cv: np.ndarray,
                 scene_json: dict,
                 objects_msg,
                 ref_img_w: int,
                 ref_img_h: int) -> np.ndarray:
    """
    Dessine l'overlay complet et retourne l'image annotée.

    Layout par objet
    ────────────────
    Au-dessus de la bbox :
      [label  conf%]          ← badge couleur de l'objet

    À droite (ou en dessous si débord) — bloc compact :
      [ZED x y z m]           ← position brute ZED, une ligne
      [x.xm R  y.ym↑  [ref]] ← coords locales (mur) ou dist+latéral (sol)
      [H=+x.xm ABS/REL]       ← hauteur

    Pas de flèche normale (trop difficile à lire).
    """
    out    = image_cv.copy()
    h_img, w_img = out.shape[:2]

    scale_u = w_img / ref_img_w
    scale_v = h_img / ref_img_h
    sc      = _scale_factors(w_img, h_img)

    # Index JSON par label
    targets_by_label: dict[str, list] = {}
    for t in scene_json.get('targets', []):
        targets_by_label.setdefault(t['label'].lower(), []).append(t)

    if objects_msg is not None:
        for obj in objects_msg.objects:
            if not hasattr(obj, 'bounding_box_2d') or \
               not obj.bounding_box_2d.corners:
                continue

            # ── Géométrie bbox ─────────────────────────────────────────────
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

            # ── Bbox + point central ───────────────────────────────────────
            _draw_bbox(out, x1, y1, x2, y2, color, sc['th_box'])
            cv2.circle(out, (cx, cy), max(3, sc['th_box']),
                       color, -1, cv2.LINE_AA)

            # ── Badge label + confiance (au-dessus de la bbox) ────────────
            conf      = getattr(obj, 'confidence', 0.0)
            lbl_text  = f'{label}  {conf:.0f}%'
            (_, lh_), _ = cv2.getTextSize(lbl_text, FONT, sc['fs_lg'], sc['th_txt'])
            y_lbl = max(lh_ + 4, y1 - lh_ - 6)
            _badge(out, x1, y_lbl, lbl_text, color, sc['fs_lg'], sc['th_txt'])

            # ── Bloc info compact ──────────────────────────────────────────
            # Positionné à droite de la bbox si la place le permet,
            # sinon en dessous.
            BLOCK_MARGIN = 6
            bx = x2 + BLOCK_MARGIN   # tentative à droite

            lo      = label.lower()
            jt_list = targets_by_label.get(lo, [])
            best_t  = jt_list[0] if jt_list else None

            info_lines = []  # liste de (text, color)

            # ── Ligne 1 : position brute ZED compactée ────────────────────
            if hasattr(obj, 'position') and len(obj.position) >= 3:
                px, py, pz = obj.position[0], obj.position[1], obj.position[2]
                info_lines.append(
                    (f'ZED {px:+.1f} {py:+.1f} {pz:+.1f}m', C_RAW))

            if best_t is not None:
                surface = best_t.get('surface', 'wall')
                coords  = best_t.get('local_coords')
                ref     = best_t.get('reference')
                h_m     = best_t.get('height_m')
                h_src   = best_t.get('height_source', 'relative_cam')

                # ── Ligne 2 : coords locales ──────────────────────────────
                if coords and ref:
                    ref_lbl = ref['label']
                    if surface == 'floor':
                        # Sol : dist_wall + latéral
                        dist_w = coords.get('dist_wall', coords.get('z', 0.0))
                        x_lat  = coords.get('x', 0.0)
                        x_dir  = 'R' if x_lat >= 0 else 'L'
                        info_lines.append((
                            f'{abs(dist_w):.1f}m wall  '
                            f'{abs(x_lat):.1f}m {x_dir}  [{ref_lbl}]',
                            C_COORDS
                        ))
                    else:
                        # Mur : droite/gauche + haut/bas
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

                # ── Ligne 3 : hauteur ─────────────────────────────────────
                if h_m is not None:
                    tag = 'ABS' if h_src == 'absolute' else 'REL'
                    info_lines.append((f'H {h_m:+.2f}m {tag}', C_HEIGHT))

            if not info_lines:
                continue

            # Vérifier si le bloc tient à droite de l'image
            max_tw = max(
                cv2.getTextSize(t, FONT, sc['fs_md'], sc['th_txt'])[0][0]
                for t, _ in info_lines
            )
            if bx + max_tw + sc['pad'] * 2 > w_img:
                bx = max(0, x1)   # rabat en dessous-gauche
                by = y2 + BLOCK_MARGIN
            else:
                by = y1           # aligne en haut avec la bbox

            _info_block(out, bx, by, info_lines, sc)

    # ── HUD + légende ──────────────────────────────────────────────────────
    _draw_hud(out, scene_json, sc)
    _draw_legend(out, sc)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# NODE ROS2
# ══════════════════════════════════════════════════════════════════════════════

class OverlayNode(Node):

    def __init__(self):
        super().__init__('overlay_node')

        # ── Paramètres ────────────────────────────────────────────────────
        self.declare_parameter('image_topic',      '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('objects_topic',    '/aeac/test/objects')
        self.declare_parameter('scene_desc_topic', '/aeac/internal/scene_description')
        self.declare_parameter('scene_text_topic', '/aeac/internal/scene_text')
        self.declare_parameter('image_width',      448)
        self.declare_parameter('image_height',     256)
        self.declare_parameter('buffer_size',       60)
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
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(Image, image_topic,
                                 self._image_cb, qos)

        if ZED_MSGS_AVAILABLE:
            self.create_subscription(ObjectsStamped, objects_topic,
                                     self._objects_cb, qos)
            self.get_logger().info(f'Objects → {objects_topic}')
        else:
            self.get_logger().warn('zed_msgs non disponible — bbox désactivées')

        self.create_subscription(String, scene_desc_topic,
                                 self._scene_cb, qos)
        self.create_subscription(String, scene_text_topic,
                                 self._text_cb, qos)

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
            f'  image      : {image_topic}\n'
            f'  objects    : {objects_topic}\n'
            f'  scene_desc : {scene_desc_topic}\n'
            f'  scene_text : {scene_text_topic}\n'
            f'  bbox_ref   : {self.ref_w}×{self.ref_h}\n'
            f'  SceneFrame : {"OK" if CUSTOM_MSG_AVAILABLE else "DÉSACTIVÉ"}\n'
            f'  save       : {self.do_save}  '
            f'dir={self.save_dir}  max={self.max_saved}'
        )

    # ── Buffers ───────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        self.image_buffer.append((stamp_to_sec(msg.header.stamp), msg))

    def _objects_cb(self, msg):
        self.objects_buffer.append((stamp_to_sec(msg.header.stamp), msg))

    def _text_cb(self, msg: String):
        self.last_scene_text = msg.data

    # ── Déclencheur ───────────────────────────────────────────────────────────

    def _scene_cb(self, msg: String):
        try:
            scene_json = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON invalide: {e}')
            return

        ts_scene    = scene_json.get('timestamp', 0.0)
        image_stamp = scene_json.get('image_stamp')
        sync_target = image_stamp if image_stamp is not None else ts_scene
        sync_mode   = 'image_stamp' if image_stamp else 'timestamp(fallback)'

        image_msg, img_diff = find_closest(self.image_buffer, sync_target)
        if image_msg is None or img_diff > IMG_MATCH_TOL:
            self.cnt_no_img += 1
            self.get_logger().warn(
                f'Image non trouvée (sync={sync_mode}, '
                f'diff={img_diff * 1000:.0f}ms)')
            return

        objects_msg, obj_diff = find_closest(self.objects_buffer, ts_scene)
        if obj_diff > 0.5:
            objects_msg = None
            self.get_logger().warn(
                f'Objets désynchronisés ({obj_diff * 1000:.0f}ms)')

        try:
            image_cv = self.bridge.imgmsg_to_cv2(
                image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge: {e}')
            return

        annotated = draw_overlay(image_cv, scene_json, objects_msg,
                                 self.ref_w, self.ref_h)

        try:
            overlay_ros        = self.bridge.cv2_to_imgmsg(
                annotated, encoding='bgr8')
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
            f'Overlay #{self.cnt_frames} | {sync_mode} | '
            f'img_diff={img_diff * 1000:.1f}ms | '
            f'{"avec bbox" if objects_msg else "sans bbox"}')

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def _save(self, img: np.ndarray, ts: float, scene_json: dict):
        """Sauvegarde image + .txt avec wall-clock timestamp pour éviter
        les doublons lors des boucles de rosbag."""
        ts_ms    = int(time.time() * 1000)
        stem     = f'aeac_{ts_ms:015d}'
        img_path = self.save_dir / f'{stem}.jpg'
        txt_path = self.save_dir / f'{stem}.txt'

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

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
            lines.append(f'── Target [{i+1}]  ({surface}) '
                         + '─' * 30)
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
        else:
            lines += ['═' * 60,
                      'SCENE DESCRIPTION : (pas encore reçue)',
                      '═' * 60, '']

        txt_path.write_text('\n'.join(lines), encoding='utf-8')

        # Rotation
        existing = sorted(self.save_dir.glob('aeac_*.jpg'))
        while len(existing) > self.max_saved:
            old = existing.pop(0)
            old.unlink(missing_ok=True)
            old.with_suffix('.txt').unlink(missing_ok=True)

    # ── Stats ─────────────────────────────────────────────────────────────────

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