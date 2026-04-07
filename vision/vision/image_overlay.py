#!/usr/bin/env python3
"""
overlay_node.py — Overlay visuel synchronisé avec le pipeline AEAC

CHANGEMENTS vs version précédente
──────────────────────────────────
  1. QoS BEST_EFFORT pour les topics ZED image
     → Matcher le wrapper pour éviter la non-réception silencieuse

  2. Synchronisation sur now() timestamp
     → system_transformation publie scene_description avec un timestamp
        correspondant au moment de publication (now() de stereo_yolo).
        L'overlay cherche l'image la plus proche de ce timestamp dans son buffer.

  3. Buffer image maxlen=10 (10Hz pub_frame_rate → 1s d'historique)
     → Suffisant pour absorber les délais du pipeline sans garder du périmé

  4. IMG_MATCH_TOL augmentée à 0.3s
     → La chaîne stereo_yolo (~200ms) + system_transformation (~100ms) = ~300ms
        entre la capture de l'image et la publication de scene_description.
        L'overlay doit trouver une image dans ce délai.

TOPICS
──────
  Subscribe:
    /aeac/internal/target_detected        ← ObjectsStamped (RELIABLE)
    /aeac/internal/scene_description      ← JSON (RELIABLE)
    /aeac/internal/scene_text             ← texte (RELIABLE)
  Publish:
    /aeac/external/overlay_image          ← Image annotée
    /aeac/external/scene_frame            ← SceneFrame (si custom_interfaces dispo)
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

FONT = cv2.FONT_HERSHEY_SIMPLEX

REFERENCE_LABELS = {'window', 'door', 'fenetre', 'porte'}
TARGET_LABELS = {
    'person', 'target', 'cible',
    'red_target',   'green', 'yellow', 'red', 'blue',  'blue_target',    'green_target',
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


IMG_MATCH_TOL = 1


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


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVES DESSIN
# ══════════════════════════════════════════════════════════════════════════════

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


def _info_block(img, x, y, lines: list, sc: dict) -> int:
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


def _draw_hud(img, scene_json: dict, sc: dict):
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
    out      = image_cv.copy()
    h_img, w_img = out.shape[:2]

    scale_u = w_img / ref_img_w
    scale_v = h_img / ref_img_h
    sc      = _scale_factors(w_img, h_img)

    targets_by_label: dict[str, list] = {}
    for t in scene_json.get('targets', []):
        targets_by_label.setdefault(t['label'].lower(), []).append(t)

    if objects_msg is not None:
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
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            _draw_bbox(out, x1, y1, x2, y2, color, sc['th_box'])


            conf     = getattr(obj, 'confidence', 0.0)
            lbl_text = f'{label}  {conf:.0f}%'
            (_, lh_), _ = cv2.getTextSize(lbl_text, FONT, sc['fs_lg'], sc['th_txt'])
            y_lbl = max(lh_ + 4, y1 - lh_ - 6)
            _badge(out, x1, y_lbl, lbl_text, color, sc['fs_lg'], sc['th_txt'])

            BLOCK_MARGIN = 6
            bx      = x2 + BLOCK_MARGIN
            lo      = label.lower()

            if lo not in targets_by_label:
                best_t = None
            elif len(targets_by_label[lo]) == 1:
                best_t = targets_by_label[lo][0]
            else:
                # Pop le premier élément pour que le prochain objet du même label
                # prenne la target suivante
                best_t = targets_by_label[lo].pop(0)

            info_lines = []

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

            if not info_lines:
                continue

            max_tw = max(
                cv2.getTextSize(t, FONT, sc['fs_md'], sc['th_txt'])[0][0]
                for t, _ in info_lines
            )
            if bx + max_tw + sc['pad'] * 2 > w_img:
                bx = max(0, x1)
                by = y2 + BLOCK_MARGIN
            else:
                by = y1

            _info_block(out, bx, by, info_lines, sc)

    _draw_hud(out, scene_json, sc)
    _draw_legend(out, sc)

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
        # Buffer image : maxlen=10 → 1s à 10Hz
        self.image_buffer   = deque(maxlen=50)
        self.objects_buffer = deque(maxlen=10)
        self.last_scene_text = ''
        self.cnt_frames = self.cnt_no_img = 0

        if self.do_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # QoS BEST_EFFORT pour les topics ZED
        zed_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_subscription(
            Image, image_topic, self._image_cb, zed_qos)

        # Objets stereo_yolo → RELIABLE
        if ZED_MSGS_AVAILABLE:
            self.create_subscription(
                ObjectsStamped, objects_topic,
                self._objects_cb, reliable_qos)
        else:
            self.get_logger().warn('zed_msgs non disponible — bbox désactivées')

        # Topics internes → RELIABLE
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
            f'  image      : {image_topic} (BEST_EFFORT)\n'
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