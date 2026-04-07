#!/usr/bin/env python3
"""
scene_description.py — Génération de descriptions en langage naturel des cibles

Aucun changement fonctionnel vs la version précédente.
Ajout des labels white_circle / black_circle dans _target_name().

TOPICS
──────
  Subscribe : /aeac/internal/scene_description  (std_msgs/String JSON)
  Publish   : /aeac/internal/scene_text         (std_msgs/String texte)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
import json
import math
from std_msgs.msg import String


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS D'ORIENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _wall_face_cardinal(wall_normal: list, heading_deg: float) -> str:
    nx, ny = wall_normal[0], wall_normal[1]
    
    # Si la normale est quasi-verticale (sim_camera Z=Forward),
    # nx et ny sont proches de zéro → atan2 instable
    # Dans ce cas la face cardinale dépend du heading uniquement
    if abs(nx) < 0.1 and abs(ny) < 0.1:
        # La caméra regarde dans une direction déterminée par heading
        # Le mur est face à la caméra → face = direction opposée au heading
        face_angle = (heading_deg + 180.0) % 360.0
    else:
        normal_angle_cam   = math.degrees(math.atan2(ny, nx))
        normal_angle_world = (heading_deg + normal_angle_cam) % 360
        face_angle         = (normal_angle_world + 180.0) % 360.0
    
    dirs = [
        (0.0,   'north'), (45.0,  'north-east'), (90.0,  'east'),
        (135.0, 'south-east'), (180.0, 'south'), (225.0, 'south-west'),
        (270.0, 'west'), (315.0, 'north-west'),
    ]
    best, best_diff = 'north', float('inf')
    for angle, name in dirs:
        diff = abs((face_angle - angle + 180.0) % 360.0 - 180.0)
        if diff < best_diff:
            best_diff, best = diff, name
    return best


def _round_dm(v: float) -> float:
    return round(v, 1)


def _target_name(label: str) -> str:
    lo = label.lower().replace('-', '_')

    if lo.endswith('_target'):
        color = lo[:-7].replace('_', ' ').strip()
        return f'{color} target' if color else 'target'

    if lo.endswith(' target') or ' target' in lo:
        return lo.strip()

    if lo.startswith('circle_'):
        color = lo[7:].replace('_', ' ').strip()
        return f'{color} circle' if color else 'circle'

    if lo.endswith('_circle'):
        color = lo[:-7].replace('_', ' ').strip()
        return f'{color} circle' if color else 'circle'

    # "red circle", "blue circle", etc. (avec espace)
    if lo.endswith(' circle'):
        return lo.strip()

    return lo.replace('_', ' ')


# ══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def _describe_target(target: dict, heading_deg: float) -> str:
    label     = target['label']
    h         = target.get('height_m', 0.0)
    reference = target.get('reference')
    coords    = target.get('local_coords')
    normal    = target.get('wall_normal')
    surface   = target.get('surface', 'wall')

    name = _target_name(label)

    source    = target.get('height_source', 'relative_cam')
    h_abs     = abs(round(h, 1))
    h_dir     = 'above' if h >= 0 else 'below'
    ref_point = 'the ground' if source == 'absolute' else 'the camera'
    height_str = f'{h_abs}m {h_dir} {ref_point}'

    if normal and len(normal) >= 2:
        face = _wall_face_cardinal(normal, heading_deg)
    else:
        face = None

    # ── CAS 1 : cible au sol ─────────────────────────────────────────────────
    if surface == 'floor':

        if reference is not None and coords is not None:
            ref_name  = reference['label'].replace('_', ' ')
            x_m       = coords.get('x', 0.0)
            dist_wall = coords.get('dist_wall', coords.get('z', 0.0))

            x_dm  = _round_dm(abs(x_m))
            dw_dm = _round_dm(dist_wall)

            ref_face = face if face else 'adjacent'

            dist_str = (f'{dw_dm}m away from the {ref_face} face of the structure'
                        if ref_face and ref_face != 'adjacent'
                        else f'{dw_dm}m away from the wall')

            if x_dm < 0.1:
                lat_str = f'directly in front of the {ref_name}'
            else:
                side    = 'right' if x_m >= 0 else 'left'
                lat_str = (f'{x_dm}m to the {side} of the {ref_name}'
                           f' when facing the wall from outside')

            return (
                f'The {name} is on the ground, '
                f'{dist_str}, '
                f'{lat_str}. '
                f'It is {height_str}.'
            )
        else:
            loc_str = (f'near the {face} face of the structure'
                       if face else 'on the ground')
            return (
                f'The {name} is on the ground {loc_str}. '
                f'No landmark reference was found nearby. '
                f'It is {height_str}.'
            )

    # ── CAS 2 : cible sur mur ────────────────────────────────────────────────
    face_desc = (f'the {face} face of the structure'
                 if face else 'the structure (wall face undetermined)')

    if reference is not None and coords is not None:
        ref_name = reference['label'].replace('_', ' ')
        x = coords.get('x', 0.0)
        y = coords.get('y', 0.0)

        x_dm = _round_dm(abs(x))
        y_dm = _round_dm(abs(y))

        if x_dm < 0.1:
            horiz = f'directly aligned with the {ref_name}'
        else:
            side  = 'right' if x >= 0 else 'left'
            horiz = (f'{x_dm}m to the {side} of the {ref_name}'
                     f' when facing the wall from outside')

        if y_dm < 0.1:
            vert = f'at the same height as the {ref_name}'
        else:
            v_dir = 'above' if y >= 0 else 'below'
            vert  = f'{y_dm}m {v_dir} the {ref_name}'

        return (
            f'The {name} is located on {face_desc}, '
            f'{horiz}, '
            f'{vert}. '
            f'It is {height_str}.'
        )

    return (
        f'The {name} is located on {face_desc}. '
        f'No landmark reference was found on this wall face. '
        f'It is {height_str}.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# NODE ROS2
# ══════════════════════════════════════════════════════════════════════════════

class SceneDescriptorNode(Node):

    def __init__(self):
        super().__init__('scene_description_node')

        self.declare_parameter('drone_heading_deg', 0.0)
        self.heading = self.get_parameter('drone_heading_deg').value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            String, '/aeac/internal/scene_description',
            self._scene_cb, qos)

        self.pub = self.create_publisher(
            String, '/aeac/internal/scene_text', 10)

        self.get_logger().info(
            f'SceneDescriptor démarré\n'
            f'  drone_heading={self.heading}°\n'
            f'  Subscribe : /aeac/internal/scene_description\n'
            f'  Publish   : /aeac/internal/scene_text'
        )

    def _scene_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON invalide : {e}')
            return

        targets   = data.get('targets', [])
        timestamp = data.get('timestamp', 0.0)
        heading   = data.get('drone_heading', self.heading)

        if not targets:
            return

        lines = [f'Target Localization Report — t={timestamp:.2f}s\n']

        for i, target in enumerate(targets):
            desc = _describe_target(target, heading)
            lines.append(f'Target {i + 1}:')
            lines.append(desc)
            lines.append('')
            self.get_logger().info(f'[Target {i+1}] {desc}')

        out      = String()
        out.data = '\n'.join(lines)
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(SceneDescriptorNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()