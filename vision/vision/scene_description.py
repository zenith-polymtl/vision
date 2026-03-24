#!/usr/bin/env python3
"""
scene_description.py — Génération de descriptions en langage naturel des cibles

PRINCIPE
────────
Subscribe à /aeac/internal/scene_description (JSON de system_transformation).
Pour chaque frame, génère une description textuelle en anglais pour chaque cible.

TROIS CAS DE DESCRIPTION
─────────────────────────
  1. Cible sur mur, avec référence (fenêtre/porte sur même plan) :
       "The blue target is located on the north face of the structure,
        1.3m to the right of the door when facing the wall from outside,
        0.1m below the door. It is 0.9m above the camera."

  2. Cible au sol, avec référence murale proche :
       "The yellow target is on the ground, 5.2m away from the west face
        of the structure, 0.2m to the left of the door when facing the wall
        from outside. It is 0.0m above the ground."

  3. Cible sans référence (mur ou sol) :
       "The red target is located on the west face of the structure.
        No landmark reference was found on this wall face.
        It is 1.1m above the camera."

DÉTECTION SOL VS MUR
─────────────────────
  Basée sur le champ "surface" du JSON, calculé dans system_transformation
  à partir de la normale RANSAC : |nz| > 0.7 → sol, sinon → mur.

COORDONNÉES LOCALES (champ local_coords du JSON)
─────────────────────────────────────────────────
  Cible sur mur :
    x  = droite sur le mur  (+ = droite, face au mur depuis l'extérieur)
    y  = haut sur le mur    (+ = haut)
    z  = profondeur         (non utilisé pour la description)

  Cible au sol :
    x         = latéral le long du mur de référence (+ = droite face au mur)
    y         = Δ hauteur   (quasi-nul, non utilisé)
    z / dist_wall = distance perpendiculaire au plan du mur (toujours positive)

ORIENTATION
───────────
  drone_heading_deg : cap de la caméra (0=Nord, 90=Est, 180=Sud, 270=Ouest).
  La face du mur = direction opposée à la normale (normale pointe vers la caméra).

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
    """
    Retourne la face cardinale du mur (ex: 'north', 'south-east').

    wall_normal  : [nx, ny, nz] dans le repère caméra REP-103 (X=Forward, Y=Left, Z=Up)
                   La normale pointe VERS la caméra.
    heading_deg  : cap de la caméra (0=Nord, 90=Est, 180=Sud, 270=Ouest)

    La face du mur = direction depuis laquelle on voit le mur = opposé de la normale.
    """
    nx, ny = wall_normal[0], wall_normal[1]
    normal_angle_cam   = math.degrees(math.atan2(ny, nx))
    normal_angle_world = (heading_deg + normal_angle_cam) % 360
    face_angle         = (normal_angle_world + 180.0) % 360.0

    dirs = [
        (0.0,   'north'),
        (45.0,  'north-east'),
        (90.0,  'east'),
        (135.0, 'south-east'),
        (180.0, 'south'),
        (225.0, 'south-west'),
        (270.0, 'west'),
        (315.0, 'north-west'),
    ]
    best, best_diff = 'north', float('inf')
    for angle, name in dirs:
        diff = abs((face_angle - angle + 180.0) % 360.0 - 180.0)
        if diff < best_diff:
            best_diff, best = diff, name
    return best


def _round_dm(v: float) -> float:
    """Arrondit au décimètre (1 décimale)."""
    return round(v, 1)


def _target_name(label: str) -> str:
    """
    Extrait le nom lisible de la cible depuis son label.
      'red_target'   → 'red target'
      'blue target'  → 'blue target'
      'circle_red'   → 'red circle'
      'red_circle'   → 'red circle'
      'target'       → 'target'
      'person'       → 'person'
    """
    lo = label.lower().replace('-', '_')

    if lo.endswith('_target'):
        color = lo[:-7].replace('_', ' ').strip()
        return f'{color} target' if color else 'target'

    # Gère 'blue target' (avec espace, pas underscore)
    if lo.endswith(' target') or ' target' in lo:
        return lo.strip()

    if lo.startswith('circle_'):
        color = lo[7:].replace('_', ' ').strip()
        return f'{color} circle' if color else 'circle'

    if lo.endswith('_circle'):
        color = lo[:-7].replace('_', ' ').strip()
        return f'{color} circle' if color else 'circle'

    return lo.replace('_', ' ')


# ══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def _describe_target(target: dict, heading_deg: float) -> str:
    """
    Génère une description en langage naturel, claire et non ambiguë.
    Gère les cibles sur mur (surface='wall') et au sol (surface='floor').
    """
    label     = target['label']
    h         = target.get('height_m', 0.0)
    reference = target.get('reference')
    coords    = target.get('local_coords')
    normal    = target.get('wall_normal')
    surface   = target.get('surface', 'wall')

    name = _target_name(label)

    # ── Hauteur ───────────────────────────────────────────────────────────────
    source    = target.get('height_source', 'relative_cam')
    h_abs     = abs(round(h, 1))
    h_dir     = 'above' if h >= 0 else 'below'
    ref_point = 'the ground' if source == 'absolute' else 'the camera'
    height_str = f'{h_abs}m {h_dir} {ref_point}'

    # ── Face du mur ───────────────────────────────────────────────────────────
    # Pour une cible au sol : la normale est celle du plan sol (quasi-verticale
    # en Z), pas du mur. On utilise la normale du mur de référence pour la face.
    # Si pas de référence, on utilise la normale de la cible elle-même (approx).
    if normal and len(normal) >= 2:
        face = _wall_face_cardinal(normal, heading_deg)
    else:
        face = None

    # ════════════════════════════════════════════════════════════════════════
    # CAS 1 — CIBLE AU SOL
    # ════════════════════════════════════════════════════════════════════════
    if surface == 'floor':

        if reference is not None and coords is not None:
            ref_name  = reference['label'].replace('_', ' ')
            x_m       = coords.get('x', 0.0)          # latéral le long du mur
            dist_wall = coords.get('dist_wall',
                        coords.get('z', 0.0))           # distance au mur

            x_dm   = _round_dm(abs(x_m))
            dw_dm  = _round_dm(dist_wall)

            # Face du mur de référence — on utilise la normale de la cible sol
            # qui pointe vers le haut, donc la face doit venir de la référence.
            # On a la wall_normal de la cible (sol), pas du mur adjacent.
            # On utilise donc heading_deg + la direction approximative vers le mur.
            # Si on a la normale du mur de ref dans le JSON, on s'en sert.
            # Ici on utilise la normale transmise (celle du plan sol, approx).
            ref_face = face if face else 'adjacent'

            # Distance au mur
            dist_str = (f'{dw_dm}m away from the {ref_face} face of the structure'
                        if ref_face and ref_face != 'adjacent'
                        else f'{dw_dm}m away from the wall')

            # Position latérale
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
            # Sol sans référence
            loc_str = (f'near the {face} face of the structure'
                       if face else 'on the ground')
            return (
                f'The {name} is on the ground {loc_str}. '
                f'No landmark reference was found nearby. '
                f'It is {height_str}.'
            )

    # ════════════════════════════════════════════════════════════════════════
    # CAS 2 — CIBLE SUR MUR
    # ════════════════════════════════════════════════════════════════════════
    face_desc = (f'the {face} face of the structure'
                 if face else 'the structure (wall face undetermined)')

    if reference is not None and coords is not None:
        ref_name = reference['label'].replace('_', ' ')
        x = coords.get('x', 0.0)
        y = coords.get('y', 0.0)

        x_dm = _round_dm(abs(x))
        y_dm = _round_dm(abs(y))

        # Horizontal (gauche/droite de la référence)
        if x_dm < 0.1:
            horiz = f'directly aligned with the {ref_name}'
        else:
            side  = 'right' if x >= 0 else 'left'
            horiz = (f'{x_dm}m to the {side} of the {ref_name}'
                     f' when facing the wall from outside')

        # Vertical (haut/bas de la référence)
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

    # Mur sans référence
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
            f'  drone_heading={self.heading}° (0=Nord, 90=Est, 180=Sud, 270=Ouest)\n'
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

        full_text = '\n'.join(lines)

        out      = String()
        out.data = full_text
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(SceneDescriptorNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()