#!/usr/bin/env python3
"""
scene_descriptor.py — Génération de descriptions en langage naturel des cibles

PRINCIPE
────────
Subscribe à /scene_description (JSON de wall_detector).
Pour chaque frame, génère une description textuelle en anglais pour chaque cible.

ORIENTATION
───────────
drone_heading_deg : direction vers laquelle la caméra pointe (0=Nord, 90=Est...).
La face du mur est calculée à partir de la normale du plan (wall_normal dans le JSON)
et du cap du drone. La face = direction opposée à la normale (la normale pointe
vers la caméra, donc la face du bâtiment regarde dans la direction opposée).

COORDONNÉES LOCALES (quand référence disponible)
────────────────────────────────────────────────
  Origine = centroïde de la référence (fenêtre/porte)
  x > 0 = droite de la référence (face au mur depuis l'extérieur)
  x < 0 = gauche de la référence
  y > 0 = au-dessus de la référence
  y < 0 = en dessous de la référence
  (z = profondeur, non utilisé pour la description)

TOPICS
──────
  Subscribe : /scene_description  (std_msgs/String JSON)
  Publish   : /scene_text         (std_msgs/String texte naturel)

LANCEMENT
─────────
  ros2 run <pkg> scene_descriptor --ros-args -p drone_heading_deg:=0.0
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import json
import math
from std_msgs.msg import String


# ── HELPERS D'ORIENTATION ──────────────────────────────────────────────────────

def _wall_face_cardinal(wall_normal: list, heading_deg: float) -> str:
    """
    Retourne la face cardinale du mur (ex: 'north', 'south-east').

    wall_normal  : [nx, ny, nz] dans le repère caméra REP-103 (X=Forward, Y=Left, Z=Up)
                   La normale pointe VERS la caméra.
    heading_deg  : cap de la caméra (0=Nord, 90=Est, 180=Sud, 270=Ouest)

    La face du mur = direction depuis laquelle on voit le mur = opposé de la normale.
    """
    nx, ny = wall_normal[0], wall_normal[1]  # composantes horizontales

    # Angle de la normale dans le repère caméra (plan horizontal)
    # atan2(Y_cam, X_cam) : X=Forward, Y=Left
    normal_angle_cam = math.degrees(math.atan2(ny, nx))

    # Angle dans le repère monde
    normal_angle_world = (heading_deg + normal_angle_cam) % 360

    # La face = opposé de la normale (la normale pointe vers la caméra,
    # donc la face pointe dans la direction d'où on regarde le mur)
    face_angle = (normal_angle_world + 180.0) % 360.0

    # Quantification en 8 directions
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


# ── GÉNÉRATION DE DESCRIPTION ──────────────────────────────────────────────────

def _describe_target(target: dict, heading_deg: float) -> str:
    """
    Génère la description complète d'une cible en anglais.

    Cas 1 — référence disponible (même plan que fenêtre/porte) :
      "Target (color: red): located on the north face of the structure.
       X.Xm to the right of the window (when facing the wall from outside).
       Y.Ym above the window.
       Height relative to camera: Z.Zm above the camera."

    Cas 2 — sans référence :
      "Target (color: red): no landmark reference found on this wall face.
       Height relative to camera: Z.Zm above/below the camera."
    """
    label     = target['label']
    h_rel     = target['h_rel_cam']
    reference = target['reference']
    coords    = target['local_coords']
    normal    = target.get('wall_normal')

    # Extraction de la couleur depuis le label
    # ex: 'red_target' → 'red', 'target' → 'unspecified color'
    parts = label.lower().replace('-', '_').split('_')
    if len(parts) >= 2 and parts[-1] == 'target':
        color = ' '.join(parts[:-1])
    elif label.lower() != 'target':
        color = label.lower()
    else:
        color = 'unspecified color'

    # Face du mur
    if normal and len(normal) >= 2:
        face = _wall_face_cardinal(normal, heading_deg)
        face_desc = f'the {face} face of the structure'
    else:
        face_desc = 'the structure (face undetermined)'

    h_dm  = _round_dm(abs(h_rel))
    h_dir = 'above' if h_rel >= 0 else 'below'
    height_str = f'{h_dm}m {h_dir} the camera'

    # ── CAS 1 : référence trouvée ──────────────────────────────────────────
    if reference is not None and coords is not None:
        ref_name = reference['label'].replace('_', ' ')
        x = coords['x']
        y = coords['y']

        x_dm = _round_dm(abs(x))
        y_dm = _round_dm(abs(y))

        # Position horizontale sur le mur
        if x_dm < 0.1:
            horiz = f'directly in front of the {ref_name}'
        else:
            h_side = 'right' if x >= 0 else 'left'
            horiz  = (f'{x_dm}m to the {h_side} of the {ref_name}'
                      f' (when facing the wall from outside)')

        # Position verticale par rapport à la référence
        if y_dm < 0.1:
            vert = f'at the same height as the {ref_name}'
        else:
            v_dir = 'above' if y >= 0 else 'below'
            vert  = f'{y_dm}m {v_dir} the {ref_name}'

        return (
            f'Target (color: {color}): located on {face_desc}. '
            f'{horiz.capitalize()}. '
            f'{vert.capitalize()}. '
            f'Height relative to camera: {height_str}.'
        )

    # ── CAS 2 : sans référence ─────────────────────────────────────────────
    return (
        f'Target (color: {color}): located on {face_desc}. '
        f'No landmark reference found on this wall face. '
        f'Height relative to camera: {height_str}.'
    )


# ── NODE ───────────────────────────────────────────────────────────────────────

class SceneDescriptorNode(Node):
    def __init__(self):
        super().__init__('scene_description')

        self.declare_parameter('drone_heading_deg', 0.0)
        self.heading = self.get_parameter('drone_heading_deg').value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(String, '/scene_description', self._scene_cb, qos)
        self.pub = self.create_publisher(String, '/scene_text', 10)

        self.get_logger().info(
            f'SceneDescriptor démarré\n'
            f'  drone_heading={self.heading}° (0=Nord, 90=Est, 180=Sud, 270=Ouest)\n'
            f'  Subscribe : /scene_description\n'
            f'  Publish   : /scene_text'
        )

    def _scene_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON invalide : {e}')
            return

        targets   = data.get('targets', [])
        timestamp = data.get('timestamp', 0.0)

        if not targets:
            return

        lines = [f'=== Target Localization Report (t={timestamp:.2f}s) ===']

        for i, target in enumerate(targets):
            desc = _describe_target(target, self.heading)
            lines.append(f'\n[Target {i + 1}]')
            lines.append(desc)
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