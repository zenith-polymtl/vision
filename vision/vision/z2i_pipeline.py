import rclpy
from rclpy.node import Node
from rclpy.time import Time
from zed_msgs.msg import ObjectsStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import os
import json
from datetime import datetime


# ── Tuning knobs ──────────────────────────────────────────────────────────────
SYNC_SLOP_SEC   = 0.1   # max allowed timestamp difference for "same frame"
QUEUE_DEPTH     = 30     # how many messages to buffer for the synchroniser
MIN_CONFIDENCE  = 30.0   # skip objects below this confidence (%)
SAVE_DIR        = "/water_ws/Pictures/zed_sdk_detections"

# Colour palette – one colour per label (cycles if more labels than colours)
PALETTE = [
    (0, 255, 0),    # green
    (255, 100, 0),  # orange
    (0, 150, 255),  # sky-blue
    (200, 0, 200),  # purple
    (0, 255, 255),  # cyan
    (255, 255, 0),  # yellow
]
# ─────────────────────────────────────────────────────────────────────────────


def euclidean_dist(pos) -> float:
    """3-D Euclidean distance from position array [x, y, z]."""
    return float((pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) ** 0.5)


def sanitise_label(label: str) -> str:
    """Make label safe for use inside a filename."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in label)


class ZEDObjectSubscriber(Node):
    """
    Subscribes to ZED object-detection and rectified RGB topics, synchronises
    them by header timestamp, annotates detections onto the image, and saves
    each annotated frame together with a companion JSON file.

    Sync strategy
    ─────────────
    Uses message_filters.ApproximateTimeSynchronizer so that every callback
    receives an (image, objects) pair whose header stamps are within SYNC_SLOP_SEC
    of each other.  This is the preferred ROS 2 approach and avoids the "stale
    image" problem of the original code.

    Saved artefacts  (per detected object in each frame)
    ────────────────────────────────────────────────────
    • <timestamp>_<label>_<dist>m_<conf>pct.jpg   – annotated JPEG crop + full frame
    • <timestamp>_frame.jpg                        – full annotated frame
    • <timestamp>_detections.json                  – structured metadata
    """

    def __init__(self):
        super().__init__("zed_object_subscriber")
        self.bridge = CvBridge()
        self.label_colour_map: dict[str, tuple] = {}
        self.frame_count = 0

        os.makedirs(SAVE_DIR, exist_ok=True)

        # ── Synchronised subscribers ──────────────────────────────────────────
        img_sub = message_filters.Subscriber(
            self,
            Image,
            "/zed/zed_node/rgb/color/rect/image",
        )
        obj_sub = message_filters.Subscriber(
            self,
            ObjectsStamped,
            "/zed/zed_node/obj_det/objects",
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, obj_sub],
            queue_size=QUEUE_DEPTH,
            slop=SYNC_SLOP_SEC,
        )
        self._sync.registerCallback(self._synced_callback)

        self.get_logger().info(
            f"ZEDObjectSubscriber ready – saving to {SAVE_DIR!r}  "
            f"(slop={SYNC_SLOP_SEC}s, min_conf={MIN_CONFIDENCE}%)"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _colour_for(self, label: str) -> tuple:
        """Assign a consistent colour to every unique label string."""
        if label not in self.label_colour_map:
            idx = len(self.label_colour_map) % len(PALETTE)
            self.label_colour_map[label] = PALETTE[idx]
        return self.label_colour_map[label]

    @staticmethod
    def _bbox_corners(obj):
        """
        Return (top_left, bottom_right) pixel tuples from the 2-D bounding box.
        ZED corner order: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left.
        """
        tl = obj.bounding_box_2d.corners[0].kp
        br = obj.bounding_box_2d.corners[2].kp
        return (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1]))

    # ── Main callback ─────────────────────────────────────────────────────────

    def _synced_callback(self, img_msg: Image, obj_msg: ObjectsStamped):
        """
        Called with a temporally aligned (image, objects) pair.
        Annotates the image and saves per-frame + per-object artefacts.
        """
        self.get_logger().info(
            f"Images and thime synced: img stamp={img_msg.header.stamp.sec}.{img_msg.header.stamp.nanosec:09d}, "
        )
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        annotated = cv_image.copy()

        # Stamp string for filenames  →  e.g. "20250312_143022_456"
        stamp_ns  = Time.from_msg(obj_msg.header.stamp).nanoseconds
        dt        = datetime.fromtimestamp(stamp_ns / 1e9)
        stamp_str = dt.strftime("%Y%m%d_%H%M%S_") + f"{dt.microsecond // 1000:03d}"

        valid_objects = [
            o for o in obj_msg.objects if o.confidence >= MIN_CONFIDENCE
        ]

        self.get_logger().info(
            f"Frame {self.frame_count:05d} | "
            f"{len(valid_objects)}/{len(obj_msg.objects)} objects "
            f"(above {MIN_CONFIDENCE}% confidence)"
        )

        detections_meta = []

        for obj in valid_objects:
            label  = obj.label  # e.g. "Person", "Car", …
            conf   = float(obj.confidence)
            dist   = euclidean_dist(obj.position)
            colour = self._colour_for(label)

            tl, br = self._bbox_corners(obj)

            # ── Draw bounding box ────────────────────────────────────────────
            cv2.rectangle(annotated, tl, br, colour, 2)

            # ── Overlay text: label / distance / confidence ──────────────────
            lines = [
                f"{label}",
                f"Dist : {dist:.2f} m",
                f"Conf : {conf:.1f} %",
            ]
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness  = 2
            line_h     = 20
            bg_pad     = 4

            # Measure widest line for background rectangle
            max_w = max(
                cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines
            )
            bg_tl = (tl[0], tl[1] - len(lines) * line_h - bg_pad * 2)
            bg_br = (tl[0] + max_w + bg_pad * 2, tl[1])
            cv2.rectangle(annotated, bg_tl, bg_br, colour, cv2.FILLED)

            for i, line in enumerate(lines):
                y = tl[1] - (len(lines) - i) * line_h
                cv2.putText(
                    annotated, line,
                    (tl[0] + bg_pad, y),
                    font, font_scale, (0, 0, 0), thickness,
                )

            # ── Save per-object crop ─────────────────────────────────────────
            h, w = cv_image.shape[:2]
            x1, y1 = max(0, tl[0]), max(0, tl[1])
            x2, y2 = min(w, br[0]), min(h, br[1])

            if x2 > x1 and y2 > y1:
                crop = cv_image[y1:y2, x1:x2]
                crop_name = (
                    f"{stamp_str}"
                    f"_{sanitise_label(label)}"
                    f"_{dist:.2f}m"
                    f"_{conf:.0f}pct"
                    f"_crop.jpg"
                )
                cv2.imwrite(os.path.join(SAVE_DIR, crop_name), crop)

            # ── Collect metadata ─────────────────────────────────────────────
            detections_meta.append(
                {
                    "label":      label,
                    "confidence": round(conf, 2),
                    "distance_m": round(dist, 4),
                    "position":   {
                        "x": round(float(obj.position[0]), 4),
                        "y": round(float(obj.position[1]), 4),
                        "z": round(float(obj.position[2]), 4),
                    },
                    "bbox_2d": {
                        "top_left":     list(tl),
                        "bottom_right": list(br),
                    },
                    "tracking_id": int(obj.tracking_id),
                    "tracking_state": int(obj.tracking_state),
                }
            )

        # ── Save full annotated frame ────────────────────────────────────────
        frame_path = os.path.join(SAVE_DIR, f"{stamp_str}_frame.jpg")
        cv2.imwrite(frame_path, annotated)

        # ── Save JSON metadata ───────────────────────────────────────────────
        meta = {
            "frame":      self.frame_count,
            "timestamp":  stamp_str,
            "stamp_ns":   stamp_ns,
            "n_objects":  len(valid_objects),
            "detections": detections_meta,
        }
        json_path = os.path.join(SAVE_DIR, f"{stamp_str}_detections.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.frame_count += 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ZEDObjectSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()