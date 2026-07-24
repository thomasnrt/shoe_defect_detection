#!/usr/bin/env python3
"""
grasp_detector.py — Two-finger grasp on hole rim using YOLO segmentation mask.
MODIFIED FOR HACKATHON: Now uses a ROS 2 Trigger service to process only 1 frame at a time.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger  # <-- NEW: Import for the trigger service
from cv_bridge import CvBridge
from ultralytics import YOLO
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  
import message_filters

CAMERA_FRAME = "camera_color_optical_frame"
ROBOT_FRAME  = "base_link"


# ─────────────────────────────────────────────────────────────
# Geometry helpers (Unchanged)
# ─────────────────────────────────────────────────────────────
def mask_pca(mask_uint8):
    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) < 10:
        return None
    pts = np.column_stack((xs, ys)).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])
    minor = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    proj_major = centered @ major
    proj_minor = centered @ minor
    half_major = float(np.max(np.abs(proj_major)))
    half_minor = float(np.max(np.abs(proj_minor)))
    return mean, major, minor, half_major, half_minor


def rim_points_along_axis(mask_uint8, center, axis_unit, max_extent):
    h, w = mask_uint8.shape
    cx, cy = center
    left, right = None, None

    for t in np.linspace(0.0, max_extent + 5.0, int(max_extent) + 6):
        px = int(round(cx + t * axis_unit[0]))
        py = int(round(cy + t * axis_unit[1]))
        if not (0 <= px < w and 0 <= py < h):
            break
        if mask_uint8[py, px] == 0:
            right = (px, py)
            break

    for t in np.linspace(0.0, max_extent + 5.0, int(max_extent) + 6):
        px = int(round(cx - t * axis_unit[0]))
        py = int(round(cy - t * axis_unit[1]))
        if not (0 <= px < w and 0 <= py < h):
            break
        if mask_uint8[py, px] == 0:
            left = (px, py)
            break

    return left, right


def median_depth(depth_img, px, py, win=2):
    h, w = depth_img.shape[:2]
    if not (0 <= px < w and 0 <= py < h):
        return None
    patch = depth_img[max(py - win, 0):py + win + 1,
                      max(px - win, 0):px + win + 1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0 


def backproject(px, py, depth_m, fx, fy, cx, cy):
    X = (px - cx) * depth_m / fx
    Y = (py - cy) * depth_m / fy
    Z = depth_m
    return X, Y, Z


# ─────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────
class GraspDetector(Node):
    def __init__(self):
        super().__init__("grasp_detector")

        # Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("target_class", "hole")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("caminfo_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("show_window", True)

        model_path        = self.get_parameter("model_path").value
        self.target_class = self.get_parameter("target_class").value
        self.conf         = float(self.get_parameter("conf_threshold").value)
        color_topic       = self.get_parameter("color_topic").value
        depth_topic       = self.get_parameter("depth_topic").value
        caminfo_topic     = self.get_parameter("caminfo_topic").value
        self.show_window  = bool(self.get_parameter("show_window").value)

        if not model_path:
            raise ValueError("model_path parameter is required")

        self.bridge = CvBridge()
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.get_logger().info(f"Class names: {self.model.names}")
        self.get_logger().info(f"Targeting class: '{self.target_class}'")

        self.fx = self.fy = self.cx = self.cy = None

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pub_center = self.create_publisher(PointStamped, "/shoe_detector/grasp_point_base", 10)
        self.pub_left   = self.create_publisher(PointStamped, "/shoe_detector/grasp_left_base",  10)
        self.pub_right  = self.create_publisher(PointStamped, "/shoe_detector/grasp_right_base", 10)
        self.pub_annot  = self.create_publisher(Image, "/shoe_detector/image_annotated", 10)

        # Camera info subscriber
        self.create_subscription(CameraInfo, caminfo_topic, self.camera_info_cb, 10)

        # Synced color + depth
        c_sub = message_filters.Subscriber(self, Image, color_topic)
        d_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [c_sub, d_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_cb)

        # ─── NEW: Trigger Switch and Service ───
        self.process_next_frame = False
        self.srv = self.create_service(Trigger, '/shoe_detector/trigger_grasp', self.trigger_cb)

        self.get_logger().info("GraspDetector ready. Waiting for trigger service call...")

    # ─── NEW: Trigger Callback ───
    def trigger_cb(self, request, response):
        self.get_logger().info("Trigger received! Processing ONE frame...")
        self.process_next_frame = True
        response.success = True
        response.message = "YOLO triggered successfully."
        return response

    # ─────────────────────────────────────────────────────────
    def camera_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(
                f"Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}"
            )

    # ─────────────────────────────────────────────────────────
    def image_cb(self, color_msg: Image, depth_msg: Image):
        # ─── NEW: Gatekeeper ───
        if not self.process_next_frame:
            return  # Ignore all frames until the trigger is pulled!
        
        # Turn off the switch so we only process this single frame
        self.process_next_frame = False

        if self.fx is None:
            return

        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        H, W = color.shape[:2]
        vis = color.copy()

        # ── YOLO inference (with masks) ─────────────────────────
        results = self.model(color, conf=self.conf, verbose=False)[0]

        if results.masks is None or results.boxes is None:
            cv2.putText(vis, "No mask output from model", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self._publish_vis(vis, color_msg.header)
            return

        # Pick highest-confidence detection of target_class
        best_idx, best_conf = -1, 0.0
        for i, box in enumerate(results.boxes):
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            if cls_name == self.target_class and conf > best_conf:
                best_conf = conf
                best_idx = i

        if best_idx < 0:
            cv2.putText(vis, f"No '{self.target_class}' detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self._publish_vis(vis, color_msg.header)
            return

        # ── Extract mask, resize to color resolution if needed ──
        mask = results.masks.data[best_idx].cpu().numpy().astype(np.uint8) 
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask * 255).astype(np.uint8)

        # ── PCA → major/minor axes ──────────────────────────────
        pca = mask_pca(mask_u8)
        if pca is None:
            cv2.putText(vis, "Mask too small for PCA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self._publish_vis(vis, color_msg.header)
            return
        center_xy, major, minor, half_major, half_minor = pca
        cx_px, cy_px = float(center_xy[0]), float(center_xy[1])

        # ── Walk the minor axis to find the two rim points ──────
        left_px, right_px = rim_points_along_axis(mask_u8, (cx_px, cy_px), minor, half_minor)
        if left_px is None or right_px is None:
            cv2.putText(vis, "Could not find rim points", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self._publish_vis(vis, color_msg.header)
            return

        # ── Depth lookup at center + both rim points ────────────
        d_c = median_depth(depth, int(round(cx_px)), int(round(cy_px)))
        d_l = median_depth(depth, left_px[0],  left_px[1])
        d_r = median_depth(depth, right_px[0], right_px[1])

        # ── Robust depth fallbacks ──────────────────────────────
        if d_c is None:
            valid_rim = [v for v in (d_l, d_r) if v is not None]
            if valid_rim:
                d_c = float(np.mean(valid_rim))
                self.get_logger().info(f"Center depth invalid; using rim avg = {d_c:.3f} m")
            else:
                d_c = median_depth(depth, int(round(cx_px)), int(round(cy_px)), win=8)
                if d_c is None:
                    self.get_logger().warn(
                        "No valid depth found anywhere near hole; skipping frame"
                    )
                    cv2.putText(vis, "No valid depth (try moving closer)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self._publish_vis(vis, color_msg.header)
                    return
                self.get_logger().info(
                    f"Center depth invalid; wide-window fallback = {d_c:.3f} m"
                )

        if d_l is None: d_l = d_c
        if d_r is None: d_r = d_c

        # ── Back-project all 3 points ───────────────────────────
        cam_c = backproject(cx_px,        cy_px,        d_c, self.fx, self.fy, self.cx, self.cy)
        cam_l = backproject(left_px[0],   left_px[1],   d_l, self.fx, self.fy, self.cx, self.cy)
        cam_r = backproject(right_px[0],  right_px[1],  d_r, self.fx, self.fy, self.cx, self.cy)

        # ── Transform to base_link ──────────────────────────────
        try:
            base_c = self._to_base(cam_c, color_msg.header.stamp)
            base_l = self._to_base(cam_l, color_msg.header.stamp)
            base_r = self._to_base(cam_r, color_msg.header.stamp)
        except Exception as exc:
            self.get_logger().error(f"TF transform failed: {exc}")
            return

        # ── Publish ─────────────────────────────────────────────
        self.pub_center.publish(base_c)
        self.pub_left.publish(base_l)
        self.pub_right.publish(base_r)

        # ── Compute physical gripper opening ────────────────────
        dx = base_l.point.x - base_r.point.x
        dy = base_l.point.y - base_r.point.y
        dz = base_l.point.z - base_r.point.z
        opening_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        self.get_logger().info(
            f"[{self.target_class} @ {best_conf:.2f}] "
            f"center_base=({base_c.point.x:+.3f},{base_c.point.y:+.3f},{base_c.point.z:+.3f}) "
            f"opening={opening_m * 1000:.1f} mm"
        )

        # ── Visualize ───────────────────────────────────────────
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        def draw_axis(img, center, axis, length, color):
            p1 = (int(round(center[0] - length * axis[0])),
                  int(round(center[1] - length * axis[1])))
            p2 = (int(round(center[0] + length * axis[0])),
                  int(round(center[1] + length * axis[1])))
            cv2.line(img, p1, p2, color, 2)

        draw_axis(vis, (cx_px, cy_px), major, half_major, (255, 200, 0))   
        draw_axis(vis, (cx_px, cy_px), minor, half_minor, (0, 165, 255))   

        cv2.circle(vis, left_px,  8, (0, 0, 255), -1)
        cv2.circle(vis, right_px, 8, (0, 0, 255), -1)
        cv2.circle(vis, (int(round(cx_px)), int(round(cy_px))), 6, (255, 255, 255), -1)
        cv2.line(vis, left_px, right_px, (0, 255, 255), 2)

        cv2.putText(vis, f"opening: {opening_m * 1000:.0f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"depth_c: {d_c:.3f} m", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis,
                    f"base=({base_c.point.x:+.2f},{base_c.point.y:+.2f},{base_c.point.z:+.2f})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        self._publish_vis(vis, color_msg.header)

    # ─────────────────────────────────────────────────────────
    def _to_base(self, cam_xyz, stamp):
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = stamp
        pt.point.x = float(cam_xyz[0])
        pt.point.y = float(cam_xyz[1])
        pt.point.z = float(cam_xyz[2])
        # Use time=0 to force it to use the latest available transform, ignoring the 10-second lag
        pt.header.stamp = rclpy.time.Time().to_msg() 
        return self.tf_buffer.transform(pt, ROBOT_FRAME, timeout=Duration(seconds=1.0))

    def _publish_vis(self, vis, header):
        out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out.header = header
        self.pub_annot.publish(out)
        if self.show_window:
            cv2.imshow("Grasp Detector", vis)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = GraspDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
