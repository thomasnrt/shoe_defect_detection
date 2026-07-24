import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
import cv2


class ShoeLocalizationNode(Node):

    def __init__(self):
        super().__init__('shoe_localization_node')

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.pose_pub = self.create_publisher(PoseStamped, '/shoe_pose', 10)
        self.camera_frame = "camera_color_optical_frame"

        # Camera intrinsics
        self.fx = 554.38
        self.fy = 554.38
        self.cx = 320.0
        self.cy = 240.0

        self.get_logger().info('ShoeLocalizationNode started')
        self.z_buffer = []
        self.buffer_size = 5
        self.yaw_buffer = []

    def depth_callback(self, msg: Image):

        try:
            depth = np.frombuffer(msg.data, dtype=np.uint16)\
                .reshape(msg.height, msg.width)\
                .astype(np.float32) / 1000.0
        except Exception:
            return

        h, w = depth.shape

        # --- 1) Floor detection ---
        zone_floor = depth[int(h * 0.85):, :]
        valid_floor = zone_floor[(zone_floor > 0.1) & (zone_floor < 1.2)]

        if valid_floor.size == 0:
            self.publish_empty()
            return

        Z_floor = float(np.median(valid_floor))

        # --- 2) Mask ---
        MIN_DEPTH = 0.2
        MAX_DEPTH = 1.2
        HEIGHT_THRESHOLD = 0.03

        mask = np.zeros_like(depth, dtype=np.uint8)
        mask[
            (depth > MIN_DEPTH) &
            (depth < MAX_DEPTH) &
            (depth < (Z_floor - HEIGHT_THRESHOLD))
        ] = 255

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- 3) Contours ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.publish_empty()
            return

        valid_contours = []
        img_area = h * w
        margin = 5

        for c in contours:
            area = cv2.contourArea(c)

            if area < 1000 or area > (img_area * 0.35):
                continue

            x, y, w_rect, h_rect = cv2.boundingRect(c)
            if x < margin or y < margin or (x + w_rect) > (w - margin) or (y + h_rect) > (h - margin):
                continue

            valid_contours.append(c)

        if not valid_contours:
            self.publish_empty()
            return

        c = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        confidence = min(1.0, area / (h * w * 0.2))  # normalize roughly

        # --- 4) Geometry ---
        rect = cv2.minAreaRect(c)
        u, v = int(rect[0][0]), int(rect[0][1])
        angle_deg = rect[2]

        if rect[1][0] < rect[1][1]:
            angle_deg += 90
        # --- Yaw smoothing ---
        self.yaw_buffer.append(angle_deg)

        if len(self.yaw_buffer) > self.buffer_size:
                self.yaw_buffer.pop(0)

        angle_deg = sum(self.yaw_buffer) / len(self.yaw_buffer)

        # --- Depth refinement ---
        chunk = depth[max(0, v-5):min(h, v+5), max(0, u-5):min(w, u+5)]
        valid_chunk = chunk[(chunk > 0.1) & (chunk < 1.0)]

        if valid_chunk.size > 0:
            Z = float(np.median(valid_chunk))
        else:
            Z = float(depth[v, u]) if (v < h and u < w) else 0.5

        if Z <= 0.1 or Z > 0.8:
            self.publish_empty()
            return
        # --- Z smoothing ---
        self.z_buffer.append(Z)
        if len(self.z_buffer) > self.buffer_size:
                self.z_buffer.pop(0)

        Z = sum(self.z_buffer) / len(self.z_buffer)
                # --- 3D Projection ---
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # --- LOG (VERY IMPORTANT) ---
        self.get_logger().info(
                f'[CAMERA FRAME] '
                f'X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m | '
                f'u={u} v={v} | '
                f'Yaw={angle_deg:.1f}° | '
                f'Conf={confidence:.2f}')

        # --- PoseStamped ---
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.camera_frame

        pose_msg.pose.position.x = X
        pose_msg.pose.position.y = Y
        pose_msg.pose.position.z = Z

        yaw_rad = math.radians(angle_deg)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
        pose_msg.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.pose_pub.publish(pose_msg)

    def publish_empty(self):
        self.get_logger().info("[camera_frame] Shoe NOT found in current frame")


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ShoeLocalizationNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
