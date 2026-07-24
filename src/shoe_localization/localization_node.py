import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
import json

class ShoeLocalizationNode(Node):
    def __init__(self):
        super().__init__('shoe_localization_node')

        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10
        )
        self.pose_pub = self.create_publisher(String, '/shoe_pose', 10)

        # Intrinsèques Caméra
        self.fx = 554.38
        self.fy = 554.38
        self.cx = 320.0
        self.cy = 240.0

        self.get_logger().info('ShoeLocalizationNode started (Border Filter DISABLED)')

    def depth_callback(self, msg: Image):
        try:
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        except Exception:
            return

        
        h, w = depth.shape

        # 1) Sol
        zone_floor = depth[int(h * 0.85):, :] 
        # On exige que le sol soit entre 10cm et 1.2m. Si on pointe vers l'horizon (mur), on rejette le sol.
        valid_floor = zone_floor[(zone_floor > 0.1) & (zone_floor < 1.2)]
        if valid_floor.size == 0: 
            self.publish_empty()
            return
        Z_floor = float(np.median(valid_floor))

        # 2) Masque
        MIN_DEPTH = 0.2
        MAX_DEPTH = 1.2  # Pas besoin d'aller chercher au-delà de 1.2m
        HEIGHT_THRESHOLD = 0.03  # Doit dépasser d'au moins 3cm du sol
        mask = np.zeros_like(depth, dtype=np.uint8)
        mask[(depth > MIN_DEPTH) & (depth < MAX_DEPTH) & (depth < (Z_floor - HEIGHT_THRESHOLD))] = 255
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 3) Filtrage intelligent des Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.publish_empty()
            return

        valid_contours = []
        img_area = h * w
        margin = 5

        for c in contours:
            area = cv2.contourArea(c)
            # Sécurité 1 & 2 : Taille
            if area < 1000 or area > (img_area * 0.35):
                continue
            
            # Sécurité 3 : Bordures
            x, y, w_rect, h_rect = cv2.boundingRect(c)
            if x < margin or y < margin or (x + w_rect) > (w - margin) or (y + h_rect) > (h - margin):
                continue
                
            valid_contours.append(c)

        if not valid_contours:
            self.publish_empty()
            return

        # On prend le plus gros contour PARMI CEUX VALIDES
        c = max(valid_contours, key=cv2.contourArea)

        # 4) Calculs
        rect = cv2.minAreaRect(c)
        u, v = int(rect[0][0]), int(rect[0][1])
        angle_deg = rect[2]
        
        # Normalisation de l'angle pour OpenCV (dépend des versions, mais ceci stabilise souvent)
        if rect[1][0] < rect[1][1]: 
            angle_deg += 90

        # Profondeur précise
        chunk = depth[max(0,v-5):min(h,v+5), max(0,u-5):min(w,u+5)]
        valid_chunk = chunk[chunk > 0]
        if valid_chunk.size > 0:
            Z = float(np.median(valid_chunk))
        else:
            if v < h and u < w: Z = float(depth[v, u])
            else: Z = 0.5

        if Z <= 0.1 or Z > 0.8: 
            self.publish_empty()
            return

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # --- LOG avec YAW affiché ---
        # --- LOGGING ---
        # This will show you BOTH the physical meters (X/Y) and the pixels (u/v)
        log_text = (
            f'[camera_frame] '
            f'Physical: X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m | '
            f'Pixels: u={u} v={v} | '
            f'Yaw={angle_deg:.1f}°'
        )
        self.get_logger().info(log_text)

        # --- JSON DATA ---
        shoe_info = {
            "present": True,
            # FOR YOUR FRIEND: Physical real-world coordinates in meters
            "x": float(X), 
            "y": float(Y), 
            "z": float(Z),
            
            # FOR YOUR UI: Screen pixels for drawing the green box
            "u": int(u), 
            "v": int(v),
            
            # Extra data
            "depth": float(Z),
            "yaw_deg": float(angle_deg)
        }
        
        msg = String()
        msg.data = json.dumps(shoe_info)
        self.pose_pub.publish(msg)

    def publish_empty(self):
        msg = String()
        msg.data = json.dumps({"present": False})
        self.pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ShoeLocalizationNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
