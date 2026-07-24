#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class CameraSimulator(Node):
    def __init__(self):
        super().__init__('simulateur_camera')
        # Topic sur lequel on publie les images
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Timer : s'exécute toutes les 10 secondes
        self.timer = self.create_timer(10.0, self.timer_callback)
        self.bridge = CvBridge()
        
        # Chemins vers tes deux images
        dossier = os.path.expanduser('~/shoe_ws/src/defect_detection/defect_detection/')
        self.images = [
            os.path.join(dossier, 'CHAUSSURE-TROUEE.jpg'),
            os.path.join(dossier, 'cuir-creux-defaut-600x450.jpg')
        ]
        self.index = 0

    def timer_callback(self):
        img_path = self.images[self.index]
        
        if os.path.exists(img_path):
            cv_img = cv2.imread(img_path)
            # Conversion OpenCV (BGR) vers message ROS
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            self.publisher_.publish(msg)
            self.get_logger().info(f'📸 Image publiée : {os.path.basename(img_path)}')
        else:
            self.get_logger().error(f'❌ Image introuvable : {img_path}')
        
        # On passe à l'image suivante pour le prochain tour (0 -> 1 -> 0 -> 1...)
        self.index = (self.index + 1) % len(self.images)

def main(args=None):
    rclpy.init(args=args)
    node = CameraSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
