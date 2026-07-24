import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class ImageTester(Node):
    def __init__(self):
        super().__init__('image_tester')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # Chemins vers deux images de test sur ton bureau
        self.img_paths = [
            os.path.expanduser('~/shoe_ws/src/defect_detection/defect_detection/cuir-creux-defaut-600x450.jpg'),
            os.path.expanduser('~/shoe_ws/src/defect_detection/defect_detection/image.jpeg')
        ]

    def run_test(self):
        for path in self.img_paths:
            img = cv2.imread(path)
            if img is not None:
                self.get_logger().info(f"Publishing: {path}")
                self.publisher.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
                time.sleep(20) # Intervalle de 20 secondes
            else:
                self.get_logger().error(f"Image not found at {path}")

def main():
    rclpy.init()
    tester = ImageTester()
    tester.run_test()
    rclpy.shutdown()
