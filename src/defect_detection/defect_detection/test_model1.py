#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from torchvision import transforms
import cv2
import os

class DefectDetectionNode(Node):
    def __init__(self):
        super().__init__('node_def_detect')
        self.bridge = CvBridge()
        self.device = torch.device('cpu') 
        
        # On pointe vers le fichier .pth d'origine (l'archive ZIP)
        model_path = os.path.expanduser('~/shoe_ws/src/defect_detection/modeles/resnet18_shoe_defect.pth')

        # 1. Chargement du modèle
        try:
            if os.path.exists(model_path):
                # On utilise weights_only=False car le fichier contient la structure complète du ResNet-18
                self.model1 = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model1.eval()
                self.get_logger().info("✅ Model 1 (ResNet-18) loaded successfully from archive!")
            else:
                self.get_logger().error(f"❌ Model file not found at: {model_path}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")

        # 2. Preprocessing standard pour ResNet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.subscription = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
            10)

    def image_callback(self, msg):
        self.get_logger().info("📸 Image received! Inferencing...")
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Inférence
        input_tensor = self.preprocess(cv_img).unsqueeze(0)
        with torch.no_grad():
            output = self.model1(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()

        if prediction == 1:
            self.get_logger().warn(f"🚨 RESULT: DEFECT DETECTED (Label {prediction})")
        else:
            self.get_logger().info(f"✨ RESULT: SHOE HEALTHY (Label {prediction})")

def main(args=None):
    rclpy.init(args=args)
    node = DefectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
