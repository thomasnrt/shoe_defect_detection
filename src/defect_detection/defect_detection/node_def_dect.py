import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os

# Définition du chemin vers ton dossier de modèles
model_folder = os.path.expanduser('~/shoe_ws/src/defect_detection/modeles/')

# Chargement du modèle 1 (le fameux dossier/fichier .pth)
try:
    # On pointe directement sur le nom du fichier/dossier
    self.model1 = torch.load(os.path.join(model_folder, 'resnet18_shoe_defect.pth'), map_location=self.device)
    self.model1.eval()
    self.get_logger().info("✅ Model 1 loaded successfully")
except Exception as e:
    self.get_logger().error(f"❌ Error loading Model 1: {e}")

# Fais la même chose pour model2 et model3 avec leurs noms respectifs


class DefectDetectionNode(Node):
    def __init__(self):
        super().__init__('node_def_detect')
        self.bridge = CvBridge()
        
        # 1. Configuration du matériel et des chemins [cite: 234, 235]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_folder = os.path.expanduser('~/ur3e_ws/src/defect_detection/models/')
        
        # 2. Chargement des 3 modèles pré-entraînés [cite: 217, 230]
        try:
            # Stage 1: Classification binaire (ResNet-18) [cite: 234]
            self.model1 = torch.load(os.path.join(model_folder, 'model1_resnet.pth'), map_location=self.device)
            # Stage 2: Segmentation (U-Net) [cite: 221]
            self.model2 = torch.load(os.path.join(model_folder, 'model2_unet.pth'), map_location=self.device)
            # Stage 3: Type de défaut (ResNet-18 spécialisé) [cite: 294]
            self.model3 = torch.load(os.path.join(model_folder, 'model3_resnet.pth'), map_location=self.device)
            
            self.model1.eval()
            self.model2.eval()
            self.model3.eval()
            self.get_logger().info(" All 3 models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {str(e)}")

        # 3. Prétraitement des images (Transformations)
        self.transform_cls = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 4. Subscriber & Publisher
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/inspection/visual_result', 10)

    def image_callback(self, msg):
        """Pipeline de traitement déclenché à chaque réception d'image."""
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # --- STAGE 1 : Binary Classification (Gatekeeper) [cite: 237, 238] ---
        if self.predict_presence(cv_img) == 1:
            self.get_logger().warn("Defect detected. Analyzing...")
            
            # --- STAGE 2 : Segmentation ---
            global_mask = self.predict_segmentation(cv_img)
            
            # --- TRAITEMENT : Séparation des défauts (Ton code) ---
            individual_masks = self.separer_les_defauts(global_mask)
            
            final_visual = cv_img.copy()
            
            # Analyse de chaque défaut isolé
            for mask in individual_masks:
                # --- STAGE 3 : Classification du type (Hole vs Scratch) [cite: 294] ---
                label = self.predict_type(cv_img, mask)
                
                # --- VISUALISATION : Fusion couleur (Ton code) [cite: 316, 317] ---
                final_visual = self.generer_visualisation_couleur(final_visual, mask, label)
            
            # Publication du résultat final
            self.publisher.publish(self.bridge.cv2_to_imgmsg(final_visual, "bgr8"))
        else:
            self.get_logger().info("Shoe is healthy.")

    # --- MÉTHODES D'INFÉRENCE ---
    def predict_presence(self, img):
        """Détermine si un défaut est présent (Stage 1)."""
        tensor = self.transform_cls(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model1(tensor)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def predict_segmentation(self, img):
        """Génère le masque binaire global (Stage 2)."""
        input_img = cv2.resize(img, (512, 512))
        tensor = transforms.ToTensor()(input_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model2(tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        return cv2.resize(binary_mask, (img.shape[1], img.shape[0]))

    def predict_type(self, original_img, individual_mask):
        """Classifie un défaut spécifique (Stage 3)."""
        # On utilise le masque pour isoler la zone sur l'image d'origine
        res = cv2.bitwise_and(original_img, original_img, mask=individual_mask)
        tensor = self.transform_cls(res).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model3(tensor)
            _, predicted = torch.max(output, 1)
        return "hole" if predicted.item() == 0 else "scratch"

    # --- TES FONCTIONS INTÉGRÉES ---
    def separer_les_defauts(self, mask_binaire):
        """Sépare un masque global en une liste de masques individuels."""
        mask_uint8 = mask_binaire.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        liste_images_defauts = []
        for i in range(1, num_labels):
            image_solo = np.zeros_like(mask_uint8)
            image_solo[labels == i] = 255
            liste_images_defauts.append(image_solo)
        return liste_images_defauts

    def generer_visualisation_couleur(self, image_base, mask, label):
        """Superpose une couleur sur l'image : Bleu (hole) ou Rouge (scratch)."""
        img_rgb = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
        color_layer = np.zeros_like(img_rgb)
        
        # Définition des couleurs [cite: 316, 317]
        couleur = (0, 0, 255) if label.lower() == "hole" else (255, 0, 0)
        
        color_layer[mask > 127] = couleur
        output_rgb = cv2.addWeighted(img_rgb, 0.7, color_layer, 0.3, 0)
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

def main(args=None):
    rclpy.init(args=args)
    node = DefectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
