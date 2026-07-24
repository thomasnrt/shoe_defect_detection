#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
from torchvision import transforms
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from PIL import Image as PILImage
import os
import shutil
import sys
import traceback

# ==========================================
# MODÈLE 3 (CNN CLASSIQUE) 
# ==========================================
class ClassicCNN(nn.Module):
    def __init__(self):
        super(ClassicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# ==========================================

class DefectDetectionOfflineNode(Node):
    def __init__(self):
        super().__init__('node_dd_offline')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 1. SETTINGS ---
        self.target_count = 4 
        
        # Seuil de surface (0.1% de l'image doit être rouge pour compter comme un défaut)
        self.seuil_pourcentage_masque = 0.1 

        # --- PATHS ADAPTED TO YOUR WORKSPACE ---
        # INPUT: Where your robot saved the photos
        self.input_folder = os.path.expanduser('~/new_ws_compressed/new_ws/inspection_results')
        # OUTPUT: Where the AI will save the annotated results
        self.result_folder = os.path.expanduser('~/new_ws_compressed/new_ws/src/defect_detection/results')

        # Reset ONLY the output folder
        if os.path.exists(self.result_folder):
            shutil.rmtree(self.result_folder)
        os.makedirs(self.result_folder)

        # --- 2. MODEL INITIALIZATION ---
        
        # Modèle 2 : Détourage (U-Net)
        self.model_unet = smp.Unet(encoder_name="resnet34", classes=1, activation=None)
        
        # Modèle 3 : Classification (CNN Classique)
        self.model_classif = ClassicCNN()

        model_folder = os.path.expanduser('~/new_ws_compressed/new_ws/src/defect_detection/modeles/')
        
        unet_path = os.path.join(model_folder, 'model_BEST_val.pth')
        cnn_path = os.path.join(model_folder, 'cnn_classifier_BEST.pth')

        self.get_logger().info(f"Chargement des modèles sur : {self.device}...")
        
        # --- CHARGEMENT U-NET ---
        try:
            self.model_unet.load_state_dict(torch.load(unet_path, map_location=self.device))
            self.get_logger().info("✅ [SUCCES] U-Net chargé avec succès.")
        except Exception as e:
            self.get_logger().warn(f"⚠️ [ATTENTION] Erreur U-Net : {e}. Tentative strict=False...")
            try:
                self.model_unet.load_state_dict(torch.load(unet_path, map_location=self.device), strict=False)
                self.get_logger().info("✅ [SUCCES] U-Net chargé en mode alternatif.")
            except Exception as e2:
                self.get_logger().error("❌ [ERREUR FATALE] Impossible de charger le U-Net.")
                traceback.print_exc()
                sys.exit(1)

        # --- CHARGEMENT CNN ---
        try:
            self.model_classif.load_state_dict(torch.load(cnn_path, map_location=self.device))
            self.get_logger().info("✅ [SUCCES] CNN chargé avec succès.")
        except Exception as e:
            self.get_logger().error(f"❌ Erreur chargement CNN: {e}")
            sys.exit(1)

        self.model_unet.to(self.device).eval()
        self.model_classif.to(self.device).eval()
        self.get_logger().info("🚀 Modèles prêts. Lancement de l'analyse hors-ligne...")

        # --- 3. PRÉTRAITEMENTS ---
        self.preprocess_unet = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        self.preprocess_classif = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # --- 4. EXECUTE BATCH AND EXIT ---
        self.run_batch_inference()
        self.get_logger().info("👋 Traitement terminé. Fermeture du nœud.")
        sys.exit(0)

    def run_batch_inference(self):
        if not os.path.exists(self.input_folder):
            self.get_logger().error(f"Dossier introuvable : {self.input_folder}")
            return
            
        # Grab all image files from the robot's output folder
        all_photos = sorted([f for f in os.listdir(self.input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        photos = all_photos[:self.target_count]
        
        if not photos:
            self.get_logger().warn(f"Aucune image trouvée dans {self.input_folder}")
            return
            
        for photo_name in photos:
            self.get_logger().info(f"🧪 Analyse de {photo_name}...")
            path = os.path.join(self.input_folder, photo_name)
            cv_img = cv2.imread(path)
            if cv_img is None: continue
            
            img_pil = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            final_img = cv_img.copy()
            
            input_tensor = self.preprocess_unet(img_pil).unsqueeze(0).to(self.device)

            # ==========================================
            # ETAPE 1 : DÉTOURAGE (U-Net)
            # ==========================================
            with torch.no_grad():
                out_seg = self.model_unet(input_tensor)
                prob_mask = torch.sigmoid(out_seg).squeeze().cpu().numpy()
            
            h, w = cv_img.shape[:2]
            prob_mask_resized = cv2.resize(prob_mask, (w, h))
            
            binary_mask = (prob_mask_resized > 0.2).astype(np.uint8)
            
            total_pixels = h * w
            pixels_blancs = np.sum(binary_mask)
            pourcentage_blanc = (pixels_blancs / total_pixels) * 100.0
            
            self.get_logger().info(f"   -> Surface détectée : {pourcentage_blanc:.3f}%")

            if pourcentage_blanc >= self.seuil_pourcentage_masque:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
                color_layer = np.zeros_like(cv_img)
                
                # ==========================================
                # ETAPE 2 : CLASSIFICATION (CNN)
                # ==========================================
                for i in range(1, num_labels):
                    mask_i = (labels == i).astype(np.uint8) * 255
                    kernel = np.ones((15, 15), np.uint8)
                    mask_thick = cv2.dilate(mask_i, kernel, iterations=1)
                    
                    img_iso_cv = cv2.bitwise_and(cv_img, cv_img, mask=mask_i)
                    img_iso_pil = PILImage.fromarray(cv2.cvtColor(img_iso_cv, cv2.COLOR_BGR2RGB))
                    
                    input_class = self.preprocess_classif(img_iso_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        out_class = self.model_classif(input_class)
                        prob = torch.sigmoid(out_class).item()
                    
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    
                    if prob > 0.5:
                        defect_name = "SCRATCH"
                        color = (0, 0, 255) # Rouge
                    else:
                        defect_name = "HOLE"
                        color = (255, 0, 0) # Bleu
                    
                    color_layer[mask_thick == 255] = color
                    cv2.putText(final_img, defect_name, (cx - 30, cy - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                final_img = cv2.addWeighted(final_img, 0.7, color_layer, 0.3, 0)
            
            else:
                cv2.putText(final_img, "0 DEFECT", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            # ==========================================
            # ETAPE 3 : SAUVEGARDE ET AFFICHAGE
            # ==========================================
            res_path = os.path.join(self.result_folder, f"result_{photo_name}")
            cv2.imwrite(res_path, final_img)
            
            cv2.imshow("IA RESULTS", final_img)
            cv2.waitKey(1000)

        self.get_logger().info(f"🏁 Analysis Finished. Check folder: {self.result_folder}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = DefectDetectionOfflineNode()
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
