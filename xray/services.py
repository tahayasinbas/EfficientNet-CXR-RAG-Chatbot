"""
X-Ray Prediction Service
Integrates with the PyTorch model for disease prediction
"""

import sys
import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Add model directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
model_dir = str(BASE_DIR / 'model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

import model as model_module
import config
from model import MultimodalChestXrayModel


class XRayPredictionService:
    """Service for predicting diseases from X-Ray images"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD
            )
        ])
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = BASE_DIR / 'model' / 'best_model.pth'

            # Create model architecture
            self.model = MultimodalChestXrayModel(
                num_diseases=config.NUM_DISEASES,
                demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
                pretrained=False,
                dropout=config.DROPOUT_RATE,
                use_attention=config.USE_ATTENTION
            )

            # Load checkpoint
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"[OK] Model loaded successfully on {self.device}")
            val_auc = checkpoint.get('val_auc', 0.0)
            print(f"   Val AUC: {val_auc:.4f}")

        except Exception as e:
            print(f"[ERROR] Model loading error: {e}")
            raise

    def encode_demographics(self, age, gender, position):
        """
        Encode demographic information
        Must match training encoding from dataset.py
        """
        features = []

        # 1. Age features (3)
        age_normalized = age / 100.0
        features.append(age_normalized)
        features.append(np.log1p(age) / np.log1p(100))
        features.append(age_normalized ** 2)

        # 2. Age bins (4)
        features.append(1.0 if age < 18 else 0.0)
        features.append(1.0 if 18 <= age < 45 else 0.0)
        features.append(1.0 if 45 <= age < 65 else 0.0)
        features.append(1.0 if age >= 65 else 0.0)

        # 3. Gender (2)
        features.append(1.0 if gender.upper() in ['M', 'MALE'] else 0.0)
        features.append(1.0 if gender.upper() in ['F', 'FEMALE'] else 0.0)

        # 4. View position (3)
        view = position.upper()
        features.append(1.0 if view == 'PA' else 0.0)
        features.append(1.0 if view == 'AP' else 0.0)
        features.append(1.0 if view not in ['PA', 'AP'] else 0.0)

        return torch.tensor([features], dtype=torch.float32)

    def predict(self, image_path, age, gender, position):
        """
        Predict diseases from X-Ray image

        Args:
            image_path: Path to X-Ray image
            age: Patient age (0-120)
            gender: Patient gender ('M' or 'F')
            position: View position ('PA' or 'AP')

        Returns:
            List of predictions with disease name, confidence, and risk level
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Encode demographics
            demographics_tensor = self.encode_demographics(age, gender, position).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor, demographics_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

            # Format results
            results = []
            for i, disease in enumerate(config.DISEASE_CLASSES):
                confidence = float(probabilities[i])
                results.append({
                    'disease_name': disease,
                    'confidence': confidence,
                    'percentage': confidence * 100,
                    'risk_level': self._get_risk_level(confidence)
                })

            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)

            return results

        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            raise

    def _get_risk_level(self, confidence):
        """Determine risk level based on confidence"""
        if confidence >= 0.7:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        else:
            return 'low'


# Singleton instance
_prediction_service = None


def get_prediction_service():
    """Get or create prediction service singleton"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = XRayPredictionService()
    return _prediction_service
