"""
Göğüs Hastalıkları KDS - FastAPI Backend
EfficientNet-B3 + 300x300 Model
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from typing import List, Dict
import sys
import os

# Add parent directory to path for importing model.py and config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import asıl model.py dosyasını kullan
from model import MultimodalChestXrayModel
import config

# ==================== CONFIG ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
IMG_SIZE = config.IMG_SIZE  # 300x300

DISEASES = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
    "Pneumonia",
    "Hernia"
]


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Göğüs Hastalıkları KDS",
    description="EfficientNet-B3 tabanlı Göğüs Röntgeni Analiz Sistemi",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prod'da sadece frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model = None

# ==================== IMAGE PREPROCESSING ====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Görüntüyü model için hazırla"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def encode_demographics(age: int, gender: str, view_position: str) -> torch.Tensor:
    """
    Demografik bilgileri encode et
    ⚠️ UYARI: EĞİTİMDEKİ İLE AYNI OLMALI! (dataset.py)
    """
    features = []
    
    # 1. Yaş özellikleri (4 adet)
    age_normalized = age / 100.0
    features.append(age_normalized)  # 1. age normalized
    features.append(np.log1p(age) / np.log1p(100))  # 2. log transform
    features.append(age_normalized ** 2)  # 3. squared term
    
    # 2. Yaş grupları (4 bins)
    features.append(1.0 if age < 18 else 0.0)    # 4. Çocuk
    features.append(1.0 if 18 <= age < 45 else 0.0)  # 5. Genç yetişkin
    features.append(1.0 if 45 <= age < 65 else 0.0)  # 6. Orta yaş
    features.append(1.0 if age >= 65 else 0.0)   # 7. Yaşlı
    
    # 3. Cinsiyet (one-hot, 2 adet)
    gender_upper = gender.upper()
    features.append(1.0 if gender_upper == 'MALE' or gender_upper == 'M' else 0.0)  # 8. Male
    features.append(1.0 if gender_upper == 'FEMALE' or gender_upper == 'F' else 0.0)  # 9. Female
    
    # 4. View position (one-hot, 3 adet)
    view = view_position.upper()
    features.append(1.0 if view == 'PA' else 0.0)  # 10. PA
    features.append(1.0 if view == 'AP' else 0.0)  # 11. AP
    features.append(1.0 if view not in ['PA', 'AP'] else 0.0)  # 12. Other (L, LL, etc.)
    
    return torch.tensor([features], dtype=torch.float32)


# ==================== ENDPOINTS ====================
@app.on_event("startup")
async def load_model():
    """Model yükleme"""
    global model
    
    print("🚀 Model yükleniyor...")
    
    try:
        # Model mimarisini config.py'den al (AYNI ŞEKİLDE!)
        model = MultimodalChestXrayModel(
            num_diseases=config.NUM_DISEASES,
            demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
            pretrained=False,  # Sadece mimari, ağırlıklar checkpoint'ten gelecek
            dropout=config.DROPOUT_RATE,
            use_attention=True  # Eğitimde True olduğu için
        )
        
        # Load checkpoint (PyTorch 2.6+ için weights_only=False)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(DEVICE)
        model.eval()
        
        val_auc = checkpoint.get('val_auc', 0.0)
        print(f"✅ Model yüklendi! (Val AUC: {val_auc:.4f})")
        print(f"✅ Device: {DEVICE}")
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """API bilgisi"""
    return {
        "service": "Göğüs Hastalıkları KDS",
        "model": "EfficientNet-B3 + 300x300",
        "version": "1.0.0",
        "status": "online" if model is not None else "model_not_loaded"
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "device": str(DEVICE),
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    view_position: str = Form(...)
):
    """
    Tahmin yap
    
    Args:
        file: Göğüs röntgeni görüntüsü (PNG, JPG, JPEG)
        age: Hasta yaşı (0-100)
        gender: Cinsiyet (Male/Female)
        view_position: Görüntü pozisyonu (PA/AP)
    
    Returns:
        JSON: Top 5 hastalık tahmini + detaylar
    """
    
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model henüz yüklenmedi"}
        )
    
    try:
        # Validate inputs
        if age < 0 or age > 100:
            return JSONResponse(
                status_code=400,
                content={"error": "Yaş 0-100 arasında olmalı"}
            )
        
        if gender.lower() not in ['male', 'female']:
            return JSONResponse(
                status_code=400,
                content={"error": "Cinsiyet 'Male' veya 'Female' olmalı"}
            )
        
        if view_position.upper() not in ['PA', 'AP']:
            return JSONResponse(
                status_code=400,
                content={"error": "Pozisyon 'PA' veya 'AP' olmalı"}
            )
        
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        image_tensor = preprocess_image(image_bytes).to(DEVICE)
        demographics_tensor = encode_demographics(age, gender, view_position).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor, demographics_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Format results
        results = []
        for i, disease in enumerate(DISEASES):
            results.append({
                "disease": disease,
                "probability": float(probabilities[i]),
                "percentage": float(probabilities[i] * 100),
                "risk_level": get_risk_level(probabilities[i])
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Top 5
        top5 = results[:5]
        
        # Response
        response = {
            "status": "success",
            "patient_info": {
                "age": age,
                "gender": gender,
                "view_position": view_position
            },
            "top5_predictions": top5,
            "all_predictions": results,
            "summary": generate_summary(top5)
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Tahmin hatası: {str(e)}"}
        )


def get_risk_level(probability: float) -> str:
    """Risk seviyesi belirle"""
    if probability >= 0.7:
        return "Yüksek"
    elif probability >= 0.4:
        return "Orta"
    else:
        return "Düşük"


def generate_summary(top5: List[Dict]) -> str:
    """Özet rapor oluştur"""
    if top5[0]['disease'] == 'No Finding' and top5[0]['probability'] > 0.5:
        return "Röntgen görüntüsünde belirgin bir patolojik bulguya rastlanmamıştır."
    
    high_risk = [d['disease'] for d in top5 if d['risk_level'] == 'Yüksek']
    
    if high_risk:
        return f"Yüksek risk: {', '.join(high_risk)}. Detaylı değerlendirme önerilir."
    else:
        return f"En olası bulgular: {', '.join([d['disease'] for d in top5[:3]])}."


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

