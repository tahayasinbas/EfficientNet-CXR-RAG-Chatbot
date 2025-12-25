# KDS - Göğüs Röntgeni Analiz ve RAG Chatbot Sistemi

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.4-green)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)](https://www.tensorflow.org/)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://reactjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](LICENSE.md)

Göğüs röntgeni görüntülerini yapay zeka ile analiz eden ve tıbbi doküman tabanlı RAG (Retrieval-Augmented Generation) chatbot sistemi içeren web uygulaması. NIH Chest X-ray Dataset üzerinde eğitilmiş EfficientNet-B3 modeli ile 15 farklı hastalığın tespitini yapar (Macro AUC: 0.82+).

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Model Performansı](#model-performansı)
- [Veri Seti](#veri-seti)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Teknoloji Stack](#teknoloji-stack)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [Kullanım](#kullanım)
- [Model Eğitimi](#model-eğitimi)
- [API Endpoints](#api-endpoints)
- [RAG Chatbot Sistemi](#rag-chatbot-sistemi)
- [Proje Yapısı](#proje-yapısı)
- [Geliştirme](#geliştirme)
- [Sorun Giderme](#sorun-giderme)
- [Performans İpuçları](#performans-ipuçları)
- [Tıbbi Sorumluluk Reddi](#tıbbi-sorumluluk-reddi)
- [Lisans](#lisans)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Kaynaklar](#kaynaklar)
- [Teşekkürler](#teşekkürler)

## 🎯 Özellikler

### X-Ray Görüntü Analizi
- **Göğüs röntgeni yükleme**: Çoklu format desteği (PNG, JPG, JPEG, DICOM)
- **AI Tabanlı Analiz**: EfficientNet-B3 tabanlı derin öğrenme modeli
- **Multi-label Sınıflandırma**: Bir görüntüde birden fazla hastalığın eş zamanlı tespiti
- **15 Hastalık Tespiti**: NIH Chest X-ray veri seti üzerinde eğitilmiş (Macro AUC: 0.82+)
  - **No Finding** (Normal), **Infiltration** (İnfiltrasyon)
  - **Effusion** (Efüzyon/Sıvı Birikimi), **Atelectasis** (Atelektazi)
  - **Nodule** (Nodül), **Mass** (Kitle)
  - **Pneumothorax** (Pnömotoraks), **Consolidation** (Konsolidasyon)
  - **Pleural Thickening** (Plevra Kalınlaşması), **Cardiomegaly** (Kardiyomegali)
  - **Emphysema** (Amfizem), **Edema** (Ödem)
  - **Fibrosis** (Fibrozis), **Pneumonia** (Pnömoni), **Hernia** (Herni)
- **Multimodal Yaklaşım**: Görüntü verisi + demografik bilgiler (yaş, cinsiyet, görüntü pozisyonu)
- **Risk Seviyesi Değerlendirmesi**: Low, Medium, High, Very High
- **Hasta Bilgileri**: Yaş, cinsiyet, pozisyon kaydı

### RAG Chatbot Sistemi
- **Hafızalı Konuşma**: Önceki mesajları hatırlayan chatbot
- **Tıbbi Doküman Tabanlı**: 44,349 tıbbi makale ile desteklenen yanıtlar
- **Hybrid Search**: BM25 (keyword) + Semantic search kombinasyonu
- **Google Gemini Integration**: gemini-2.5-flash modeli
- **X-ray Sonuç Yorumlama**: Model tahminlerini açıklama ve tedavi önerileri
- **Güvenlik Uyarıları**: Tıbbi sorumluluk reddi otomatik eklenir

### Kullanıcı Arayüzü
- **Modern React Frontend**: Responsive tasarım
- **Real-time Analiz**: Canlı sonuç görüntüleme
- **Görsel Raporlama**: Grafik ve chart'larla sonuç sunumu
- **Geçmiş Kayıtlar**: Tüm analizlerin saklanması ve görüntülenmesi
### Arayüz Fotoları 
<img width="946" height="548" alt="image" src="https://github.com/user-attachments/assets/6b63ad9a-deb1-4cb9-a1e3-656b2a14a88f" />

<img width="963" height="522" alt="image" src="https://github.com/user-attachments/assets/11672589-40f2-494a-90b7-e3993dcb04a6" />


## 📊 Model Performansı

### Test Seti Sonuçları (17,448 görüntü)

| Hastalık | AUC | Precision | Recall | F1-Score |
|----------|-----|-----------|--------|----------|
| **Emphysema** | 0.935 | 0.396 | 0.659 | 0.495 |
| **Cardiomegaly** | 0.910 | 0.318 | 0.579 | 0.411 |
| **Edema** | 0.886 | 0.140 | 0.455 | 0.214 |
| **Pneumothorax** | 0.884 | 0.340 | 0.522 | 0.412 |
| **Hernia** | 0.868 | 0.215 | 0.412 | 0.283 |
| **Effusion** | 0.856 | 0.818 | 0.004 | 0.009 |
| **Mass** | 0.834 | 0.347 | 0.328 | 0.337 |
| **Macro Average** | **0.820** | - | - | **0.177** |

### Eğitim Detayları

- **Veri Seti**: NIH Chest X-ray Dataset (112,120 görüntü)
- **Train/Val/Test Split**: 70%/15%/15% (Patient-level split)
- **Model**: EfficientNet-B3 (12M parametreler)
- **Görüntü Boyutu**: 300×300 piksel
- **Eğitim Platformu**: Kaggle (GPU: Tesla T4 x2)
- **Eğitim Süresi**: 5.3 saat (18 epoch)
- **Batch Size**: 36
- **Optimizasyon**: Adam optimizer + Cosine Annealing LR
- **Loss Function**: Focal Loss + Class Weights
- **Data Augmentation**: Medium (rotation, shift, scale, flip)
- **Test-Time Augmentation (TTA)**: 5x augmentation (+0.0025 AUC artışı)

### Güçlü Yönler

✅ **Yüksek Performans**:
- Emphysema: AUC 0.935
- Cardiomegaly: AUC 0.910
- Pneumothorax: AUC 0.884
- Edema: AUC 0.886

✅ **Data Leakage Önleme**:
- Patient-level split ile güvenilir sonuçlar
- Train-Val-Test overlap: 0

✅ **Multi-label Handling**:
- Bir görüntüde birden fazla hastalık tespiti
- Focal Loss + Class Weights ile dengesiz veri yönetimi

✅ **Multimodal Approach**:
- Görüntü + demografik bilgiler
- Attention mechanism ile modalite ağırlıklandırma

### İyileştirme Alanları

⚠️ **Düşük Performanslı Hastalıklar**:
- Infiltration: AUC 0.690 (veri belirsizliği)
- Pneumonia: AUC 0.761 (az örnek sayısı)
- Nodule: AUC 0.730 (küçük lezyon tespiti zor)

⚠️ **F1-Score Düşük**:
- Macro F1: 0.177 (precision-recall trade-off)
- Threshold optimization gerekli

⚠️ **Class Imbalance**:
- Hernia: Sadece 227 örnek (%0.2)
- No Finding: 60,361 örnek (%53.8)
- Class weights kısmen çözüm sağladı

## 📦 Veri Seti

### NIH Chest X-ray Dataset

**Kaynak**: [Kaggle - NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**Özellikler**:
- **Toplam Görüntü**: 112,120 frontal göğüs röntgeni
- **Hasta Sayısı**: 30,805 benzersiz hasta
- **Görüntü Formatı**: PNG (1024×1024 gri tonlama)
- **Multi-label**: Görüntülerin %18.5'inde birden fazla hastalık mevcut
- **Veri Dağılımı**:
  - No Finding: 60,361 (%53.8)
  - Infiltration: 19,894 (%17.7)
  - Effusion: 13,317 (%11.9)
  - Atelectasis: 11,559 (%10.3)
  - Diğer hastalıklar: %7.3

**Data Leakage Önleme**:
- Patient-level split stratejisi kullanıldı
- Aynı hastanın görüntüleri farklı setlere (train/val/test) karıştırılmadı
- Train-Val-Test overlap: 0 (doğrulandı)

## 🏗️ Sistem Mimarisi

### Model Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    Multimodal Model                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Image Encoder   │         │ Demographic      │          │
│  │  EfficientNet-B3 │         │ Encoder (MLP)    │          │
│  │  (1536 features) │         │  (64 features)   │          │
│  └────────┬─────────┘         └─────────┬────────┘          │
│           │                             │                   │
│           │    ┌───────────────────┐   │                    │
│           └────┤ Attention Fusion  ├───┘                    │
│                └─────────┬─────────┘                        │
│                          │                                  │
│                ┌─────────▼─────────┐                        │
│                │  Fusion MLP       │                        │
│                │  (512→256→128)    │                        │
│                └─────────┬─────────┘                        │
│                          │                                  │
│                ┌─────────▼─────────┐                        │
│                │ Output Layer (15) │                        │
│                │   Multi-label     │                        │
│                └───────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Teknik Detaylar

**Image Encoder (EfficientNet-B3)**:
- Pre-trained on ImageNet
- 1,536 dimensional feature vector
- Backbone freeze: İlk 2 epoch

**Demographic Encoder**:
- 12 demografik özellik:
  - Yaş özellikleri (4): normalized, log, squared, age_bins
  - Cinsiyet (2): Male/Female (one-hot)
  - Görüntü pozisyonu (3): PA/AP/Other (one-hot)
  - Yaş grupları (4): <18, 18-45, 45-65, 65+ (one-hot)
- 3-layer MLP (12→128→128→64)
- Batch normalization + Dropout

**Attention Fusion**:
- Görüntü ve demografik özellikleri için öğrenilebilir attention weights
- Modelin hangi modaliteye daha çok odaklanacağını dinamik olarak seçmesi

**Fusion Network**:
- 3-layer deep MLP (1600→512→256→128)
- Batch normalization + Dropout (0.55)
- ReLU aktivasyon

**Output Layer**:
- 15 hastalık için sigmoid aktivasyon
- Multi-label classification (BCEWithLogitsLoss)

## 🛠 Teknoloji Stack

### Backend
- **Framework**: Django 5.2.4 + Django REST Framework
- **Database**:
  - SQLite (Ana veritabanı)
  - PostgreSQL 16 + pgvector (RAG sistem için)
- **AI/ML**:
  - TensorFlow 2.18.0
  - OpenCV 4.10.0
  - EfficientNet-B3 (Görüntü sınıflandırma)
  - BAAI/bge-m3 (Embedding modeli)
- **RAG System**:
  - LangChain (Core, Community, Postgres)
  - Google Generative AI (Gemini)
  - BM25Retriever + Vector Search
  - LangSmith (Tracing)

### Frontend
- **Framework**: React 18
- **UI Library**: Material-UI
- **State Management**: React Hooks
- **HTTP Client**: Axios
- **Charts**: Recharts

### Infrastructure
- **Containerization**: Docker + Docker Compose (PostgreSQL)
- **Python Version**: 3.13
- **Node Version**: 18+

## 💻 Sistem Gereksinimleri

### Minimum
- **CPU**: 4 cores
- **RAM**: 8 GB
- **GPU**: CUDA destekli GPU (önerilen) veya CPU
- **Disk**: 10 GB boş alan

### Önerilen
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU (CUDA 11.8+)
- **Disk**: 20 GB SSD

## 📦 Kurulum

### 1. Repository'yi Klonlayın

```bash
git clone <repository-url>
cd kds_django_fantezi
```

### 2. Python Sanal Ortamı Oluşturun

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Python Bağımlılıklarını Kurun

```bash
pip install -r requirements.txt
```

**Not**: CUDA destekli GPU kullanıyorsanız, PyTorch CUDA versiyonunu kurun:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. PostgreSQL + pgvector'ü Başlatın

RAG chatbot sistemi için PostgreSQL gereklidir:

```bash
cd Rag_Chatbot
docker-compose up -d
cd ..
```

**Veritabanı Bilgileri:**
- Host: localhost
- Port: 5433
- Database: rag_db
- User: admin
- Password: sifre123

### 5. Frontend Bağımlılıklarını Kurun

```bash
cd frontend
npm install
cd ..
```

### 6. .env Dosyasını Yapılandırın

`.env.example` dosyasını `.env` olarak kopyalayın ve API keylerini ekleyin:

```bash
# Kök dizinde
cp .env.example .env

# Rag_Chatbot klasöründe
cd Rag_Chatbot
cp .env.example .env
cd ..

# Frontend klasöründe
cd frontend
cp .env.example .env
cd ..
```

Ardından `.env` dosyalarını düzenleyin:

**Kök dizin `.env`:**
```env
# Google Gemini API Key
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Django Secret Key (Değiştirin!)
SECRET_KEY=your-secret-django-key-here

# PostgreSQL Database (RAG Chatbot)
POSTGRES_USER=admin
POSTGRES_PASSWORD=sifre123
POSTGRES_DB=rag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
```

**Rag_Chatbot/.env:**
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Google Gemini API Key Alma:**
1. [Google AI Studio](https://aistudio.google.com/app/apikey) adresine gidin
2. API key oluşturun
3. Tüm `.env` dosyalarına ekleyin

### 7. Django Veritabanını Hazırlayın

```bash
python manage.py migrate
```

### 8. Uygulamayı Başlatın

**Backend:**
```bash
python manage.py runserver
# Backend: http://localhost:8000
```

**Frontend (Yeni terminal):**
```bash
cd frontend
npm start
# Frontend: http://localhost:3000
```

## ⚙️ Yapılandırma

### Django Settings (`kdsweb/settings.py`)

**RAG Chatbot Konfigürasyonu:**
```python
RAG_CHATBOT_CONFIG = {
    'CONNECTION_STRING': 'postgresql://admin:sifre123@localhost:5433/rag_db',
    'COLLECTION_NAME': 'makaleler_vectors',
    'MODEL_NAME': 'BAAI/bge-m3',
    'GEMINI_MODEL': 'gemini-2.5-flash',
    'GEMINI_TEMPERATURE': 0.4,
}
```

**CORS Ayarları:**
```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
```

### Frontend Konfigürasyonu (`frontend/src/services/api.js`)

**API Base URL:**
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
```

**Chat Timeout:**
```javascript
timeout: 120000 // 2 dakika (RAG ilk yükleme için)
```

## 🚀 Kullanım

### 1. Göğüs Röntgeni Analizi

1. Ana sayfaya gidin: http://localhost:3000
2. "Yeni Analiz" butonuna tıklayın
3. Röntgen görüntüsünü yükleyin
4. Hasta bilgilerini girin (yaş, cinsiyet, pozisyon)
5. "Analiz Et" butonuna tıklayın
6. Sonuçları görüntüleyin

### 2. Chatbot ile Etkileşim

**Analiz Sonrası:**
1. Analiz sonuçları sayfasında "Chatbot" sekmesine geçin
2. Sorunuzu yazın (örn: "Bu sonuçlar ne anlama geliyor?")
3. Chatbot, model tahminlerini ve tıbbi dokümanları kullanarak yanıt verir
4. Önceki konuşmalar hatırlanır (hafıza sistemi)

**Örnek Sorular:**
- "Bu tanı ne anlama geliyor?"
- "Tedavi protokolü nedir?"
- "Bu hastalığın belirtileri nelerdir?"
- "Benzer vakalar nasıl tedavi ediliyor?"

### 3. Geçmiş Kayıtlar

1. Ana sayfada "Geçmiş Analizler" bölümüne gidin
2. Önceki analizleri görüntüleyin
3. Detaylar için bir analiz seçin

## 🎓 Model Eğitimi

### Kaggle'da Eğitim

Model, Kaggle platformunda Tesla T4 GPU kullanılarak eğitilmiştir. Eğitim çıktıları `egitim-ciktilari/kaggle-ciktisi.txt` dosyasında mevcuttur.

### Eğitim Adımları

#### 1. Veri Hazırlama (`01_data_preparation.py`)
```bash
python 01_data_preparation.py
```
- 112,120 görüntünün analizi
- Patient-level stratified split (70/15/15)
- Multi-label distribution kontrolü
- CSV dosyaları oluşturma (train/val/test)

#### 2. Model Eğitimi (`04_train.py`)
```bash
python 04_train.py
```
- EfficientNet-B3 eğitimi
- 18 epoch, batch size 36
- Focal Loss + Class Weights
- Cosine Annealing LR Scheduler
- Mixed Precision Training
- Early stopping (patience=9)
- Checkpoint saving

#### 3. Model Değerlendirme (`05_evaluate.py`)
```bash
python 05_evaluate.py
```
- Test seti üzerinde metrik hesaplama
- ROC curves ve confusion matrices
- CSV export (predictions + metrics)

#### 4. Test-Time Augmentation (`05_evaluate_with_tta.py`)
```bash
python 05_evaluate_with_tta.py
```
- 5x augmentation ile tahmin
- Ensemble averaging
- +0.0025 AUC improvement

### Konfigürasyon

Tüm hyperparameter'lar `egitim-dosyalari/config.py` dosyasında tanımlıdır:

```python
IMG_SIZE = 300
BATCH_SIZE = 36
EPOCHS = 18
LEARNING_RATE = 0.0003
DROPOUT_RATE = 0.55
PRETRAINED_MODEL = "efficientnet_b3"
FREEZE_BACKBONE_EPOCHS = 2
USE_FOCAL_LOSS = True
USE_CLASS_WEIGHTS = True
AUGMENTATION_STRENGTH = 'medium'
```

### Eğitim Süreçleri

**Epoch İlerlemesi**:
```
Epoch 1/18 - Val AUC: 0.6870
Epoch 5/18 - Val AUC: 0.7917
Epoch 10/18 - Val AUC: 0.8170
Epoch 18/18 - Val AUC: 0.8231 ⭐ Best
```

**Süre Dağılımı**:
- Veri hazırlama: ~6 dakika
- Eğitim: 5.3 saat (18 epoch)
- Değerlendirme: ~3 dakika
- TTA: ~35 dakika
- **Toplam**: ~6.2 saat

## 📡 API Endpoints

### X-Ray Endpoints

**Tüm X-Ray'leri Listele**
```http
GET /api/xrays/
```

**Yeni X-Ray Yükle**
```http
POST /api/xrays/
Content-Type: multipart/form-data

{
  "image": <file>,
  "age": 45,
  "gender": "M",
  "position": "PA"
}
```

**X-Ray Analiz Et**
```http
POST /api/xrays/{id}/analyze/

Response:
{
  "id": 1,
  "is_analyzed": true,
  "analyzed_at": "2025-12-03T20:00:00Z",
  "diagnoses": [
    {
      "disease_name": "Pneumonia",
      "percentage": 89.5,
      "risk_level": "High"
    }
  ]
}
```

**X-Ray Detayları**
```http
GET /api/xrays/{id}/
```

### Chat Endpoints

**Mesaj Gönder**
```http
POST /api/chat/send/

{
  "session_id": 1,          // Opsiyonel
  "xray_id": 1,             // Opsiyonel
  "message": "Bu sonuçlar ne anlama geliyor?"
}

Response:
{
  "session_id": 1,
  "user_message": {
    "id": 1,
    "sender": "user",
    "content": "Bu sonuçlar ne anlama geliyor?",
    "created_at": "2025-12-03T20:00:00Z"
  },
  "ai_message": {
    "id": 2,
    "sender": "ai",
    "content": "Modelin analizine göre...",
    "rag_source": "RAG System",
    "created_at": "2025-12-03T20:00:05Z"
  },
  "success": true
}
```

**Chat Session Oluştur**
```http
POST /api/chat/sessions/

{
  "xray": 1  // Opsiyonel
}
```

**Tüm Chat Session'ları Listele**
```http
GET /api/chat/sessions/
```

## 🤖 RAG Chatbot Sistemi

### Mimari

```
┌─────────────┐
│   Kullanıcı │
└──────┬──────┘
       │ Soru
       ▼
┌──────────────────┐
│  Django Backend  │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────┐
│   RAG Chatbot Service        │
│  (chatbot/services.py)       │
└──┬───────────────────────┬───┘
   │                       │
   │ 1. Retrieve Docs      │ 2. Get History
   ▼                       ▼
┌──────────────┐     ┌──────────────┐
│ Hybrid       │     │  SQLite DB   │
│ Retriever    │     │  (History)   │
└──┬────────┬──┘     └──────────────┘
   │        │
   │        │ 3. Query
   ▼        ▼
┌────────┐ ┌──────────┐
│ BM25   │ │ Semantic │
│ Search │ │ (Vector) │
└───┬────┘ └────┬─────┘
    │           │
    └─────┬─────┘
          │
          ▼
    ┌──────────────┐
    │ PostgreSQL + │
    │  pgvector    │
    │ (44,349 docs)│
    └──────────────┘
          │
          │ 4. Retrieved Docs
          ▼
    ┌──────────────┐
    │ Build Prompt │
    │ + Context    │
    └──────┬───────┘
           │
           │ 5. Generate
           ▼
    ┌──────────────┐
    │ Google Gemini│
    │ 2.5-flash    │
    └──────┬───────┘
           │
           │ 6. Response
           ▼
    ┌──────────────┐
    │  Kullanıcı   │
    └──────────────┘
```

### Özellikler

**1. Hybrid Search (BM25 + Semantic)**
- **BM25**: Keyword tabanlı arama (İstatistiksel)
- **Semantic**: Anlamsal benzerlik araması (Vector)
- **Fusion**: Reciprocal Rank Fusion ile sonuçları birleştirir

**2. Memory System**
- Son 10 mesaj veritabanından yüklenir
- LangChain message formatı (SystemMessage, HumanMessage, AIMessage)
- Her konuşma bağımsız session'da saklanır

**3. Context Integration**
- X-ray model tahminleri otomatik eklenir
- Hasta bilgileri (yaş, cinsiyet, pozisyon)
- Risk seviyesi değerlendirmeleri

**4. Safety Features**
- Tıbbi sorumluluk reddi otomatik eklenir
- "Kesin teşhis" ifadeleri engellenir
- Profesyonel ton ve dil kuralları

### İlk Kullanımda Yükleme Süresi

**Beklenen Süreler:**
- Embedding modeli yükleme: ~30 saniye
- 44,349 doküman yükleme: ~45 saniye
- **Toplam ilk yükleme**: ~90-120 saniye

**Sonraki kullanımlar**: 2-5 saniye (cache'den)

### Performans Optimizasyonu

```python
# chatbot/services.py

# 1. Lazy initialization - Sadece ilk kullanımda yüklenir
if not self._initialized:
    self._initialize()

# 2. Singleton pattern - Tek instance
_chatbot_service = None

# 3. Cache - BM25 retriever bellekte tutulur
self.retriever = HybridRetriever(...)
```

## 📁 Proje Yapısı

```
kds_project/
│
├── .env                          # Environment variables (GİZLİ - Git'e eklenmez)
├── .env.example                  # Environment variables şablonu
├── .gitignore                    # Git ignore dosyası
├── requirements.txt              # Python dependencies
├── manage.py                     # Django management script
├── db.sqlite3                    # SQLite database (Git'e eklenmez)
├── README.md                     # Bu dosya
│
├── kdsweb/                       # Django project settings
│   ├── settings.py               # Ana ayarlar
│   ├── urls.py                   # Root URL configuration
│   └── wsgi.py                   # WSGI configuration
│
├── xray/                         # X-ray analiz uygulaması
│   ├── models.py                 # XRay ve Diagnosis modelleri
│   ├── views.py                  # API views
│   ├── serializers.py            # DRF serializers
│   ├── urls.py                   # URL routing
│   └── ai_analyzer.py            # AI model entegrasyonu
│
├── chatbot/                      # RAG chatbot uygulaması
│   ├── models.py                 # ChatSession ve ChatMessage
│   ├── views.py                  # Chat API views
│   ├── serializers.py            # DRF serializers
│   ├── urls.py                   # URL routing
│   └── services.py               # RAG chatbot servisi ⭐
│
├── model/                        # ML model dosyaları
│   └── model.weights.h5          # EfficientNet-B3 weights (294MB)
│
├── media/                        # Yüklenen dosyalar
│   └── xrays/                    # X-ray görüntüleri
│
├── egitim-dosyalari/            # Model Eğitim Scriptleri ⭐
│   ├── 01_data_preparation.py   # Veri hazırlama ve split
│   ├── 04_train.py              # Model eğitimi (Kaggle)
│   ├── 05_evaluate.py           # Model değerlendirme
│   ├── 05_evaluate_with_tta.py  # TTA değerlendirme
│   ├── config.py                # Hyperparameter konfigürasyonu
│   ├── dataset.py               # Dataset loader
│   ├── model.py                 # Model mimarisi (Multimodal)
│   └── run_kaggle.py            # Kaggle runner script
│
├── egitim-ciktilari/            # Eğitim Çıktıları ve Metrikler ⭐
│   ├── kaggle-ciktisi.txt       # Detaylı eğitim logları (18 epoch)
│   ├── confusion_matrices.png   # Confusion matrices (15 hastalık)
│   ├── roc_curves.png           # ROC eğrileri
│   ├── test_metrics.csv         # Test metrikleri (AUC, F1, etc.)
│   ├── test_metrics_tta.csv     # TTA metrikleri
│   ├── test_predictions.csv     # Test tahminleri (17,448 görüntü)
│   └── best_model.pth           # Model checkpoint (PyTorch)
│
├── frontend/                     # React Frontend
│   ├── public/                   # Static files
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── services/             # API services
│   │   │   └── api.js            # Axios configuration
│   │   ├── constants/            # Constants
│   │   └── App.js                # Main app component
│   ├── package.json              # Node dependencies
│   └── README.md                 # Frontend README
│
└── Rag_Chatbot/                  # RAG Sistem Dosyaları
    ├── docker-compose.yml        # PostgreSQL + pgvector
    ├── database.ipynb            # Veritabanı kurulum notebook
    ├── hafizaliRag.ipynb         # RAG sistem test notebook
    ├── requirements.txt          # RAG dependencies
    ├── .env                      # Gemini API key (GİZLİ - Git'e eklenmez)
    └── .env.example              # API key şablonu
```

## 🔧 Geliştirme

### Backend Geliştirme

**Yeni Model Ekleme:**
```bash
python manage.py makemigrations
python manage.py migrate
```

**Django Shell:**
```bash
python manage.py shell
```

**Testler:**
```bash
python manage.py test
```

### Frontend Geliştirme

**Development Server:**
```bash
cd frontend
npm start
```

**Build for Production:**
```bash
cd frontend
npm run build
```

**Linting:**
```bash
cd frontend
npm run lint
```

### RAG Sistem Geliştirme

**Jupyter Notebook ile Test:**
```bash
cd Rag_Chatbot
jupyter notebook hafizaliRag.ipynb
```

**Yeni Doküman Ekleme:**
1. Dokümanları PostgreSQL'e yükleyin
2. Embedding'leri oluşturun
3. Chatbot otomatik olarak yeni dokümanları kullanır

## 🐛 Sorun Giderme

### 1. PostgreSQL Bağlantı Hatası

**Hata:**
```
psycopg2.OperationalError: could not connect to server
```

**Çözüm:**
```bash
cd Rag_Chatbot
docker-compose ps  # Container durumunu kontrol et
docker-compose up -d  # Container'ı başlat
```

### 2. Gemini API Key Hatası

**Hata:**
```
Your default credentials were not found
```

**Çözüm:**
1. `.env` dosyasının kök dizinde olduğundan emin olun
2. `GOOGLE_API_KEY` değişkeninin doğru olduğunu kontrol edin
3. Django sunucusunu yeniden başlatın

### 3. RAG İlk Yükleme Timeout

**Hata:**
```
Broken pipe / Connection timeout
```

**Çözüm:**
- Frontend timeout'u artırıldı (120 saniye)
- İlk kullanımda sabırlı olun (~2 dakika)
- Sonraki istekler çok hızlı olacak

### 4. CUDA / GPU Hatası

**Hata:**
```
CUDA out of memory
```

**Çözüm:**
```python
# kdsweb/settings.py veya environment variable ile

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU kullan
```

### 5. Model Dosyası Bulunamadı

**Hata:**
```
FileNotFoundError: model.weights.h5
```

**Çözüm:**
- Model dosyasının `model/model.weights.h5` konumunda olduğundan emin olun
- Model dosyasını indirin ve doğru konuma yerleştirin

### 6. Frontend CORS Hatası

**Hata:**
```
Access to XMLHttpRequest blocked by CORS policy
```

**Çözüm:**
```python
# kdsweb/settings.py

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
```

## 📊 Performans İpuçları

### Backend Optimizasyonu
1. **Database Indexing**: X-ray ve Chat sorguları için index oluşturun
2. **Caching**: Redis ile API response cache'i
3. **Async Processing**: Celery ile arka plan görevleri

### RAG Sistem Optimizasyonu
1. **Doküman Limitı**: `k=50000` yerine `k=10000` kullanın (daha hızlı)
2. **BM25 Weight**: `bm25_weight=0.4` optimal değer
3. **Embedding Cache**: Model weights'i GPU memory'de tutun

### Frontend Optimizasyonu
1. **Code Splitting**: React lazy loading kullanın
2. **Image Optimization**: Yüklenen görüntüleri sıkıştırın
3. **Debouncing**: Chat input için debounce ekleyin

## ⚠️ Tıbbi Sorumluluk Reddi

**Dikkat**: Bu sistem eğitim ve araştırma amaçlıdır. Klinik karar verme için kullanılmamalıdır. Tüm tanılar lisanslı radyologlar tarafından onaylanmalıdır.

**Performans Uyarısı**: Model performansı kullanılan görüntü kalitesine, çekim tekniğine ve hasta popülasyonuna bağlı olarak değişebilir. External validation yapılmamıştır.

## 📝 Lisans

Bu proje eğitim amaçlıdır. Ticari kullanım için lütfen lisans alın.

## 👥 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📚 Kaynaklar

### Dataset
- Wang X, Peng Y, Lu L, et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017.
- NIH Clinical Center: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

### Model Architecture
- Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.
- https://arxiv.org/abs/1905.11946

### Related Work
- Rajpurkar P, et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv 2017.
- Irvin J, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels. AAAI 2019.

## 🙏 Teşekkürler

- **NIH Clinical Center**: Dataset sağladığı için
- **Kaggle**: GPU kaynakları için
- **Google Gemini**: LLM entegrasyonu için
- **LangChain**: RAG framework için
- **PyTorch, TensorFlow ve timm**: Kütüphaneleri için
- **Django & React**: Framework'ler için
- **Tüm açık kaynak topluluğu**

---

**Son Güncelleme**: 25 Aralık 2025
**Versiyon**: 1.0.0
**Geliştirici**: KDS Ekibi

