# KDS - Göğüs Röntgeni Analiz ve RAG Chatbot Sistemi

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.4-green)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)](https://www.tensorflow.org/)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://reactjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](LICENSE.md)

Göğüs röntgeni görüntülerini yapay zeka ile analiz eden ve tıbbi doküman tabanlı RAG (Retrieval-Augmented Generation) chatbot sistemi içeren web uygulaması. NIH Chest X-ray Dataset üzerinde eğitilmiş EfficientNet-B3 modeli ile 15 farklı hastalığın tespitini yapar (Macro AUC: 0.8342, TTA ile, 15 sınıf).

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
- **15 Hastalık Tespiti**: NIH Chest X-ray veri seti üzerinde eğitilmiş (Macro AUC: 0.8342, TTA ile)
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

| Hastalık | AUC | AP | Precision | Recall (Sens.) | F1 | Destek |
|----------|-----|-----|-----------|----------------|-----|--------|
| **Emphysema** | 0.9258 | 0.4084 | 0.2565 | 0.7564 | 0.3830 | 472 |
| **Cardiomegaly** | 0.9033 | 0.3229 | 0.1940 | 0.7440 | 0.3078 | 496 |
| **Pneumothorax** | 0.8952 | 0.3363 | 0.2285 | 0.7760 | 0.3530 | 835 |
| **Edema** | 0.8913 | 0.1717 | 0.0837 | 0.7994 | 0.1516 | 334 |
| **Effusion** | 0.8779 | 0.5162 | 0.2610 | 0.9004 | 0.4047 | 2,038 |
| **Mass** | 0.8510 | 0.3004 | 0.1573 | 0.7211 | 0.2582 | 760 |
| **Hernia** | 0.8491 | 0.2171 | 0.1009 | 0.3235 | 0.1538 | 34 |
| **Atelectasis** | 0.8188 | 0.3622 | 0.1966 | 0.8595 | 0.3200 | 1,758 |
| **Fibrosis** | 0.8032 | 0.0956 | 0.0783 | 0.4585 | 0.1337 | 277 |
| **Pleural Thickening** | 0.8009 | 0.1443 | 0.1028 | 0.6250 | 0.1766 | 536 |
| **Consolidation** | 0.7974 | 0.1366 | 0.1056 | 0.7938 | 0.1864 | 771 |
| **No Finding** | 0.7875 | 0.8028 | 0.7842 | 0.6785 | 0.7276 | 9,434 |
| **Nodule** | 0.7780 | 0.2514 | 0.1240 | 0.7038 | 0.2109 | 979 |
| **Pneumonia** | 0.7705 | 0.0437 | 0.0479 | 0.4190 | 0.0859 | 210 |
| **Infiltration** | 0.7144 | 0.3560 | 0.2102 | 0.9062 | 0.3412 | 3,093 |
| **Macro Average** | **0.8309** | **0.2977** | — | **0.6977** | **0.2796** | — |
| **Macro Average (+TTA)** | **0.8342** | **0.3016** | — | **0.7001** | **0.2817** | — |

> **Not:** Yukarıdaki değerler varsayılan eşik τ = 0.5 içindir. Sınıf başına eşikler
> **yalnızca validation seti üzerinde** kalibre edildiğinde (Youden's J) macro
> sensitivity 0.6977 → **0.7662**, F1-maksimize eden eşikle macro F1 0.2796 →
> **0.3640** olur. Tam tablo: `egitim-ciktilari/threshold_optimized_metrics.csv`
> (üreten script: `06_calibration_and_thresholds.py`).

### Kontrollü Ablation (8 konfigürasyon)

Hepsi **aynı** patient-level split, ön işleme, optimizer, seed ve **10-epoch bütçesiyle** eğitildi;
yalnızca satırda belirtilen faktör değişiyor (`run_ablations.py`).

| Konfigürasyon | Test macro AUC | Test macro F1 | Test macro Sens. |
|---|---|---|---|
| **Önerilen model (gating)** | 0.8323 | 0.2740 | 0.6957 |
| Sadece görüntü | 0.8307 | 0.2801 | 0.6915 |
| Sadece demografik | 0.6124 | 0.1219 | 0.5229 |
| Basit birleştirme (concat) | 0.8320 | 0.2707 | 0.6891 |
| Çoklu-baş self-attention | 0.8119 | 0.2334 | 0.6848 |
| Naive class weight (eski formül) | 0.8112 | 0.1431 | 0.1920 |
| Focal loss kapalı (düz BCE) | 0.8257 | 0.3026 | 0.6073 |
| Augmentation kapalı | 0.8193 | 0.2660 | 0.6863 |
| *Önerilen model (18 epoch, referans)* | *0.8309* | *0.2796* | *0.6977* |

**Gürültü tabanı:** Aynı konfigürasyon 10 epoch'ta 0.8323, 18 epoch'ta 0.8309 veriyor —
hiçbir şey değişmeden **0.0014** fark. Tek-seed'li koşumların oynaklığı bu mertebede, dolayısıyla:

- gating (0.8323) vs concat (0.8320) → **+0.0003**, gürültünün altında → füzyon mekanizması seçimi ölçülemiyor
- tam model (0.8323) vs sadece görüntü (0.8307) → **+0.0016**, gürültüyle aynı mertebede → demografik katkı iddia edilmiyor
- sadece demografik (0.6124) vs sadece görüntü (0.8307) → **0.2183**, ~150× → sinyal radyografiden geliyor, demografik kısayoldan değil
- naive class weight → sensitivity **0.6957 → 0.1920** (3.6× düşüş), AUC ise büyük ölçüde korunuyor

Ham sonuçlar: `egitim-ciktilari/ablation_results.csv`

### Eğitim Detayları

- **Veri Seti**: NIH Chest X-ray Dataset (112,120 görüntü)
- **Train/Val/Test Split**: 70%/15%/15% (Patient-level split)
- **Model**: EfficientNet-B3 (12M parametreler)
- **Görüntü Boyutu**: 300×300 piksel
- **Eğitim Platformu**: Kaggle (tek NVIDIA Tesla T4, 16 GB)
- **Eğitim**: 18 epoch; en iyi validation macro-AUC epoch 12'de (0.8392), test sonuçları bu checkpoint'ten
- **Batch Size**: 36
- **Optimizasyon**: AdamW (lr 3e-4, weight decay 1e-4) + Cosine Annealing + warmup
- **Karışık Hassasiyet**: float16 AMP (T4 native BF16 desteklemez) + dinamik loss scaling
- **Loss Function**: Focal Loss (α=0.25, γ=2.0) + sınıf başına `pos_weight = N_neg/N_pos` (15'te sınırlı)
- **Kararlılık**: Gradient clipping (max_norm=5.0) + NaN-loss batch atlama
- **Data Augmentation**: Medium (rotation, shift, scale, flip, CLAHE, gauss noise, coarse dropout)
- **Test-Time Augmentation (TTA)**: 5x augmentation, **+0.0032 macro AUC** (paired bootstrap %95 CI [+0.0025, +0.0039], 15/15 sınıfta iyileşme)

### Güçlü Yönler

✅ **Yüksek Performans**:
- Emphysema: AUC 0.9258
- Cardiomegaly: AUC 0.9033
- Pneumothorax: AUC 0.8952
- Edema: AUC 0.8913

✅ **Data Leakage Önleme**:
- Patient-level split ile güvenilir sonuçlar
- Train-Val-Test overlap: 0

✅ **Multi-label Handling**:
- Bir görüntüde birden fazla hastalık tespiti
- Focal Loss + Class Weights ile dengesiz veri yönetimi

✅ **Multimodal Approach**:
- Görüntü + demografik bilgiler
- Öğrenilmiş **modalite kapılama** (modality gating) ile modalite ağırlıklandırma

✅ **Kontrollü Ablation** (8 konfigürasyon, hepsi 10 epoch, `run_ablations.py`):
- Sinyalin radyografik içerikten geldiği kanıtlandı (metadata-only 0.6124 vs image-only 0.8307)
- Sonuçlar: `egitim-ciktilari/ablation_results.csv`

### İyileştirme Alanları

⚠️ **Düşük Performanslı Hastalıklar**:
- Infiltration: AUC 0.7144 (etiket gürültüsü)
- Pneumonia: AUC 0.7705 (az örnek sayısı)
- Nodule: AUC 0.7780 (küçük lezyon, 300×300 çözünürlükte zor)

⚠️ **F1-Score Düşük (τ = 0.5'te)**:
- Macro F1: 0.2796 — nadir sınıflarda precision cezası
- Validation setinde kalibre edilmiş eşiklerle **0.3640**'a çıkıyor
  (`06_calibration_and_thresholds.py`)

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
│           └────┤ Modality Gating   ├───┘                    │
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
- 12 demografik özellik (3 + 4 + 2 + 3 = 12):
  - Sürekli yaş dönüşümleri (3): min-max (age/100), log (log(1+age)/log(101)), karesel ((age/100)²)
  - Yaş bantları (4): <18, 18-44, 45-64, ≥65 (birbirini dışlayan one-hot)
  - Cinsiyet (2): Male/Female (one-hot)
  - Görüntü pozisyonu (3): PA/AP/Other (one-hot)
- Eksik/geçersiz değer yönetimi: yaş [0, 120] aralığına clip + eksikse ortalama ile doldurma;
  tanınmayan cinsiyet/pozisyon değerleri ilgili one-hot bloğunda tamamen sıfır ("bilinmeyen")
- 3-layer MLP (12→128→128→64), her katmanda BatchNorm + ReLU + Dropout (0.30 / 0.25 / 0.20)

**Modality Gating Fusion** (`ModalityGatingFusion`):
- Bu mekanizma **self-attention DEĞİLDİR** — token dizisi oluşturmaz, Q/K/V projeksiyonu hesaplamaz.
  Birleştirilmiş vektör üzerinden **modalite başına bir skaler** üreten öğrenilmiş bir softmax kapısıdır:

  ```
  z    = [z_image ; z_demo]              ∈ ℝ¹⁶⁰⁰
  h    = ReLU(W₁ z + b₁),  W₁ ∈ ℝ⁴⁰⁰ˣ¹⁶⁰⁰
  [a_image, a_demo] = softmax(W₂ h + b₂),  W₂ ∈ ℝ²ˣ⁴⁰⁰,  a_image + a_demo = 1
  z_fused = [a_image · z_image ; a_demo · z_demo]  ∈ ℝ¹⁶⁰⁰
  ```
- Her modalite bloğu, girdiye bağlı tek bir skalerle ölçeklenip birleştirilir.
- Karşılaştırma için **gerçek** çoklu-baş self-attention (`CrossModalSelfAttention`:
  2 modalite token'ı → 256 boyut, 4 head, residual + LayerNorm → 512 boyut) da
  uygulandı ve ablation'da değerlendirildi; `config.FUSION_TYPE` ile seçilir
  (`'gating'` | `'self_attention'` | `'concat'`).

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
- **Patient-level** split (`GroupShuffleSplit`, grup anahtarı = hasta kimliği, seed 42), hedef 70/15/15
- **Not:** Sınıf-stratifikasyonu uygulanmaz — çok-etiketli bir problemde hasta-düzeyi gruplama ile
  stratifikasyon aynı anda zorlanamaz. Gerçekleşen bölünme: 78,566 / 16,106 / 17,448 görüntü
  (21,563 / 4,621 / 4,621 hasta), hasta örtüşmesi programatik olarak sıfır doğrulandı.
- `split_manifest_112k.json` üretimi (bölüm başına tam görüntü/hasta sayıları + sınıf prevalansları)
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
- 5x augmentation ile tahmin (orijinal, yatay çevirme, ±5° rastgele döndürme, sabit −5° döndürme, hafif parlaklık/kontrast)
- Ensemble averaging
- **+0.0032 macro AUC** (0.8309 → 0.8342); paired bootstrap %95 CI [+0.0025, +0.0039], 15/15 sınıfta iyileşme
- Per-sample çıktı: `test_predictions_tta.csv` (anlamlılık testi: `09_tta_significance_test.py`)

#### 5. Analiz ve Hakem-Yanıtı Script'leri

```bash
python 06_calibration_and_thresholds.py --predictions test_predictions.csv        --threshold-source val_predictions.csv --output-dir results/
python 07_model_profiling.py --output-dir results/
python 08_gradcam_visualization.py --checkpoint best_model.pth --test-csv test_112k.csv        --img-dir <nih-images> --output-dir results/gradcam/
python 09_tta_significance_test.py
python run_ablations.py --presets full_model image_only metadata_only concat_no_gating        self_attention_fusion naive_class_weights no_focal_loss no_augmentation
```

- `06` — sınıf başına eşik kalibrasyonu (**yalnızca validation setinde**), PR eğrileri,
  reliability diyagramları, Brier skorları, bootstrap %95 CI
- `07` — parametre / FLOPs / bellek / gecikme profili (DenseNet-121 karşılaştırması dahil)
- `08` — Grad-CAM ısı haritaları (hasta-özel görsel kanıt)
- `09` — TTA için paired bootstrap anlamlılık testi
- `run_ablations.py` — 8 konfigürasyonluk kontrollü ablation (hepsi aynı 10-epoch bütçesinde);
  oturumlar arası sonuçları **birleştirir**, test değerlendirmesini diskteki en iyi checkpoint'ten yapar
- `10` / `11` — makale figürleri (mimari şeması, PR+kalibrasyon paneli, Grad-CAM paneli)

### 🔬 Yeniden Üretilebilirlik — Makale Tablo/Figür Eşlemesi

| Makale öğesi | Üreten script | Çıktı dosyası |
|---|---|---|
| Tablo 1 (split istatistikleri) | `01_data_preparation.py` | `egitim-ciktilari/split_manifest_112k.json` |
| Tablo 2 (sınıf başına performans) | `05_evaluate.py` | `egitim-ciktilari/test_metrics.csv` |
| Tablo 3 (eşik kalibrasyonu) | `06_calibration_and_thresholds.py` | `egitim-ciktilari/threshold_optimized_metrics.csv`, `report.md` |
| Tablo 4 (TTA etkisi) | `05_evaluate_with_tta.py` + `09_tta_significance_test.py` | `egitim-ciktilari/test_metrics_tta.csv` |
| Tablo 5 (ablation) | `run_ablations.py` | `egitim-ciktilari/ablation_results.csv` |
| Tablo 6 (hesaplama profili) | `07_model_profiling.py` | — |
| Fig 2 (mimari) | `10_architecture_figure.py` | `egitim-ciktilari/fig2_architecture.png` |
| Fig 6 (PR + kalibrasyon) | `06` + `11_composite_figures.py` | `egitim-ciktilari/fig6_pr_calibration.png` |
| Fig 7 (Grad-CAM) | `08` + `11_composite_figures.py` | `egitim-ciktilari/fig7_gradcam_panel.png` |
| Fig 8 / 9 (ROC, confusion) | `05_evaluate.py` | `egitim-ciktilari/roc_curves.png`, `confusion_matrices.png` |

Ham tahmin dosyaları (`test_predictions.csv`, `test_predictions_tta.csv`,
`val_predictions.csv`) da paylaşılmıştır; tüm metrikler ve güven aralıkları
bunlardan bağımsız olarak yeniden hesaplanabilir.

> ℹ️ **`model/` klasörü hakkında:** o klasör web servisinin çıkarım anlık
> görüntüsüdür ve eski hiperparametre değerleri içerebilir. **Makaledeki tüm
> eğitim ve değerlendirme sonuçları `egitim-dosyalari/` altındaki kodla
> üretilmiştir** — referans alınması gereken kaynak orasıdır.

### Konfigürasyon

Tüm hyperparameter'lar `egitim-dosyalari/config.py` dosyasında tanımlıdır:

```python
IMG_SIZE = 300
BATCH_SIZE = 36
EPOCHS = 18
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.55
PRETRAINED_MODEL = "efficientnet_b3"
FREEZE_BACKBONE_EPOCHS = 2
EARLY_STOP_PATIENCE = 9
USE_FOCAL_LOSS = True
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
AUGMENTATION_STRENGTH = 'medium'
RANDOM_SEED = 42

AMP_DTYPE = torch.float16      # T4 native BF16 Tensor Core DESTEKLEMEZ
CLASS_WEIGHT_SCHEME = 'corrected'   # pos_weight = N_neg/N_pos, 15'te sınırlı
                                    # ('naive' = eski hatalı formül, ablation için saklandı)
FUSION_TYPE = 'gating'         # 'gating' | 'self_attention' | 'concat'
ABLATION_MODE = 'full'         # 'full' | 'image_only' | 'metadata_only'
ABLATION_EPOCHS = 10
```

> **Class weight düzeltmesi (önemli):** Önceki sürüm her sınıfın sayısını *tüm
> sınıfların toplamına* bölüyordu; bu, çoğunluk sınıfları için `pos_weight < 1`
> üretiyordu (ör. "No Finding" = 0.157) ve BCEWithLogitsLoss'ta pozitif tahminleri
> caydırarak τ = 0.5'te sensitivity/F1'i sıfıra düşürüyordu. Standart
> `N_negatif / N_pozitif` formülüne geçildi (15× sınır). Kontrollü ablation bunu
> doğruluyor: `naive_class_weights` preset'inde macro sensitivity 0.6957 → **0.1920**
> çöküyor, AUC ise büyük ölçüde korunuyor — yani arıza yalnızca AUC'ye bakınca
> görünmüyor. Eski formül `NAIVE_CLASS_WEIGHTS` olarak bilinçli saklandı.

### Eğitim Süreçleri

**Seçilen Checkpoint** (`best_model.pth` metadata'sından):
```
Epoch 12/18 — Val macro AUC: 0.8392  ⭐ Best  (early stopping tetiklenmedi)
  Train loss 0.0845  |  Train macro AUC 0.8450
  Val   loss 0.0871  |  Val   macro AUC 0.8392
  Train–Val AUC farkı: 0.006  → overfitting kontrol altında
```
Tüm test sonuçları bu checkpoint'ten üretilmiştir, son epoch ağırlıklarından değil.

> ⚠️ **`egitim-ciktilari/kaggle-ciktisi.txt` hakkında:** bu dosya **önceki**
> (hatalı class-weight formüllü) koşumun tam logudur ve bilinçli olarak
> saklanmıştır — düzeltilen hatanın kanıtıdır. Bu klasördeki güncel
> `best_model.pth` / `test_metrics*.csv` dosyalarını **tarif etmez**.

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

Bu depodaki kod **MIT Lisansı** altında dağıtılmaktadır — bkz. [`LICENSE`](LICENSE).

NIH ChestX-ray14 veri seti ve eğitilmiş model ağırlıkları bu lisansın kapsamı
dışındadır; veri seti NIH Clinical Center'ın kendi kullanım şartlarına tabidir.

**Akademik kullanım:** Bu depo bir dergi makalesinin yeniden üretilebilirlik
materyalidir. Kullanıyorsanız lütfen ilgili makaleye atıf verin.

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

**Son Güncelleme**: 11 Eylül 2026
**Versiyon**: 2.0.0 — hakem revizyonu sürümü (düzeltilmiş class-weight ile yeniden eğitim,
eşik kalibrasyonu, 8 konfigürasyonluk ablation, Grad-CAM, hesaplama profili)
**Geliştirici**: KDS Ekibi

