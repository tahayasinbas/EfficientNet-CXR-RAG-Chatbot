# KDS - Göğüs Röntgeni Analiz ve RAG Chatbot Sistemi

Göğüs röntgeni görüntülerini yapay zeka ile analiz eden ve tıbbi doküman tabanlı RAG (Retrieval-Augmented Generation) chatbot sistemi içeren web uygulaması.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Teknoloji Stack](#teknoloji-stack)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [Kullanım](#kullanım)
- [API Endpoints](#api-endpoints)
- [RAG Chatbot Sistemi](#rag-chatbot-sistemi)
- [Proje Yapısı](#proje-yapısı)
- [Geliştirme](#geliştirme)
- [Sorun Giderme](#sorun-giderme)

## 🎯 Özellikler

### X-Ray Görüntü Analizi
- **Göğüs röntgeni yükleme**: Çoklu format desteği (PNG, JPG, JPEG, DICOM)
- **AI Tabanlı Analiz**: EfficientNet-B3 tabanlı derin öğrenme modeli
- **14 Hastalık Tespiti**: NIH Chest X-ray veri seti üzerinde eğitilmiş
  - Atelectasis, Cardiomegaly, Effusion, Infiltration
  - Mass, Nodule, Pneumonia, Pneumothorax
  - Consolidation, Edema, Emphysema, Fibrosis
  - Pleural_Thickening, Hernia
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
kds_django_fantezi/
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
│   └── model.weights.h5          # EfficientNet-B3 weights
│
├── media/                        # Yüklenen dosyalar
│   └── xrays/                    # X-ray görüntüleri
│
├── frontend/                     # React frontend
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
└── Rag_Chatbot/                  # RAG sistem dosyaları
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

## 📝 Lisans

Bu proje eğitim amaçlıdır. Ticari kullanım için lütfen lisans alın.

## 👥 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 🙏 Teşekkürler

- **NIH Chest X-ray Dataset**: Eğitim verisi için
- **Google Gemini**: LLM entegrasyonu için
- **LangChain**: RAG framework için
- **Django & React**: Framework'ler için

---

**Son Güncelleme**: 3 Aralık 2025
**Versiyon**: 1.0.0
**Geliştirici**: KDS Ekibi
