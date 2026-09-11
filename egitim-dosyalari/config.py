"""
Göğüs Hastalıkları Karar Destek Sistemi - Kaggle Optimized
Multi-label Classification with Patient-level Split
Target: 85%+ AUC in <12 hours on Kaggle T4 GPU
"""

import torch

# ==================== PROJE BİLGİLERİ ====================
PROJECT_NAME = "chest-xray-multimodal-diagnosis"
VERSION = "5.0-kaggle-optimized"
RANDOM_SEED = 42

# ==================== KAGGLE VERİ AYARLARI ====================
# Kaggle dataset paths
# NIH dataset'i Kaggle'da bu yolda bulunur
DATA_DIR = "/kaggle/input/data"
CSV_FILE = f"{DATA_DIR}/Data_Entry_2017.csv"
BBOX_FILE = f"{DATA_DIR}/BBox_List_2017.csv"

# Images are in: /kaggle/input/data/images_001/images/, ...
IMAGES_BASE_DIR = "/kaggle/input/data"

OUTPUT_DIR = "/kaggle/working"
MODELS_DIR = f"{OUTPUT_DIR}/models"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# ==================== VERİ SEÇİMİ - TÜM DATASET (MAXIMUM PERFORMANCE) ====================
TOTAL_IMAGES = 112120  # Tüm NIH dataset + EfficientNet-B2 (MAXIMUM!)

# Hastalık dağılımı (Multi-label için dengeli)
# NOT: Bir görüntüde birden fazla hastalık olabilir!
# Gerçek dataset sayıları (README_CHESTXRAY.pdf, toplam 112,120 görüntü):
#   Infiltration: 19,871 | Effusion: 13,307 | Atelectasis: 11,535
#   Nodule: 6,323 | Mass: 5,746 | Pneumothorax: 5,298 | Consolidation: 4,667
#   Pleural_Thickening: 3,385 | Cardiomegaly: 2,776 | Emphysema: 2,516
#   Edema: 2,303 | Fibrosis: 1,686 | Pneumonia: 1,314 | Hernia: 227
DISTRIBUTION = {
    'No Finding': 60000,        # ~60K gerçek (tüm kontrol grubu)
    'Infiltration': 19871,      # 19,871 gerçek (TÜM)
    'Effusion': 13317,          # 13,317 gerçek (TÜM)
    'Atelectasis': 11559,       # 11,559 gerçek (TÜM)
    'Nodule': 6331,             # 6,331 gerçek (TÜM)
    'Mass': 5782,               # 5,782 gerçek (TÜM)
    'Pneumothorax': 5302,       # 5,302 gerçek (TÜM)
    'Consolidation': 4667,      # 4,667 gerçek (TÜM)
    'Pleural_Thickening': 3385, # 3,385 gerçek (TÜM)
    'Cardiomegaly': 2776,       # 2,776 gerçek (TÜM)
    'Emphysema': 2516,          # 2,516 gerçek (TÜM)
    'Edema': 2303,              # 2,303 gerçek (TÜM)
    'Fibrosis': 1686,           # 1,686 gerçek (TÜM)
    'Pneumonia': 1431,          # 1,431 gerçek (TÜM)
    'Hernia': 227               # 227 gerçek (TÜM - rare disease)
}

DISEASES = list(DISTRIBUTION.keys())
NUM_DISEASES = len(DISEASES)

# Veri bölünmesi (PATIENT-LEVEL SPLIT!)
TRAIN_RATIO = 0.70  # ~35,000 görüntü
VAL_RATIO = 0.15    # ~7,500 görüntü
TEST_RATIO = 0.15   # ~7,500 görüntü

# ==================== MODEL AYARLARI ====================
IMG_SIZE = 300  # B3 optimal resolution! (daha fazla detay, Nodule/Mass için)
IMG_CHANNELS = 3

# Demografik özellikler (EXPANDED)
NUM_DEMOGRAPHIC_FEATURES = 12  # 8 → 12 (daha zengin feature set)

# Model
PRETRAINED_MODEL = "efficientnet_b3"  # B2 → B3 (daha güçlü - 12M params!)
DROPOUT_RATE = 0.55  # B3 + MEDIUM aug için optimal denge! (0.60'tan dengeli)

# Freeze pretrained layers initially
FREEZE_BACKBONE_EPOCHS = 2  # B3 için 2 epoch freeze yeterli (3 → 2)

# ==================== EĞİTİM AYARLARI ====================
BATCH_SIZE = 36  # B3 + 300x300: BS36 deneyelim! (280'de çalışıyor, OOM olursa 32)
EPOCHS = 18      # B3 + 300x300 + 18 epoch: Gerçek veri göre ~8.2h (optimal!)
LEARNING_RATE = 0.0003  # 112K veri için optimal
WEIGHT_DECAY = 1e-4     # Overfitting kontrolü

# Loss fonksiyonu - Multi-label için Binary Cross Entropy + Class Weights
USE_FOCAL_LOSS = True
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Learning rate scheduler - Cosine Annealing
USE_COSINE_ANNEALING = True
LR_MIN = 1e-7
WARMUP_EPOCHS = 2  # İlk 2 epoch warmup

# Early stopping
EARLY_STOP_PATIENCE = 9  # 18 epoch için patience 9 optimal (yarısı)

# ==================== DATA AUGMENTATION ====================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Augmentation strategy (MEDIUM - 2 GPU + 8 workers ile optimal)
USE_AUGMENTATION = True
AUGMENTATION_STRENGTH = 'medium'  # B3 + 280x280: MEDIUM optimal (HEAVY gereksiz)

# ==================== HESAPLAMA AYARLARI ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # Kaggle T4 için optimal (8 worker CPU overload yapıyor!)
PIN_MEMORY = True
USE_AMP = True  # Mixed Precision Training
# AMP dtype: T4 (Kaggle GPU'su) native BF16 Tensor Core DESTEKLEMİYOR.
# GradScaler de zaten sadece float16 ile anlamlıdır (bfloat16 taşma riski
# taşımadığı için scaler gerektirmez). Bu yüzden fiili precision float16'dır
# — makalede "bfloat16" olarak geçiyorsa düzeltilmeli.
AMP_DTYPE = torch.float16
PREFETCH_FACTOR = 2
ACCUMULATION_STEPS = 1

# ==================== ABLATION / FUSION AYARLARI ====================
# Hakem #1 (madde 1) ve #3 (major, ablation talebi) için: reviewer'ın istediği
# image-only / metadata-only / concat / attention karşılaştırmasını TEK
# parametrik pipeline ile çalıştırmak amacıyla eklendi (bkz. run_ablations.py).
#
# ABLATION_MODE:
#   'full'          -> görüntü + demografik, seçili FUSION_TYPE ile birleşir (ana model)
#   'image_only'    -> sadece EfficientNet-B3 (demografik encoder tamamen bypass edilir)
#   'metadata_only' -> sadece demografik MLP (görüntü encoder tamamen bypass edilir)
ABLATION_MODE = 'full'

# FUSION_TYPE (sadece ABLATION_MODE='full' iken etkili):
#   'gating'          -> ModalityGatingFusion (mevcut/varsayılan: 2 skaler ağırlıklı softmax gate)
#   'self_attention'  -> CrossModalSelfAttention (gerçek multi-head Q/K/V attention, 2 modality-token)
#   'concat'          -> ağırlıksız basit concat ("concat_no_gating" ablation'ı bu değerle yapılır)
FUSION_TYPE = 'gating'

# CrossModalSelfAttention için
ATTENTION_NUM_HEADS = 4
ATTENTION_COMMON_DIM = 256  # img/demo projeksiyon ortak boyutu (head sayısına bölünebilir olmalı)

# Ablation ayrı çalıştırmalar için epoch bütçesi (tam 18 yerine — 12h Kaggle
# oturum limitine 8 varyantın hepsini sığdırmak için düşürülmüş bütçe)
ABLATION_EPOCHS = 10

# ==================== CLASS BALANCING ====================
# Multi-label için class weights
USE_CLASS_WEIGHTS = True  # BCE loss için class weights
USE_WEIGHTED_SAMPLING = False  # 50K veri için gerekli değil

# ==================== DEĞERLENDİRME ====================
CLASSIFICATION_THRESHOLD = 0.5
METRICS = ['AUC', 'AP', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']

# ==================== LOGLAMA ====================
USE_WANDB = False  # Kaggle'da genelde kapalı
WANDB_PROJECT = "chest-xray-multimodal"

# ==================== GRADİO ====================
GRADIO_SHARE = True
GRADIO_PORT = 7860

# ==================== BEKLENEN PERFORMANS ====================
EXPECTED_PERFORMANCE = {
    'target_auc': 0.89,          # %89+ hedef (B3 + 300x300 + MEDIUM + 18 epoch + TTA)
    'min_acceptable_auc': 0.85,  # Minimum kabul edilebilir
    'training_time_hours': 6.9,  # B3 + 300x300 + 18 epoch + BS36 → ~6.9 saat (gerçek: 23 dk/epoch)
    'tta_time_hours': 1.0,       # TTA: +60 dk (300x300)
    'total_time_hours': 8.2,     # Toplam: 8.2 saat (BOL MARJ 3.8h!)
    'gpu_memory_gb': 11.5,       # 1xT4 için (BS36 ile B3 300x300: ~11.5GB, spike 15-16GB)
    'batch_time_seconds': 1.20   # B3 + 300x300 + MEDIUM: ~23 dk/epoch (BS36)
}

# ==================== SÜRE TAHMİNİ ====================
def estimate_training_time():
    """Eğitim süresini tahmin et"""
    train_samples = int(TOTAL_IMAGES * TRAIN_RATIO)
    batches_per_epoch = train_samples // BATCH_SIZE
    seconds_per_batch = EXPECTED_PERFORMANCE['batch_time_seconds']

    epoch_time_minutes = (batches_per_epoch * seconds_per_batch) / 60
    total_time_hours = (epoch_time_minutes * EPOCHS) / 60

    return {
        'train_samples': train_samples,
        'batches_per_epoch': batches_per_epoch,
        'epoch_time_minutes': epoch_time_minutes,
        'total_time_hours': total_time_hours,
        'max_epochs_in_12h': int(12 * 60 / epoch_time_minutes)
    }

# ==================== CLASS WEIGHTS (Multi-label için) ====================
def calculate_naive_multilabel_weights():
    """
    ESKİ (HATALI) FORMÜL — sadece referans/ablation amaçlı saklanıyor.

    weight_i = sum(DISTRIBUTION) / (count_i * NUM_DISEASES)

    Sorun: sum(DISTRIBUTION) multi-label örtüşmeleri nedeniyle TOTAL_IMAGES
    değil, tüm hastalık sayılarının toplamı (141,153 > 112,120). Bu, çoğunluk
    sınıfları (No Finding, Infiltration, Effusion, Atelectasis) için
    pos_weight < 1 üretir (örn. No Finding=0.157). nn.BCEWithLogitsLoss'ta
    pos_weight, SADECE pozitif örneklerin loss katkısını ölçekler; <1 bir
    ağırlık modelin bu sınıflar için pozitif tahmin üretmesini AKTİF OLARAK
    caydırır (negatiflere göre daha az cezalandırma). Bu, eğitilmiş modelde
    (best_model.pth, val_auc=0.8231) threshold=0.5'te tam olarak bu dört
    sınıf için sensitivity/F1 ≈ 0 çıkmasının kök nedenidir (bkz.
    hakem-yanitlari/class_weight_bug_analysis.md). Bu fonksiyon SADECE
    "naive_weights" ablation preset'i için tutuluyor; gerçek eğitimde
    kullanılmamalı.
    """
    total = sum(DISTRIBUTION.values())
    weights = {}
    for disease, count in DISTRIBUTION.items():
        weights[disease] = total / (count * NUM_DISEASES)
    return weights


def calculate_class_weights_from_distribution(max_weight=15.0):
    """
    DÜZELTİLMİŞ (standart) class weight formülü — her hastalığı bağımsız
    bir binary classification problemi olarak ele alır:

        pos_weight_i = N_negative_i / N_positive_i
                     = (TOTAL_IMAGES - count_i) / count_i

    Bu, nn.BCEWithLogitsLoss(pos_weight=...) dokümantasyonundaki standart
    kullanımdır (bkz. PyTorch docs: "pos_weight > 1 azınlık sınıfını
    ağırlıklandırır"). Çoğunluk sınıfları için weight doğal olarak ~1'e
    yakın/altında çıkar (örn. No Finding≈0.87, çünkü %53.8 prevalence'ta
    negatif/pozitif oranı zaten dengeye yakın) — bu, eski formüldeki
    aşırı bastırmayı (0.157) ortadan kaldırır.

    Aşırı nadir sınıflarda (örn. Hernia: %0.2 prevalence → çıplak oran
    ~493x) gradyan kararsızlığını önlemek için üst sınır (max_weight)
    uygulanıyor; bu, capped inverse-frequency weighting olarak literatürde
    yaygın bir tekniktir.
    """
    weights = {}
    for disease, count in DISTRIBUTION.items():
        neg = TOTAL_IMAGES - count
        weight = neg / count
        weights[disease] = min(weight, max_weight)
    return weights


CLASS_WEIGHTS = calculate_class_weights_from_distribution()
NAIVE_CLASS_WEIGHTS = calculate_naive_multilabel_weights()  # sadece ablation için

# Hangi class-weight şemasının eğitimde kullanılacağını seçer.
#   'corrected' -> CLASS_WEIGHTS (düzeltilmiş, standart neg/pos oranı — VARSAYILAN)
#   'naive'     -> NAIVE_CLASS_WEIGHTS (eski hatalı formül — sadece ablation/karşılaştırma için)
CLASS_WEIGHT_SCHEME = 'corrected'

# Config başlangıç mesajı
if __name__ == "__main__":
    print("="*70)
    print(f"✓ Config {TOTAL_IMAGES//1000}K yüklendi (KAGGLE OPTIMIZED)")
    print("="*70)
    print(f"  📊 Toplam görüntü: {TOTAL_IMAGES:,}")
    print(f"  🏗️  Model: {PRETRAINED_MODEL}")
    print(f"  🎯 Hedef AUC: {EXPECTED_PERFORMANCE['target_auc']:.2f}")
    print(f"  ⏱️  Tahmini süre: {EXPECTED_PERFORMANCE['training_time_hours']:.1f} saat")
    print(f"  🖥️  Device: {DEVICE}")
    print(f"  🔋 Mixed Precision: {'✅' if USE_AMP else '❌'}")
    print(f"  ⚖️  Class Weights: {'✅' if USE_CLASS_WEIGHTS else '❌'}")
    print(f"  🧊 Freeze Backbone: {FREEZE_BACKBONE_EPOCHS} epochs")

    # Süre tahmini
    time_est = estimate_training_time()
    print(f"\n📊 Eğitim Detayları:")
    print(f"  - Train samples: {time_est['train_samples']:,}")
    print(f"  - Batches/epoch: {time_est['batches_per_epoch']:,}")
    print(f"  - Epoch süresi: ~{time_est['epoch_time_minutes']:.1f} dakika")
    print(f"  - Toplam süre: ~{time_est['total_time_hours']:.1f} saat ({EPOCHS} epoch)")
    print(f"  - 12 saatte max epoch: {time_est['max_epochs_in_12h']}")

    if time_est['total_time_hours'] > 12:
        print(f"\n⚠️  UYARI: Tahmini süre 12 saati aşıyor!")
        print(f"  Önerilen EPOCHS: {time_est['max_epochs_in_12h']}")
    else:
        print(f"\n✅ Süre 12 saat içinde ({time_est['total_time_hours']:.1f} < 12)")

    print("\n💡 Optimizasyonlar:")
    print("  ✅ Kaggle dataset yapısına uyumlu")
    print("  ✅ Patient-level split (data leakage önleme)")
    print("  ✅ Multi-label handling")
    print("  ✅ Class weights (imbalance için)")
    print("  ✅ Cosine annealing + warmup")
    print("  ✅ 50K görüntü (12 saat garantisi)")
    print("="*70)
