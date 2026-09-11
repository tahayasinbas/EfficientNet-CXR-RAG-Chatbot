# ═══════════════════════════════════════════════════════════════
# GÖĞÜS HASTALIKLARI KDS - YENİ VERSİYON (60K)
# Kaggle'da Direkt Çalıştırma
# ═══════════════════════════════════════════════════════════════

import os
import sys
import subprocess
import time

print("="*70)
print("GÖĞÜS HASTALIKLARI KDS SİSTEMİ (112K - EfficientNet-B3 v16.0 FINAL)")
print("="*70)
print("📊 112,120 görüntü (TÜM DATASET)")
print("🏗️  Model: EfficientNet-B3 (12M params - %32 daha güçlü!)")
print("🎯 Hedef AUC: 0.85+ (B3 optimal resolution!)")
print("⏱️  Tahmini süre: ~6.9 saat (eğitim) + 1.0 saat (TTA) = 8.2 saat")
print("📏 Image Size: 300×300 (B3 OPTIMAL! Nodule/Mass daha iyi!)")
print("🔧 Batch Size: 36 (gerçek veri: BS36 çalışıyor!)")
print("🔧 Epochs: 18 (23dk/epoch → 8.2h, marj 3.8h!)")
print("🔧 Augmentation: MEDIUM + TTA")
print("🔧 Dropout: 0.55 (Dengeli regularization)")
print("🔧 NUM_WORKERS: 4 (Kaggle T4 optimal)")
print("🔧 Freeze: 2 epoch (B3 hızlı öğrenir)")
print("🔧 Early Stop: patience=9 (18 epoch için)")
print("🔧 Patient-Level Split: ✅")
print("🔧 TTA: ✅ OTOMATİK")
print("="*70)
print()

# Dosyaları working directory'ye kopyala
print("📁 Python dosyaları kontrol ediliyor...")

# Gerekli dosyalar
required_files = [
    'config.py',
    'model.py',
    'dataset.py',
    '01_data_preparation.py',
    '04_train.py',
    '05_evaluate.py'
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)
    else:
        print(f"✓ {file}")

if missing_files:
    print(f"\n❌ HATA: Eksik dosyalar!")
    for file in missing_files:
        print(f"   - {file}")
    print("\n💡 Çözüm: Tüm Python dosyalarını notebook'a yükleyin")
    sys.exit(1)

print()

# DATASET INPUT KONTROLÜ
# NIH dataset'i Kaggle'da otomatik olarak /kaggle/input/data/ yolunda
data_path = "/kaggle/input/data"

if not os.path.exists(data_path):
    print(f"\n❌ HATA: Dataset bulunamadı!")
    print(f"   Beklenen: /kaggle/input/data")
    print(f"\n💡 Çözüm:")
    print("   1. Notebook → Add Data")
    print("   2. 'NIH Chest X-rays' dataset'ini bulun ve ekleyin")
    print("   3. Dataset'in Input bölümünde göründüğünden emin olun")
    sys.exit(1)

print(f"✓ Dataset bulundu: /kaggle/input/data")

# CSV kontrolü
csv_file = f"{data_path}/Data_Entry_2017.csv"
if os.path.exists(csv_file):
    print(f"✓ CSV bulundu: Data_Entry_2017.csv")
else:
    print(f"❌ CSV bulunamadı: {csv_file}")
    sys.exit(1)

# Görüntü klasörleri
image_dirs = 0
for i in range(1, 13):
    img_path = f"{data_path}/images_{i:03d}/images"
    if os.path.exists(img_path):
        image_dirs += 1

print(f"✓ Görüntü klasörleri: {image_dirs}/12 bulundu")
print()

# GPU kontrolü
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
    else:
        print("⚠️  GPU bulunamadı! CPU modunda çalışacak (ÇOK YAVAŞ)")
        print("   Çözüm: Settings → Accelerator → GPU T4 x2")
        response = input("\nDevam etmek istiyor musunuz? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
except:
    print("⚠️  PyTorch GPU kontrolü yapılamadı")

print()

# Split yöntemi - OTOMATIK OLARAK TÜM DATASET (112K - MAXIMUM!)
print("="*70)
print("📊 VERİ SPLIT YÖNTEMİ")
print("="*70)
print("\n✅ TÜM DATASET + MULTI-GPU (MAXIMUM PERFORMANCE!)")
print("   - 112,120 görüntü (TÜM NIH DATASET) + EfficientNet-B2")
print("   - 1 GPU")
print("   - Patient-level split (data leakage ÖNLENDİ)")
print("   - 300×300 piksel (224'ten daha iyi detay, dengeli!)")
print("   - MEDIUM augmentation + TTA (Overfitting önleme!)")
print("   - NUM_WORKERS: 4 (Kaggle T4 optimal)")
print("   - Batch: 36 ")
print("   - Dropout: 0.55 (Piksel artışı için)")
print("   - Beklenen AUC: 0.86-0.88+ (256x256 + MEDIUM + TTA!)")
print("   - Tahmini süre: ~9.75 saat (eğitim) + 0.75h (TTA) = 10.5h (12 saat limiti içinde ✅)")
print()
print("ℹ️  Not: Bu, NIH dataset'indeki tüm görüntüleri kullanıyor!")
print("="*70)
print()

# Otomatik olarak kendi split'i kullan
data_prep_script = "01_data_preparation.py"
use_official_split = False

# ═══════════════════════════════════════════════════════════════
# ADIM 1: VERİ HAZIRLAMA
# ═══════════════════════════════════════════════════════════════
print("="*70)
if use_official_split:
    print("ADIM 1/3: VERİ HAZIRLAMA (RESMİ NIH SPLIT)")
    print("="*70)
    print("📂 train_val_list.txt + test_list.txt kullanılıyor...")
    print("📊 ~80K-90K görüntü")
    print("✅ Patient-level split (README: resmi split patient-level)")
else:
    print("ADIM 1/3: VERİ HAZIRLAMA (112K - TÜM DATASET)")
    print("="*70)
    print("🔍 112,120 görüntü seçiliyor (TÜM NIH DATASET)...")
    print("⚖️  Stratified sampling...")
    print("👨‍⚕️ Patient-level split (data leakage ÖNLENDİ!)")
    print("📊 Tüm hastalıklar dahil")
    print("📊 Train/Val/Test: 70%/15%/15%")
print()
print("⏱️  Süre: ~10 dakika")
print("-"*70)

start_time = time.time()

result = subprocess.run(
    [sys.executable, data_prep_script],
    capture_output=False
)

if result.returncode != 0:
    print(f"\n❌ Veri hazırlama hatası! Kod: {result.returncode}")
    sys.exit(1)

prep_time = (time.time() - start_time) / 60
print(f"\n✅ Veri hazırlama tamamlandı! ({prep_time:.1f} dk)")
print("="*70)

# CSV kontrolü
import pandas as pd

print("\n📊 Oluşturulan CSV dosyaları:")
if use_official_split:
    csv_files = ['train_official.csv', 'val_official.csv', 'test_official.csv']
else:
    # CSV dosyaları config.TOTAL_IMAGES'a göre isimlendiriliyor
    csv_suffix = f"{112}k"  # 112K için 112k (hard-coded, config import edilmemiş)
    csv_files = [f'train_{csv_suffix}.csv', f'val_{csv_suffix}.csv', f'test_{csv_suffix}.csv']

for csv_name in csv_files:
    csv_path = f"/kaggle/working/{csv_name}"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  ✓ {csv_name}: {len(df):,} görüntü")
    else:
        print(f"  ❌ {csv_name} bulunamadı!")
        sys.exit(1)

# Resmi split kullanıldıysa, CSV dosyalarını standart isimlere kopyala
# Çünkü 04_train.py ve 05_evaluate.py dosyaları train_Xk.csv formatını bekliyor
if use_official_split:
    print("\n📋 CSV dosyaları standart formata dönüştürülüyor...")
    import shutil

    # Toplam görüntü sayısını hesapla
    train_df = pd.read_csv("/kaggle/working/train_official.csv")
    val_df = pd.read_csv("/kaggle/working/val_official.csv")
    test_df = pd.read_csv("/kaggle/working/test_official.csv")
    total_images = len(train_df) + len(val_df) + len(test_df)

    # config.py'yi güncelle (TOTAL_IMAGES değerini)
    print(f"  ℹ️  Toplam: {total_images:,} görüntü")
    print(f"  ⚠️  Not: config.TOTAL_IMAGES {total_images} olarak ayarlanmalı")

    # Eğitim scriptleri bu isimleri bekliyor
    suffix = f"{total_images//1000}k"
    shutil.copy("/kaggle/working/train_official.csv", f"/kaggle/working/train_{suffix}.csv")
    shutil.copy("/kaggle/working/val_official.csv", f"/kaggle/working/val_{suffix}.csv")
    shutil.copy("/kaggle/working/test_official.csv", f"/kaggle/working/test_{suffix}.csv")

    print(f"  ✓ train_{suffix}.csv oluşturuldu")
    print(f"  ✓ val_{suffix}.csv oluşturuldu")
    print(f"  ✓ test_{suffix}.csv oluşturuldu")

# ═══════════════════════════════════════════════════════════════
# ADIM 2: MODEL EĞİTİMİ
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADIM 2/3: MODEL EĞİTİMİ (112K - 36 EPOCH - TEK GPU + 300x300!)")
print("="*70)
print("🏗️  EfficientNet-B3 ")
print("📊 Train: ~78,500 görüntü (TÜM DATASET)")
print("📊 Val: ~16,100 görüntü")
print("📏 Image: 256×256 (Daha iyi detay için!)")
print("🔧 Batch: 42 (256x256 için optimal) | Dropout: 0.58 | LR: 0.0003")
print("🔧 Augmentation: MEDIUM + TTA (Overfitting önleme!)")
print("🔧 NUM_WORKERS: 4 (Kaggle T4 optimal)")
print("🔧 TTA: OTOMATİK (eğitim bitince +0.01-0.02 AUC)")
print("🔧 Focal Loss + Class Weights")
print("🔧 Cosine Annealing + Warmup (2 epoch)")
print("🔧 Backbone Freeze: 3 epoch → Unfreeze")
print("🔧 Mixed Precision: ✅")
print("🔧 Attention Fusion: ✅")
print("🔧 Multi-GPU: ❌ KAPALI (Tek GPU daha hızlı!)")
print()
print("🎯 Beklenen AUC: 0.86-0.88+ (256x256 + MEDIUM + dropout 0.58!)")
print("⏱️  Süre: ~9.75 saat (eğitim) + 0.75h (TTA) = 10.5 saat total")
print("⚠️  Notebook'u kapatabilirsiniz (Kaggle arka planda devam eder)")
print("-"*70)

train_start = time.time()

result = subprocess.run(
    [sys.executable, "04_train.py"],
    capture_output=False
)

train_time = (time.time() - train_start) / 3600

if result.returncode == 0:
    print(f"\n✅ Eğitim tamamlandı! ({train_time:.1f} saat)")
else:
    print(f"\n⚠️  Eğitim sonlandı (kod: {result.returncode})")
    print("   Model yine de kaydedilmiş olabilir, devam ediliyor...")

print("="*70)

# Model kontrolü
model_path = "/kaggle/working/models/best_model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    print("\n📊 En İyi Model:")
    print(f"  Epoch: {checkpoint['epoch']+1}")
    print(f"  Val AUC: {checkpoint['val_auc']:.4f}")

    if checkpoint['val_auc'] >= 0.86:
        print(f"  🎯 HEDEF BAŞARILI: {checkpoint['val_auc']:.4f} >= 0.86 ✅")
    elif checkpoint['val_auc'] >= 0.84:
        print(f"  ✅ İYİ: {checkpoint['val_auc']:.4f} >= 0.84")
    else:
        print(f"  ⚠️  Hedefin altında: {checkpoint['val_auc']:.4f}")
else:
    print("\n❌ Model bulunamadı!")
    print("   Eğitim tamamlanmamış olabilir")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# ADIM 3: DEĞERLENDİRME
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADIM 3/3: MODEL DEĞERLENDİRME (TEST)")
print("="*70)
print("🔬 Test: ~9,000 görüntü")
print("📊 Metrikler: AUC, F1, Sensitivity, Specificity...")
print("📈 ROC curves + Confusion matrices")
print()
print("⏱️  Süre: ~10 dakika")
print("-"*70)

eval_start = time.time()

result = subprocess.run(
    [sys.executable, "05_evaluate.py"],
    capture_output=False
)

eval_time = (time.time() - eval_start) / 60

if result.returncode == 0:
    print(f"\n✅ Değerlendirme tamamlandı! ({eval_time:.1f} dk)")
else:
    print(f"\n❌ Değerlendirme hatası!")

print("="*70)

# Sonuçları göster
results_path = "/kaggle/working/results/test_metrics.csv"
if os.path.exists(results_path):
    df_metrics = pd.read_csv(results_path, index_col=0)

    print("\n📊 TEST SONUÇLARI:")
    macro_auc = df_metrics['AUC'].mean()
    macro_f1 = df_metrics['F1'].mean()

    print(f"  Macro AUC: {macro_auc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")

    if macro_auc >= 0.86:
        print(f"  🎯 HEDEF BAŞARILI! AUC >= 0.86 ✅")
    elif macro_auc >= 0.84:
        print(f"  ✅ İYİ PERFORMANS! AUC >= 0.84")
    else:
        print(f"  ⚠️  Hedefin altında: AUC < 0.84")

    print(f"\n  Top 5 Hastalık (En Yüksek AUC):")
    for idx, row in df_metrics.nlargest(5, 'AUC').iterrows():
        print(f"    {idx:20s}: AUC {row['AUC']:.4f}")

    print(f"\n  Bottom 5 Hastalık (En Düşük AUC):")
    for idx, row in df_metrics.nsmallest(5, 'AUC').iterrows():
        print(f"    {idx:20s}: AUC {row['AUC']:.4f}")

    print(f"\n  Hernia (oversampled):")
    if 'Hernia' in df_metrics.index:
        hernia_auc = df_metrics.loc['Hernia', 'AUC']
        hernia_support = df_metrics.loc['Hernia', 'Support']
        print(f"    AUC: {hernia_auc:.4f} | Support: {hernia_support} {'✅' if hernia_auc > 0.70 else '⚠️'}")

# ═══════════════════════════════════════════════════════════════
# TTA DEĞERLENDİRME (OTOMATİK - LIGHT aug için AUC boost!)
# ═══════════════════════════════════════════════════════════════
if os.path.exists("05_evaluate_with_tta.py"):
    print("\n" + "="*70)
    print("BONUS: TEST-TIME AUGMENTATION (TTA) - OTOMATİK")
    print("="*70)
    print("🔮 5 farklı augmentation ile tahmin")
    print("⏱️  Süre: ~25-30 dakika")
    print("🎯 Beklenen AUC artışı: +0.01-0.02")
    print("💡 LIGHT aug kullanıldığı için TTA otomatik çalıştırılıyor!")
    print()

    # Otomatik TTA (kullanıcıya sorma)
    if True:  # Her zaman çalışsın
        print("\n-"*70)
        tta_start = time.time()

        result = subprocess.run(
            [sys.executable, "05_evaluate_with_tta.py"],
            capture_output=False
        )

        tta_time = (time.time() - tta_start) / 60

        if result.returncode == 0:
            print(f"\n✅ TTA tamamlandı! ({tta_time:.1f} dk)")

            # TTA sonuçlarını göster
            tta_results = "/kaggle/working/results/test_metrics_tta.csv"
            if os.path.exists(tta_results):
                df_tta = pd.read_csv(tta_results, index_col=0)
                tta_auc = df_tta['AUC'].mean()

                improvement = tta_auc - macro_auc
                print(f"\n📊 TTA Karşılaştırma:")
                print(f"  Normal AUC: {macro_auc:.4f}")
                print(f"  TTA AUC:    {tta_auc:.4f}")
                print(f"  Kazanç:     {improvement:+.4f} {'✅' if improvement > 0 else '⚠️'}")
        else:
            print(f"\n⚠️  TTA hatası!")

        print("="*70)

# ═══════════════════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════════════════
total_time = (time.time() - start_time) / 3600

print("\n" + "="*70)
print("🎉 PROJE TAMAMLANDI!")
print("="*70)
print(f"\n⏱️  Toplam Süre: {total_time:.1f} saat")
print(f"   Veri Hazırlama: {prep_time:.1f} dk")
print(f"   Eğitim: {train_time:.1f} saat")
print(f"   Değerlendirme: {eval_time:.1f} dk")

if total_time < 12:
    print(f"   ✅ 12 saat limiti içinde! ({12-total_time:.1f} saat kaldı)")
else:
    print(f"   ⚠️  12 saat limitini aştı!")

print("\n📂 Oluşturulan Dosyalar:")
print("  /kaggle/working/")
print("    ├── train_60k.csv")
print("    ├── val_60k.csv")
print("    ├── test_60k.csv")
print("    ├── models/")
print("    │   └── best_model.pth ⭐")
print("    └── results/")
print("        ├── test_metrics.csv")
print("        ├── test_predictions.csv")
print("        ├── roc_curves.png")
print("        └── confusion_matrices.png")

print("\n📥 Dosyaları İndirmek İçin:")
print("  Yöntem 1 (Otomatik):")
print("    from IPython.display import FileLink")
print("    FileLink('/kaggle/working/models/best_model.pth')")
print()
print("  Yöntem 2 (Manuel):")
print("    Output sekmesi → İlgili dosya → Download")

print("\n💡 v5.0 Yeni Özellikler:")
print("  ✅ Patient-level split (data leakage ÖNLENDİ!)")
print("  ✅ Geliştirilmiş demographic encoder (8→12 feature)")
print("  ✅ Attention fusion mechanism")
print("  ✅ Cosine annealing + warmup")
print("  ✅ Backbone freezing stratejisi (3 epoch)")
print("  ✅ Focal Loss + Class Weights")
print("  ✅ 60K veri (50K→60K)")
print("  ✅ Hedef AUC: 0.86 (0.85→0.86)")

print("\n📚 İyileştirme Fikirleri:")
print("  - TTA kullanın: +0.01-0.02 AUC kazancı")
print("  - Ensemble (3-5 model): Daha kararlı sonuçlar")
print("  - EfficientNet-B2/B3: Daha büyük model")
print("  - Epoch sayısını artırın: 25-30 epoch")

print("\n" + "="*70)
print("✨ Başarılar! 🎯")
print("="*70)
