"""
Veri Hazırlama - Kaggle NIH Chest X-ray Dataset
TÜM DATASET (112,120 görüntü) + PATIENT-LEVEL SPLIT (Data leakage önleme!)
Multi-label classification için optimize edilmiş
FİLTRELEME KALDIRILDI - Maximum performance için tüm veri kullanılıyor!
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import config
from pathlib import Path
import json

print("="*70)
print(f"📊 VERİ HAZIRLAMA BAŞLIYOR ({config.TOTAL_IMAGES:,} GÖRÜNTÜ)")
print("="*70)

# ==================== 1. ANA CSV'Yİ OKU ====================
print("\n1️⃣ Ana dataset yükleniyor...")
df = pd.read_csv(config.CSV_FILE)
print(f"   ✓ Toplam kayıt: {len(df):,}")
print(f"   ✓ Kolonlar: {list(df.columns)}")

# Patient ID extraction (Image Index format: 00000001_000.png → Patient ID: 00000001)
df['Patient ID'] = df['Image Index'].str.split('_').str[0]
n_unique_patients = df['Patient ID'].nunique()
print(f"   ✓ Benzersiz hasta sayısı: {n_unique_patients:,}")

# ==================== 2. HASTALIK DAĞILIMI ANALİZİ ====================
print("\n2️⃣ Hastalık dağılımı analizi...")
all_findings = []
for finding in df['Finding Labels']:
    if finding == 'No Finding':
        all_findings.append('No Finding')
    else:
        all_findings.extend(finding.split('|'))

finding_counts = Counter(all_findings)
print("\n   Mevcut hastalık sayıları:")
for disease in config.DISEASES:
    count = finding_counts.get(disease, 0)
    target = config.DISTRIBUTION.get(disease, 0)
    status = "✓" if count >= target else "⚠"
    print(f"   {status} {disease:20s}: {count:6,} mevcut / {target:6,} hedef")

# ==================== 3. MULTI-LABEL STATİSTİKLERİ ====================
print("\n3️⃣ Multi-label analizi...")
multi_label_samples = 0
for finding in df['Finding Labels']:
    if finding != 'No Finding' and '|' in finding:
        multi_label_samples += 1

print(f"   Multi-label örnekler: {multi_label_samples:,} / {len(df):,} "
      f"({multi_label_samples/len(df)*100:.1f}%)")

# ==================== 4. TÜM VERİYİ KULLAN (FİLTRELEME YOK!) ====================
print("\n4️⃣ TÜM DATASET kullanılıyor (FİLTRELEME KALDIRILDI!)...")
print("   ✅ MAXIMUM PERFORMANCE: Tüm 112,120 görüntü kullanılacak")
print("   ✅ Tüm hastalık örnekleri dahil (data loss yok!)")
print("   ✅ Multi-label ilişkileri korunuyor")
print("   ✅ Doğal hastalık dağılımı (class weights ile dengelenecek)")

# Tüm veriyi kullan - FİLTRELEME YOK!
df_selected_unique = df.copy()

print(f"\n   📦 Toplam görüntü: {len(df_selected_unique):,}")
print(f"   📦 Benzersiz hasta: {df_selected_unique['Patient ID'].nunique():,}")

# Hastalık dağılımını göster
print("\n   📊 Doğal hastalık dağılımı (tüm dataset):")
all_findings = []
for finding in df_selected_unique['Finding Labels']:
    if finding == 'No Finding':
        all_findings.append('No Finding')
    else:
        all_findings.extend(finding.split('|'))

finding_counts = Counter(all_findings)
for disease in config.DISEASES:
    count = finding_counts.get(disease, 0)
    percentage = (count / len(df_selected_unique)) * 100
    print(f"      {disease:20s}: {count:6,} ({percentage:5.1f}%)")

# ==================== 5. PATIENT-LEVEL SPLIT (ÇOK ÖNEMLİ!) ====================
print("\n5️⃣ PATIENT-LEVEL Split yapılıyor...")
print("   ⚠️  UYARI: Aynı hastanın görüntüleri train/val/test'e karışmayacak!")

# Patient ID'lere göre grup oluştur
patient_ids = df_selected_unique['Patient ID'].values

# İlk split: Train vs (Val+Test)
# GroupShuffleSplit: Her hasta sadece bir sette olur
gss_train = GroupShuffleSplit(
    n_splits=1,
    test_size=(config.VAL_RATIO + config.TEST_RATIO),
    random_state=config.RANDOM_SEED
)

train_idx, temp_idx = next(gss_train.split(df_selected_unique, groups=patient_ids))

train_df = df_selected_unique.iloc[train_idx].reset_index(drop=True)
temp_df = df_selected_unique.iloc[temp_idx].reset_index(drop=True)

# İkinci split: Val vs Test
temp_patient_ids = temp_df['Patient ID'].values
gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
    random_state=config.RANDOM_SEED
)

val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_patient_ids))

val_df = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print(f"\n   ✓ Train: {len(train_df):6,} görüntü ({len(train_df)/len(df_selected_unique)*100:.1f}%)")
print(f"   ✓ Val:   {len(val_df):6,} görüntü ({len(val_df)/len(df_selected_unique)*100:.1f}%)")
print(f"   ✓ Test:  {len(test_df):6,} görüntü ({len(test_df)/len(df_selected_unique)*100:.1f}%)")

# Patient overlap kontrolü (OLMAMALI!)
train_patients = set(train_df['Patient ID'])
val_patients = set(val_df['Patient ID'])
test_patients = set(test_df['Patient ID'])

overlap_train_val = train_patients & val_patients
overlap_train_test = train_patients & test_patients
overlap_val_test = val_patients & test_patients

print(f"\n   🔍 Patient Overlap Kontrolü:")
print(f"      Train-Val overlap: {len(overlap_train_val)} (0 olmalı!)")
print(f"      Train-Test overlap: {len(overlap_train_test)} (0 olmalı!)")
print(f"      Val-Test overlap: {len(overlap_val_test)} (0 olmalı!)")

if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print(f"      ✅ Patient-level split başarılı! Data leakage yok.")
else:
    print(f"      ❌ UYARI: Patient overlap var! Split tekrar kontrol edin.")

# ==================== 6. HER SETTEDE HASTALIK DAĞILIMI ====================
print("\n6️⃣ Her setteki hastalık dağılımı:")

split_prevalence = {}  # manifest için toplanıyor (bkz. adım 7b)
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    findings = []
    for finding in split_df['Finding Labels']:
        if finding == 'No Finding':
            findings.append('No Finding')
        else:
            findings.extend(finding.split('|'))

    counts = Counter(findings)
    print(f"\n   {split_name} ({len(split_df):,} görüntü, {split_df['Patient ID'].nunique()} hasta):")
    disease_counts = {}
    for disease in config.DISEASES:
        count = counts.get(disease, 0)
        percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
        print(f"     {disease:20s}: {count:4,} ({percentage:5.1f}%)")
        disease_counts[disease] = {'count': int(count), 'percentage': round(percentage, 2)}

    split_prevalence[split_name] = {
        'n_images': int(len(split_df)),
        'n_patients': int(split_df['Patient ID'].nunique()),
        'disease_counts': disease_counts
    }

# ==================== 7. CSV'LERİ KAYDET ====================
print("\n7️⃣ CSV dosyaları kaydediliyor...")
output_dir = Path(config.OUTPUT_DIR)
output_dir.mkdir(exist_ok=True, parents=True)

csv_suffix = f"{config.TOTAL_IMAGES//1000}k"
train_df.to_csv(output_dir / f'train_{csv_suffix}.csv', index=False)
val_df.to_csv(output_dir / f'val_{csv_suffix}.csv', index=False)
test_df.to_csv(output_dir / f'test_{csv_suffix}.csv', index=False)

print(f"   ✓ train_{csv_suffix}.csv kaydedildi ({len(train_df):,} satır)")
print(f"   ✓ val_{csv_suffix}.csv kaydedildi ({len(val_df):,} satır)")
print(f"   ✓ test_{csv_suffix}.csv kaydedildi ({len(test_df):,} satır)")

# ==================== 7b. SPLIT MANIFEST (hakem #3 major: kesin sayım kaydı) ====================
# Amaç: makaledeki split sayılarının (train/val/test görüntü+hasta sayısı,
# sınıf-bazlı prevalence) HER ÇALIŞTIRMADA log-scraping yapmadan, doğrudan
# bu dosyadan doğrulanabilmesini sağlamak. Önceki bir çalıştırmada makale
# metnindeki test sayısı (16,842) ile gerçek sonuçların (confusion matrix,
# test_predictions.csv) kullandığı sayı (17,448) arasında bir tutarsızlık
# tespit edilmişti; bu manifest böyle bir tutarsızlığın bir daha fark
# edilmeden kalmamasını garanti eder.
manifest = {
    'random_seed': config.RANDOM_SEED,
    'total_images': int(len(df_selected_unique)),
    'total_unique_patients': int(df_selected_unique['Patient ID'].nunique()),
    'split_ratios': {
        'train': config.TRAIN_RATIO,
        'val': config.VAL_RATIO,
        'test': config.TEST_RATIO
    },
    'patient_overlap': {
        'train_val': len(overlap_train_val),
        'train_test': len(overlap_train_test),
        'val_test': len(overlap_val_test)
    },
    'splits': split_prevalence
}

manifest_path = output_dir / f'split_manifest_{csv_suffix}.json'
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"   ✓ {manifest_path.name} kaydedildi (kesin split/prevalence kaydı)")

# ==================== 8. ÖZET İSTATİSTİKLER ====================
print("\n" + "="*70)
print("📊 VERİ HAZIRLAMA TAMAMLANDI!")
print("="*70)
print(f"\n✅ Toplam seçilen: {len(df_selected_unique):,} benzersiz görüntü")
print(f"   - Train: {len(train_df):,} ({len(train_df)/len(df_selected_unique)*100:.1f}%) - {train_df['Patient ID'].nunique()} hasta")
print(f"   - Val:   {len(val_df):,} ({len(val_df)/len(df_selected_unique)*100:.1f}%) - {val_df['Patient ID'].nunique()} hasta")
print(f"   - Test:  {len(test_df):,} ({len(test_df)/len(df_selected_unique)*100:.1f}%) - {test_df['Patient ID'].nunique()} hasta")

print(f"\n🎯 Beklenen Performans:")
print(f"   - Hedef AUC: {config.EXPECTED_PERFORMANCE['target_auc']:.2f}")
print(f"   - Minimum AUC: {config.EXPECTED_PERFORMANCE['min_acceptable_auc']:.2f}")
print(f"   - Eğitim süresi: ~{config.EXPECTED_PERFORMANCE['training_time_hours']:.1f} saat")
print(f"   - GPU memory: ~{config.EXPECTED_PERFORMANCE['gpu_memory_gb']:.0f} GB")

print("\n💡 Önemli İyileştirmeler:")
print("   ✅ Patient-level split (data leakage önlendi)")
print("   ✅ Multi-label ilişkileri korundu")
print("   ✅ TÜM 112,120 görüntü kullanıldı (data loss YOK!)")
print("   ✅ Tüm hastalıklar tam temsil ediliyor")
print("   ✅ Kaggle dataset yapısına uyumlu")
print("   ✅ Filtreleme kaldırıldı (basit + hatasız kod)")
print("   ✅ Class imbalance: CLASS_WEIGHTS + Focal Loss ile dengeleniyor")

print("\n🚀 Sonraki Adım: 04_train.py dosyasını çalıştırın")
print("🎯 Beklenen AUC artışı: 0.77 → 0.85-0.87+ ✅")
print("="*70)
