"""
Model Değerlendirme + Test-Time Augmentation (TTA)
TTA: 5 farklı augmentation, ortalamasını al
Beklenen kazanç: +0.01-0.02 AUC
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path
import config
from model import MultimodalChestXrayModel
from dataset import ChestXrayMultimodalDataset


def get_tta_transforms():
    """TTA için farklı augmentation setleri"""
    transforms = []

    # 1. No augmentation (orijinal)
    transforms.append(A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2()
    ]))

    # 2. Horizontal flip
    transforms.append(A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2()
    ]))

    # 3. Slight rotation (+5°)
    transforms.append(A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Rotate(limit=5, p=1.0),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2()
    ]))

    # 4. Slight rotation (-5°)
    transforms.append(A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Rotate(limit=(-5, -5), p=1.0),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2()
    ]))

    # 5. Brightness adjustment
    transforms.append(A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2()
    ]))

    return transforms


def evaluate_with_tta(model, test_dataset, device, n_tta=5, batch_size=8):
    """
    TTA ile değerlendirme
    Dataset'ten direkt okuyup her görüntüyü n_tta kez augment eder
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_image_ids = []

    print(f"\n🔮 Test-Time Augmentation (TTA) ile tahmin yapılıyor...")
    print(f"  Augmentation sayısı: {n_tta}")
    print(f"  Test örnekleri: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")

    tta_transforms = get_tta_transforms()[:n_tta]

    # Her sample için TTA
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc='TTA Evaluation'):
            sample = test_dataset[idx]

            demographics = sample['demographics'].unsqueeze(0).to(device)  # (1, 12)
            labels = sample['labels'].numpy()
            image_id = sample['image_id']

            # Orijinal görüntüyü al (PIL/numpy format - augmentation için)
            # Dataset'ten raw image al
            from PIL import Image
            img_path = test_dataset._find_image_path(image_id)
            original_img = np.array(Image.open(img_path).convert('RGB'))

            sample_preds = []

            # Her TTA augmentation için
            for transform in tta_transforms:
                # Augment + normalize
                augmented = transform(image=original_img)
                img_tensor = augmented['image'].unsqueeze(0).to(device)  # (1, 3, 224, 224)

                # Forward pass
                logits = model(img_tensor, demographics)
                pred = torch.sigmoid(logits).cpu().numpy()[0]  # (15,)

                sample_preds.append(pred)

            # TTA ortalaması
            pred_avg = np.mean(sample_preds, axis=0)  # (n_tta, 15) → (15,)

            all_preds.append(pred_avg)
            all_labels.append(labels)
            all_image_ids.append(image_id)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print(f"\n✓ {len(all_image_ids)} görüntü için TTA tamamlandı!")

    return all_preds, all_labels, all_image_ids


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Metrik hesaplama"""
    metrics = {}

    for i, disease in enumerate(config.DISEASES):
        y_true_disease = y_true[:, i]
        y_pred_disease = y_pred[:, i]
        y_pred_binary = (y_pred_disease >= threshold).astype(int)

        if np.sum(y_true_disease) > 0:
            try:
                auc = roc_auc_score(y_true_disease, y_pred_disease)
                ap = average_precision_score(y_true_disease, y_pred_disease)
            except:
                auc = 0.0
                ap = 0.0
        else:
            auc = 0.0
            ap = 0.0

        f1 = f1_score(y_true_disease, y_pred_binary, zero_division=0)

        metrics[disease] = {
            'AUC': auc,
            'AP': ap,
            'F1': f1,
            'Support': int(np.sum(y_true_disease))
        }

    return metrics


def main():
    """Ana fonksiyon"""
    print("="*70)
    print("🔬 MODEL DEĞERLENDİRME (WITH TTA)")
    print("="*70)

    # Model yükle
    print("\n📂 Model yükleniyor...")
    checkpoint_path = Path(config.MODELS_DIR) / 'best_model.pth'

    if not checkpoint_path.exists():
        print(f"❌ Model bulunamadı: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)

    model = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES,
        demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=False,
        dropout=config.DROPOUT_RATE,
        use_attention=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()

    print(f"✓ Model yüklendi (Val AUC: {checkpoint['val_auc']:.4f})")

    # Test dataset
    print("\n📂 Test dataset yükleniyor...")
    csv_suffix = f"{config.TOTAL_IMAGES//1000}k"
    test_csv = Path(config.OUTPUT_DIR) / f'test_{csv_suffix}.csv'

    test_dataset = ChestXrayMultimodalDataset(
        csv_file=str(test_csv),
        img_dir=config.IMAGES_BASE_DIR,
        mode='test'
    )

    print(f"✓ Test dataset yüklendi: {len(test_dataset)} görüntü")

    # TTA ile tahmin (dataset'i direkt kullanıyoruz, dataloader değil)
    y_pred_tta, y_true, image_ids = evaluate_with_tta(
        model, test_dataset, config.DEVICE, n_tta=5
    )

    # Metrikleri hesapla
    metrics_tta = calculate_metrics(y_true, y_pred_tta)

    # Sonuçları yazdır
    print("\n" + "="*70)
    print("📈 TTA SONUÇLARI")
    print("="*70)

    macro_auc = np.mean([v['AUC'] for v in metrics_tta.values()])
    print(f"\n🎯 Macro AUC (with TTA): {macro_auc:.4f}")

    # Normal tahminle karşılaştır (varsa)
    normal_results = Path(config.RESULTS_DIR) / 'test_metrics.csv'
    if normal_results.exists():
        df_normal = pd.read_csv(normal_results, index_col=0)
        normal_auc = df_normal['AUC'].mean()
        improvement = macro_auc - normal_auc

        print(f"📊 Karşılaştırma:")
        print(f"  Normal AUC:  {normal_auc:.4f}")
        print(f"  TTA AUC:     {macro_auc:.4f}")
        print(f"  Kazanç:      {improvement:+.4f} {'✅' if improvement > 0 else '⚠️'}")

    # Hastalık bazında
    print(f"\n📋 Hastalık Bazında (Top 5):")
    sorted_metrics = sorted(metrics_tta.items(), key=lambda x: x[1]['AUC'], reverse=True)
    for disease, vals in sorted_metrics[:5]:
        print(f"  {disease:20s}: AUC {vals['AUC']:.4f}")

    # Kaydet
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)

    df_metrics_tta = pd.DataFrame(metrics_tta).T
    df_metrics_tta.to_csv(results_dir / 'test_metrics_tta.csv')

    print(f"\n✅ TTA sonuçları kaydedildi:")
    print(f"  {results_dir / 'test_metrics_tta.csv'}")

    print("\n" + "="*70)
    print("🎉 TTA DEĞERLENDİRME TAMAMLANDI!")
    print("="*70)


if __name__ == '__main__':
    main()
