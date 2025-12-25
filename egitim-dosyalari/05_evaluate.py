"""
Model Değerlendirme - Test Seti Üzerinde
Detaylı metrikler, ROC eğrileri, confusion matrices
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import config
from model import MultimodalChestXrayModel
from dataset import ChestXrayMultimodalDataset


def evaluate_model(model, test_loader, device):
    """Model değerlendirme"""
    model.eval()
    all_preds = []
    all_labels = []
    all_image_ids = []
    
    print("\n🔮 Test seti üzerinde tahminler yapılıyor...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            images = batch['image'].to(device)
            demographics = batch['demographics'].to(device)
            labels = batch['labels']
            image_ids = batch['image_id']
            
            # Forward pass
            logits = model(images, demographics)
            preds = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_image_ids.extend(image_ids)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    print(f"✓ {len(all_image_ids)} görüntü için tahmin yapıldı")
    
    return all_preds, all_labels, all_image_ids


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Tüm metrikleri hesapla"""
    print("\n📊 Metrikler hesaplanıyor...")
    
    metrics = {}
    
    for i, disease in enumerate(config.DISEASES):
        y_true_disease = y_true[:, i]
        y_pred_disease = y_pred[:, i]
        y_pred_binary = (y_pred_disease >= threshold).astype(int)
        
        # Sadece pozitif örnekler varsa hesapla
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
        
        # Diğer metrikler
        f1 = f1_score(y_true_disease, y_pred_binary, zero_division=0)
        precision = precision_score(y_true_disease, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_disease, y_pred_binary, zero_division=0)
        
        # Sensitivity ve Specificity
        tn, fp, fn, tp = confusion_matrix(y_true_disease, y_pred_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[disease] = {
            'AUC': auc,
            'AP': ap,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Support': int(np.sum(y_true_disease))
        }
    
    return metrics


def plot_roc_curves(y_true, y_pred, save_path):
    """ROC eğrilerini çiz"""
    print("\n📊 ROC eğrileri çiziliyor...")
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.ravel()
    
    for i, disease in enumerate(config.DISEASES):
        ax = axes[i]
        
        y_true_disease = y_true[:, i]
        y_pred_disease = y_pred[:, i]
        
        if np.sum(y_true_disease) > 0:
            fpr, tpr, _ = roc_curve(y_true_disease, y_pred_disease)
            auc = roc_auc_score(y_true_disease, y_pred_disease)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{disease}', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No positive samples', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{disease}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Kaydedildi: {save_path}")


def plot_confusion_matrices(y_true, y_pred, threshold=0.5, save_path=None):
    """Confusion matrices çiz"""
    print("\n📊 Confusion matrices çiziliyor...")
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.ravel()
    
    for i, disease in enumerate(config.DISEASES):
        ax = axes[i]
        
        y_true_disease = y_true[:, i]
        y_pred_disease = (y_pred[:, i] >= threshold).astype(int)
        
        cm = confusion_matrix(y_true_disease, y_pred_disease)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_title(f'{disease}', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Kaydedildi: {save_path}")


def save_results(metrics, predictions_df, results_dir):
    """Sonuçları kaydet"""
    print("\n💾 Sonuçlar kaydediliyor...")
    
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Metrikler CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(results_dir / 'test_metrics.csv')
    print(f"  ✓ Metrikler: {results_dir / 'test_metrics.csv'}")
    
    # Tahminler CSV
    predictions_df.to_csv(results_dir / 'test_predictions.csv', index=False)
    print(f"  ✓ Tahminler: {results_dir / 'test_predictions.csv'}")


def print_results(metrics):
    """Sonuçları ekrana yazdır"""
    print("\n" + "="*70)
    print("📈 TEST SETİ SONUÇLARI")
    print("="*70)
    
    # Tablo başlıkları
    header = f"{'Hastalık':<20} {'AUC':>7} {'AP':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Sens':>7} {'Spec':>7} {'N':>5}"
    print(header)
    print("-" * 85)
    
    # Her hastalık için
    for disease, vals in metrics.items():
        print(f"{disease:<20} {vals['AUC']:>7.4f} {vals['AP']:>7.4f} {vals['F1']:>7.4f} "
              f"{vals['Precision']:>7.4f} {vals['Recall']:>7.4f} {vals['Sensitivity']:>7.4f} "
              f"{vals['Specificity']:>7.4f} {vals['Support']:>5d}")
    
    print("-" * 85)
    
    # Ortalamalar
    macro_auc = np.mean([v['AUC'] for v in metrics.values()])
    macro_ap = np.mean([v['AP'] for v in metrics.values()])
    macro_f1 = np.mean([v['F1'] for v in metrics.values()])
    
    print(f"{'MACRO AVERAGE':<20} {macro_auc:>7.4f} {macro_ap:>7.4f} {macro_f1:>7.4f}")
    
    print("="*70)
    
    # Hedef karşılaştırma
    target_auc = config.EXPECTED_PERFORMANCE['target_auc']
    if macro_auc >= target_auc:
        print(f"\n🎯 HEDEF BAŞARILI: AUC {macro_auc:.4f} >= {target_auc:.2f} ✅")
    else:
        print(f"\n⚠️  Hedefin altında: AUC {macro_auc:.4f} < {target_auc:.2f}")


def main():
    """Ana değerlendirme fonksiyonu"""
    print("="*70)
    print("🔬 MODEL DEĞERLENDİRME")
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
        use_attention=True  # ✅ EKLENDI: Eğitimde True olduğu için burada da True olmalı!
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    print(f"✓ Model yüklendi (Epoch: {checkpoint['epoch']+1}, Val AUC: {checkpoint['val_auc']:.4f})")
    
    # Test dataset yükle
    print("\n📂 Test dataset yükleniyor...")
    csv_suffix = f"{config.TOTAL_IMAGES//1000}k"
    test_csv = Path(config.OUTPUT_DIR) / f'test_{csv_suffix}.csv'
    
    test_dataset = ChestXrayMultimodalDataset(
        csv_file=str(test_csv),
        img_dir=config.IMAGES_BASE_DIR,
        mode='test'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Tahmin yap
    y_pred, y_true, image_ids = evaluate_model(model, test_loader, config.DEVICE)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(y_true, y_pred, threshold=config.CLASSIFICATION_THRESHOLD)
    
    # Sonuçları yazdır
    print_results(metrics)
    
    # Grafikler
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    plot_roc_curves(y_true, y_pred, save_path=results_dir / 'roc_curves.png')
    plot_confusion_matrices(y_true, y_pred, save_path=results_dir / 'confusion_matrices.png')
    
    # Tahminleri DataFrame olarak kaydet
    predictions_data = {'image_id': image_ids}
    for i, disease in enumerate(config.DISEASES):
        predictions_data[f'{disease}_true'] = y_true[:, i]
        predictions_data[f'{disease}_pred'] = y_pred[:, i]
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Kaydet
    save_results(metrics, predictions_df, results_dir)
    
    print("\n" + "="*70)
    print("✅ DEĞERLENDİRME TAMAMLANDI!")
    print("="*70)
    print(f"Sonuçlar kaydedildi: {results_dir}")
    print("="*70)


if __name__ == '__main__':
    main()