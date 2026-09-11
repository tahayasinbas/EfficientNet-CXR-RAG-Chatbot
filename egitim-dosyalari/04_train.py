"""
Model Eğitimi - Kaggle Optimized
50,000 görüntü, <12 saat, %85+ AUC hedefi
Focal Loss + Class Weights + Cosine Annealing + Backbone Freezing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path
import time
import config
from model import MultimodalChestXrayModel
from dataset import create_dataloaders


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    Helps with class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight  # Class weights

    def forward(self, inputs, targets):
        # BCE with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )

        # Focal term
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class Trainer:
    """
    Enhanced Trainer with:
    - Focal Loss + Class Weights
    - Cosine Annealing + Warmup
    - Backbone Freezing
    - Mixed Precision
    - Early Stopping
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Class weights for imbalanced data
        self.class_weights = self._compute_class_weights()

        # Loss function
        if config.USE_FOCAL_LOSS:
            self.criterion = FocalLoss(
                alpha=config.FOCAL_LOSS_ALPHA,
                gamma=config.FOCAL_LOSS_GAMMA,
                pos_weight=self.class_weights
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        if config.USE_COSINE_ANNEALING:
            # T_max hesapla: Toplam epoch - freeze epoch - warmup epoch
            # Örnek: 14 - 3 - 2 = 9 epoch için cosine annealing
            effective_epochs = max(1, config.EPOCHS - config.FREEZE_BACKBONE_EPOCHS - config.WARMUP_EPOCHS)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=effective_epochs,  # ✅ DÜZELTİLDİ!
                eta_min=config.LR_MIN
            )
            self.warmup_epochs = config.WARMUP_EPOCHS
            print(f"  📉 CosineAnnealingLR: T_max={effective_epochs} (epochs {config.FREEZE_BACKBONE_EPOCHS+config.WARMUP_EPOCHS+1}-{config.EPOCHS})")
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                min_lr=config.LR_MIN,
                verbose=True
            )
            self.warmup_epochs = 0

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.USE_AMP else None

        # Tracking
        self.best_val_auc = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.start_time = time.time()

    def _compute_class_weights(self):
        """
        config.py'de önceden hesaplanmış CLASS_WEIGHTS / NAIVE_CLASS_WEIGHTS
        sözlüğünden config.DISEASES sırasına göre tensor üretir.

        DÜZELTME NOTU: Bu fonksiyon önceden config.DISTRIBUTION'dan bağımsız
        olarak KENDİ İÇİNDE eski (hatalı) formülü tekrar hesaplıyordu —
        config.py'deki CLASS_WEIGHTS sabiti hiç kullanılmıyordu. Artık tek
        kaynak-of-truth config.py; hangi şemanın kullanılacağı
        config.CLASS_WEIGHT_SCHEME ile seçilir ('corrected' varsayılan,
        'naive' sadece ablation/karşılaştırma amaçlı).
        """
        if not config.USE_CLASS_WEIGHTS:
            return None

        weights_dict = (
            config.CLASS_WEIGHTS if config.CLASS_WEIGHT_SCHEME == 'corrected'
            else config.NAIVE_CLASS_WEIGHTS
        )
        weights = [weights_dict[disease] for disease in config.DISEASES]
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        print(f"\n⚖️  Class Weights (scheme={config.CLASS_WEIGHT_SCHEME}):")
        for disease, weight in zip(config.DISEASES, weights):
            print(f"  {disease:20s}: {weight:.3f}")

        return weights_tensor

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * lr_scale

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [TRAIN]')

        for batch in pbar:
            images = batch['image'].to(self.device)
            demographics = batch['demographics'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            # NOT: dtype AÇIKÇA config.AMP_DTYPE (float16) olarak belirtiliyor.
            # T4 (Kaggle GPU'su) native BF16 Tensor Core desteklemiyor; GradScaler
            # de zaten sadece float16 ile anlamlıdır — bu yüzden fiili precision
            # her zaman float16'dır (makalede "bfloat16" geçiyorsa hatalı).
            if config.USE_AMP:
                with autocast('cuda', dtype=config.AMP_DTYPE):
                    logits = self.model(images, demographics)
                    loss = self.criterion(logits, labels)

                # NaN/Inf loss durumunda backward/step'i tamamen ATLA — aksi halde
                # bozuk gradyanlar model ağırlıklarını kalıcı olarak NaN'a
                # çevirebilir (bkz. run_ablations.py self_attention_fusion koşumu,
                # epoch 9'da loss=nan olup bir daha düzelmedi). Bu kontrol
                # backward'dan ÖNCE yapılıyor ki hiç bozuk gradyan hesaplanmasın.
                if not torch.isfinite(loss):
                    print(f"  ⚠️  UYARI: loss NaN/Inf oldu, bu batch atlanıyor (ağırlıklar korunuyor)")
                    continue

                self.scaler.scale(loss).backward()
                # Gradient clipping: fp16'da (özellikle self-attention fusion gibi
                # sayısal olarak daha kırılgan mimarilerde) gradyan patlaması
                # riskini azaltır. unscale_() önce çağrılmalı (GradScaler ile
                # birlikte clip yapmanın doğru sırası).
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images, demographics)
                loss = self.criterion(logits, labels)

                if not torch.isfinite(loss):
                    print(f"  ⚠️  UYARI: loss NaN/Inf oldu, bu batch atlanıyor (ağırlıklar korunuyor)")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            running_loss += loss.item()

            # Predictions for AUC
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        try:
            epoch_auc = roc_auc_score(all_labels, all_preds, average='macro')
        except:
            epoch_auc = 0.0

        return epoch_loss, epoch_auc

    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                demographics = batch['demographics'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(images, demographics)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        try:
            val_auc = roc_auc_score(all_labels, all_preds, average='macro')
        except:
            val_auc = 0.0

        return val_loss, val_auc

    def train(self, num_epochs):
        """Main training loop"""
        print("\n" + "="*70)
        print(f"🚀 EĞİTİM BAŞLIYOR ({config.TOTAL_IMAGES//1000}K GÖRÜNTÜ)")
        print("="*70)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Weight decay: {config.WEIGHT_DECAY}")
        print(f"Dropout: {config.DROPOUT_RATE}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {'✅ (' + str(config.AMP_DTYPE) + ')' if config.USE_AMP else '❌'}")
        print(f"Focal Loss: {'✅' if config.USE_FOCAL_LOSS else '❌'}")
        print(f"Class Weights: {'✅ (scheme=' + config.CLASS_WEIGHT_SCHEME + ')' if config.USE_CLASS_WEIGHTS else '❌'}")
        print(f"Ablation Mode: {config.ABLATION_MODE} | Fusion Type: {config.FUSION_TYPE}")
        print(f"Cosine Annealing: {'✅' if config.USE_COSINE_ANNEALING else '❌'}")
        print(f"Backbone Freeze: {config.FREEZE_BACKBONE_EPOCHS} epochs")
        print(f"Train batches: {len(self.train_loader):,}")
        print(f"Val batches: {len(self.val_loader):,}")
        print(f"\n🎯 Hedef: AUC > {config.EXPECTED_PERFORMANCE['target_auc']:.2f}")
        print(f"⏱️  Tahmini süre: ~{config.EXPECTED_PERFORMANCE['training_time_hours']:.1f} saat")
        print(f"⏰  Maksimum süre: 12 saat")
        print("="*70 + "\n")

        # Checkpoint directory
        checkpoint_dir = Path(config.MODELS_DIR)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        for epoch in range(num_epochs):
            # Time limit check (12 hours)
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours > 11.5:
                print(f"\n⏰ SÜRE LİMİTİ: {elapsed_hours:.1f} saat geçti, eğitim durduruluyor...")
                break

            # Unfreeze backbone after initial epochs
            if epoch == config.FREEZE_BACKBONE_EPOCHS:
                self.model.unfreeze_backbone()

                # Reset optimizer with new parameters
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=config.LEARNING_RATE,
                    weight_decay=config.WEIGHT_DECAY
                )

                # ✅ FIX: Scheduler'ı da yenile! (Gemini'nin tespit ettiği bug!)
                if config.USE_COSINE_ANNEALING:
                    # T_max: Kalan epoch sayısı (warmup hariç)
                    remaining_epochs = config.EPOCHS - epoch - 1  # Epoch 3'ten sonra kalan
                    effective_T = max(1, remaining_epochs)
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=effective_T,  # ✅ DOĞRU HESAPLAMA!
                        eta_min=config.LR_MIN
                    )
                    print(f"  📉 CosineAnnealingLR reset: T_max={effective_T} (epochs {epoch+2}-{config.EPOCHS})")
                else:
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode='max',
                        factor=0.5,
                        patience=3,
                        verbose=True
                    )

                print(f"  ✅ Optimizer VE Scheduler yenilendi (LR: {config.LEARNING_RATE})")

            # Train
            train_loss, train_auc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_auc = self.validate()
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)

            # Scheduler step
            if config.USE_COSINE_ANNEALING:
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()
            else:
                self.scheduler.step(val_auc)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Progress
            elapsed = (time.time() - self.start_time) / 60
            overfitting_gap = train_auc - val_auc

            print(f"\nEpoch {epoch+1}/{num_epochs} - {elapsed:.1f}m ({elapsed/60:.1f}h)")
            print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
            print(f"  Overfitting Gap: {overfitting_gap:.4f} {'⚠️' if overfitting_gap > 0.05 else '✅'}")
            print(f"  LR: {current_lr:.2e}")

            # Best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'config': {
                        'num_diseases': config.NUM_DISEASES,
                        'demographic_features': config.NUM_DEMOGRAPHIC_FEATURES,
                        'dropout': config.DROPOUT_RATE,
                        'ablation_mode': config.ABLATION_MODE,
                        'fusion_type': config.FUSION_TYPE,
                        'class_weight_scheme': config.CLASS_WEIGHT_SCHEME,
                        'amp_dtype': str(config.AMP_DTYPE),
                    }
                }, checkpoint_dir / 'best_model.pth')

                print(f"  🏆 En iyi model kaydedildi! AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc
                }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
                print(f"  💾 Checkpoint kaydedildi: epoch_{epoch+1}.pth")

            # Early stopping
            if self.patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n⚠️  Early stopping! {config.EARLY_STOP_PATIENCE} epoch iyileşme yok.")
                break

        # Final summary
        total_time = (time.time() - self.start_time) / 60
        print("\n" + "="*70)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("="*70)
        print(f"Toplam süre: {total_time:.1f} dakika ({total_time/60:.1f} saat)")
        print(f"En iyi Val AUC: {self.best_val_auc:.4f} (Epoch {np.argmax(self.val_aucs)+1})")

        # Target check
        if self.best_val_auc >= config.EXPECTED_PERFORMANCE['target_auc']:
            print(f"🎯 HEDEF BAŞARILI: {self.best_val_auc:.4f} >= {config.EXPECTED_PERFORMANCE['target_auc']:.2f} ✅")
        else:
            print(f"⚠️  Hedefin altında: {self.best_val_auc:.4f} < {config.EXPECTED_PERFORMANCE['target_auc']:.2f}")
            print(f"   Fark: {config.EXPECTED_PERFORMANCE['target_auc'] - self.best_val_auc:.4f}")

        print("="*70)


def main():
    """Main function"""
    # Config info
    print("\n" + "="*70)
    print("⚙️  KONFİGÜRASYON")
    print("="*70)
    print(f"Proje: {config.PROJECT_NAME} v{config.VERSION}")
    print(f"Veri: {config.TOTAL_IMAGES:,} görüntü")
    print(f"Hastalık sayısı: {config.NUM_DISEASES}")
    print(f"Device: {config.DEVICE}")
    print("="*70)

    # DataLoaders
    csv_suffix = f"{config.TOTAL_IMAGES//1000}k"
    train_csv = Path(config.OUTPUT_DIR) / f'train_{csv_suffix}.csv'
    val_csv = Path(config.OUTPUT_DIR) / f'val_{csv_suffix}.csv'
    test_csv = Path(config.OUTPUT_DIR) / f'test_{csv_suffix}.csv'

    train_loader, val_loader, _ = create_dataloaders(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(test_csv),
        img_dir=config.IMAGES_BASE_DIR,
        batch_size=config.BATCH_SIZE
    )

    # Model
    print("\n" + "="*70)
    print("🗂️  MODEL OLUŞTURULUYOR")
    print("="*70)

    model = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES,
        demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=True,
        dropout=config.DROPOUT_RATE,
        use_attention=True,  # ✅ Fusion aktif (fusion_type config.FUSION_TYPE'tan gelir)
        ablation_mode=config.ABLATION_MODE,
        fusion_type=config.FUSION_TYPE
    )

    # Freeze backbone initially
    if config.FREEZE_BACKBONE_EPOCHS > 0:
        model.freeze_backbone()

    # Multi-GPU DEVRE DIŞI (overhead çok fazla, tek GPU daha hızlı!)
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"🎯 TEK GPU MODU (Overhead yok, daha hızlı!)")
    print(f"{'='*70}")
    print(f"  Mevcut GPU: {num_gpus} adet")
    print(f"  Kullanılacak: 1 GPU (DataParallel overhead önlendi!)")
    print(f"  Batch Size: {config.BATCH_SIZE} (tek GPU'ya)")
    print(f"  ✅ Tek GPU aktif!")
    print(f"{'='*70}\n")
    
    # Multi-GPU kullanma (overhead çok fazla!)
    # if num_gpus > 1:
    #     model = nn.DataParallel(model)

    model = model.to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model oluşturuldu: {config.PRETRAINED_MODEL}")
    print(f"  Toplam parametreler: {total_params:,}")
    print(f"  Eğitilebilir: {trainable_params:,}")
    print("="*70)

    # Train
    trainer = Trainer(model, train_loader, val_loader, config.DEVICE)
    trainer.train(num_epochs=config.EPOCHS)

    print(f"\n✅ Eğitim tamamlandı!")
    print(f"📂 Model: {config.MODELS_DIR}/best_model.pth")
    print(f"🎯 En iyi AUC: {trainer.best_val_auc:.4f}")


if __name__ == '__main__':
    main()
