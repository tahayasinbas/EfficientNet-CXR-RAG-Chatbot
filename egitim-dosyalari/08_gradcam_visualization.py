"""
Grad-CAM Görselleştirme — Sınıflandırıcı Açıklanabilirliği

Hakem #3 (major) yanıtı: "RAG modülü hastalık hakkında genel bilgi verebilir,
ama sınıflandırıcının BELİRLİ bir radyografta NEDEN o kararı verdiğini
açıklamaz. Localization/saliency/counterfactual değerlendirmesi yok" eleştirisi.

Bu script, EfficientNet-B3 görüntü encoder'ının SON KONVOLÜSYON BLOĞU
üzerinden Grad-CAM ısı haritası üretir — hasta-özel, görsel kanıt sağlar.
"Explainable AI" iddiası ancak bu tür hasta-özel görsel kanıtla birlikte
kullanılmalıdır (RAG'in verdiği genel hastalık bilgisiyle KARIŞTIRILMAMALI).

ÖNEMLİ: Bu script'in çalışması için GERÇEK best_model.pth ağırlıkları VE
gerçek NIH göğüs röntgeni görüntüleri gerekir — ikisi de sadece Kaggle
ortamında mevcut (yerel ortamda best_model.pth bir Git-LFS pointer'ı,
gerçek NIH görüntüleri de yok). Bu yüzden Kaggle'da çalıştırılmak üzere
hazırlanmıştır; burada SADECE sözdizimi/mantık incelemesi yapılabildi.

Kullanım (Kaggle):
    python 08_gradcam_visualization.py \
        --checkpoint /kaggle/working/models/best_model.pth \
        --test-csv /kaggle/working/test_112k.csv \
        --img-dir /kaggle/input/data \
        --n-samples 8 \
        --output-dir /kaggle/working/results/gradcam
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
from model import MultimodalChestXrayModel
from dataset import ChestXrayMultimodalDataset


class GradCAM:
    """
    EfficientNet-B3'ün son konvolüsyon bloğu (features[-1]) üzerinden Grad-CAM.

    Matematiksel tanım (Selvaraju et al. 2017):
        A^k        : hedef katmanın k. kanal aktivasyon haritası, (H,W)
        alpha_k    = (1/Z) * sum_{i,j} dY^c / dA^k_{ij}       (global-average-pooled gradyan)
        L_GradCAM  = ReLU( sum_k alpha_k * A^k )               (B, H, W)
    Burada Y^c, hedef hastalık sınıfının (c) sigmoid ÖNCESİ logit'idir.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image, demographics, class_idx):
        self.model.zero_grad()
        logits = self.model(image, demographics)
        score = logits[:, class_idx].sum()
        score.backward()

        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = F.relu((alpha * self.activations).sum(dim=1))    # (B,H,W)

        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()


def overlay_cam_on_image(image_np, cam, alpha=0.4):
    """image_np: (H,W,3) uint8 [0,255]. cam: (h,w) [0,1]."""
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image_np.shape[1], image_np.shape[0]), Image.BILINEAR)) / 255.0
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--test-csv', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--n-samples', type=int, default=8)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    device = config.DEVICE
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES, demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=False, dropout=config.DROPOUT_RATE, use_attention=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # EfficientNet-B3'ün son konv bloğu: efficientnet.features[-1]
    target_layer = model.image_encoder.efficientnet.features[-1]
    gradcam = GradCAM(model, target_layer)

    dataset = ChestXrayMultimodalDataset(csv_file=args.test_csv, img_dir=args.img_dir, mode='test')

    rng = np.random.RandomState(config.RANDOM_SEED)
    indices = rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        demographics = sample['demographics'].unsqueeze(0).to(device)
        image_id = sample['image_id']

        # Gradyan hesaplamak için requires_grad gerekmiyor (backward image_encoder
        # ağırlıklarına akar), ama batchnorm eval modunda kalmalı.
        img_path = dataset._find_image_path(image_id)
        raw_img = np.array(Image.open(img_path).convert('RGB').resize((config.IMG_SIZE, config.IMG_SIZE)))

        true_labels = sample['labels'].numpy()
        # En yüksek gerçek pozitif sınıf (yoksa en yüksek tahmin edilen sınıf) hedeflenir
        positive_idx = np.where(true_labels == 1)[0]
        with torch.no_grad():
            probs_preview = torch.sigmoid(model(image, demographics)).cpu().numpy()[0]
        target_class = int(positive_idx[0]) if len(positive_idx) > 0 else int(np.argmax(probs_preview))

        cam, probs = gradcam.generate(image, demographics, target_class)
        overlay = overlay_cam_on_image(raw_img, cam[0])

        disease_name = config.DISEASES[target_class]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_img); axes[0].set_title(f'{image_id}\n(orijinal)'); axes[0].axis('off')
        axes[1].imshow(overlay)
        axes[1].set_title(f'Grad-CAM: {disease_name} (p={probs[0][target_class]:.3f})')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f'gradcam_{image_id.replace(".png","")}_{disease_name}.png', dpi=150)
        plt.close(fig)
        print(f"✓ {image_id} -> {disease_name} (p={probs[0][target_class]:.3f})")

    print(f"\n✅ Grad-CAM görselleri kaydedildi: {out_dir}")
    print("NOT: Bu görseller RAG'in verdiği genel bilgiden farklı olarak HASTA-ÖZEL")
    print("görsel kanıt sağlar; makalede 'explainable AI' iddiası bu tür çıktılarla desteklenmelidir.")


if __name__ == '__main__':
    main()
