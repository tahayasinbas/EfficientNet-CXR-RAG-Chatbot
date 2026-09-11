"""
Multimodal Chest X-ray Model (Improved)
- Image Encoder: EfficientNet-B0/B2/B3 (pretrained ImageNet) - DYNAMIC!
- Demographic Encoder: Enhanced MLP (12 features, bkz. dataset.py:_extract_demographic_features)
- Fusion: ModalityGatingFusion (varsayılan) veya CrossModalSelfAttention (opsiyonel) + MLP
- Multi-label classification (15 diseases)
- Ablation desteği: ABLATION_MODE ('full' | 'image_only' | 'metadata_only') ile
  hakem-istekli baseline karşılaştırmaları (bkz. config.py, run_ablations.py)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import config  # Import config to read PRETRAINED_MODEL


class ImageEncoder(nn.Module):
    """
    EfficientNet image encoder (DYNAMIC: B0/B2/B3)
    Reads model type from config.PRETRAINED_MODEL
    """
    def __init__(self, pretrained=True, model_name=None):
        super(ImageEncoder, self).__init__()

        # Use config.PRETRAINED_MODEL if not specified
        if model_name is None:
            model_name = config.PRETRAINED_MODEL

        # EfficientNet model selection
        if model_name == "efficientnet_b0":
            if pretrained:
                from torchvision.models import EfficientNet_B0_Weights
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                self.efficientnet = models.efficientnet_b0(weights=weights)
            else:
                self.efficientnet = models.efficientnet_b0(weights=None)
            self.num_features = 1280  # B0 output

        elif model_name == "efficientnet_b2":
            if pretrained:
                from torchvision.models import EfficientNet_B2_Weights
                weights = EfficientNet_B2_Weights.IMAGENET1K_V1
                self.efficientnet = models.efficientnet_b2(weights=weights)
            else:
                self.efficientnet = models.efficientnet_b2(weights=None)
            self.num_features = 1408  # B2 output

        elif model_name == "efficientnet_b3":
            if pretrained:
                from torchvision.models import EfficientNet_B3_Weights
                weights = EfficientNet_B3_Weights.IMAGENET1K_V1
                self.efficientnet = models.efficientnet_b3(weights=weights)
            else:
                self.efficientnet = models.efficientnet_b3(weights=None)
            self.num_features = 1536  # B3 output

        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'efficientnet_b0', 'efficientnet_b2', or 'efficientnet_b3'")

        # Remove classifier, keep only feature extractor
        self.efficientnet.classifier = nn.Identity()

        print(f"✅ ImageEncoder: {model_name} loaded (output dim: {self.num_features})")

    def forward(self, x):
        """
        Input: (batch_size, 3, IMG_SIZE, IMG_SIZE)
        Output: (batch_size, num_features)
        """
        features = self.efficientnet(x)
        return features


class DemographicEncoder(nn.Module):
    """
    ENHANCED Demographic Encoder
    Input: 12 features [age_norm, age_log, age_squared, 4x age_bins, 2x gender, 3x view]
    (tam formülasyon: dataset.py:_extract_demographic_features)
    Output: 64-dim dense representation (was 32 → now 64 for more capacity)
    """
    def __init__(self, input_features=12, hidden_features=128, output_features=64):
        super(DemographicEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(input_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Layer 2 (ADDED - more capacity)
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            # Layer 3 (Output)
            nn.Linear(hidden_features, output_features),
            nn.BatchNorm1d(output_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.output_features = output_features

    def forward(self, x):
        """
        Input: (batch_size, 12)
        Output: (batch_size, 64)
        """
        return self.encoder(x)


class ModalityGatingFusion(nn.Module):
    """
    Öğrenilmiş modalite kapılama (gating) mekanizması.

    ÖNEMLİ (hakem #1 madde 2 ve #3 self-attention eleştirisine yanıt):
    Bu modül klasik "self-attention" (Q/K/V, multi-head, tokenization) DEĞİLDİR.
    Görüntü ve demografik özellik vektörlerini birleştirip (concat), üzerine
    küçük bir MLP + softmax uygulayarak İKİ SKALER ağırlık üretir ve her
    modalitenin tüm kanallarını bu tek skalerle ölçekler. Tam matematiksel
    tanım:

        z            = concat(f_img, f_demo)                       ∈ R^(d_img+d_demo)
        h            = ReLU(W1 z + b1)                              ∈ R^((d_img+d_demo)/4)
        [a_img,a_demo] = softmax(W2 h + b2)                          ∈ R^2, a_img+a_demo=1
        output       = concat(a_img * f_img, a_demo * f_demo)      ∈ R^(d_img+d_demo)

    Yani "attention" burada iki modalite BLOĞU arasında bir skaler
    ağırlıklandırma (soft gating) anlamına gelir; token-seviyesinde
    Q/K/V hesaplayan bir öz-dikkat (self-attention) mekanizması değildir.
    Gerçek çoklu-baş öz-dikkat için bkz. CrossModalSelfAttention.
    """
    def __init__(self, img_features, demo_features):
        super(ModalityGatingFusion, self).__init__()

        total_features = img_features + demo_features

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_features // 4, 2),  # 2 weights: img, demo
            nn.Softmax(dim=1)
        )

    def forward(self, img_feat, demo_feat):
        """
        Input:
            img_feat: (B, img_dim)
            demo_feat: (B, demo_dim)
        Output:
            weighted_feat: (B, img_dim + demo_dim)
        """
        # Concatenate
        combined = torch.cat([img_feat, demo_feat], dim=1)

        # Compute attention weights
        weights = self.attention(combined)  # (B, 2)

        # Weight original features
        img_weight = weights[:, 0].unsqueeze(1)  # (B, 1)
        demo_weight = weights[:, 1].unsqueeze(1)  # (B, 1)

        # Apply weights (broadcast)
        weighted_img = img_feat * img_weight
        weighted_demo = demo_feat * demo_weight

        # Return concatenated weighted features
        return torch.cat([weighted_img, weighted_demo], dim=1)


# Geriye dönük uyumluluk: eski kod/checkpoint'ler AttentionFusion adını arayabilir.
# Not: state_dict anahtarları modül ATTRIBUTE adına bağlıdır (örn.
# `attention_fusion.attention.0.weight`), sınıf adına değil — bu yüzden bu
# yeniden adlandırma mevcut best_model.pth'nin yüklenmesini ETKİLEMEZ.
AttentionFusion = ModalityGatingFusion


class CrossModalSelfAttention(nn.Module):
    """
    Gerçek çoklu-baş öz-dikkat (multi-head self-attention) tabanlı füzyon.

    Hakem #3'ün talep ettiği tam mimari tanım:

    1. Tokenization: İki modalite ayrı birer "token" olarak ele alınır ve
       ortak bir boyuta (common_dim) projekte edilir:
           t_img  = W_img  f_img  + b_img    ∈ R^common_dim
           t_demo = W_demo f_demo + b_demo   ∈ R^common_dim
           T = stack([t_img, t_demo])         ∈ R^(B, 2, common_dim)   (sequence_len=2)

    2. Q/K/V + multi-head self-attention (nn.MultiheadAttention, batch_first=True):
           Q = K = V = T
           A = MultiHeadAttention(Q, K, V; num_heads=h)   ∈ R^(B, 2, common_dim)
       Her head boyutu: common_dim / num_heads.

    3. Residual + LayerNorm (standart post-norm transformer bloğu):
           T' = LayerNorm(T + A)                          ∈ R^(B, 2, common_dim)

    4. Çıktı: iki token'ın (görüntü, demografik) düzleştirilmiş birleşimi:
           output = concat(T'[:,0,:], T'[:,1,:])           ∈ R^(B, 2*common_dim)

    config.py: FUSION_TYPE='self_attention', ATTENTION_NUM_HEADS, ATTENTION_COMMON_DIM
    ile aktifleştirilir. Mevcut eğitilmiş checkpoint (ModalityGatingFusion ile
    eğitildi) bu modülle UYUMLU DEĞİLDİR — kullanmak için yeniden eğitim gerekir
    (bkz. run_ablations.py, Kaggle'da çalıştırılmalı).
    """
    def __init__(self, img_features, demo_features, common_dim=256, num_heads=4, dropout=0.1):
        super(CrossModalSelfAttention, self).__init__()

        if common_dim % num_heads != 0:
            raise ValueError(f"common_dim ({common_dim}) num_heads'e ({num_heads}) bölünebilir olmalı")

        self.common_dim = common_dim
        self.img_proj = nn.Linear(img_features, common_dim)
        self.demo_proj = nn.Linear(demo_features, common_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(common_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_dim = 2 * common_dim

    def forward(self, img_feat, demo_feat):
        """
        Input:
            img_feat:  (B, img_dim)
            demo_feat: (B, demo_dim)
        Output:
            fused: (B, 2 * common_dim)
        """
        t_img = self.img_proj(img_feat)    # (B, common_dim)
        t_demo = self.demo_proj(demo_feat)  # (B, common_dim)

        tokens = torch.stack([t_img, t_demo], dim=1)  # (B, 2, common_dim)

        attn_out, _ = self.mha(tokens, tokens, tokens, need_weights=False)  # (B, 2, common_dim)
        tokens = self.norm(tokens + self.dropout(attn_out))  # residual + post-norm

        return tokens.reshape(tokens.size(0), -1)  # (B, 2*common_dim)


class MultimodalChestXrayModel(nn.Module):
    """
    IMPROVED Multimodal Model
    - Image + Demographics fusion (ModalityGatingFusion / CrossModalSelfAttention / concat)
    - Deeper fusion network
    - Multi-label output (15 diseases), her sınıf BAĞIMSIZ sigmoid ile üretilir
      ("No Finding" 14 hastalığın eşik-altı kombinasyonundan TÜRETİLMEZ; 15
      sınıftan biri olarak kendi başına öğrenilir — bkz. dataset.py:_extract_labels)
    - Ablation desteği: ablation_mode ile image-only / metadata-only varyantları
      (hakem #1 madde 1, #3 major ablation talebi)
    """
    def __init__(self, num_diseases=15, demographic_features=12,
                 pretrained=True, dropout=0.5, use_attention=False,
                 ablation_mode=None, fusion_type=None):
        super(MultimodalChestXrayModel, self).__init__()

        # Geriye dönük uyumluluk: ablation_mode/fusion_type verilmezse config'ten
        # ya da eski use_attention bayrağından türet (mevcut checkpoint'in
        # mimarisini birebir yeniden üretmek için).
        self.ablation_mode = ablation_mode or getattr(config, 'ABLATION_MODE', 'full')
        if fusion_type is None:
            fusion_type = getattr(config, 'FUSION_TYPE', 'gating') if use_attention else 'concat'
        self.fusion_type = fusion_type
        self.num_diseases = num_diseases

        if self.ablation_mode not in ('full', 'image_only', 'metadata_only'):
            raise ValueError(f"Bilinmeyen ablation_mode: {self.ablation_mode}")

        # ---- Encoders (ablation_mode'a göre koşullu inşa) ----
        self.image_encoder = None
        self.demographic_encoder = None
        img_features = 0
        demo_features = 0

        if self.ablation_mode in ('full', 'image_only'):
            self.image_encoder = ImageEncoder(pretrained=pretrained)
            img_features = self.image_encoder.num_features

        if self.ablation_mode in ('full', 'metadata_only'):
            self.demographic_encoder = DemographicEncoder(
                input_features=demographic_features,
                hidden_features=128,
                output_features=64
            )
            demo_features = self.demographic_encoder.output_features

        # ---- Fusion (sadece ablation_mode='full' iken birden fazla modalite var) ----
        self.fusion_module = None
        if self.ablation_mode == 'full':
            if self.fusion_type == 'gating':
                self.attention_fusion = ModalityGatingFusion(img_features, demo_features)
                fusion_input = img_features + demo_features
            elif self.fusion_type == 'self_attention':
                self.fusion_module = CrossModalSelfAttention(
                    img_features, demo_features,
                    common_dim=getattr(config, 'ATTENTION_COMMON_DIM', 256),
                    num_heads=getattr(config, 'ATTENTION_NUM_HEADS', 4)
                )
                fusion_input = self.fusion_module.output_dim
            elif self.fusion_type == 'concat':
                fusion_input = img_features + demo_features
            else:
                raise ValueError(f"Bilinmeyen fusion_type: {self.fusion_type}")
        elif self.ablation_mode == 'image_only':
            fusion_input = img_features
        else:  # metadata_only
            fusion_input = demo_features

        # use_attention eski API'yi korumak için tutuluyor (True <=> fusion_type != 'concat')
        self.use_attention = self.ablation_mode == 'full' and self.fusion_type != 'concat'

        # ---- Fusion MLP (sınıflandırma başlığı) ----
        self.fusion = nn.Sequential(
            # Layer 1
            nn.Linear(fusion_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),

            # Layer 3 (ADDED - deeper network)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),

            # Output layer (multi-label)
            nn.Linear(128, num_diseases)
        )

    def forward(self, image=None, demographics=None):
        """
        Input:
            image: (batch_size, 3, IMG_SIZE, IMG_SIZE) — ablation_mode='metadata_only' iken
                   yok sayılır/None olabilir (API tutarlılığı için parametre yine de kabul edilir)
            demographics: (batch_size, 12) — ablation_mode='image_only' iken yok sayılır/None olabilir
        Output:
            logits: (batch_size, num_diseases) - raw scores for multi-label
        """
        if self.ablation_mode == 'image_only':
            combined = self.image_encoder(image)

        elif self.ablation_mode == 'metadata_only':
            combined = self.demographic_encoder(demographics)

        else:  # 'full'
            img_features = self.image_encoder(image)
            demo_features = self.demographic_encoder(demographics)

            if self.fusion_type == 'gating':
                combined = self.attention_fusion(img_features, demo_features)
            elif self.fusion_type == 'self_attention':
                combined = self.fusion_module(img_features, demo_features)
            else:  # concat
                combined = torch.cat([img_features, demo_features], dim=1)

        logits = self.fusion(combined)
        return logits

    def freeze_backbone(self):
        """Freeze EfficientNet backbone (for initial training)"""
        if self.image_encoder is None:
            print("ℹ️  image_encoder yok (metadata_only ablation) — freeze atlanıyor")
            return
        for param in self.image_encoder.efficientnet.parameters():
            param.requires_grad = False
        print("✅ Image encoder (EfficientNet) frozen")

    def unfreeze_backbone(self):
        """Unfreeze EfficientNet backbone (for fine-tuning)"""
        if self.image_encoder is None:
            print("ℹ️  image_encoder yok (metadata_only ablation) — unfreeze atlanıyor")
            return
        for param in self.image_encoder.efficientnet.parameters():
            param.requires_grad = True
        print("✅ Image encoder (EfficientNet) unfrozen")


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Test
    print("="*70)
    print("MODEL TEST")
    print("="*70)

    # Create model
    model = MultimodalChestXrayModel(
        num_diseases=15,
        demographic_features=12,
        pretrained=False,
        dropout=0.5,
        use_attention=False
    )

    # Dummy input
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    demographics = torch.randn(batch_size, 12)

    # Forward pass
    output = model(image, demographics)

    print(f"\n✓ Forward pass successful")
    print(f"  Image shape: {image.shape}")
    print(f"  Demographics shape: {demographics.shape}")
    print(f"  Output shape: {output.shape}")

    # Parameters
    total, trainable = count_parameters(model)
    print(f"\n✓ Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")

    # Test freeze/unfreeze
    print(f"\n✓ Testing freeze/unfreeze:")
    model.freeze_backbone()
    _, trainable_frozen = count_parameters(model)
    print(f"  Trainable (frozen): {trainable_frozen:,}")

    model.unfreeze_backbone()
    _, trainable_unfrozen = count_parameters(model)
    print(f"  Trainable (unfrozen): {trainable_unfrozen:,}")

    print("="*70)
