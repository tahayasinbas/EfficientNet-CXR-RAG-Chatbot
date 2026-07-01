"""
Multimodal Chest X-ray Model
- Image Encoder: EfficientNet-B3 (pretrained ImageNet)
- Demographic Encoder: MLP (12 features → 64 dim)
- Fusion: Attention-based + MLP
- Multi-label classification (15 diseases)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import config


class ImageEncoder(nn.Module):
    """
    EfficientNet-B3 image encoder.
    Removes the classifier head; output dim = 1536.
    """

    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()

        if pretrained:
            from torchvision.models import EfficientNet_B3_Weights
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            self.efficientnet = models.efficientnet_b3(weights=weights)
        else:
            self.efficientnet = models.efficientnet_b3(weights=None)

        self.num_features = 1536  # B3 output dimension

        # Remove classifier, keep only feature extractor
        self.efficientnet.classifier = nn.Identity()

        print(f"[OK] ImageEncoder: EfficientNet-B3 loaded (output dim: {self.num_features})")

    def forward(self, x):
        """
        x: (batch_size, 3, IMG_SIZE, IMG_SIZE)
        returns: (batch_size, 1536)
        """
        return self.efficientnet(x)


class DemographicEncoder(nn.Module):
    """
    Demographic Encoder.
    Input: 12 features [age_norm, age_log, age_squared, 4x age_bins, 2x gender, 3x view]
    Output: 64-dim representation
    """

    def __init__(self, input_features=12, hidden_features=128, output_features=64):
        super(DemographicEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(hidden_features, output_features),
            nn.BatchNorm1d(output_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.output_features = output_features

    def forward(self, x):
        """
        x: (batch_size, 12)
        returns: (batch_size, 64)
        """
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """
    Attention mechanism to weight image vs demographic features before fusion.
    """

    def __init__(self, img_features, demo_features):
        super(AttentionFusion, self).__init__()

        total_features = img_features + demo_features  # 1536 + 64 = 1600

        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_features // 4, 2),  # weights for img and demo
            nn.Softmax(dim=1),
        )

    def forward(self, img_feat, demo_feat):
        """
        img_feat:  (B, 1536)
        demo_feat: (B, 64)
        returns:   (B, 1600)  — attention-weighted concatenation
        """
        combined = torch.cat([img_feat, demo_feat], dim=1)   # (B, 1600)
        weights = self.attention(combined)                    # (B, 2)

        weighted_img  = img_feat  * weights[:, 0].unsqueeze(1)
        weighted_demo = demo_feat * weights[:, 1].unsqueeze(1)

        return torch.cat([weighted_img, weighted_demo], dim=1)  # (B, 1600)


class MultimodalChestXrayModel(nn.Module):
    """
    Multimodal model: EfficientNet-B3 image features + demographic features.
    Fusion dim: 1536 (image) + 64 (demographics) = 1600
    Output: 15-class multi-label logits
    """

    def __init__(
        self,
        num_diseases=config.NUM_DISEASES,
        demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=True,
        dropout=config.DROPOUT_RATE,
        use_attention=config.USE_ATTENTION,
    ):
        super(MultimodalChestXrayModel, self).__init__()

        self.image_encoder = ImageEncoder(pretrained=pretrained)
        img_features = self.image_encoder.num_features  # 1536 (B3)

        self.demographic_encoder = DemographicEncoder(
            input_features=demographic_features,
            hidden_features=128,
            output_features=64,
        )
        demo_features = self.demographic_encoder.output_features  # 64

        self.use_attention = use_attention
        if use_attention:
            self.attention_fusion = AttentionFusion(img_features, demo_features)

        fusion_input = img_features + demo_features  # 1536 + 64 = 1600

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),

            nn.Linear(128, num_diseases),
        )

        self.num_diseases = num_diseases

    def forward(self, image, demographics):
        """
        image:        (batch_size, 3, 224, 224)
        demographics: (batch_size, 12)
        returns:      (batch_size, 15)  — raw logits for multi-label
        """
        img_features  = self.image_encoder(image)           # (B, 1536)
        demo_features = self.demographic_encoder(demographics)  # (B, 64)

        if self.use_attention:
            combined = self.attention_fusion(img_features, demo_features)  # (B, 1600)
        else:
            combined = torch.cat([img_features, demo_features], dim=1)    # (B, 1600)

        return self.fusion(combined)  # (B, 15)

    def freeze_backbone(self):
        for param in self.image_encoder.efficientnet.parameters():
            param.requires_grad = False
        print("[OK] EfficientNet-B3 backbone frozen")

    def unfreeze_backbone(self):
        for param in self.image_encoder.efficientnet.parameters():
            param.requires_grad = True
        print("[OK] EfficientNet-B3 backbone unfrozen")


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("=" * 60)
    print("MODEL TEST — EfficientNet-B3")
    print("=" * 60)

    model = MultimodalChestXrayModel(pretrained=False)

    batch_size   = 4
    image        = torch.randn(batch_size, 3, config.IMG_SIZE, config.IMG_SIZE)
    demographics = torch.randn(batch_size, config.NUM_DEMOGRAPHIC_FEATURES)

    output = model(image, demographics)

    print(f"\n[OK] Forward pass successful")
    print(f"     Image input:      {image.shape}")
    print(f"     Demographics:     {demographics.shape}")
    print(f"     Output (logits):  {output.shape}")

    total, trainable = count_parameters(model)
    print(f"\n[OK] Parameters:")
    print(f"     Total:     {total:,}")
    print(f"     Trainable: {trainable:,}")

    model.freeze_backbone()
    _, frozen = count_parameters(model)
    print(f"\n[OK] Frozen backbone trainable params: {frozen:,}")

    model.unfreeze_backbone()
    _, unfrozen = count_parameters(model)
    print(f"[OK] Unfrozen backbone trainable params: {unfrozen:,}")

    print("=" * 60)
