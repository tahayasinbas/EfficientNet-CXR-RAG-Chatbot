"""
Multimodal Chest X-ray Model (Improved)
- Image Encoder: EfficientNet-B0/B2/B3 (pretrained ImageNet) - DYNAMIC!
- Demographic Encoder: Enhanced MLP (12 features)
- Fusion: Attention-based + MLP
- Multi-label classification (15 diseases)
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

        print(f"[OK] ImageEncoder: {model_name} loaded (output dim: {self.num_features})")

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


class AttentionFusion(nn.Module):
    """
    Simple attention mechanism to weight image vs demographic features
    """
    def __init__(self, img_features, demo_features):
        super(AttentionFusion, self).__init__()

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


class MultimodalChestXrayModel(nn.Module):
    """
    IMPROVED Multimodal Model
    - Image + Demographics fusion with attention
    - Deeper fusion network
    - Multi-label output (15 diseases)
    """
    def __init__(self, num_diseases=15, demographic_features=12,
                 pretrained=True, dropout=0.5, use_attention=False):
        super(MultimodalChestXrayModel, self).__init__()

        # Encoders
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        img_features = self.image_encoder.num_features  # 1280

        self.demographic_encoder = DemographicEncoder(
            input_features=demographic_features,
            hidden_features=128,  # 64 → 128
            output_features=64    # 32 → 64
        )
        demo_features = self.demographic_encoder.output_features  # 64

        # Attention fusion (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention_fusion = AttentionFusion(img_features, demo_features)

        # Fusion MLP
        fusion_input = img_features + demo_features  # 1280 + 64 = 1344

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

        self.num_diseases = num_diseases

    def forward(self, image, demographics):
        """
        Input:
            image: (batch_size, 3, 224, 224)
            demographics: (batch_size, 12)
        Output:
            logits: (batch_size, 15) - raw scores for multi-label
        """
        # Encode
        img_features = self.image_encoder(image)          # (B, 1280)
        demo_features = self.demographic_encoder(demographics)  # (B, 64)

        # Fusion with optional attention
        if self.use_attention:
            combined = self.attention_fusion(img_features, demo_features)
        else:
            combined = torch.cat([img_features, demo_features], dim=1)  # (B, 1344)

        # Classification
        logits = self.fusion(combined)  # (B, 15)

        return logits

    def freeze_backbone(self):
        """Freeze EfficientNet backbone (for initial training)"""
        for param in self.image_encoder.efficientnet.parameters():
            param.requires_grad = False
        print("[OK] Image encoder (EfficientNet) frozen")

    def unfreeze_backbone(self):
        """Unfreeze EfficientNet backbone (for fine-tuning)"""
        for param in self.image_encoder.efficientnet.parameters():
            param.requires_grad = True
        print("[OK] Image encoder (EfficientNet) unfrozen")


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
