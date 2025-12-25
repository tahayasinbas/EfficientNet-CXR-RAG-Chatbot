"""
Configuration for Multimodal Chest X-ray Model
"""

# Model architecture
PRETRAINED_MODEL = "efficientnet_b3"  # Options: efficientnet_b0, efficientnet_b2, efficientnet_b3
NUM_DISEASES = 15

# Disease classes (15 classes) - ORDER MATTERS! Must match training order.
DISEASE_CLASSES = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
    "Pneumonia",
    "Hernia"
]

# Image preprocessing
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Demographics
DEMOGRAPHIC_FEATURES = 12  # [age_norm, age_log, age_squared, 4x age_bins, 2x gender, 3x view]
NUM_DEMOGRAPHIC_FEATURES = 12  # Alias for compatibility
AGE_BINS = [0, 20, 40, 60, 100]  # Age group boundaries

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DROPOUT = 0.5
DROPOUT_RATE = 0.5  # Alias for compatibility

# Attention
USE_ATTENTION = True  # Set to True to use attention fusion
