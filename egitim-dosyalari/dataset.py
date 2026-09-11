"""
MULTIMODAL DATASET - Görüntü + Demografik Veri
Kaggle NIH Chest X-ray Dataset için optimize edilmiş
Multi-label classification + Patient-level split
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import config


class ChestXrayMultimodalDataset(Dataset):
    """
    Chest X-ray görüntüleri + demografik bilgiler
    Multi-label classification için
    """

    def __init__(self, csv_file, img_dir, mode='train'):
        """
        Args:
            csv_file: CSV dosya yolu (train/val/test)
            img_dir: Görüntü base dizini
            mode: 'train', 'val', veya 'test'
        """
        self.df = pd.read_csv(csv_file)
        self.img_base_dir = Path(img_dir)
        self.mode = mode
        self.diseases = config.DISEASES
        self.transform = self._get_transforms()

        # Görüntü dizinlerini ön yükle
        self.image_dirs = self._get_image_directories()

        print(f"✓ {mode.upper()} Dataset yüklendi: {len(self.df):,} görüntü")
        print(f"  Görüntü klasörleri: {len(self.image_dirs)} adet")

        # Multi-label statistics
        self._print_label_statistics()

    def _get_image_directories(self):
        """Kaggle: images_001/images/, images_002/images/, ... images_012/images/"""
        dirs = []

        # images_001 ~ images_012
        for i in range(1, 13):
            subdir = self.img_base_dir / f'images_{i:03d}' / 'images'
            if subdir.exists():
                dirs.append(subdir)

        if len(dirs) == 0:
            # Fallback: Belki direkt images/ klasörü var?
            fallback = self.img_base_dir / 'images'
            if fallback.exists():
                dirs.append(fallback)

        return sorted(dirs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = row['Image Index']

        # Görüntüyü yükle
        try:
            img_path = self._find_image_path(img_filename)
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            # Hata durumunda siyah görüntü (nadiren olur)
            print(f"⚠️  {img_filename}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Transform uygula
        if self.transform:
            image = self.transform(image=image)['image']

        # Demografik özellikler (EXPANDED - 12 features)
        demographics = self._extract_demographic_features(row)
        demographics = torch.FloatTensor(demographics)

        # Etiketler (MULTI-LABEL)
        labels = self._extract_labels(row)
        labels = torch.FloatTensor(labels)

        return {
            'image': image,
            'demographics': demographics,
            'labels': labels,
            'image_id': img_filename
        }

    def _find_image_path(self, img_filename):
        """Görüntüyü images_XXX/images/ klasörlerinde ara"""
        for img_dir in self.image_dirs:
            img_path = img_dir / img_filename
            if img_path.exists():
                return img_path

        raise FileNotFoundError(f"Bulunamadı: {img_filename}")

    def _extract_demographic_features(self, row):
        """
        Demografik özellikleri çıkar ve normalize et (12 boyutlu vektör).

        Hakem #3 (major) sorusuna yanıt: 12 boyutun tam kaynağı ve eksik/geçersiz
        değer yönetimi:

            [0] age_norm     = clip(age,0,120) / 100          — yaş, [0,1.2] aralığına normalize
            [1] age_log      = log1p(clip(age)) / log1p(100)  — log-dönüşüm (uç yaşları yumuşatır)
            [2] age_squared  = age_norm ** 2                  — doğrusal olmayan yaş etkisi
            [3] age_bin_child      = 1[age<18]
            [4] age_bin_young      = 1[18<=age<45]
            [5] age_bin_middle     = 1[45<=age<65]
            [6] age_bin_senior     = 1[age>=65]                — 4 bin birbirini dışlar, toplamı=1
            [7] gender_M     = 1[gender=='M']
            [8] gender_F     = 1[gender=='F']                  — ikisi de 0 ise "bilinmeyen cinsiyet"
            [9]  view_PA     = 1[view=='PA']
            [10] view_AP     = 1[view=='AP']
            [11] view_other  = 1[view not in {PA,AP}]          — L/LL/vb. VEYA eksik değer

        Geçersiz/eksik değer yönetimi:
        - Yaş: NaN veya [0,120] dışına düşen değerler clip edilir (age.py'de
          hiçbir hasta 120'den büyük olamayacağından bu bir güvenlik sınırıdır,
          NIH ChestX-ray14'te bilinen birkaç hatalı '>100' yaş kaydı için de
          koruma sağlar). NaN → dataset ortalaması (yaklaşık 46) ile doldurulur.
        - Cinsiyet: 'M'/'F' dışındaki (veya eksik) değerler için gender_M=gender_F=0
          olarak bırakılır — bu, modele "bilinmeyen cinsiyet" durumunu örtük
          biçimde iletir (aynı mantık backend/main.py:encode_demographics'te
          tutarlı şekilde uygulanır — train/serve skew'i önlemek için).
        - View position: PA/AP dışındaki her değer (L, LL, eksik, vb.)
          view_other=1 ile tek bir "diğer" kategorisine toplanır.
        """
        features = []

        # Yaş: NaN/geçersiz değerleri sanitize et
        age = row['Patient Age']
        if pd.isna(age):
            age = self.df['Patient Age'].mean()
        age = float(np.clip(age, 0, 120))

        # Yaş özellikleri (4 adet)
        features.append(age / 100.0)  # Normalize
        features.append(np.log1p(age) / np.log1p(100))  # Log transform
        features.append((age / 100.0) ** 2)  # Squared term

        # Yaş grupları (4 bins)
        features.append(1.0 if age < 18 else 0.0)    # Çocuk
        features.append(1.0 if 18 <= age < 45 else 0.0)  # Genç yetişkin
        features.append(1.0 if 45 <= age < 65 else 0.0)  # Orta yaş
        features.append(1.0 if age >= 65 else 0.0)   # Yaşlı

        # Cinsiyet (one-hot, 2 adet) — bilinmeyen/eksik değer -> [0,0]
        gender = row['Patient Gender']
        features.append(1.0 if gender == 'M' else 0.0)
        features.append(1.0 if gender == 'F' else 0.0)

        # View position (one-hot, 3 adet) — PA/AP dışı ve eksik -> view_other
        view = row['View Position']
        features.append(1.0 if view == 'PA' else 0.0)
        features.append(1.0 if view == 'AP' else 0.0)
        features.append(1.0 if view not in ['PA', 'AP'] else 0.0)  # L, LL, NaN, vb.

        return np.array(features, dtype=np.float32)

    def _extract_labels(self, row):
        """
        Multi-label etiketleri çıkar
        Output: 15 boyutlu binary vector

        ÖNEMLİ: Bir görüntüde birden fazla hastalık olabilir!
        Örnek: "Infiltration|Effusion" → [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]

        Hakem #2 (madde 3) sorusuna yanıt: "No Finding" ayrı ve BAĞIMSIZ bir
        sınıf olarak (index 0) ele alınır; diğer 14 hastalığın olasılıklarının
        eşik-altında kalmasından TÜRETİLMEZ. Model çıktısı 15 bağımsız sigmoid
        (softmax değil) olduğundan her sınık kendi öğrenilmiş kararını verir;
        "No Finding" olasılığı diğer 14 sınıfın tahminlerinden matematiksel
        olarak bağımsızdır (bkz. model.py:MultimodalChestXrayModel.forward,
        çıkış katmanı nn.Linear(128, num_diseases), aktivasyon yok — sigmoid
        eğitim/değerlendirme kodunda ayrıca uygulanır).
        """
        labels = np.zeros(len(self.diseases), dtype=np.float32)
        findings = row['Finding Labels']

        if findings == 'No Finding':
            # Sadece 'No Finding' varsa, ilk label'ı 1 yap
            labels[0] = 1.0
        else:
            # Multi-label: birden fazla hastalık olabilir
            disease_list = findings.split('|')
            for disease in disease_list:
                disease = disease.strip()
                if disease in self.diseases:
                    idx = self.diseases.index(disease)
                    labels[idx] = 1.0

        return labels

    def _print_label_statistics(self):
        """Multi-label istatistiklerini yazdır"""
        if self.mode != 'train':
            return

        print(f"\n📊 {self.mode.upper()} Multi-label Statistics:")

        # Her hastalığın sayısını hesapla
        disease_counts = {disease: 0 for disease in self.diseases}
        multi_label_count = 0

        for _, row in self.df.iterrows():
            findings = row['Finding Labels']
            if findings != 'No Finding':
                diseases = findings.split('|')
                if len(diseases) > 1:
                    multi_label_count += 1
                for disease in diseases:
                    disease = disease.strip()
                    if disease in disease_counts:
                        disease_counts[disease] += 1
            else:
                disease_counts['No Finding'] += 1

        print(f"  Multi-label örnekler: {multi_label_count} / {len(self.df)} "
              f"({multi_label_count/len(self.df)*100:.1f}%)")

        print(f"\n  Hastalık dağılımı:")
        for disease, count in disease_counts.items():
            pct = count / len(self.df) * 100
            print(f"    {disease:20s}: {count:5,} ({pct:5.1f}%)")

    def _get_transforms(self):
        """Data augmentation ve normalization"""
        if self.mode == 'train' and config.USE_AUGMENTATION:
            strength = config.AUGMENTATION_STRENGTH

            if strength == 'light':
                # Hafif augmentation
                return A.Compose([
                    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.3
                    ),
                    A.Normalize(
                        mean=config.NORMALIZE_MEAN,
                        std=config.NORMALIZE_STD
                    ),
                    ToTensorV2()
                ])

            elif strength == 'medium':
                # Orta seviye augmentation (ÖNERİLEN)
                return A.Compose([
                    A.Resize(config.IMG_SIZE, config.IMG_SIZE),

                    # Geometrik
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.4),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.4
                    ),

                    # Piksel seviye
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=0.5
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=0.3
                    ),
                    A.GaussNoise(
                        var_limit=(5.0, 20.0),
                        p=0.2
                    ),
                    A.CoarseDropout(
                        max_holes=5,
                        max_height=16,
                        max_width=16,
                        min_holes=2,
                        min_height=8,
                        min_width=8,
                        fill_value=0,
                        p=0.2
                    ),

                    A.Normalize(
                        mean=config.NORMALIZE_MEAN,
                        std=config.NORMALIZE_STD
                    ),
                    ToTensorV2()
                ])

            else:  # 'heavy'
                # Ağır augmentation (overfitting çok fazlaysa)
                return A.Compose([
                    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.15,
                        scale_limit=0.15,
                        rotate_limit=20,
                        p=0.5
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=0.5
                    ),
                    A.CLAHE(
                        clip_limit=3.0,
                        tile_grid_size=(8, 8),
                        p=0.4
                    ),
                    A.GaussNoise(
                        var_limit=(5.0, 30.0),
                        p=0.3
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=0.2
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=20,
                        max_width=20,
                        min_holes=3,
                        min_height=8,
                        min_width=8,
                        fill_value=0,
                        p=0.3
                    ),
                    A.Normalize(
                        mean=config.NORMALIZE_MEAN,
                        std=config.NORMALIZE_STD
                    ),
                    ToTensorV2()
                ])
        else:
            # Val/Test: Sadece normalize
            return A.Compose([
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.Normalize(
                    mean=config.NORMALIZE_MEAN,
                    std=config.NORMALIZE_STD
                ),
                ToTensorV2()
            ])


def create_dataloaders(train_csv, val_csv, test_csv, img_dir, batch_size):
    """
    Train/Val/Test dataloader'larını oluştur
    """
    print("\n" + "="*70)
    print("📦 DATALOADER'LAR OLUŞTURULUYOR")
    print("="*70)

    # Dataset'leri oluştur
    train_dataset = ChestXrayMultimodalDataset(train_csv, img_dir, mode='train')
    val_dataset = ChestXrayMultimodalDataset(val_csv, img_dir, mode='val')
    test_dataset = ChestXrayMultimodalDataset(test_csv, img_dir, mode='test')

    # DataLoader'lar
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Random shuffle
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    print(f"\n✓ DataLoader'lar hazır:")
    print(f"  - Train batches: {len(train_loader):,}")
    print(f"  - Val batches: {len(val_loader):,}")
    print(f"  - Test batches: {len(test_loader):,}")
    print("="*70)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test
    print("Dataset test ediliyor...")

    # Test için dummy CSV oluştur
    test_data = {
        'Image Index': ['00000001_000.png', '00000002_000.png'],
        'Finding Labels': ['No Finding', 'Infiltration|Effusion'],
        'Patient Age': [58, 45],
        'Patient Gender': ['M', 'F'],
        'View Position': ['PA', 'AP']
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('/tmp/test.csv', index=False)

    dataset = ChestXrayMultimodalDataset(
        csv_file='/tmp/test.csv',
        img_dir='/kaggle/input/nih-chest-xrays/data',
        mode='train'
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Diseases: {dataset.diseases}")
