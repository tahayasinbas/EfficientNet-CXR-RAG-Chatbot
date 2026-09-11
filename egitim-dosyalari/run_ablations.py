"""
Ablation Driver — Hakem #1 (madde 1) ve #3 (major) yanıtı

Şu karşılaştırmaları TEK Kaggle oturumunda, AYNI split/hiperparametrelerle
(sadece belirtilen değişkenler farklı) sırayla çalıştırıp sonuçları
ablation_results.csv'ye toplar:

  1. full_model              -> mevcut ana model (referans, gating fusion)
  2. image_only               -> sadece EfficientNet-B3 (demografik katkısını izole eder — Hakem #1.1)
  3. metadata_only             -> sadece demografik MLP
  4. concat_no_gating          -> görüntü+demografik, füzyon ağırlıklandırması olmadan
  5. self_attention_fusion     -> CrossModalSelfAttention ile füzyon
  6. corrected_class_weights   -> (=full_model ile aynı; referans için açıkça listelendi)
  7. naive_class_weights       -> ESKİ (hatalı) class-weight formülüyle — bug'ın etkisini gösterir
  8. no_focal_loss             -> plain BCEWithLogitsLoss (focal loss kapalı)
  9. no_augmentation           -> augmentation kapalı

ÖNEMLİ — GPU BÜTÇESİ: 04_train.py'deki tam model 18 epoch'ta ~5.3 saat sürdü
(bkz. kaggle-ciktisi.txt). 9 varyantı TAM epoch bütçesiyle tek 12h Kaggle
oturumuna sığdırmak MÜMKÜN DEĞİL. Bu script config.ABLATION_EPOCHS (varsayılan
10) kullanır ve her preset'i ayrı ayrı, ~2.5-3h/preset bütçesiyle çalıştırır —
yine de TÜMÜNÜ bir oturumda bitirmek zor olabilir; PRESETS listesini bölüp
birden fazla Kaggle oturumuna (haftalık 30h ücretsiz GPU kotası) yaymanız
önerilir (--presets ile alt küme seçin).

Bu script Kaggle'da (gerçek 112K NIH verisiyle) çalıştırılmak üzere
hazırlanmıştır; yerel ortamda veri/GPU bütçesi olmadığı için burada
çalıştırılmadı — sadece statik/mantık incelemesiyle doğrulandı.

Kullanım (Kaggle):
    python run_ablations.py --presets image_only metadata_only concat_no_gating
    python run_ablations.py  # tüm presetler
"""

import argparse
import copy
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).parent
train_mod = _load_module('train_mod', HERE / '04_train.py')
eval_mod = _load_module('eval_mod', HERE / '05_evaluate.py')

from model import MultimodalChestXrayModel
from dataset import create_dataloaders


PRESETS = {
    'full_model': {},  # referans: config.py'deki mevcut varsayılanlar
    'image_only': {'ABLATION_MODE': 'image_only'},
    'metadata_only': {'ABLATION_MODE': 'metadata_only'},
    'concat_no_gating': {'FUSION_TYPE': 'concat'},
    'self_attention_fusion': {'FUSION_TYPE': 'self_attention'},
    'naive_class_weights': {'CLASS_WEIGHT_SCHEME': 'naive'},
    'no_focal_loss': {'USE_FOCAL_LOSS': False},
    'no_augmentation': {'USE_AUGMENTATION': False},
}


def apply_preset(overrides):
    for key, value in overrides.items():
        setattr(config, key, value)


def run_single_ablation(preset_name, overrides, train_csv, val_csv, test_csv, results_rows):
    print("\n" + "#" * 80)
    print(f"# ABLATION: {preset_name}  overrides={overrides}")
    print("#" * 80)

    apply_preset(overrides)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=str(train_csv), val_csv=str(val_csv), test_csv=str(test_csv),
        img_dir=config.IMAGES_BASE_DIR, batch_size=config.BATCH_SIZE
    )

    model = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES, demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=True, dropout=config.DROPOUT_RATE, use_attention=True,
        ablation_mode=config.ABLATION_MODE, fusion_type=config.FUSION_TYPE
    )
    if config.FREEZE_BACKBONE_EPOCHS > 0:
        model.freeze_backbone()
    model = model.to(config.DEVICE)

    trainer = train_mod.Trainer(model, train_loader, val_loader, config.DEVICE)
    t0 = time.time()
    trainer.train(num_epochs=config.ABLATION_EPOCHS)
    train_hours = (time.time() - t0) / 3600

    # KRİTİK: test değerlendirmesi bellekte kalan SON epoch'un ağırlıklarıyla
    # DEĞİL, diskteki EN İYİ (val_auc en yüksek) checkpoint'le yapılmalı —
    # 05_evaluate.py'nin yaptığı gibi. Son epoch'lar (özellikle
    # self_attention_fusion gibi sayısal olarak daha kırılgan varyantlarda)
    # NaN'a diverge edebilir; bellekteki `model` o durumda bozuk olur, ama
    # diskteki best_model.pth (val_auc en yüksekken kaydedilmiş) temiz kalır.
    checkpoint_path = Path(config.MODELS_DIR) / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ℹ️  Test değerlendirmesi için diskteki en iyi checkpoint yeniden yüklendi "
          f"(epoch={checkpoint['epoch']+1}, val_auc={checkpoint['val_auc']:.4f})")

    y_pred, y_true, _ = eval_mod.evaluate_model(model, test_loader, config.DEVICE)
    metrics = eval_mod.calculate_metrics(y_true, y_pred, threshold=config.CLASSIFICATION_THRESHOLD)

    macro_auc = float(np.mean([v['AUC'] for v in metrics.values()]))
    macro_f1 = float(np.mean([v['F1'] for v in metrics.values()]))
    macro_sens = float(np.mean([v['Sensitivity'] for v in metrics.values()]))

    row = {
        'preset': preset_name, 'overrides': str(overrides),
        'best_val_auc': trainer.best_val_auc, 'test_macro_auc': macro_auc,
        'test_macro_f1@0.5': macro_f1, 'test_macro_sensitivity@0.5': macro_sens,
        'train_hours': round(train_hours, 2), 'epochs': config.ABLATION_EPOCHS,
        'ablation_mode': config.ABLATION_MODE, 'fusion_type': config.FUSION_TYPE,
        'class_weight_scheme': config.CLASS_WEIGHT_SCHEME,
        'use_focal_loss': config.USE_FOCAL_LOSS, 'use_augmentation': config.USE_AUGMENTATION
    }
    results_rows.append(row)
    print(f"\n✓ {preset_name}: test_macro_auc={macro_auc:.4f} test_macro_f1@0.5={macro_f1:.4f} "
          f"({train_hours:.2f}h)")

    del model, trainer, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--presets', nargs='*', default=list(PRESETS.keys()),
                     choices=list(PRESETS.keys()),
                     help='Çalıştırılacak preset alt kümesi (varsayılan: hepsi)')
    ap.add_argument('--output', default=str(Path(config.RESULTS_DIR) / 'ablation_results.csv'))
    args = ap.parse_args()

    csv_suffix = f"{config.TOTAL_IMAGES // 1000}k"
    train_csv = Path(config.OUTPUT_DIR) / f'train_{csv_suffix}.csv'
    val_csv = Path(config.OUTPUT_DIR) / f'val_{csv_suffix}.csv'
    test_csv = Path(config.OUTPUT_DIR) / f'test_{csv_suffix}.csv'

    # Her preset'ten sonra config'i tam olarak orijinal haline döndürmek için
    # başlangıç değerlerini sakla (Python modülleri process boyunca paylaşıldığı
    # için preset'ler arası state sızıntısını önlemek amacıyla).
    baseline = {k: copy.deepcopy(getattr(config, k)) for k in
                ['ABLATION_MODE', 'FUSION_TYPE', 'CLASS_WEIGHT_SCHEME', 'USE_FOCAL_LOSS', 'USE_AUGMENTATION']}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ÇOK ÖNEMLİ: Ablation'lar genellikle BİRDEN FAZLA Kaggle oturumuna
    # yayılır (12h limiti + GPU bütçesi). Her yeni oturumda /kaggle/working
    # SIFIRDAN başladığı için, önceki oturumun ürettiği ablation_results.csv
    # burada YOKSA bu run onu kaybeder/üzerine yazar. Bu yüzden: eğer
    # --output yolunda ÖNCEKİ bir sonuç dosyası varsa (kullanıcı onu bu
    # oturuma geri yüklediyse), onu preset-adına göre bir sözlüğe okuyup yeni
    # sonuçlarla BİRLEŞTİRİYORUZ (overwrite değil, merge) — böylece dosya her
    # zaman o ana kadar çalıştırılmış TÜM preset'lerin kümülatif sonucunu
    # içerir. Önceki bir sonucu bu oturuma geri yüklemediysen, script sadece
    # bu oturumdaki preset'leri içeren bir dosyayla başlar (veri kaybı olmaz,
    # sadece birleştirme fırsatı kaçmış olur — o zaman CSV'leri elle/yerel
    # olarak birleştirmen gerekir).
    results_by_preset = {}
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        for _, row in existing_df.iterrows():
            results_by_preset[row['preset']] = row.to_dict()
        print(f"✓ Önceki sonuç dosyası bulundu, {len(results_by_preset)} preset zaten var: {out_path}")
    else:
        print(f"ℹ️  Önceki sonuç dosyası yok ({out_path}), sıfırdan başlanıyor")

    for preset_name in args.presets:
        overrides = PRESETS[preset_name]
        results_rows = []
        try:
            run_single_ablation(preset_name, overrides, train_csv, val_csv, test_csv, results_rows)
        except Exception as e:
            print(f"❌ {preset_name} HATA: {e}")
            results_rows.append({'preset': preset_name, 'overrides': str(overrides), 'error': str(e)})
        finally:
            # config'i sıfırla (bir sonraki preset temiz başlasın)
            for k, v in baseline.items():
                setattr(config, k, v)

        for row in results_rows:
            results_by_preset[row['preset']] = row  # bu preset'in önceki kaydı varsa üzerine yazar

        # Her preset sonrası ara kayıt (Kaggle 12h limitine takılırsa veri kaybını önler)
        pd.DataFrame(list(results_by_preset.values())).to_csv(out_path, index=False)
        print(f"  💾 Kümülatif sonuç kaydedildi ({len(results_by_preset)} preset): {out_path}")

    print("\n" + "=" * 80)
    print("TÜM ABLATION SONUÇLARI (bu oturum + önceden yüklenmiş sonuçlar)")
    print("=" * 80)
    print(pd.DataFrame(list(results_by_preset.values())).to_string(index=False))
    print(f"\n✅ Nihai sonuçlar: {out_path}")


if __name__ == '__main__':
    main()
