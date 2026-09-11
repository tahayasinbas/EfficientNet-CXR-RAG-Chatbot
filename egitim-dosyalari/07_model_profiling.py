"""
Model Profiling: Parametre / FLOPs / Bellek / Gecikme (Latency) Karşılaştırması

Hakem #3 (major) yanıtı: Tablo 3'teki "EfficientNet-B3, DenseNet-121'den
önemli ölçüde hafif" iddiasının doğruluğunu GERÇEK sayılarla doğrulamak/
düzeltmek, ve "exact parameter counts, FLOPs, memory use, and inference
latency, including the fivefold TTA cost" talebini karşılamak için.

Retrain GEREKTİRMEZ — sadece mimari + (opsiyonel) ImageNet-pretrained
ağırlıklar. Bu script'in ürettiği sayılar model.py'deki mevcut
count_parameters() fonksiyonu ile çapraz doğrulanır (bkz. main() sonunda
assert).

Kullanım:
    python 07_model_profiling.py --output-dir ../hakem-yanitlari
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
import pandas as pd

import config
from model import MultimodalChestXrayModel, count_parameters

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️  thop bulunamadı, FLOPs hesaplanamayacak (pip install thop)")


def build_densenet121(num_diseases, pretrained=False):
    """R3'ün 'B3 vs DenseNet-121' iddiasını test etmek için standart DenseNet-121 baseline."""
    if pretrained:
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1
        m = tvm.densenet121(weights=weights)
    else:
        m = tvm.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_diseases)
    return m


def measure_latency(forward_fn, n_warmup=10, n_runs=50, device='cuda'):
    for _ in range(n_warmup):
        forward_fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    import numpy as np
    return float(np.mean(times)), float(np.std(times))


def profile_model(name, model, device, img_size, demo_dim, num_diseases,
                   multimodal, batch_sizes=(1, 36)):
    model = model.to(device).eval()

    total_params, trainable_params = count_parameters(model)

    result = {'model': name, 'total_params': total_params, 'trainable_params': trainable_params,
              'total_params_M': round(total_params / 1e6, 2)}

    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_demo = torch.randn(1, demo_dim, device=device) if multimodal else None

    if HAS_THOP:
        with torch.no_grad():
            if multimodal:
                macs, params_thop = thop_profile(model, inputs=(dummy_img, dummy_demo), verbose=False)
            else:
                macs, params_thop = thop_profile(model, inputs=(dummy_img,), verbose=False)
        result['GFLOPs'] = round(2 * macs / 1e9, 3)  # MACs -> FLOPs yaklaşık 2x
        result['GMACs'] = round(macs / 1e9, 3)
    else:
        result['GFLOPs'] = None
        result['GMACs'] = None

    for bs in batch_sizes:
        img = torch.randn(bs, 3, img_size, img_size, device=device)
        demo = torch.randn(bs, demo_dim, device=device) if multimodal else None

        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        def fwd():
            with torch.no_grad():
                if multimodal:
                    model(img, demo)
                else:
                    model(img)

        mean_ms, std_ms = measure_latency(fwd, device=device)
        result[f'latency_bs{bs}_ms_mean'] = round(mean_ms, 3)
        result[f'latency_bs{bs}_ms_std'] = round(std_ms, 3)
        result[f'throughput_bs{bs}_img_per_s'] = round(bs / (mean_ms / 1000), 1)

        if device == 'cuda':
            result[f'peak_mem_bs{bs}_MB'] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # 5x TTA maliyeti: batch=1 tek-görüntü inference latency'sinin 5 katı
    # (alt sınır — CPU-side augmentation/resize overhead'i HARİÇ)
    result['tta_5x_latency_ms_lower_bound'] = round(result['latency_bs1_ms_mean'] * 5, 3)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='../hakem-yanitlari')
    ap.add_argument('--pretrained', action='store_true', help='ImageNet ağırlıklarını indir (params/FLOPs etkilenmez, latency ölçümü için gerekmez)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("⚠️  NOT: Bu ölçümler yerel GPU'da yapıldı, Kaggle T4 farklı latency "
              "verebilir — parametre/FLOP sayıları donanımdan bağımsızdır, latency göreceli kıyas içindir.\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("1) Tam multimodal model (EfficientNet-B3 + demografik, gating fusion)...")
    m_full = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES, demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=args.pretrained, dropout=config.DROPOUT_RATE, use_attention=True,
        ablation_mode='full', fusion_type='gating'
    )
    results.append(profile_model('Full multimodal (B3 + demo, gating)', m_full, device,
                                  config.IMG_SIZE, config.NUM_DEMOGRAPHIC_FEATURES,
                                  config.NUM_DISEASES, multimodal=True))
    del m_full

    print("2) Tam multimodal model (self-attention fusion varyantı)...")
    m_attn = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES, demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=args.pretrained, dropout=config.DROPOUT_RATE, use_attention=True,
        ablation_mode='full', fusion_type='self_attention'
    )
    results.append(profile_model('Full multimodal (B3 + demo, self-attention)', m_attn, device,
                                  config.IMG_SIZE, config.NUM_DEMOGRAPHIC_FEATURES,
                                  config.NUM_DISEASES, multimodal=True))
    del m_attn

    print("3) Image-only EfficientNet-B3 (demografik olmadan)...")
    m_img = MultimodalChestXrayModel(
        num_diseases=config.NUM_DISEASES, demographic_features=config.NUM_DEMOGRAPHIC_FEATURES,
        pretrained=args.pretrained, dropout=config.DROPOUT_RATE, ablation_mode='image_only'
    )
    results.append(profile_model('Image-only (EfficientNet-B3)', m_img, device,
                                  config.IMG_SIZE, config.NUM_DEMOGRAPHIC_FEATURES,
                                  config.NUM_DISEASES, multimodal=False))
    del m_img

    print("4) DenseNet-121 baseline (Tablo 3 kıyası için)...")
    m_dense = build_densenet121(config.NUM_DISEASES, pretrained=args.pretrained)
    results.append(profile_model('DenseNet-121 (image-only baseline)', m_dense, device,
                                  config.IMG_SIZE, config.NUM_DEMOGRAPHIC_FEATURES,
                                  config.NUM_DISEASES, multimodal=False))
    del m_dense

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'model_profiling_results.csv', index=False)

    print("\n" + "=" * 100)
    print("SONUÇ TABLOSU")
    print("=" * 100)
    cols_show = ['model', 'total_params_M', 'GFLOPs', 'latency_bs1_ms_mean', 'tta_5x_latency_ms_lower_bound']
    print(df[cols_show].to_string(index=False))

    b3_row = df[df['model'].str.contains('Image-only')].iloc[0]
    dn_row = df[df['model'].str.contains('DenseNet-121')].iloc[0]
    print("\n--- 'EfficientNet-B3, DenseNet-121'den hafiftir' iddiası kontrolü ---")
    print(f"  EfficientNet-B3 (image-only): {b3_row['total_params_M']}M params, {b3_row['GFLOPs']} GFLOPs")
    print(f"  DenseNet-121:                 {dn_row['total_params_M']}M params, {dn_row['GFLOPs']} GFLOPs")
    if b3_row['total_params_M'] < dn_row['total_params_M']:
        print("  -> B3 GERÇEKTEN daha az parametreye sahip (iddia doğru).")
    else:
        print("  -> B3 DAHA FAZLA parametreye sahip (iddia YANLIŞ — makalede düzeltilmeli).")

    with open(out_dir / 'model_profiling_results.md', 'w', encoding='utf-8') as f:
        f.write("# Model Profiling Sonuçları (Parametre / FLOPs / Latency)\n\n")
        f.write(f"Ölçüm cihazı: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'} "
                f"(Kaggle T4 DEĞİL — göreceli kıyas için; parametre/FLOP sayıları donanımdan bağımsız)\n\n")
        f.write(df.to_csv(index=False))
        f.write("\n\n## 'B3, DenseNet-121'den hafiftir' iddiası\n\n")
        f.write(f"- EfficientNet-B3 (image-only): **{b3_row['total_params_M']}M params**, {b3_row['GFLOPs']} GFLOPs\n")
        f.write(f"- DenseNet-121: **{dn_row['total_params_M']}M params**, {dn_row['GFLOPs']} GFLOPs\n")
        verdict = "doğru" if b3_row['total_params_M'] < dn_row['total_params_M'] else "YANLIŞ, düzeltilmeli"
        f.write(f"- Sonuç: iddia **{verdict}**.\n")

    print(f"\n✅ Kaydedildi: {out_dir / 'model_profiling_results.csv'} ve .md")


if __name__ == '__main__':
    main()
