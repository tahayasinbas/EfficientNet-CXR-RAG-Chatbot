"""
TTA İstatistiksel Anlamlılık Testi (Paired Bootstrap)

Hakem #3 (major) yanıtı: "Table 2 reports only four classes, although the
text states that all classes improved... an AUC increase of 0.0025 cannot be
described as clinically meaningful without confidence intervals or
statistical testing."

Bu script, normal ve TTA per-sample tahminlerini (`test_predictions.csv` /
`test_predictions_tta.csv` — aynı image_id sırasıyla, `05_evaluate.py` /
`05_evaluate_with_tta.py` çıktıları) EŞLEŞTİRİLMİŞ (paired) bootstrap ile
karşılaştırır: her bootstrap tekrarında AYNI resample indeksleri hem normal
hem TTA tahminlerine uygulanır, macro AUC farkı hesaplanır — bu, TTA'nın
gerçekten sistematik bir iyileşme mi yoksa örnekleme gürültüsü mü olduğunu
ayırt eder.

Kullanım:
    python 09_tta_significance_test.py \
        --predictions test_predictions.csv \
        --tta-predictions test_predictions_tta.csv \
        --output-dir ../hakem-yanitlari
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(path):
    df = pd.read_csv(path)
    diseases = [c[:-5] for c in df.columns if c.endswith('_true')]
    diseases = [d for d in diseases if f'{d}_pred' in df.columns]
    return df, diseases


def macro_auc(df, diseases, idx):
    aucs = []
    for d in diseases:
        yt = df[f'{d}_true'].values[idx]
        ys = df[f'{d}_pred'].values[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    return float(np.mean(aucs)), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True, help='Normal (TTA'"'"'siz) test_predictions.csv')
    ap.add_argument('--tta-predictions', required=True, help='test_predictions_tta.csv (aynı image_id sırası)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n-bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_n, diseases_n = load(args.predictions)
    df_t, diseases_t = load(args.tta_predictions)
    diseases = [d for d in diseases_n if d in diseases_t]

    if not (df_n['image_id'].values == df_t['image_id'].values).all():
        raise ValueError("image_id sıraları eşleşmiyor — normal ve TTA dosyaları aynı test setinden, aynı sırayla üretilmiş olmalı.")

    n = len(df_n)
    print(f"✓ {n:,} örnek, {len(diseases)} sınıf eşleştirildi")

    auc_n_full, per_class_n = macro_auc(df_n, diseases, np.arange(n))
    auc_t_full, per_class_t = macro_auc(df_t, diseases, np.arange(n))
    print(f"Normal macro AUC: {auc_n_full:.4f}")
    print(f"TTA    macro AUC: {auc_t_full:.4f}")
    print(f"Gözlenen fark: {auc_t_full - auc_n_full:+.4f}")

    per_class_rows = []
    for d in diseases:
        yt = df_n[f'{d}_true'].values
        if len(np.unique(yt)) < 2:
            continue
        a_n = roc_auc_score(yt, df_n[f'{d}_pred'].values)
        a_t = roc_auc_score(df_t[f'{d}_true'].values, df_t[f'{d}_pred'].values)
        per_class_rows.append({'disease': d, 'auc_normal': a_n, 'auc_tta': a_t, 'delta': a_t - a_n})
    per_class_df = pd.DataFrame(per_class_rows).sort_values('delta', ascending=False)
    n_improved = int((per_class_df['delta'] > 0).sum())
    print(f"\n{n_improved}/{len(per_class_df)} sınıfta TTA ile AUC iyileşti")

    # Paired bootstrap: her tekrarda AYNI indeksler her iki dosyaya da uygulanır
    rng = np.random.RandomState(args.seed)
    deltas = []
    for _ in range(args.n_bootstrap):
        idx = rng.randint(0, n, n)
        a_n, _ = macro_auc(df_n, diseases, idx)
        a_t, _ = macro_auc(df_t, diseases, idx)
        deltas.append(a_t - a_n)
    deltas = np.array(deltas)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_positive = float((deltas > 0).mean())
    significant = ci_low > 0

    print(f"\nPaired bootstrap ({args.n_bootstrap} tekrar):")
    print(f"  Ortalama fark: {deltas.mean():+.4f}")
    print(f"  %95 CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  P(TTA daha iyi): {p_positive:.3f}")
    print(f"  İstatistiksel olarak anlamlı (CI 0'ı içermiyor): {'EVET' if significant else 'HAYIR'}")

    per_class_df.round(4).to_csv(out_dir / 'tta_significance_per_class.csv', index=False)

    with open(out_dir / 'tta_significance_report.md', 'w', encoding='utf-8') as f:
        f.write("# TTA İstatistiksel Anlamlılık Testi (Paired Bootstrap)\n\n")
        f.write(f"- Normal: `{args.predictions}`\n- TTA: `{args.tta_predictions}`\n")
        f.write(f"- Örnek sayısı: {n:,}, bootstrap tekrarı: {args.n_bootstrap}\n\n")
        f.write(f"**Macro AUC:** normal={auc_n_full:.4f}, TTA={auc_t_full:.4f}, "
                f"fark={auc_t_full - auc_n_full:+.4f}\n\n")
        f.write(f"**Paired bootstrap %95 CI:** [{ci_low:+.4f}, {ci_high:+.4f}] "
                f"(P(TTA daha iyi)={p_positive:.3f})\n\n")
        f.write(f"**Sonuç:** Fark istatistiksel olarak {'ANLAMLI' if significant else 'ANLAMLI DEĞİL'} "
                f"(CI {'sıfırı içermiyor' if significant else 'sıfırı içeriyor'}). "
                f"{n_improved}/{len(per_class_df)} sınıfta AUC iyileşti (tüm sınıflar için tablo: "
                f"`tta_significance_per_class.csv`).\n\n")
        f.write("**Önemli not (aşırı yorumlamamak için):** İstatistiksel anlamlılık, etki büyüklüğünün "
                "KLİNİK olarak önemli olduğu anlamına gelmez — mutlak iyileşme küçüktür "
                f"(~{(auc_t_full - auc_n_full):.3f} AUC). Makalede TTA'nın faydası 'istatistiksel olarak "
                "tutarlı ama mütevazı' şeklinde, abartılmadan ifade edilmelidir.\n\n")
        f.write("## Sınıf Bazlı AUC Değişimi\n\n")
        f.write(per_class_df.round(4).to_string(index=False))
        f.write("\n")

    print(f"\n✅ Kaydedildi: {out_dir / 'tta_significance_report.md'}, {out_dir / 'tta_significance_per_class.csv'}")


if __name__ == '__main__':
    main()
