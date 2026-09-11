"""
Sınıf-Bazlı Threshold Optimizasyonu + Kalibrasyon Analizi

Hakem #2 (madde 5) ve #3 (major) yanıtı: eşik=0.5'te No Finding / Atelectasis /
Infiltration için sensitivity=F1=0 çıkması üzerine — bu script RETRAIN
GEREKTİRMEDEN, mevcut tahmin CSV'lerinden (05_evaluate.py / 05_evaluate_with_tta.py
çıktıları) şunları üretir:

  1. Sınıf-bazlı optimal threshold (Youden's J istatistiği VE F1-maksimize eden nokta)
  2. Bu threshold'larda yeniden hesaplanmış sensitivity/specificity/precision/F1
  3. Precision-Recall eğrileri (15 sınıf)
  4. Kalibrasyon (reliability) diyagramları + Brier score
  5. Bootstrap %95 güven aralığı (AUC ve F1 için)

ÖNEMLİ KAVEAT (R3'ün doğrudan talebi): threshold'lar --threshold-source ile
verilen dosyadan (idealde val_predictions.csv) türetilmelidir; ayrı bir dosya
verilmezse thresholdlar aynı (test) dosyadan türetilir ve script bunu açıkça
"TEŞHİS AMAÇLI / LEAKAGE RİSKİ" uyarısıyla işaretler — makaledeki nihai tabloda
mutlaka val-set türevli threshold kullanılmalıdır.

Kullanım:
    python 06_calibration_and_thresholds.py \
        --predictions ../egitim-ciktilari/test_predictions.csv \
        --threshold-source ../egitim-ciktilari/val_predictions.csv \
        --output-dir ../hakem-yanitlari/calibration_threshold_analysis

    # threshold-source verilmezse (val henüz yoksa) test-üzerinden teşhis modu:
    python 06_calibration_and_thresholds.py \
        --predictions ../egitim-ciktilari/test_predictions.csv \
        --output-dir ../hakem-yanitlari/calibration_threshold_analysis
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, brier_score_loss, confusion_matrix
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


def df_to_markdown(df):
    """tabulate bağımlılığı olmadan basit bir markdown tablo üretici."""
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(str(v) for v in row.values) + ' |')
    return '\n'.join(lines)


def load_predictions(path):
    """CSV'den <Disease>_true / <Disease>_pred sütun çiftlerini otomatik keşfet."""
    df = pd.read_csv(path)
    diseases = [c[:-5] for c in df.columns if c.endswith('_true')]
    diseases = [d for d in diseases if f'{d}_pred' in df.columns]
    return df, diseases


def youden_optimal_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thr[idx]), float(tpr[idx]), float(1 - fpr[idx])


def f1_optimal_threshold(y_true, y_score):
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve thr has len(precision)-1 entries
    f1 = np.zeros_like(precision)
    denom = precision + recall
    nonzero = denom > 0
    f1[nonzero] = 2 * precision[nonzero] * recall[nonzero] / denom[nonzero]
    idx = np.argmax(f1[:-1]) if len(thr) > 0 else 0
    if len(thr) == 0:
        return 0.5, float(precision[0]), float(recall[0]), float(f1[0])
    return float(thr[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])


def metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'f1': f1, 'tp': int(tp), 'fp': int(fp),
        'fn': int(fn), 'tn': int(tn)
    }


def bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=1000, seed=42, alpha=0.05):
    """Percentile-method bootstrap %95 CI. metric_fn(y_true, y_score) -> float."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUC tanımsız olur, atla
        try:
            values.append(metric_fn(yt, ys))
        except Exception:
            continue
    if len(values) < 10:
        return float('nan'), float('nan')
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True, help='Metriklerin hesaplanacağı tahmin CSV (örn. test_predictions.csv)')
    ap.add_argument('--threshold-source', default=None, help='Threshold türetmek için ayrı CSV (idealde val_predictions.csv). Verilmezse --predictions kullanılır (leakage uyarısıyla).')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n-bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, diseases = load_predictions(args.predictions)
    print(f"✓ {args.predictions}: {len(df):,} satır, {len(diseases)} sınıf")

    leakage_warning = args.threshold_source is None
    if leakage_warning:
        thr_df, thr_diseases = df, diseases
        print("\n⚠️  UYARI: --threshold-source verilmedi. Threshold'lar AYNI dosyadan "
              "(--predictions) türetiliyor. Bu, teşhis/keşif amaçlıdır ve leakage "
              "riski taşır — makaledeki NİHAİ tabloda mutlaka val_predictions.csv "
              "gibi ayrı bir kaynak kullanılmalıdır.\n")
    else:
        thr_df, thr_diseases = load_predictions(args.threshold_source)
        print(f"✓ Threshold kaynağı: {args.threshold_source}: {len(thr_df):,} satır")

    rows = []
    n_diseases = len(diseases)
    fig_pr, axes_pr = plt.subplots(5, 3, figsize=(15, 20))
    axes_pr = axes_pr.ravel()
    fig_cal, axes_cal = plt.subplots(5, 3, figsize=(15, 20))
    axes_cal = axes_cal.ravel()

    for i, disease in enumerate(diseases):
        y_true = df[f'{disease}_true'].values.astype(int)
        y_score = df[f'{disease}_pred'].values.astype(float)

        if disease in thr_diseases:
            yt_thr = thr_df[f'{disease}_true'].values.astype(int)
            ys_thr = thr_df[f'{disease}_pred'].values.astype(float)
        else:
            yt_thr, ys_thr = y_true, y_score

        n_pos = int(y_true.sum())
        if n_pos == 0 or n_pos == len(y_true):
            print(f"  ⚠️  {disease}: tek sınıf var, atlanıyor")
            continue

        auc = roc_auc_score(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        brier = brier_score_loss(y_true, y_score)

        # Fixed threshold=0.5 (makaledeki mevcut sonuç, karşılaştırma için)
        m_fixed = metrics_at_threshold(y_true, y_score, 0.5)

        # Youden J threshold (val/thr kaynağından türetildi)
        youden_thr, _, _ = youden_optimal_threshold(yt_thr, ys_thr)
        m_youden = metrics_at_threshold(y_true, y_score, youden_thr)

        # F1-optimal threshold (val/thr kaynağından türetildi)
        f1_thr, _, _, _ = f1_optimal_threshold(yt_thr, ys_thr)
        m_f1opt = metrics_at_threshold(y_true, y_score, f1_thr)

        # Bootstrap CI (auc + f1@youden)
        auc_lo, auc_hi = bootstrap_ci(y_true, y_score, roc_auc_score, args.n_bootstrap, args.seed)
        f1_lo, f1_hi = bootstrap_ci(
            y_true, y_score,
            lambda yt, ys: f1_score(yt, (ys >= youden_thr).astype(int), zero_division=0),
            args.n_bootstrap, args.seed
        )

        rows.append({
            'disease': disease, 'support': n_pos, 'auc': auc, 'auc_ci_low': auc_lo, 'auc_ci_high': auc_hi,
            'ap': ap_score, 'brier_score': brier,
            'sens@0.5': m_fixed['sensitivity'], 'spec@0.5': m_fixed['specificity'], 'f1@0.5': m_fixed['f1'],
            'youden_threshold': youden_thr, 'sens@youden': m_youden['sensitivity'],
            'spec@youden': m_youden['specificity'], 'precision@youden': m_youden['precision'],
            'f1@youden': m_youden['f1'], 'f1@youden_ci_low': f1_lo, 'f1@youden_ci_high': f1_hi,
            'f1_optimal_threshold': f1_thr, 'sens@f1opt': m_f1opt['sensitivity'],
            'spec@f1opt': m_f1opt['specificity'], 'precision@f1opt': m_f1opt['precision'],
            'f1@f1opt': m_f1opt['f1'],
            'threshold_source': 'same_file_LEAKAGE_RISK' if leakage_warning else Path(args.threshold_source).name
        })

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax = axes_pr[i]
        ax.plot(recall, precision, linewidth=2, label=f'AP={ap_score:.3f}')
        ax.axhline(y_true.mean(), color='gray', linestyle='--', linewidth=1, label='baseline (prevalence)')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(disease, fontsize=10, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7); ax.grid(alpha=0.3)

        # Calibration curve (10 bins)
        bins = np.linspace(0, 1, 11)
        bin_ids = np.clip(np.digitize(y_score, bins) - 1, 0, 9)
        bin_true_mean, bin_pred_mean = [], []
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() > 0:
                bin_true_mean.append(y_true[mask].mean())
                bin_pred_mean.append(y_score[mask].mean())
        ax2 = axes_cal[i]
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='perfect calibration')
        ax2.plot(bin_pred_mean, bin_true_mean, 'o-', linewidth=2, label=f'Brier={brier:.4f}')
        ax2.set_xlabel('Ortalama tahmin edilen olasılık'); ax2.set_ylabel('Gözlenen frekans')
        ax2.set_title(disease, fontsize=10, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=7); ax2.grid(alpha=0.3)

        print(f"  {disease:20s} AUC={auc:.4f} [{auc_lo:.4f},{auc_hi:.4f}]  "
              f"F1@0.5={m_fixed['f1']:.4f}  F1@youden({youden_thr:.3f})={m_youden['f1']:.4f}  "
              f"sens@0.5={m_fixed['sensitivity']:.4f}->sens@youden={m_youden['sensitivity']:.4f}")

    # Boş kalan subplot'ları kapat
    for ax in axes_pr[len(diseases):]:
        ax.axis('off')
    for ax in axes_cal[len(diseases):]:
        ax.axis('off')

    fig_pr.tight_layout()
    fig_pr.savefig(out_dir / 'pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig_pr)

    fig_cal.tight_layout()
    fig_cal.savefig(out_dir / 'calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig_cal)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / 'threshold_optimized_metrics.csv', index=False)

    macro_row = {
        'disease': 'MACRO AVERAGE', 'support': results_df['support'].sum(),
        'auc': results_df['auc'].mean(), 'ap': results_df['ap'].mean(),
        'sens@0.5': results_df['sens@0.5'].mean(), 'sens@youden': results_df['sens@youden'].mean(),
        'f1@0.5': results_df['f1@0.5'].mean(), 'f1@youden': results_df['f1@youden'].mean(),
        'f1@f1opt': results_df['f1@f1opt'].mean()
    }

    print("\n" + "=" * 90)
    print("ÖZET (macro ortalama)")
    print("=" * 90)
    print(f"  Sensitivity @0.5 sabit eşik : {macro_row['sens@0.5']:.4f}")
    print(f"  Sensitivity @Youden eşiği   : {macro_row['sens@youden']:.4f}")
    print(f"  F1 @0.5 sabit eşik          : {macro_row['f1@0.5']:.4f}")
    print(f"  F1 @Youden eşiği            : {macro_row['f1@youden']:.4f}")
    print(f"  F1 @F1-optimal eşiği        : {macro_row['f1@f1opt']:.4f}")
    print("=" * 90)

    with open(out_dir / 'report.md', 'w', encoding='utf-8') as f:
        f.write("# Sınıf-Bazlı Threshold Optimizasyonu ve Kalibrasyon Raporu\n\n")
        f.write(f"- Değerlendirilen dosya: `{args.predictions}`\n")
        f.write(f"- Threshold kaynağı: `{args.threshold_source or args.predictions}`"
                f"{' **(AYNI DOSYA — leakage riski, sadece teşhis amaçlı)**' if leakage_warning else ' (ayrı val seti — leakage yok)'}\n")
        f.write(f"- Bootstrap tekrar sayısı: {args.n_bootstrap}\n\n")
        f.write("## Macro Özet\n\n")
        f.write(f"| Metrik | @0.5 (sabit) | @Youden (optimize) | @F1-optimal |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| Sensitivity | {macro_row['sens@0.5']:.4f} | {macro_row['sens@youden']:.4f} | - |\n")
        f.write(f"| F1 | {macro_row['f1@0.5']:.4f} | {macro_row['f1@youden']:.4f} | {macro_row['f1@f1opt']:.4f} |\n\n")
        f.write("## Sınıf Bazlı Tablo\n\n")
        f.write(df_to_markdown(results_df.round(4)))
        f.write("\n")

    print(f"\n✅ Çıktılar kaydedildi: {out_dir}")
    print(f"   - threshold_optimized_metrics.csv")
    print(f"   - pr_curves.png, calibration_curves.png")
    print(f"   - report.md")


if __name__ == '__main__':
    main()
