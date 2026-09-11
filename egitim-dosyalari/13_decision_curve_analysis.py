"""
Decision Curve Analysis (DCA) — Hakem #3 (major) yanıtı.

Hakem: "Precision-recall curves, calibration curves, Brier scores, confidence
intervals, and decision-curve analysis would be more informative than AUC alone."

DCA (Vickers & Elkin, 2006), bir modelin KLİNİK fayda sağlayıp sağlamadığını
"herkesi tedavi et" ve "kimseyi tedavi etme" varsayılan stratejilerine karşı
ölçer. Yeniden eğitim gerektirmez — sadece örnek-bazlı tahminler yeterli.

Eşik olasılığı p_t, klinisyenin ödünleşimini kodlar: p_t'de hekim, 1 kaçırılmış
pozitifi p_t/(1-p_t) yanlış pozitife denk görür.

    Net Benefit(model) = TP/N - (FP/N) * p_t/(1-p_t)
    Net Benefit(hepsi) = prevalans - (1-prevalans) * p_t/(1-p_t)
    Net Benefit(hiçbiri) = 0

Model, NB(model) > max(NB(hepsi), 0) olan p_t aralığında klinik olarak faydalıdır.

Kullanım:
    python 13_decision_curve_analysis.py \
        --predictions ../egitim-ciktilari/test_predictions.csv \
        --output-dir ../egitim-ciktilari
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression


def net_benefit_model(y_true, y_prob, thresholds):
    """NB = TP/N - (FP/N) * p_t/(1-p_t), her p_t için."""
    n = len(y_true)
    out = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred = y_prob >= pt
        tp = np.sum((y_true == 1) & pred)
        fp = np.sum((y_true == 0) & pred)
        out[i] = tp / n - (fp / n) * (pt / (1.0 - pt))
    return out


def net_benefit_all(prevalence, thresholds):
    """Herkesi pozitif say stratejisi."""
    return prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))


# Klinik olarak anlamlı kabul edilen en küçük net fayda: 1000 hastada 1 ek
# doğru pozitif (aynı sayıda yanlış pozitifle). Bundan küçük "kazançlar"
# sayısal gürültüdür ve aralık olarak raporlanmamalıdır.
MIN_MEANINGFUL_GAIN = 0.001


def useful_range(thresholds, nb_model, nb_all, tol=MIN_MEANINGFUL_GAIN):
    """Modelin her iki varsayılan stratejiyi ANLAMLI ölçüde geçtiği p_t aralığı."""
    better = (nb_model > np.maximum(nb_all, 0.0) + tol)
    if not better.any():
        return None
    idx = np.where(better)[0]
    # en uzun kesintisiz bloğu döndür
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    block = max(splits, key=len)
    return thresholds[block[0]], thresholds[block[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', default='../egitim-ciktilari/test_predictions.csv')
    ap.add_argument('--calibration-source', default='../egitim-ciktilari/val_predictions.csv',
                    help='Kalibratörün eğitileceği AYRI set (leakage yok). Boş verilirse sadece ham analiz.')
    ap.add_argument('--output-dir', default='../egitim-ciktilari')
    ap.add_argument('--pt-min', type=float, default=0.001)
    ap.add_argument('--pt-max', type=float, default=0.70)
    ap.add_argument('--pt-steps', type=int, default=400)
    a = ap.parse_args()

    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.predictions)
    pred_cols = [c for c in df.columns if c.endswith('_pred')]
    diseases = [c[:-5] for c in pred_cols]
    print(f'{len(df):,} örnek, {len(diseases)} sınıf')

    # ── Kalibrasyon ───────────────────────────────────────────────────────
    # Sınıf ağırlıklandırma (pos_weight) olasılıkları sistematik olarak YUKARI
    # kaydırır: ör. Cardiomegaly'de prevalans %2.8 iken ortalama tahmin %26.5.
    # DCA sıralamaya değil KALİBRASYONA duyarlı olduğu için, ham olasılıklarla
    # model düşük p_t'de herkesi işaretler ve "treat all" ile aynileşir.
    # Bu yüzden analiz hem HAM hem de validation setinde öğrenilmiş izotonik
    # kalibrasyondan geçmiş olasılıklarla yapılır (test setine hiç bakılmaz).
    cal = {}
    if a.calibration_source:
        vdf = pd.read_csv(a.calibration_source)
        for d in diseases:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            iso.fit(vdf[f'{d}_pred'].values.astype(float),
                    vdf[f'{d}_true'].values.astype(int))
            cal[d] = iso
        print(f'✓ izotonik kalibrasyon {len(vdf):,} validation örneğinde eğitildi '
              f'({a.calibration_source})')

    # Log aralıklı ızgara: prevalansı %0.2 ile %54 arasında değişen sınıfların
    # hepsine adil olmak için (nadir sınıflarda klinik olarak anlamlı p_t çok küçüktür).
    thresholds = np.unique(np.concatenate([
        np.geomspace(a.pt_min, a.pt_max, a.pt_steps),
        np.linspace(a.pt_min, a.pt_max, a.pt_steps),
    ]))

    rows, curves = [], {}
    for d in diseases:
        y = df[f'{d}_true'].values.astype(int)
        p = df[f'{d}_pred'].values.astype(float)
        prev = y.mean()

        nb_m = net_benefit_model(y, p, thresholds)
        nb_a = net_benefit_all(prev, thresholds)

        if d in cal:
            pc = cal[d].predict(p)
            nb_c = net_benefit_model(y, pc, thresholds)
            gains_c = nb_c - np.maximum(nb_a, 0.0)
            kc = int(np.argmax(gains_c))
            rng_c = useful_range(thresholds, nb_c, nb_a)
        else:
            nb_c, gains_c, kc, rng_c = None, None, None, None
        curves[d] = (nb_m, nb_a, nb_c)

        rng = useful_range(thresholds, nb_m, nb_a)
        # p_t = prevalans: "tarafsız" referans noktası (burada treat-all'ın NB'si tam 0'dır).
        # KIRPMA YOK — grid'e değil, gerçek prevalansa göre doğrudan hesaplanır.
        nb_m_ref = float(net_benefit_model(y, p, np.array([prev]))[0])
        nb_a_ref = float(net_benefit_all(prev, np.array([prev]))[0])
        gain = nb_m_ref - max(nb_a_ref, 0.0)

        gains = nb_m - np.maximum(nb_a, 0.0)
        k = int(np.argmax(gains))

        rows.append({
            'disease': d,
            'prevalence': round(prev, 4),
            'max_nb_gain': round(float(gains[k]), 5),
            'pt_at_max_gain': round(float(thresholds[k]), 3),
            'clinically_useful': bool(gains[k] > MIN_MEANINGFUL_GAIN),
            'pt_useful_low': None if rng is None else round(rng[0], 3),
            'pt_useful_high': None if rng is None else round(rng[1], 3),
            'nb_model_at_prev': round(nb_m_ref, 5),
            'nb_treat_all_at_prev': round(nb_a_ref, 5),
            'nb_gain_at_prev': round(gain, 5),
            # Aynı sayıda yanlış pozitifte 1000 hastada kazanılan net doğru pozitif
            'net_tp_per_1000_at_prev': round(gain * 1000, 1),
            # ── validation-kalibre edilmiş olasılıklarla ──
            'cal_max_nb_gain': None if nb_c is None else round(float(gains_c[kc]), 5),
            'cal_pt_at_max_gain': None if nb_c is None else round(float(thresholds[kc]), 3),
            'cal_clinically_useful': None if nb_c is None else bool(gains_c[kc] > MIN_MEANINGFUL_GAIN),
            'cal_pt_useful_low': None if rng_c is None else round(rng_c[0], 3),
            'cal_pt_useful_high': None if rng_c is None else round(rng_c[1], 3),
            'cal_net_tp_per_1000_at_max': None if nb_c is None else round(float(gains_c[kc]) * 1000, 1),
        })

    res = pd.DataFrame(rows).sort_values('prevalence', ascending=False)
    csv_path = outdir / 'decision_curve_analysis.csv'
    res.to_csv(csv_path, index=False)
    print(f'✓ {csv_path}')
    print(res.to_string(index=False))

    print(f"\nKlinik olarak anlamlı net fayda (her iki varsayılan stratejiyi de geçen sınıf sayısı):")
    print(f"  HAM olasılıklar      : {int(res['clinically_useful'].sum())}/{len(res)}")
    if res['cal_clinically_useful'].notna().any():
        print(f"  KALİBRE edilmiş      : {int(res['cal_clinically_useful'].fillna(False).sum())}/{len(res)}")
        print(f"  -> Fark tamamen kalibrasyondan kaynaklanıyor; sıralama (AUC) değişmedi.")

    # ── figür 1: makale için kompakt panel (prevalansı kapsayan 4 temsilci sınıf) ──
    FEATURED = ['No Finding', 'Effusion', 'Pneumothorax', 'Cardiomegaly']
    feat = [d for d in FEATURED if d in curves]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for ax, d in zip(axes.ravel(), feat):
        nb_m, nb_a, nb_c = curves[d]
        prev = float(res.loc[res.disease == d, 'prevalence'].iloc[0])
        if nb_c is not None:
            ax.plot(thresholds, nb_c, lw=2.0, color='tab:blue', label='Model (calibrated)')
        ax.plot(thresholds, nb_m, lw=1.4, color='tab:orange', label='Model (raw)')
        ax.plot(thresholds, nb_a, lw=1.2, ls='--', color='gray', label='Treat all')
        ax.axhline(0.0, lw=1.2, ls=':', color='black', label='Treat none')
        allc = nb_m if nb_c is None else np.concatenate([nb_m, nb_c])
        ax.set_ylim(max(-0.04, float(np.nanmin(allc)) * 1.2),
                    max(float(np.nanmax(allc)) * 1.25, 0.01))
        ax.set_xlim(0, 0.7)
        ax.set_title(f"{d.replace('_', ' ')}  (prevalence {prev*100:.1f}%)", fontsize=10)
        ax.set_xlabel('Threshold probability $p_t$', fontsize=9)
        ax.set_ylabel('Net benefit', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes.ravel()[0].legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    p_feat = outdir / 'fig_decision_curves.png'
    fig.savefig(p_feat, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ {p_feat}  (makale figürü, {len(feat)} temsilci sınıf)')

    # ── figür 2: tüm 15 sınıf (depo eki) ──
    ncols, nrows = 3, int(np.ceil(len(diseases) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
    axes = axes.ravel()
    for ax, d in zip(axes, res['disease'].tolist()):
        nb_m, nb_a, nb_c = curves[d]
        if nb_c is not None:
            ax.plot(thresholds, nb_c, lw=1.9, color='tab:blue', label='Model (calibrated)')
        ax.plot(thresholds, nb_m, lw=1.3, color='tab:orange', label='Model (raw)')
        ax.plot(thresholds, nb_a, lw=1.1, ls='--', color='gray', label='Treat all')
        ax.axhline(0.0, lw=1.1, ls=':', color='black', label='Treat none')
        allc = nb_m if nb_c is None else np.concatenate([nb_m, nb_c])
        ax.set_ylim(max(-0.05, float(np.nanmin(allc)) * 1.2),
                    max(float(np.nanmax(allc)) * 1.25, 0.01))
        ax.set_xlim(0, 0.7)
        ax.set_title(d.replace('_', ' '), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[len(diseases):]:
        ax.axis('off')
    axes[0].legend(fontsize=7, loc='upper right')
    fig.supxlabel('Threshold probability $p_t$', fontsize=10)
    fig.supylabel('Net benefit', fontsize=10)
    plt.tight_layout()
    p_all = outdir / 'decision_curves.png'
    fig.savefig(p_all, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ {p_all}  (tüm 15 sınıf)')


if __name__ == '__main__':
    main()
