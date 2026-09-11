"""
Fig 6 (PR eğrileri + kalibrasyon diyagramları) ve Fig 7 (Grad-CAM paneli)
birleşik figürlerini üretir — makaleye tek görsel olarak yerleştirmek için.

Kullanım:
    python 11_composite_figures.py --input-dir ../egitim-ciktilari --output-dir ../egitim-ciktilari
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def make_fig6(in_dir: Path, out_dir: Path):
    pr = in_dir / 'pr_curves.png'
    cal = in_dir / 'calibration_curves.png'
    if not (pr.exists() and cal.exists()):
        print(f'⚠️  Fig 6 atlandı (eksik dosya): {pr.exists()=} {cal.exists()=}')
        return

    img_pr = mpimg.imread(pr)
    img_cal = mpimg.imread(cal)

    # İki dikey panel yan yana
    h = max(img_pr.shape[0], img_cal.shape[0])
    w = img_pr.shape[1] + img_cal.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(w / 260, h / 260))
    for ax, img, label in zip(axes, (img_pr, img_cal), ('(a)', '(b)')):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label, loc='left', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=0.8)
    p = out_dir / 'fig6_pr_calibration.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ {p}')


def make_fig7(in_dir: Path, out_dir: Path, n=4, ncols=2):
    gdir = in_dir / 'gradcam'
    if not gdir.exists():
        print('⚠️  Fig 7 atlandı: gradcam klasörü yok')
        return

    # Çeşitlilik için tercih sırası: farklı patolojiler
    preferred = ['Pneumothorax', 'Effusion', 'Infiltration', 'No Finding',
                 'Pleural_Thickening']
    files = sorted(gdir.glob('*.png'))
    picked = []
    for key in preferred:
        for f in files:
            if key in f.name and f not in picked:
                picked.append(f)
                break
        if len(picked) == n:
            break
    for f in files:  # yedek doldurma
        if len(picked) == n:
            break
        if f not in picked:
            picked.append(f)

    if not picked:
        print('⚠️  Fig 7 atlandı: uygun görsel bulunamadı')
        return

    # Sayfaya sığması için ızgara düzeni (4x1 dikey panel A4'e sığmıyordu)
    rows = (len(picked) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols, figsize=(5.5 * ncols, 3.0 * rows))
    axes = [axes] if rows * ncols == 1 else list(axes.ravel())
    for ax, f, lab in zip(axes, picked, 'abcdefgh'):
        ax.imshow(mpimg.imread(f))
        ax.axis('off')
        ax.set_title(f'({lab})', loc='left', fontsize=13, fontweight='bold')
    for ax in axes[len(picked):]:
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    p = out_dir / 'fig7_gradcam_panel.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ {p}  (kullanılan: {[f.name for f in picked]})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='../egitim-ciktilari')
    ap.add_argument('--output-dir', default='../egitim-ciktilari')
    a = ap.parse_args()
    ind, outd = Path(a.input_dir), Path(a.output_dir)
    outd.mkdir(parents=True, exist_ok=True)
    make_fig6(ind, outd)
    make_fig7(ind, outd)
