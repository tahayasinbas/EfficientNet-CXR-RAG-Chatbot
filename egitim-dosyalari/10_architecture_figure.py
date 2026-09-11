"""
Fig 2 — Multimodal architecture diagram (publication figure).

Makaledeki eski Fig 2, füzyonu "self-attention" olarak gösteriyordu; bu script
gerçek mimariyi (ModalityGatingFusion: 2 skaler softmax kapısı) doğru şekilde
çizer. Hakem #1 madde 2 ve #3 (self-attention tanımsız) yanıtının görsel ayağı.

Kullanım:
    python 10_architecture_figure.py --output-dir ../egitim-ciktilari
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Grayscale-friendly palette (baskıda da ayırt edilebilir)
C_IMG = '#cfe0f3'   # görüntü kolu
C_DEMO = '#ffe3c2'  # demografik kol
C_GATE = '#dcd0f0'  # kapılama
C_HEAD = '#d7ede0'  # sınıflandırıcı
C_IO = '#eaeaea'    # giriş/çıkış
EDGE = '#333333'


def box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=EDGE, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal',
            linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, style='-|>', lw=1.2, ls='-'):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
        linewidth=lw, linestyle=ls, color=EDGE, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='../egitim-ciktilari')
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.4); ax.axis('off')

    # ---------------- Görüntü kolu (üst) ----------------
    box(ax, 0.15, 4.35, 1.55, 1.15,
        'Chest radiograph\n300 × 300 × 3', C_IO)
    arrow(ax, 1.70, 4.93, 2.15, 4.93)

    box(ax, 2.15, 4.35, 1.95, 1.15,
        'EfficientNet-B3\n(ImageNet-1K pretrained)', C_IMG, bold=True)
    arrow(ax, 4.10, 4.93, 4.55, 4.93)

    box(ax, 4.55, 4.35, 1.55, 1.15,
        'Global Avg Pool\n+ Dropout (0.55)', C_IMG)
    arrow(ax, 6.10, 4.93, 6.60, 4.93)

    box(ax, 6.60, 4.50, 1.05, 0.85, r'$\mathbf{z}_{img}$' + '\n' + r'$\in \mathbb{R}^{1536}$',
        C_IMG, fontsize=9, bold=True)

    # ---------------- Demografik kol (alt) ----------------
    box(ax, 0.15, 0.95, 1.55, 1.35,
        'Patient metadata\nage · sex · view', C_IO)
    arrow(ax, 1.70, 1.63, 2.15, 1.63)

    box(ax, 2.15, 0.85, 1.95, 1.55,
        'Feature encoding (12-D)\n3 age transforms\n4 age bands\n2 sex · 3 projection',
        C_DEMO, fontsize=8)
    arrow(ax, 4.10, 1.63, 4.55, 1.63)

    box(ax, 4.55, 0.95, 1.55, 1.35,
        'MLP\n12→128→128→64\nBN + ReLU + Dropout', C_DEMO, fontsize=8, bold=True)
    arrow(ax, 6.10, 1.63, 6.60, 1.63)

    box(ax, 6.60, 1.20, 1.05, 0.85, r'$\mathbf{z}_{demo}$' + '\n' + r'$\in \mathbb{R}^{64}$',
        C_DEMO, fontsize=9, bold=True)

    # ---------------- Kapılama (orta) ----------------
    # Her iki koldan kapı hesaplayıcıya
    arrow(ax, 7.65, 4.93, 8.10, 3.90)
    arrow(ax, 7.65, 1.63, 8.10, 2.70)

    box(ax, 8.10, 2.55, 1.85, 1.50,
        'Modality gating\n' + r'$\mathbf{h}=\mathrm{ReLU}(\mathbf{W}_1[\mathbf{z}_{img};\mathbf{z}_{demo}])$' +
        '\n' + r'$[a_{img},a_{demo}]=\mathrm{softmax}(\mathbf{W}_2\mathbf{h})$',
        C_GATE, fontsize=8, bold=True)

    # Kapı ağırlıkları çarpım düğümlerine
    ax.text(10.02, 3.62, r'$a_{img}$', fontsize=9, ha='center', va='center')
    ax.text(10.02, 2.28, r'$a_{demo}$', fontsize=9, ha='center', va='center')
    arrow(ax, 9.95, 3.42, 10.35, 4.62, ls='--', lw=1.0)
    arrow(ax, 9.95, 2.55, 10.35, 1.72, ls='--', lw=1.0)

    # Çarpım düğümleri
    for cy in (4.93, 1.63):
        ax.add_patch(Circle((10.50, cy), 0.155, facecolor='white',
                            edgecolor=EDGE, linewidth=1.1, zorder=3))
        ax.text(10.50, cy, '×', ha='center', va='center', fontsize=12, zorder=4)

    arrow(ax, 7.65, 4.93, 10.34, 4.93)
    arrow(ax, 7.65, 1.63, 10.34, 1.63)

    # Concat
    arrow(ax, 10.66, 4.93, 11.05, 3.75)
    arrow(ax, 10.66, 1.63, 11.05, 2.85)

    box(ax, 11.05, 2.75, 1.05, 1.05,
        'concat\n' + r'$\mathbb{R}^{1600}$', C_GATE, fontsize=8.5, bold=True)

    # ---------------- Sınıflandırıcı ----------------
    arrow(ax, 12.10, 3.28, 12.40, 3.28)
    box(ax, 12.40, 1.55, 0.95, 3.45,
        'Classifier\n\n1600 → 512\n512 → 256\n256 → 128\n128 → 15\n\n(BN + ReLU\n+ Dropout)\n\nSigmoid',
        C_HEAD, fontsize=7.6, bold=False)

    ax.text(12.875, 1.15, '15 independent\nclass probabilities\n(14 pathologies + No Finding)',
            ha='center', va='center', fontsize=7.6, style='italic')

    # ---------------- Kol etiketleri ----------------
    ax.text(0.15, 5.85, 'Visual pathway', fontsize=10, fontweight='bold', color='#1f4e79')
    ax.text(0.15, 2.62, 'Demographic pathway', fontsize=10, fontweight='bold', color='#9c5700')
    ax.text(8.10, 4.28, 'Learned modality gating',
            fontsize=10, fontweight='bold', color='#4b2d83')

    plt.tight_layout()
    for ext, dpi in (('png', 300), ('pdf', 300)):
        p = out_dir / f'fig2_architecture.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'✓ kaydedildi: {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
