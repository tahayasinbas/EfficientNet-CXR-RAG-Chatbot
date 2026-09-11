"""
Revize edilmiş makaleyi .docx olarak üretir.

Orijinal dosyayı ASLA değiştirmez: kopya üzerinde çalışır, böylece Word
biçimlendirmesi, stilleri, gömülü şekilleri ve bölüm ayarları korunur.
Sadece (a) metin paragraflarının içeriği, (b) veri tablolarının hücreleri,
(c) üç figürün görsel içeriği değiştirilir; yeni bölümler/tablolar/figürler
doğru konumlara eklenir.

Kullanım (proje kökünden):
    python egitim-dosyalari/12_build_revised_docx.py
"""

from pathlib import Path
import copy
import re
import shutil

import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
WP = '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}'

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'SCI_Gögüs_Hastaligi_Türkçe_özv2 (2).docx'
DST = ROOT / 'SCI_Gogus_Hastaligi_REVIZE.docx'
FIGDIR = ROOT / 'egitim-ciktilari'


# ─────────────────────────── yardımcılar ───────────────────────────

def set_text(par, text):
    """Paragraf metnini, ilk run'ın biçimini koruyarak değiştir."""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return par
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    return par


# Orijinal metinde, muhtemelen bir Oxford-virgül temizliği sırasında oluşmuş
# "…and" birleşmeleri (ör. "exposureand"). Hakem #3 zaten yazım hataları için
# şikâyet ettiği için hepsi düzeltiliyor. Sadece boşluk eklenir, içerik değişmez.
MISSING_SPACE_STEMS = [
    'exposure', 'cardiomegaly', 'variability', 'scalable', 'history', 'sex',
    'maintainability', 'layer', 'updated', 'visualization', 'routing',
    'outputs', 'integrity', 'inference', 'maintenance', 'maintainable',
    'patient', 'Django', 'editing',
]
_MISSING_SPACE_RE = re.compile(
    r'\b(' + '|'.join(MISSING_SPACE_STEMS) + r')(and)\b')


def fix_missing_spaces(par):
    """Run biçimlerini bozmadan, run sınırlarını aşan '…and' birleşmelerini onar.

    Paragrafın tam metni üzerinden eşleşme aranır, ardından eklenecek boşluğun
    denk geldiği run bulunup boşluk oraya yerleştirilir. Böylece italik/üst
    simge gibi run-içi biçimler korunur.
    """
    runs = par.runs
    if not runs:
        return 0
    n = 0
    while True:
        full = ''.join(r.text for r in runs)
        m = _MISSING_SPACE_RE.search(full)
        if m is None:
            return n
        pos = m.start(2)          # "and"in başladığı indeks -> buraya boşluk
        off = 0
        for r in runs:
            ln = len(r.text)
            if off <= pos <= off + ln:
                k = pos - off
                r.text = r.text[:k] + ' ' + r.text[k:]
                break
            off += ln
        else:
            return n
        n += 1


def clone_par_after(anchor_el, template_par, text):
    """template_par'ın biçimini kopyalayan yeni bir paragrafı anchor'dan sonra ekle."""
    new_el = copy.deepcopy(template_par._element)
    anchor_el.addnext(new_el)
    from docx.text.paragraph import Paragraph
    p = Paragraph(new_el, template_par._parent)
    # tüm run'ları temizleyip tek run bırak
    for r in list(p.runs)[1:]:
        r._element.getparent().remove(r._element)
    if p.runs:
        p.runs[0].text = text
        # gömülü çizim varsa temizle
        for dr in p.runs[0]._element.findall('.//' + W + 'drawing'):
            dr.getparent().remove(dr)
        for pc in p.runs[0]._element.findall('.//' + W + 'pict'):
            pc.getparent().remove(pc)
    else:
        p.add_run(text)
    return p


# Sayfa yazı alanı genişliği (A4, mevcut kenar boşluklarıyla): 6.27 inç.
# Tabloların bunu aşmaması gerekir, yoksa Word sağ marja taşırır.
TEXT_WIDTH_IN = 6.27

# Tablo başına sütun genişlikleri (inç). Toplamları TEXT_WIDTH_IN'i aşmamalı.
COL_WIDTHS = {
    'split':    [1.71, 1.52, 1.52, 1.52],                          # Tablo 1
    'perclass': [1.35, 0.72, 0.72, 0.80, 0.90, 0.90, 0.88],        # Tablo 2
    'thresh':   [1.15, 1.42, 0.50, 0.52, 0.68, 0.68, 0.62, 0.70],  # Tablo 3
    'tta':      [1.80, 1.20, 1.20, 0.80],                          # Tablo 4
    'ablation': [1.80, 1.10, 0.90, 0.87, 0.83, 0.77],              # Tablo 5
    'profile':  [1.72, 0.72, 0.55, 0.78, 0.82, 0.68, 1.00],        # Tablo 6
    'sota':     [1.15, 0.40, 0.85, 1.55, 1.12, 0.62, 0.58],        # Tablo 7
}

# Metin içeren (sayısal olmayan) sütunlar sola yaslanmalı
LEFT_COLS = {
    'ablation': (1, 2),          # Modalities, Fusion
    'sota':     (2, 3, 4),       # Input, Architecture, Split protocol
}


def set_table_widths(tbl, widths_in):
    """tblGrid + her hücrenin tcW değerini inç cinsinden sabitle (sabit düzen)."""
    twips = [int(round(w * 1440)) for w in widths_in]
    tblPr = tbl._element.tblPr

    # sabit düzen: Word sütunları içeriğe göre genişletmesin
    for tag in ('tblLayout', 'tblW'):
        old = tblPr.find(W + tag)
        if old is not None:
            tblPr.remove(old)
    layout = tblPr.makeelement(W + 'tblLayout', {W + 'type': 'fixed'})
    tblPr.append(layout)
    tw = tblPr.makeelement(W + 'tblW',
                           {W + 'w': str(sum(twips)), W + 'type': 'dxa'})
    tblPr.append(tw)

    grid = tbl._element.find(W + 'tblGrid')
    if grid is not None:
        cols = grid.findall(W + 'gridCol')
        for gc, t in zip(cols, twips):
            gc.set(W + 'w', str(t))
    for row in tbl.rows:
        for j, cell in enumerate(row.cells):
            if j >= len(twips):
                continue
            tcPr = cell._tc.get_or_add_tcPr()
            old = tcPr.find(W + 'tcW')
            if old is not None:
                tcPr.remove(old)
            tcPr.append(tcPr.makeelement(
                W + 'tcW', {W + 'w': str(twips[j]), W + 'type': 'dxa'}))
    return tbl


def make_table(doc, data, style_src_tbl, anchor_el, bold_header=True,
               bold_last_row=False, font_pt=8, left_cols=()):
    """data: list[list[str]] — ilk satır başlık. anchor_el'den sonra ekle."""
    rows, cols = len(data), len(data[0])
    tbl = doc.add_table(rows=rows, cols=cols)
    try:
        tbl.style = style_src_tbl.style
    except Exception:
        pass
    if style_src_tbl is not None and style_src_tbl._element.tblPr is not None:
        new_pr = copy.deepcopy(style_src_tbl._element.tblPr)
        old_pr = tbl._element.tblPr
        tbl._element.replace(old_pr, new_pr)

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(font_pt)
            if (i == 0 and bold_header) or (bold_last_row and i == rows - 1):
                run.font.bold = True
            if j > 0 and j not in left_cols:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    anchor_el.addnext(tbl._element)
    return tbl


def resize_table(tbl, n_rows, n_cols):
    """Var olan tabloyu hedef boyuta getir (satır/sütun ekle)."""
    while len(tbl.rows) < n_rows:
        tbl.add_row()
    while len(tbl.columns) < n_cols:
        tbl.add_column(Inches(0.8))
    return tbl


def fill_table(tbl, data, bold_header=True, bold_last_row=False, font_pt=8,
               left_cols=()):
    """Var olan tabloyu verilerle doldur (biçimi korunur)."""
    resize_table(tbl, len(data), len(data[0]))
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(font_pt)
            if (i == 0 and bold_header) or (bold_last_row and i == len(data) - 1):
                run.font.bold = True
            if j > 0 and j not in left_cols:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def replace_image(doc, par, new_path):
    """Paragraftaki gömülü görselin ikili verisini değiştir, en-boy oranını koru."""
    blips = par._element.findall('.//' + A + 'blip')
    if not blips:
        return False
    rid = blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    if rid is None:
        return False
    image_part = doc.part.related_parts[rid]
    data = Path(new_path).read_bytes()
    image_part._blob = data

    with Image.open(new_path) as im:
        nw, nh = im.size
    ratio = nh / nw

    for ext in par._element.findall('.//' + WP + 'extent'):
        cx = int(ext.get('cx'))
        ext.set('cy', str(int(cx * ratio)))
    for ext in par._element.findall('.//' + A + 'ext'):
        if ext.get('cx') is None:
            continue
        cx = int(ext.get('cx'))
        ext.set('cy', str(int(cx * ratio)))
    return True


def scale_images_in(element, factor):
    """Bir öğedeki (tablo/paragraf) tüm gömülü görselleri oranı koruyarak ölçekle."""
    n = 0
    for ext in element.findall('.//' + WP + 'extent'):
        ext.set('cx', str(int(int(ext.get('cx')) * factor)))
        ext.set('cy', str(int(int(ext.get('cy')) * factor)))
        n += 1
    for ext in element.findall('.//' + A + 'ext'):
        if ext.get('cx') is None or ext.get('cy') is None:
            continue
        ext.set('cx', str(int(int(ext.get('cx')) * factor)))
        ext.set('cy', str(int(int(ext.get('cy')) * factor)))
    return n


def insert_picture_par(doc, anchor_el, template_par, image_path, width_in=6.2):
    """anchor'dan sonra, ortalanmış görsel içeren yeni bir paragraf ekle."""
    p = clone_par_after(anchor_el, template_par, '')
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_in))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


# ─────────────────────────── metinler ───────────────────────────
# (Kaynak: SCI_makale_REVIZE.md — RAG bölümleri değiştirilmiyor)

T = {}

T[2] = (
    "Although chest X-ray interpretation remains a cornerstone of thoracic disease diagnosis, it continues to "
    "face critical challenges including escalating radiologist workload, substantial inter-observer variability "
    "and the inherent limitations of unimodal analytical approaches that disregard clinical context. This study "
    "proposes an end-to-end clinical decision support system that integrates a multimodal deep learning "
    "architecture with Retrieval-Augmented Generation to address these gaps simultaneously. The proposed "
    "framework extends an ImageNet-pretrained EfficientNet-B3 backbone with a learned modality-gating fusion "
    "mechanism that jointly encodes radiographic features alongside structured patient metadata (specifically "
    "age, sex and imaging position), enabling shared representation learning across visual and clinical "
    "modalities. Model training and evaluation were conducted on the NIH ChestX-ray14 dataset with strict "
    "patient-level data splitting, explicitly preventing data leakage, a methodological concern frequently "
    "overlooked in the literature. Augmented with Test Time Augmentation, the proposed model achieves a "
    "macro-average AUC of 0.8342 across all 15 classes (the 14 thoracic pathologies plus the “No Finding” "
    "category), corresponding to 0.8374 when averaged over the 14 pathology labels used by comparable studies, "
    "together with a macro-average F1 of 0.2817 and a macro-average sensitivity of 0.7001 at the standard "
    "decision threshold. A controlled ablation study spanning eight epoch-matched configurations — the "
    "proposed model together with image-only, metadata-only, simple concatenation, multi-head self-attention "
    "fusion, class-weighting scheme and loss/augmentation variants — was conducted under identical data "
    "splits and optimization settings to quantify the contribution of each component, and per-class decision thresholds were calibrated "
    "exclusively on the validation set to report clinically usable operating points. Gradient-weighted class "
    "activation mapping was additionally employed to provide patient-specific visual evidence for individual "
    "predictions. Complementing the classification module, a hybrid BM25-vector retrieval pipeline was "
    "constructed over a corpus of 42,457 biomedical text chunks utilizing OpenAI’s text-embedding-3-small "
    "model, yielding a Hit Rate@10 of 56.0%. By leveraging the GPT-4o mini model for clinical response "
    "generation, the system achieved an average LLM-as-a-Judge score of 3.76/5.0, thereby generating clinically "
    "coherent, explainable and hallucination-reduced diagnostic narratives. The complete system was deployed as "
    "a real-time web application built on a React, Django and FastAPI microservices architecture. Collectively, "
    "these results demonstrate that the proposed framework constitutes a clinically applicable, interpretable "
    "and extensible AI-assisted diagnostic platform suitable for integration into real-world healthcare workflows."
)

T[3] = ("Keywords: Chest X-ray interpretation, Clinical decision support, EfficientNet, Explainable AI, "
        "Multimodal deep learning, Retrieval-Augmented Generation, Ablation study")

T[6] = (
    "Recent advances in deep learning, coupled with the availability of large-scale annotated datasets, have "
    "driven substantial performance improvements in AI-based CXR analysis systems. The NIH ChestX-ray14 dataset, "
    "introduced by Wang et al. [1] (originally released as ChestX-ray8 with eight pathology labels and "
    "subsequently expanded to 14 thoracic disease classes), has emerged as one of the most widely adopted "
    "benchmark resources in thoracic imaging research. Building upon this foundation, Rajpurkar et al. [5] "
    "demonstrated that a DenseNet-121-based architecture — CheXNet — could achieve radiologist-level "
    "performance in pneumonia detection, establishing the representational capacity of deep convolutional neural "
    "networks (CNNs) for radiographic pathology recognition. Subsequent studies leveraging architectures such as "
    "DenseNet, Xception and EfficientNet further advanced performance through transfer learning, compound scaling "
    "strategies and deep feature extraction [6-8]. CNN-based approaches have also demonstrated promising results "
    "in CXR-based COVID-19 screening [9,10]. More recent work has continued to advance this benchmark through "
    "hybrid convolutional-transformer designs [32], pretraining-diversity and clinical-metric optimization "
    "strategies [33], systematic reproduction and improvement of CheXNet [34] and transfer-learning pipelines "
    "based on modern residual backbones [35]."
)

T[8] = (
    "A second fundamental limitation of current deep learning-based clinical systems concerns the lack of "
    "interpretability in model outputs. Although modern CNN-based classifiers can achieve high predictive "
    "accuracy, they predominantly yield probability scores without providing clinically meaningful "
    "justifications for their decisions. This opacity constrains clinician trust in AI systems and impedes their "
    "integration into routine clinical workflows. Two complementary forms of interpretability are required in "
    "this setting and should be clearly distinguished. The first is model-level visual evidence: an explanation "
    "of why the classifier reached a particular conclusion for a particular radiograph, which can be obtained "
    "through saliency methods such as Gradient-weighted Class Activation Mapping (Grad-CAM) [36]. The second is "
    "knowledge-level grounding: an explanation of what a predicted pathology means clinically, which requires "
    "access to verifiable biomedical evidence. Large Language Models (LLMs) offer considerable potential for the "
    "latter; however, their propensity for producing unverified content — commonly referred to as "
    "hallucination — necessitates careful deployment in medical contexts. Retrieval-Augmented Generation "
    "(RAG) architectures have emerged as a principled solution to this problem by grounding model outputs in "
    "verifiable scientific sources, thereby substantially reducing hallucination risk [14,15]. By anchoring the "
    "generation process to peer-reviewed biomedical literature and authoritative clinical documents, RAG "
    "frameworks enable the production of more interpretable, consistent and clinically trustworthy narratives "
    "[16]. Hybrid retrieval strategies combining BM25-based lexical search with dense vector-based semantic "
    "retrieval have demonstrated robust performance in complex information retrieval tasks [17,18]. Building on "
    "this technological foundation, adapting such multi-stage architectures to the medical domain provides a "
    "highly effective solution for its unique terminological demands. In the radiology domain, RAG-based "
    "approaches have also demonstrated promising results in clinically interpretable applications such as "
    "image-text alignment and automated CXR report generation [19]. The present work implements both forms of "
    "interpretability and reports them separately, rather than treating literature retrieval as a substitute for "
    "visual explanation."
)

T[9] = (
    "In response to the limitations outlined above, this study proposes an integrated clinical decision support "
    "framework that unifies: (i) a multimodal EfficientNet-B3 architecture with a learned modality-gating fusion "
    "mechanism, trained on the NIH ChestX-ray14 dataset; (ii) patient-level data partitioning to rigorously "
    "prevent data leakage, with exact per-partition image, patient and class-prevalence counts reported; (iii) a "
    "controlled eight-configuration, epoch-matched ablation study together with validation-set-calibrated "
    "per-class decision "
    "thresholds, calibration analysis and Grad-CAM-based visual evidence; (iv) a hybrid BM25-vector RAG pipeline "
    "constructed over a corpus of 1,577 Turkish biomedical documents (yielding 42,457 text chunks) utilizing "
    "OpenAI’s text-embedding-3-small model and the GPT-4o mini LLM backend; and (v) a production-grade web "
    "platform implemented on a React, Django and FastAPI microservices architecture — all within a single "
    "end-to-end framework. The proposed model achieves a macro-average AUC of 0.8342 with Test Time Augmentation "
    "(TTA), while the integrated RAG module attains a Hit Rate@10 of 56.0% and a mean LLM-as-a-Judge score of "
    "3.76/5.0. Collectively, these results demonstrate that the proposed system constitutes not merely a "
    "high-accuracy image classification model, but a clinically explainable, reliable and deployable AI-assisted "
    "decision support platform."
)

T[11] = (
    "This section provides a comprehensive account of the development pipeline for the proposed multimodal "
    "clinical decision support system. The subsections address, in turn: the dataset and patient-level data "
    "partitioning strategy; image preprocessing and augmentation; the multimodal deep learning architecture; "
    "training configuration and evaluation protocol; the ablation, calibration and interpretability protocols; "
    "the biomedical text corpus and vector database construction; the hybrid RAG architecture and retrieval "
    "strategy and the microservices-based deployment architecture."
)

T[13] = (
    "All experiments were conducted on the NIH ChestX-ray14 dataset [20], which comprises 112,120 frontal-view "
    "chest radiographs acquired from 30,805 unique patients. Each image is annotated in a multi-label format "
    "across 14 thoracic pathology classes — Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, "
    "Emphysema, Fibrosis, Pleural Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass and Hernia "
    "— as well as a “No Finding” category. No filtering, subsampling or class balancing was applied "
    "at the dataset level; all 112,120 images were retained."
)

T[14] = (
    "A critical but frequently overlooked methodological challenge in ChestX-ray14-based research is the presence "
    "of multiple radiographs per patient within the dataset. Conventional image-level random splitting strategies "
    "can inadvertently place images from the same patient in both the training and test sets, introducing data "
    "leakage and leading to artificially inflated performance estimates. To rigorously eliminate this confound, "
    "all data partitioning in this study was performed at the patient level using grouped random splitting "
    "(scikit-learn GroupShuffleSplit, grouping key = patient identifier, random seed = 42), ensuring complete "
    "mutual exclusivity between subsets. Each patient was assigned exclusively to one partition. Because grouping "
    "constrains the split at the patient rather than the image level, the realized partition sizes deviate "
    "slightly from the nominal 70/15/15 targets, yielding 78,566 images from 21,563 patients for training "
    "(70.1%), 16,106 images from 4,621 patients for validation (14.4%) and 17,448 images from 4,621 patients for "
    "testing (15.6%). Patient-overlap between all pairs of partitions was verified programmatically to be exactly "
    "zero. It should be explicitly noted that class-stratified allocation was not applied, as stratification and "
    "patient-level grouping cannot be simultaneously enforced without violating group exclusivity in a "
    "multi-label setting. Nevertheless, because the split operates over a large number of patient groups, the "
    "empirical pathology prevalences remain closely matched across partitions (e.g., “No Finding”: "
    "53.7% / 54.1% / 54.1% and Infiltration: 17.7% / 18.2% / 17.7% for train / validation / test, respectively). "
    "Complete per-partition image counts, patient counts and class prevalences are reported in Table 1. To "
    "support full reproducibility, the partitioning script emits a machine-readable split manifest recording "
    "these counts, the random seed and the verified zero patient-overlap for every run; this manifest and the "
    "partitioning code are available together with the model weights and evaluation scripts (see Data Statement)."
)

T[20] = (
    "The pipeline was organized into three complementary categories, with all transformation parameters specified "
    "explicitly to support reproduction. Geometric transformations: horizontal flipping (p = 0.5), random "
    "rotation within ±15° (p = 0.4) and random shift–scale–rotate (shift limit = 0.1, scale "
    "limit = 0.1, rotation limit = 15°, p = 0.4) were applied to simulate variability in patient positioning "
    "and acquisition conditions. Pixel-space transformations: random brightness and contrast perturbations "
    "(limits = ±0.25, p = 0.5), Contrast Limited Adaptive Histogram Equalization (CLAHE; clip limit = 2.0, "
    "tile grid = 8 × 8, p = 0.3) and additive Gaussian noise (variance range = 5–20, p = 0.2) were "
    "employed to improve robustness to acquisition-related intensity variations. Regularization-based "
    "augmentation: Coarse Dropout (2–5 rectangular masks of 8–16 pixels per side, p = 0.2) was applied "
    "to discourage over-reliance on localized anatomical regions, thereby promoting the learning of more "
    "distributed and generalizable spatial representations. The use of horizontal flipping in chest radiography "
    "warrants explicit justification, as it alters anatomical laterality. Three considerations support its "
    "inclusion here. First, none of the 15 ChestX-ray14 labels encodes a laterality-specific diagnosis; the "
    "annotation schema records the presence of a finding rather than the side on which it occurs, so flipping "
    "does not invalidate the label. Second, situs anomalies such as dextrocardia — the principal clinical "
    "scenario in which left–right orientation is diagnostically decisive — are rare in the general "
    "population and are not represented as a distinct class in this dataset. Third, horizontal flipping is "
    "standard practice in the ChestX-ray14 literature, including in the CheXNet training pipeline [5], which "
    "makes its use consistent with the comparison methods reported in Section 3.7. The same reasoning applies to "
    "the flip transformation used at inference time within the TTA ensemble (Section 2.4). Systems intended for "
    "deployment in settings where laterality must be preserved should nevertheless remove this transformation; "
    "the ablation reported in Section 3.4 quantifies the overall contribution of the augmentation pipeline."
)

T[22] = (
    "The proposed system is built upon a multimodal architecture that jointly encodes radiographic image features "
    "and structured demographic metadata through a learned modality-gating mechanism. The overall model "
    "architecture and data flow are depicted in Figure 2."
)

T[24] = ("Fig 2. Multimodal model architecture. The visual and demographic pathways are encoded separately and "
         "combined by a learned gating module that produces one scalar weight per modality; the weighted "
         "embeddings are concatenated and passed to the multi-label classifier head.")

T[25] = (
    "Visual encoder. EfficientNet-B3 [8], pretrained on ImageNet-1K [22], was adopted as the visual backbone. The "
    "original classification head was removed and replaced with a Global Average Pooling layer, followed by a "
    "dropout layer (rate = 0.55) to regularize the learned representations. This yields a compact visual "
    "embedding vector z_image ∈ ℝ¹⁵³⁶ for each input radiograph."
)

T[26] = (
    "Demographic encoder. Structured patient metadata comprised three variables: age, sex and imaging position "
    "(AP/PA/other). These were expanded into a 12-dimensional input vector as follows: three continuous age "
    "transformations — min–max normalization (age/100), a logarithmic transformation "
    "(log(1+age)/log(101)) and a quadratic term (age/100)² — followed by four mutually exclusive "
    "age-band indicators (< 18, 18–44, 45–64, ≥ 65 years), a two-dimensional one-hot encoding of "
    "sex and a three-dimensional one-hot encoding of projection (PA, AP, other). Missing or out-of-range values "
    "are handled deterministically: ages are clipped to the physiologically plausible interval [0, 120] years and "
    "missing ages are mean-imputed, while unrecognized or missing sex and projection values yield an all-zero "
    "encoding within the corresponding one-hot block, which the network can learn to interpret as an "
    "“unknown” state rather than as a spurious category. The 12-dimensional vector was passed through a "
    "three-layer multilayer perceptron — Linear(12→128) → BatchNorm → ReLU → "
    "Dropout(0.30) → Linear(128→128) → BatchNorm → ReLU → Dropout(0.25) → "
    "Linear(128→64) → BatchNorm → ReLU → Dropout(0.20) — producing a demographic "
    "embedding z_demo ∈ ℝ⁶⁴."
)

T[27] = (
    "Modality-gating fusion. The two embeddings are combined by a learned gating mechanism that assigns a scalar "
    "weight to each modality as a function of the joint representation. Formally, given the concatenated vector "
    "z = [z_image ; z_demo] ∈ ℝ¹⁶⁰⁰, the module computes h = ReLU(W₁ z + "
    "b₁) with W₁ ∈ ℝ⁴⁰⁰ˣ¹⁶⁰⁰, then [a_image, a_demo] "
    "= softmax(W₂ h + b₂) with W₂ ∈ ℝ²ˣ⁴⁰⁰ and a_image + "
    "a_demo = 1, and finally z_fused = [a_image · z_image ; a_demo · z_demo] ∈ "
    "ℝ¹⁶⁰⁰. Each modality block is therefore rescaled by a single input-dependent scalar "
    "before concatenation, allowing the network to modulate the relative influence of imaging and demographic "
    "evidence on a per-sample basis. We deliberately describe this mechanism as modality gating rather than "
    "self-attention: it does not construct token sequences, nor does it compute query, key and value "
    "projections, and it therefore differs fundamentally from dot-product attention as formulated by Vaswani et "
    "al. [12]. For completeness, a genuine multi-head self-attention variant was also implemented, in which the "
    "two embeddings are linearly projected into a shared 256-dimensional space to form a two-token sequence, "
    "processed by four-head self-attention with residual connection and layer normalization, and flattened to a "
    "512-dimensional fused representation. This variant is evaluated as an ablation condition in Section 3.4."
)

T[28] = (
    "Classifier head. Following fusion, a three-stage fully connected network was applied: Linear(1600→512) "
    "→ BatchNorm → ReLU → Dropout → Linear(512→256) → BatchNorm → ReLU → "
    "Dropout → Linear(256→128) → BatchNorm → ReLU → Dropout → "
    "Linear(128→15). Given the multi-label nature of the classification task, a sigmoid activation was "
    "applied at the output layer to produce independent probability estimates for each class. It is important to "
    "emphasize that “No Finding” is treated as an independent fifteenth output unit with its own "
    "learned decision function; it is not derived post hoc from the absence of the other 14 pathology "
    "predictions. Consequently, the model can, in principle, assign high probability to both “No "
    "Finding” and a pathology label for the same radiograph, and the two are reconciled at the reporting "
    "stage rather than by architectural constraint."
)

T[30] = (
    "Model training was conducted on the Kaggle cloud infrastructure using an NVIDIA Tesla T4 GPU (16 GB VRAM). "
    "Automatic Mixed Precision (AMP) was enabled to improve computational efficiency, using float16 as the "
    "reduced-precision format together with dynamic gradient scaling. We note explicitly that bfloat16 was not "
    "used, as the Tesla T4 (Turing architecture) does not provide native BF16 Tensor Core support; float16 with "
    "loss scaling is the appropriate mixed-precision configuration for this hardware. The batch size was set to "
    "36, the number of data loader workers to 4 and a global random seed of 42 was fixed to ensure "
    "reproducibility."
)

T[31] = (
    "The ChestX-ray14 dataset exhibits severe class imbalance, with per-class prevalences ranging from 0.2% "
    "(Hernia) to 53.8% (“No Finding”). Two complementary strategies were employed to address this. "
    "First, Focal Loss [23] was adopted as the primary objective function, with hyperparameters α = 0.25 and "
    "γ = 2.0, which down-weights well-classified examples and concentrates gradient updates on hard "
    "instances. Second, per-class positive weighting was incorporated into the loss computation. Each class i was "
    "assigned the standard negative-to-positive ratio wᵢ = N_negative,i / N_positive,i, capped at w_max = "
    "15, where the cap prevents the extremely rare classes (most notably Hernia, whose uncapped ratio exceeds "
    "490) from producing unstable gradient magnitudes. This yields weights ranging from 0.87 for “No "
    "Finding” to 15.0 for the rarest pathologies. We emphasize this formulation because an alternative "
    "weighting scheme — normalizing each class count by the sum of all class counts across the multi-label "
    "annotation rather than by its own negative count — produces weights below unity for the majority "
    "classes and, as demonstrated in the ablation of Section 3.4, causes a severe collapse in sensitivity; the "
    "correct pos-weight parameterization is therefore not a cosmetic implementation detail but a determinant of "
    "clinically usable behaviour."
)

T[32] = (
    "The AdamW optimizer [24] was used with an initial learning rate of 3 × 10⁻⁴ and weight decay "
    "of 1 × 10⁻⁴. Gradient-norm clipping at a maximum norm of 5.0 was applied at every "
    "optimization step, and any mini-batch producing a non-finite loss was skipped without updating the "
    "parameters; these safeguards were introduced after the multi-head self-attention fusion variant was observed "
    "to diverge numerically under float16 in preliminary runs. The full training pipeline was implemented in "
    "PyTorch [25]."
)

T[33] = (
    "A two-phase fine-tuning protocol was adopted. During the initial two epochs, the backbone was frozen and "
    "only the demographic encoder, fusion module and classification head were trained to stabilize early feature "
    "adaptation. In the subsequent 16 epochs, all parameters were unfrozen and trained jointly under a cosine "
    "annealing learning rate schedule, with the optimizer and scheduler reinitialized at the unfreezing boundary. "
    "To prevent overfitting, validation-AUC-based model checkpointing and early stopping with a patience of 9 "
    "epochs were applied throughout training; all reported test-set results were obtained from the checkpoint "
    "achieving the highest validation macro-AUC, never from the final-epoch weights."
)

T[34] = (
    "Model performance was assessed on the held-out test set using macro-averaged ROC-AUC, Average Precision "
    "(AP), F1-score, sensitivity (recall) and specificity. Two-sided 95% confidence intervals were obtained by "
    "percentile bootstrap with 1,000 resamples of the test set [37]. To improve prediction stability at inference "
    "time, Test Time Augmentation (TTA) was applied: five variants of each test image were generated — the "
    "unmodified image, a horizontally flipped copy, a random rotation drawn uniformly from ±5°, a fixed "
    "−5° rotation, and a mild brightness/contrast perturbation (limits = ±0.1) — with all other "
    "preprocessing identical to training-time inference — and their predicted probabilities were averaged "
    "to produce the final output. The statistical significance of the TTA effect was assessed by a paired "
    "bootstrap over the test set, in which each resample index set was applied identically to the non-TTA and TTA "
    "prediction matrices and the difference in macro-AUC was recorded."
)

# Bölüm numarası kaydırmaları (RAG bölümleri içerik olarak korunuyor)
T[35] = "2.6 Biomedical Text Corpus and Vector Database Construction"
T[38] = "2.7 RAG Architecture and Hybrid Retrieval Strategy"
T[47] = "2.8 System Architecture and Deployment"

T[60] = "3. RESULTS"   # orijinalde "3.RESULTS" (noktadan sonra boşluk yok)

T[61] = (
    "The proposed multimodal EfficientNet-B3 based system exhibited stable and consistent convergence behaviour "
    "throughout training. The model was trained for a total of 18 epochs with the early-stopping mechanism "
    "enabled; the stopping criterion was not triggered, as no sustained deterioration in validation performance "
    "occurred. The highest validation macro-AUC of 0.8392 was attained at epoch 12, and this checkpoint — "
    "rather than the final-epoch weights — was used for all subsequently reported test-set results. At the "
    "selected checkpoint the training loss was 0.0845 against a validation loss of 0.0871, and the training "
    "macro-AUC was 0.8450 against a validation macro-AUC of 0.8392; the train–validation AUC gap of 0.006 "
    "indicates that the data augmentation strategy, dropout regularization and patient-level partitioning "
    "together provided effective control of overfitting. This contrast is made explicit by the no-augmentation "
    "ablation reported in Section 3.4, in which the same gap widens progressively to 0.141 by epoch 10. Training "
    "was performed on a single NVIDIA Tesla T4 GPU."
)

T[63] = ("The per-class performance of the proposed model on the held-out test set is reported in Table 2. "
         "Results correspond to standard inference at the default decision threshold τ = 0.5, without Test "
         "Time Augmentation.")

T[64] = ("Table 2. Per-class classification performance on the NIH ChestX-ray14 test set (n = 17,448 images from "
         "4,621 patients), reported at the default threshold τ = 0.5 and without TTA. AP = average "
         "precision; Support = number of positive cases in the test set.")

T[65] = (
    "The model achieves a macro-average AUC of 0.8309 across all 15 classes, comprising the 14 thoracic pathology "
    "categories together with the “No Finding” class. Restricting the average to the 14 pathology "
    "labels — the convention adopted by the comparison methods discussed in Section 3.7 — yields a "
    "macro-AUC of 0.8340, since “No Finding” (AUC = 0.7875) falls below the overall mean. Both figures "
    "are reported to make the comparison in Section 3.7 explicit rather than implicit. Performance varies "
    "considerably across pathology types, a pattern that is both expected and clinically interpretable given the "
    "heterogeneous nature of the classification task."
)

T[66] = (
    "The highest discriminative performance was obtained for Emphysema (AUC = 0.9258) and Cardiomegaly (AUC = "
    "0.9033) — two pathologies characterized by prominent, large-scale structural alterations in thoracic "
    "anatomy. In Emphysema, global morphological changes such as diaphragmatic flattening and pulmonary "
    "hyperinflation produce distinctive textural and shape signatures that CNN-based feature extractors can "
    "reliably learn. Similarly, in Cardiomegaly, the enlargement of the cardiac silhouette constitutes a "
    "spatially diffuse and visually salient alteration, facilitating high specificity and sensitivity. Strong "
    "performance was also observed for Pneumothorax (AUC = 0.8952) and Edema (AUC = 0.8913), consistent with the "
    "comparatively prominent density changes and anatomical disruptions these conditions produce on radiographs."
)

T[67] = (
    "In contrast, Infiltration (AUC = 0.7144), Pneumonia (AUC = 0.7705) and Nodule (AUC = 0.7780) represented the "
    "most challenging classes. Infiltration is a well-recognized diagnostic challenge in the ChestX-ray14 "
    "literature, exhibiting substantial radiographic overlap with pneumonia, consolidation and atelectasis. "
    "Furthermore, the NLP-based automated labeling pipeline used to annotate the dataset is known to introduce "
    "label noise disproportionately affecting this class [28], which likely contributes to its comparatively "
    "limited learnability."
)

T[68] = (
    "The reduced performance on the Nodule class is attributable primarily to input resolution constraints: at "
    "300 × 300 pixels, small pulmonary nodules occupy only a few pixels and are readily confounded with "
    "vascular cross-sections or image noise, resulting in elevated false-negative rates. This limitation "
    "underscores a known challenge in low-resolution CXR analysis and motivates future work incorporating "
    "higher-resolution inputs or lesion-specific detection modules."
)

T[69] = (
    "The macro-average sensitivity of 0.6977 obtained at the default threshold indicates that the model produces "
    "positive predictions across all classes, including the high-prevalence categories, and the macro-average F1 "
    "of 0.2796 reflects the precision penalty inherent to detecting low-prevalence pathologies within a large "
    "negative population rather than a failure of discrimination. The relationship between threshold selection "
    "and this precision–recall trade-off is examined systematically in Section 3.2. The comparatively low F1 "
    "values for rare classes such as Pneumonia (prevalence 1.2%) and Fibrosis (1.6%) are a direct consequence of "
    "prevalence: even a highly specific classifier accumulates a substantial number of false positives when "
    "screening 17,448 images for a condition present in fewer than 300 of them."
)

T[70] = "3.3 Impact of Test Time Augmentation"

T[71] = ("The effect of Test Time Augmentation on model performance is summarized in Table 4, reported for all 15 "
         "classes rather than a selected subset. TTA was implemented by generating five augmented variants of "
         "each test image, as specified in Section 2.4, and averaging the resulting predicted probability "
         "distributions.")

T[72] = "Table 4. Effect of Test Time Augmentation on per-class AUC (all 15 classes)."

T[73] = (
    "TTA improved AUC in all 15 of 15 classes, raising the macro-average AUC from 0.8309 to 0.8342 and the "
    "macro-average AP from 0.2977 to 0.3016. A paired bootstrap over the test set (1,000 resamples, identical "
    "resample indices applied to both prediction sets) yields a mean macro-AUC difference of +0.0032 with a 95% "
    "confidence interval of [+0.0025, +0.0039]; the interval excludes zero in every resample, indicating that the "
    "improvement is systematic rather than an artifact of sampling variation. We nevertheless emphasize that "
    "statistical consistency should not be conflated with clinical significance: an absolute AUC gain of "
    "approximately 0.003 is small, and TTA is best characterized as a low-cost inference-time stabilization "
    "technique rather than a substantive performance improvement. Its computational cost is quantified in "
    "Section 3.5."
)

T[74] = ("The largest per-class improvement was observed for the Hernia class (+0.0073), consistent with the "
         "expectation that low-frequency classes exhibit higher prediction variance and therefore benefit most "
         "from averaging across augmented views; the smallest gains occurred for Edema (+0.0007) and Infiltration "
         "(+0.0008).")

T[75] = "3.7 Comparison with Reference Methods"

T[76] = ("The performance of the proposed system was benchmarked against reference methods evaluated on the NIH "
         "ChestX-ray14 dataset. Comparative results are presented in Table 7.")

T[77] = ("Table 7. Comparison with reference methods on NIH ChestX-ray14. Macro-AUC for the proposed model is "
         "reported over the 14 pathology labels for consistency with the comparison methods; the 15-class figure "
         "including “No Finding” is 0.8342. N/R = not reported in the cited work.")

T[78] = (
    "An important methodological caveat must accompany this table, and it applies with particular force to the "
    "most recent entries. The cited studies differ from the present work — and from one another — in "
    "input resolution, preprocessing, data partitioning protocol, label handling, threshold selection and "
    "evaluation convention. The spread of reported values is itself informative: within the same benchmark and "
    "the same 14 labels, published macro-AUC figures range from 0.738 to 0.97 and macro-F1 from 0.39 to 0.92. "
    "Such a range cannot plausibly reflect model quality alone. Results in the upper part of this range are "
    "typically associated with one or more of the following: image-level rather than patient-level partitioning, "
    "which permits radiographs of the same patient to appear in both training and test sets; per-class thresholds "
    "optimized on the test set rather than on a held-out validation set; or averaging conventions under which the "
    "overwhelming negative majority of a rare class inflates the reported score. We were unable to verify the "
    "partitioning protocol for two of the recent entries from the publicly available versions of those papers, "
    "and we therefore reproduce their published figures without endorsing their comparability. For this reason, "
    "the values in Table 7 are indicative rather than strictly commensurable, and no claim of superiority is made "
    "on the basis of this comparison. A definitive ranking would require retraining all methods under a single "
    "common protocol — identical patient-level split, identical resolution and identical threshold-selection "
    "procedure — which is beyond the scope of this study but which we would encourage as a community "
    "benchmark."
)

T[79] = (
    "Within these limits, the proposed model’s 14-class macro-AUC of 0.8374 exceeds the original NIH "
    "baseline of Wang et al. [1] (0.738), the CNN-LSTM architecture of Yao et al. [29] (0.798), the multimodal "
    "ResNet-50 system of Baltruschat et al. [11] (0.806) and the Xception-based system of Majkowska et al. [7] "
    "(0.816); is essentially identical to the CvT-graph hybrid of Lu et al. [32] (0.8376); and falls marginally "
    "below CheXNet [5] (0.841) and the patient-level-split CheXNet reproduction of Strick et al. [34] (0.8527), "
    "the latter being the most directly comparable recent entry in the table since it explicitly adopts "
    "patient-level partitioning. The comparison with Baltruschat et al. is the most methodologically similar in "
    "terms of input modality, as both systems combine imaging with structured metadata; however, in light of the "
    "ablation results in Section 3.4 — which show that metadata contributes no measurable AUC gain in our "
    "setting — we attribute the difference primarily to the backbone and training pipeline rather than to "
    "the fusion strategy. Macro-F1 is reported for the proposed model to address the limited informativeness of "
    "AUC alone under severe class imbalance; where the comparison methods did not report this metric, the "
    "corresponding cells are marked as not reported rather than estimated."
)

T[80] = "3.8 RAG Module Evaluation"
T[81] = ("The performance of the proposed Retrieval-Augmented Generation module was assessed through a two-stage "
         "evaluation framework encompassing retrieval quality and response generation quality. Evaluation was "
         "conducted against a curated reference set of 25 expert-level clinical questions spanning pulmonary "
         "medicine, radiology and infectious diseases. Quantitative results are reported in Table 8.")
T[82] = "Table 8. RAG system evaluation results (n=25 clinical reference questionnaires)"

T[87] = ("Per-class discriminative performance is visualized through ROC curves presented in Figure 8. The "
         "majority of curves are positioned in close proximity to the upper-left corner of the ROC space, "
         "confirming strong discriminative capacity across most pathology classes.")
T[89] = "Fig 8. Disease-Based ROC Curves"
T[91] = "Confusion matrices for all classes at the default threshold are presented in Figure 9."
T[93] = "Fig 9. Confusion matrices"

T[94] = (
    "Consistent with the sensitivity values reported in Table 2, the model produces substantial true-positive "
    "counts across all classes, including the high-prevalence “No Finding”, Atelectasis and "
    "Infiltration categories for which the earlier weighting configuration had suppressed positive predictions "
    "entirely (Section 3.4). As expected from the per-class AUC analysis, the Nodule class exhibits comparatively "
    "elevated false-negative rates, reflecting the inherent difficulty of detecting small lesions at low input "
    "resolution. This observation reinforces the conclusion that sub-centimeter pulmonary structure detection at "
    "300 × 300 pixel resolution remains a significant technical challenge and one that warrants dedicated "
    "attention in future system iterations."
)

T[96] = (
    "The proposed multimodal clinical decision support system demonstrates competitive performance on the NIH "
    "ChestX-ray14 benchmark, achieving a macro-average AUC of 0.8342 over all 15 classes (0.8374 over the 14 "
    "pathology labels) following Test Time Augmentation, while using a computationally efficient EfficientNet-B3 "
    "backbone. This places the system in the same performance range as established reference methods evaluated on "
    "this dataset, obtained under a strict patient-level partitioning protocol that precludes the optimistic bias "
    "introduced by image-level splitting. The controlled ablation study reported in Section 3.4 requires us to "
    "state the contribution of multimodality more precisely than is customary in this literature. Under identical "
    "training conditions and an identical epoch budget, an image-only EfficientNet-B3 attains a macro-AUC of "
    "0.8307 against 0.8323 for the full multimodal model and 0.8320 for simple feature concatenation — "
    "separations of 0.0016 and 0.0003 respectively, measured against a 0.0014 spread observed for one and the "
    "same configuration trained to two different epoch budgets. We therefore do not claim that demographic "
    "fusion improves discriminative performance in this setting; the evidence does not support such a claim. What the ablation does establish is twofold and, in our view, more "
    "useful: first, that the model’s predictive signal derives predominantly from radiographic content "
    "rather than from demographic or acquisition-related shortcuts, as evidenced by the 0.22 AUC gap between "
    "metadata-only (0.6124) and image-only performance; and second, that additional fusion complexity is not "
    "automatically beneficial — a genuine multi-head self-attention module underperformed both gating and "
    "plain concatenation (0.8119) and exhibited numerical instability under mixed-precision training. Multimodal "
    "architectures for CXR classification should therefore be justified by measured ablation rather than by "
    "architectural appeal, and we would encourage reporting of image-only baselines as standard practice in this "
    "area."
)

T[97] = (
    "Examination of the per-class performance profile reveals a diagnostically coherent pattern. Pathologies "
    "associated with large-scale, structurally prominent anatomical alterations — namely Emphysema and "
    "Cardiomegaly — are detected with high accuracy, while conditions characterized by smaller, more "
    "diffuse, or visually ambiguous anomalies — including Nodule, Infiltration and Pneumonia — prove "
    "more challenging. Critically, this performance gradient is not merely an architectural artifact; it mirrors "
    "the inherent visual ambiguity of chest radiography and the well-documented inter-observer variability among "
    "human radiologists interpreting these findings [28]. The model’s error patterns therefore exhibit "
    "substantial correspondence with those of clinical experts, suggesting that the remaining limitations reflect "
    "fundamental constraints of the imaging modality and of the label-generation process rather than "
    "straightforwardly correctable model deficiencies."
)

T[98] = (
    "A finding of particular methodological significance concerns the interaction between class-imbalance "
    "correction and threshold-based metrics. In an earlier configuration of this system, several classes — "
    "including the high-prevalence “No Finding”, Atelectasis and Infiltration categories — "
    "exhibited zero sensitivity and zero F1 at the fixed threshold τ = 0.5 despite acceptable AUC values."
)

T[99] = (
    "Detailed analysis identified the cause precisely: the positive-class weights had been computed by "
    "normalizing each class count against the sum of all class counts in the multi-label annotation rather than "
    "against that class’s own negative count, which produced positive weights below unity (as low as 0.157) "
    "for exactly the high-prevalence classes. Because the positive weight in a weighted binary cross-entropy "
    "objective scales the loss contribution of positive examples only, weights below unity penalize missed "
    "positives less than false positives, systematically driving predicted probabilities for these classes below "
    "the decision threshold while leaving the ranking — and hence AUC — intact. Correcting the "
    "weighting to the standard negative-to-positive ratio raised macro-average sensitivity from 0.2845 to 0.6977 "
    "and macro-average F1 from 0.1769 to 0.2796, and the controlled ablation in Section 3.4 reproduces the "
    "failure mode on demand: reinstating the naive scheme under otherwise identical conditions collapses "
    "macro-sensitivity to 0.1920. We report this diagnosis in full because the failure is invisible to AUC-based "
    "evaluation and, given the prevalence of both focal loss and class weighting in the medical imaging "
    "literature, is unlikely to be unique to this implementation."
)

T[100] = (
    "This finding underscores a critical distinction in multi-label medical imaging evaluation: AUC, being a "
    "threshold-independent metric, is a more reliable indicator of discriminative capacity in heavily reweighted, "
    "imbalanced settings, whereas F1 and sensitivity are highly sensitive to the choice of decision threshold and "
    "should be reported alongside AUC precisely because they surface calibration failures that ranking metrics "
    "conceal. Related to this, the per-class threshold calibration reported in Section 3.2 establishes that a "
    "single global threshold is inappropriate for multi-label CXR classification. Calibrating thresholds on the "
    "validation set alone and applying them unchanged to the test set raises macro-sensitivity to 0.7662 at a "
    "macro-specificity of 0.7515, or macro-F1 to 0.3640 under an F1-maximizing criterion. These operating points, "
    "rather than τ = 0.5, are the appropriate basis for any discussion of clinical utility, and we report "
    "them explicitly with bootstrap confidence intervals and Brier scores so that the reliability of the "
    "underlying probabilities can be assessed independently of the chosen threshold."
)

T[102] = (
    "In this study, all partitioning was performed exclusively at the patient identifier level, guaranteeing "
    "complete mutual exclusivity between training, validation and test sets, with zero patient overlap verified "
    "programmatically and the exact per-partition image counts, patient counts and class prevalences reported in "
    "Table 1 together with a machine-readable split manifest. This design provides a more faithful estimate of "
    "the model’s capacity to generalize to truly novel patients — the operationally relevant "
    "performance criterion for any real-world clinical AI system. The consistent performance observed across "
    "validation and test sets, together with the absence of overfitting during training, corroborates the "
    "effectiveness of this approach. It is the position of this work that patient-level partitioning should be "
    "adopted as the minimum methodological standard for evaluation on multi-image clinical imaging datasets and "
    "that results obtained under image-level splitting should be interpreted with caution when benchmarking "
    "against patient-level results. The system additionally provides two distinct and complementary forms of "
    "interpretability, which we consider it important not to conflate. Grad-CAM analysis (Section 3.6) addresses "
    "the question of why this prediction for this radiograph, producing patient-specific spatial evidence that a "
    "clinician can inspect against the image in front of them. The RAG module, by contrast, addresses the "
    "question of what this predicted finding means clinically. Knowledge retrieval is not a substitute for visual "
    "explanation, and we therefore report the two mechanisms separately rather than presenting retrieval alone as "
    "“explainable AI”."
)

T[110] = (
    "Third, while the modality-gating mechanism proved stable and interpretable, more advanced multimodal "
    "architectures — including vision-language transformers and cross-modal attention frameworks — may "
    "offer additional representational capacity when trained at larger scale than was feasible here. In "
    "particular, medically pretrained vision-language models such as BioViL-T [13], which learn joint image-text "
    "representations within a shared embedding space, could provide stronger semantic alignment between "
    "radiographic findings and clinical descriptions. A further limitation concerns the experimental protocol of "
    "the ablation study itself: all eight configurations were trained under a reduced 10-epoch budget to remain "
    "within available compute constraints, and all results derive from single training runs rather than seed "
    "ensembles. The 0.0014 macro-AUC difference between the proposed model at 10 and 18 epochs gives one "
    "estimate of that variability, and the separations between image-only, concatenation and gating "
    "(0.0003–0.0016) fall at or below it, so they should not be over-interpreted. Replication over multiple "
    "random seeds would place proper confidence intervals on these comparisons. Finally, the comparison with reference methods in "
    "Table 7 aggregates studies that differ in split protocol, resolution, preprocessing and evaluation "
    "convention; as discussed in Section 3.7, these values are indicative rather than strictly commensurable, and "
    "no superiority claim is made on their basis."
)

T[113] = "5. CONCLUSION"

T[114] = (
    "This study presented an integrated multimodal clinical decision support framework for automated thoracic "
    "disease recognition, combining an EfficientNet-B3 architecture with learned modality gating and a "
    "Retrieval-Augmented Generation pipeline. The proposed system achieves a macro-average AUC of 0.8342 across "
    "15 classes on the NIH ChestX-ray14 benchmark (0.8374 across the 14 pathology labels) under strict "
    "patient-level data partitioning, ensuring that reported results reflect genuine generalization to unseen "
    "patients rather than patient-specific memorization artifacts. Beyond the headline metric, this work "
    "contributes a controlled eight-configuration, epoch-matched ablation study establishing that the "
    "discriminative signal "
    "originates predominantly from radiographic content rather than demographic shortcuts, that additional fusion "
    "complexity is not automatically beneficial, and that the parameterization of positive-class weighting "
    "— rather than the choice of loss function or fusion mechanism — is the single most consequential "
    "determinant of clinically usable sensitivity. Per-class decision thresholds calibrated exclusively on the "
    "validation set raise macro-average sensitivity to 0.7662 at a macro-specificity of 0.7515, and are reported "
    "together with bootstrap confidence intervals, Brier scores and precision–recall curves so that "
    "operating points can be selected according to clinical context rather than by convention."
)

T[116] = (
    "Collectively, these results indicate that multimodal deep learning combined with retrieval-augmented "
    "generation offers an interpretable and deployable path toward AI-assisted chest radiography, while also "
    "showing that careful ablation and threshold calibration are necessary to characterize such systems honestly. "
    "Future work will focus on expert-annotated dataset validation, multi-seed ablation "
    "replication, quantitative localization evaluation against expert bounding boxes, advanced vision-language "
    "architectures and prospective multicenter clinical evaluation."
)

T[118] = (
    "The NIH ChestX-ray14 dataset is publicly accessible via the Kaggle platform [20] "
    "(https://www.kaggle.com/datasets/nih-chest-xrays/data) and ultimately through the NIH Clinical Centre. The "
    "DergiPark corpus used to build the RAG knowledge base is accessible via the DergiPark platform "
    "(https://dergipark.org.tr) subject to individual journal open access policies. Model weights, training code, "
    "the patient-level partitioning script, the generated split manifest recording exact per-partition image and "
    "patient counts, the ablation driver and all evaluation scripts (including the threshold-calibration, "
    "bootstrap-significance and Grad-CAM utilities) are available from the corresponding author upon reasonable "
    "request."
)


# ─────────────────────────── tablo verileri ───────────────────────────

TBL_SPLIT = [
    ['Class', 'Train, n (%)', 'Validation, n (%)', 'Test, n (%)'],
    ['No Finding', '42,210 (53.7)', '8,717 (54.1)', '9,434 (54.1)'],
    ['Infiltration', '13,868 (17.7)', '2,933 (18.2)', '3,093 (17.7)'],
    ['Effusion', '9,438 (12.0)', '1,841 (11.4)', '2,038 (11.7)'],
    ['Atelectasis', '8,190 (10.4)', '1,611 (10.0)', '1,758 (10.1)'],
    ['Nodule', '4,432 (5.6)', '920 (5.7)', '979 (5.6)'],
    ['Mass', '4,260 (5.4)', '762 (4.7)', '760 (4.4)'],
    ['Pneumothorax', '3,770 (4.8)', '697 (4.3)', '835 (4.8)'],
    ['Consolidation', '3,228 (4.1)', '668 (4.1)', '771 (4.4)'],
    ['Pleural Thickening', '2,379 (3.0)', '470 (2.9)', '536 (3.1)'],
    ['Cardiomegaly', '1,891 (2.4)', '389 (2.4)', '496 (2.8)'],
    ['Emphysema', '1,666 (2.1)', '378 (2.3)', '472 (2.7)'],
    ['Edema', '1,630 (2.1)', '339 (2.1)', '334 (1.9)'],
    ['Fibrosis', '1,156 (1.5)', '253 (1.6)', '277 (1.6)'],
    ['Pneumonia', '995 (1.3)', '226 (1.4)', '210 (1.2)'],
    ['Hernia', '158 (0.2)', '35 (0.2)', '34 (0.2)'],
    ['Total images', '78,566 (70.1)', '16,106 (14.4)', '17,448 (15.6)'],
    ['Unique patients', '21,563', '4,621', '4,621'],
]

TBL_PERCLASS = [
    ['Disease Class', 'AUC', 'AP', 'F1 Score', 'Sensitivity', 'Specificity', 'Support'],
    ['Emphysema', '0.9258', '0.4084', '0.3830', '0.7564', '0.9390', '472'],
    ['Cardiomegaly', '0.9033', '0.3229', '0.3078', '0.7440', '0.9096', '496'],
    ['Pneumothorax', '0.8952', '0.3363', '0.3530', '0.7760', '0.8683', '835'],
    ['Edema', '0.8913', '0.1717', '0.1516', '0.7994', '0.8293', '334'],
    ['Effusion', '0.8779', '0.5162', '0.4047', '0.9004', '0.6629', '2,038'],
    ['Mass', '0.8510', '0.3004', '0.2582', '0.7211', '0.8241', '760'],
    ['Hernia', '0.8491', '0.2171', '0.1538', '0.3235', '0.9944', '34'],
    ['Atelectasis', '0.8188', '0.3622', '0.3200', '0.8595', '0.6064', '1,758'],
    ['Fibrosis', '0.8032', '0.0956', '0.1337', '0.4585', '0.9129', '277'],
    ['Pleural Thickening', '0.8009', '0.1443', '0.1766', '0.6250', '0.8272', '536'],
    ['Consolidation', '0.7974', '0.1366', '0.1864', '0.7938', '0.6892', '771'],
    ['No Finding', '0.7875', '0.8028', '0.7276', '0.6785', '0.7803', '9,434'],
    ['Nodule', '0.7780', '0.2514', '0.2109', '0.7038', '0.7045', '979'],
    ['Pneumonia', '0.7705', '0.0437', '0.0859', '0.4190', '0.8984', '210'],
    ['Infiltration', '0.7144', '0.3560', '0.3412', '0.9062', '0.2662', '3,093'],
    ['Macro Average', '0.8309', '0.2977', '0.2796', '0.6977', '0.7808', '—'],
]

TBL_THRESH = [
    ['Disease Class', 'AUC [95% CI]', 'Brier', 'Youden τ', 'Sensitivity', 'Specificity', 'PPV', 'F1'],
    ['Emphysema', '0.9258 [0.9119–0.9390]', '0.0825', '0.412', '0.8263', '0.8840', '0.1653', '0.2754'],
    ['Cardiomegaly', '0.9033 [0.8877–0.9178]', '0.0958', '0.329', '0.8770', '0.7672', '0.0993', '0.1784'],
    ['Pneumothorax', '0.8952 [0.8839–0.9067]', '0.1300', '0.351', '0.8814', '0.7448', '0.1479', '0.2534'],
    ['Edema', '0.8913 [0.8736–0.9063]', '0.1227', '0.441', '0.8443', '0.7925', '0.0736', '0.1353'],
    ['Effusion', '0.8779 [0.8702–0.8858]', '0.2049', '0.591', '0.8386', '0.7685', '0.3239', '0.4673'],
    ['Mass', '0.8510 [0.8368–0.8650]', '0.1650', '0.456', '0.7605', '0.7776', '0.1347', '0.2289'],
    ['Hernia', '0.8491 [0.7667–0.9185]', '0.0223', '0.142', '0.7647', '0.7311', '0.0055', '0.0110'],
    ['Atelectasis', '0.8188 [0.8088–0.8289]', '0.2396', '0.538', '0.8146', '0.6660', '0.2146', '0.3397'],
    ['Fibrosis', '0.8032 [0.7747–0.8293]', '0.1091', '0.356', '0.7184', '0.7376', '0.0423', '0.0799'],
    ['Pleural Thickening', '0.8009 [0.7814–0.8215]', '0.1599', '0.427', '0.7649', '0.7245', '0.0809', '0.1463'],
    ['Consolidation', '0.7974 [0.7826–0.8124]', '0.1972', '0.468', '0.8184', '0.6485', '0.0972', '0.1737'],
    ['No Finding', '0.7875 [0.7806–0.7937]', '0.2065', '0.492', '0.6938', '0.7694', '0.7798', '0.7343'],
    ['Nodule', '0.7780 [0.7624–0.7937]', '0.2226', '0.513', '0.6782', '0.7362', '0.1326', '0.2218'],
    ['Pneumonia', '0.7705 [0.7381–0.8011]', '0.1143', '0.399', '0.6381', '0.7713', '0.0329', '0.0625'],
    ['Infiltration', '0.7144 [0.7048–0.7238]', '0.2933', '0.625', '0.5739', '0.7530', '0.3336', '0.4219'],
    ['Macro Average', '0.8309', '0.1577', '0.436', '0.7662', '0.7515', '—', '0.2487'],
]

TBL_TTA = [
    ['Disease Class', 'AUC before TTA', 'AUC after TTA', 'Δ AUC'],
    ['Emphysema', '0.9258', '0.9285', '+0.0028'],
    ['Cardiomegaly', '0.9033', '0.9074', '+0.0041'],
    ['Pneumothorax', '0.8952', '0.8985', '+0.0033'],
    ['Edema', '0.8913', '0.8919', '+0.0007'],
    ['Effusion', '0.8779', '0.8792', '+0.0014'],
    ['Mass', '0.8510', '0.8543', '+0.0033'],
    ['Hernia', '0.8491', '0.8564', '+0.0073'],
    ['Atelectasis', '0.8188', '0.8221', '+0.0033'],
    ['Fibrosis', '0.8032', '0.8073', '+0.0041'],
    ['Pleural Thickening', '0.8009', '0.8043', '+0.0034'],
    ['Consolidation', '0.7974', '0.8001', '+0.0027'],
    ['No Finding', '0.7875', '0.7893', '+0.0018'],
    ['Nodule', '0.7780', '0.7834', '+0.0053'],
    ['Pneumonia', '0.7705', '0.7748', '+0.0043'],
    ['Infiltration', '0.7144', '0.7152', '+0.0008'],
    ['Macro Average', '0.8309', '0.8342', '+0.0032'],
]

TBL_ABLATION = [
    ['Configuration', 'Modalities', 'Fusion', 'Test macro AUC', 'Test macro F1', 'Test macro Sens.'],
    ['Proposed model (10 epochs)', 'image + metadata', 'gating', '0.8323', '0.2740', '0.6957'],
    ['Image-only', 'image', '—', '0.8307', '0.2801', '0.6915'],
    ['Metadata-only', 'metadata', '—', '0.6124', '0.1219', '0.5229'],
    ['Simple concatenation', 'image + metadata', 'concat', '0.8320', '0.2707', '0.6891'],
    ['Multi-head self-attention', 'image + metadata', 'self-attention', '0.8119', '0.2334', '0.6848'],
    ['Naive class weighting', 'image + metadata', 'gating', '0.8112', '0.1431', '0.1920'],
    ['No focal loss (plain BCE)', 'image + metadata', 'gating', '0.8257', '0.3026', '0.6073'],
    ['No augmentation', 'image + metadata', 'gating', '0.8193', '0.2660', '0.6863'],
    ['Proposed model (18 epochs, ref.)', 'image + metadata', 'gating', '0.8309', '0.2796', '0.6977'],
]

TBL_PROFILE = [
    ['Model', 'Parameters (M)', 'GFLOPs', 'Latency bs=1 (ms)', 'Latency bs=36 (ms)', '5× TTA (ms)', 'Peak mem. bs=36 (MB)'],
    ['Proposed (B3 + metadata, gating)', '12.35', '3.86', '20.2', '114.4', '101.1', '1,111'],
    ['B3 + metadata, self-attention', '11.83', '3.86', '18.2', '114.5', '90.8', '1,159'],
    ['Image-only EfficientNet-B3', '11.65', '3.86', '15.5', '114.1', '77.3', '1,204'],
    ['DenseNet-121 (image-only baseline)', '6.97', '10.09', '24.1', '126.6', '120.3', '825'],
]

TBL_SOTA = [
    ['Work', 'Year', 'Input', 'Architecture', 'Split protocol', 'Macro AUC', 'Macro F1'],
    ['Wang et al. [1]', '2017', 'Image only', 'ResNet-50', 'Official patient-level', '0.738', 'N/R'],
    ['Rajpurkar et al. [5]', '2017', 'Image only', 'DenseNet-121', 'Patient-level', '0.841', 'N/R'],
    ['Yao et al. [29]', '2018', 'Image only', 'CNN + LSTM', 'Patient-level', '0.798', 'N/R'],
    ['Baltruschat et al. [11]', '2019', 'Image + metadata', 'ResNet-50 + MLP', 'Patient-level', '0.806', 'N/R'],
    ['Majkowska et al. [7]', '2020', 'Image only', 'Xception', 'Patient-level', '0.816', 'N/R'],
    ['Lu et al. [32]', '2024', 'Image only', 'CvT + graph (CvTGNet)', 'Not stated', '0.8376', 'N/R'],
    ['Strick et al. [34]', '2025', 'Image only', 'DenseNet variant', 'Own patient-level split', '0.8527', '0.3861'],
    ['Fisher [33]', '2025', 'Image only', 'ConvNeXt 3-model ensemble', 'Not verified', '0.940', '0.821'],
    ['Gejje et al. [35]', '2025', 'Image only', 'ResNet-50 transfer learning', 'Not verified', '0.97', '0.917'],
    ['Proposed model', '2026', 'Image + metadata', 'EfficientNet-B3 + gating + TTA', 'Patient-level (this work)', '0.8374', '0.2817'],
]

NEW_REFS = [
    "[32] Lu, Y., Hu, Y., Li, L., Xu, Z., Liu, H., Liang, H., & Fu, X. (2024). CvTGNet: A novel framework for "
    "chest X-ray multi-label classification. In Proceedings of the 21st ACM International Conference on Computing "
    "Frontiers (pp. 12–20).",
    "[33] Fisher, G. (2025). Pretraining diversity and clinical metric optimization achieve state-of-the-art "
    "performance on ChestX-ray14. medRxiv, 2025-10.",
    "[34] Strick, D. J., Garcia, C., Huang, A., & Gardos, T. (2025). Reproducing and improving CheXNet: Deep "
    "learning for chest X-ray disease classification. arXiv preprint arXiv:2505.06646.",
    "[35] Gejje, S., Joshi, A., Kaur, J., Divyaa, N., Bakyarani, E. S., & Begum, M. B. (2025). Transfer learning "
    "with ResNet-50 for multi-label chest X-ray classification on the NIH ChestX-Ray14 dataset. In 2025 2nd Asian "
    "Conference on Intelligent Technologies (ACOIT) (pp. 1–7). IEEE.",
    "[36] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual "
    "explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International "
    "Conference on Computer Vision (ICCV) (pp. 618–626).",
    "[37] Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.",
]


# ─────────────────────────── ana akış ───────────────────────────

def main():
    shutil.copy(SRC, DST)
    doc = docx.Document(DST)

    from docx.text.paragraph import Paragraph
    from docx.table import Table

    pars, tbls = [], []
    for child in doc.element.body.iterchildren():
        tag = child.tag.split('}')[1]
        if tag == 'p':
            pars.append(Paragraph(child, doc))
        elif tag == 'tbl':
            tbls.append(Table(child, doc))

    print(f'kaynak: {len(pars)} paragraf, {len(tbls)} tablo')

    # 1) metin değişimleri
    for idx, text in T.items():
        set_text(pars[idx], text)
    print(f'✓ {len(T)} paragraf metni güncellendi')

    # 2) mevcut tabloları doldur  (T1=per-class, T2=TTA, T3=SOTA)
    fill_table(tbls[1], TBL_PERCLASS, bold_last_row=True, font_pt=8)
    fill_table(tbls[2], TBL_TTA, bold_last_row=True, font_pt=8)
    fill_table(tbls[3], TBL_SOTA, bold_last_row=True, font_pt=7.5,
               left_cols=LEFT_COLS['sota'])
    set_table_widths(tbls[1], COL_WIDTHS['perclass'])
    set_table_widths(tbls[2], COL_WIDTHS['tta'])
    set_table_widths(tbls[3], COL_WIDTHS['sota'])
    print('✓ mevcut 3 veri tablosu güncellendi (per-class, TTA, SOTA)')

    # 2b) Fig 5 panel tablosu orijinalde de sağ marja taşıyordu (6.75 > 6.27 inç):
    #     görselleri hafifçe küçültüp ızgarayı yazı alanına sığdır.
    scale_images_in(tbls[0]._element, 0.93)
    set_table_widths(tbls[0], [2.09, 2.09, 2.09])
    # Fig 5.a görseli de orijinalde 6.34 inçti (yazı alanı 6.27)
    scale_images_in(pars[53]._element, 6.20 / 6.34)
    print('✓ Fig 5 panel tablosu + Fig 5.a sayfa genişliğine sığdırıldı')

    style_src = tbls[1]

    # 3) YENİ Tablo 1 (split) — 2.1 paragrafından (P014) sonra
    cap1 = clone_par_after(pars[14]._element, pars[64],
                           'Table 1. Per-partition image counts, patient counts and class prevalence for the '
                           'patient-level split of NIH ChestX-ray14 (random seed = 42). Percentages are computed '
                           'with respect to the number of images in the corresponding partition; because the '
                           'dataset is multi-label, column percentages sum to more than 100%.')
    tbl1 = make_table(doc, TBL_SPLIT, style_src, pars[14]._element, bold_last_row=True, font_pt=8)
    set_table_widths(tbl1, COL_WIDTHS['split'])
    print('✓ Tablo 1 (split istatistikleri) eklendi')

    # 4) YENİ Bölüm 2.5 — P034'ten sonra
    sec25_texts = [
        ("Ablation protocol. To isolate the contribution of each architectural and training component, eight "
         "configurations were trained under strictly identical conditions — same patient-level split, same "
         "preprocessing, same optimizer, learning rate schedule, batch size, random seed and epoch budget — "
         "the proposed model itself plus seven variants, each differing in exactly one factor: (i) image-only, in which the demographic branch is removed entirely; "
         "(ii) metadata-only, in which the visual backbone is removed; (iii) simple concatenation, in which the "
         "gating weights are omitted; (iv) multi-head self-attention fusion, as described in Section 2.3; (v) "
         "naive class weighting, using the sum-normalized weighting scheme discussed in Section 2.4; (vi) no "
         "focal loss, substituting plain weighted binary cross-entropy; and (vii) no augmentation, disabling the "
         "entire training-time augmentation pipeline. Because the ablation study requires eight independent "
         "training runs, every configuration — the proposed model included — was trained for the same "
         "reduced budget of 10 epochs, so that all rows of Table 5 are directly comparable. The proposed model "
         "is additionally reported at its complete 18-epoch budget; the difference between its 10-epoch and "
         "18-epoch test results, arising from an identical configuration, provides an empirical estimate of "
         "run-to-run variability against which the between-configuration differences should be judged."),
        ("Per-class threshold calibration. Reporting sensitivity and F1 at a single fixed threshold of τ = "
         "0.5 is inappropriate for a multi-label task with prevalences spanning two orders of magnitude. "
         "Class-specific thresholds were therefore optimized exclusively on the validation set — never on "
         "the test set — using two criteria: Youden’s J statistic (maximizing sensitivity + "
         "specificity − 1), which targets a balanced operating point, and the F1-maximizing threshold, "
         "which targets precision–recall balance. The selected thresholds were then applied unchanged to "
         "the held-out test set, and sensitivity, specificity, precision (positive predictive value) and F1 were "
         "recomputed at these operating points. Calibration quality was assessed with reliability diagrams and "
         "per-class Brier scores, and precision–recall curves were generated for all 15 classes."),
        ("Visual explainability. Gradient-weighted Class Activation Mapping (Grad-CAM) [36] was applied to the "
         "final convolutional block of the EfficientNet-B3 backbone. For a target class c, channel importance "
         "weights are obtained by global average pooling of the gradients of the pre-sigmoid logit with respect "
         "to each activation map, and the localization map is computed as the ReLU of their weighted sum, "
         "min–max normalized and upsampled to the input resolution for overlay. This provides "
         "patient-specific visual evidence for individual predictions and is reported separately from the "
         "literature-grounded explanations produced by the RAG module, which address a different question."),
        ("Computational profiling. Parameter counts, multiply–accumulate operations, peak GPU memory and "
         "inference latency were measured directly for the proposed model, the image-only variant, the "
         "self-attention variant and a DenseNet-121 baseline with a 15-unit output head. FLOPs were obtained "
         "with a standard profiler and cross-validated against a direct parameter count; latency was measured "
         "after ten warm-up iterations as the mean over fifty timed forward passes with explicit device "
         "synchronization, at batch sizes 1 and 36, and the five-fold TTA cost is reported as five times the "
         "single-image latency."),
    ]
    anchor = pars[34]._element
    head25 = clone_par_after(anchor, pars[29], '2.5 Ablation, Threshold Calibration and Interpretability Protocols')
    anchor = head25._element
    for t in sec25_texts:
        p = clone_par_after(anchor, pars[34], t)
        anchor = p._element
    print('✓ Bölüm 2.5 (protokoller) eklendi')

    # 5) YENİ Bölüm 3.2 (threshold kalibrasyonu) — P069'dan sonra
    anchor = pars[69]._element
    h32 = clone_par_after(anchor, pars[62], '3.2 Threshold Calibration and Probability Reliability')
    anchor = h32._element
    p = clone_par_after(anchor, pars[63],
                        'Because a single fixed threshold cannot be appropriate across classes whose prevalence '
                        'spans two orders of magnitude, per-class operating points were calibrated on the '
                        'validation set and then applied unchanged to the test set, as described in Section 2.5. '
                        'Table 3 reports, for each class, the test-set AUC with bootstrap confidence intervals, '
                        'the Brier score, the validation-derived Youden threshold and the resulting test-set '
                        'sensitivity, specificity, precision and F1.')
    anchor = p._element
    tbl3 = make_table(doc, TBL_THRESH, style_src, anchor, bold_last_row=True, font_pt=7)
    set_table_widths(tbl3, COL_WIDTHS['thresh'])
    anchor = tbl3._element  # caption tablodan SONRA gelmeli
    cap3 = clone_par_after(anchor, pars[64],
                           'Table 3. Per-class threshold calibration and probability reliability on the test '
                           'set. Thresholds were selected on the validation set only. CI = percentile bootstrap '
                           '95% confidence interval (1,000 resamples). PPV = positive predictive value.')
    anchor = cap3._element

    for t in [
        ("Three observations follow. First, threshold optimization at the balanced (Youden) operating point "
         "raises macro-average sensitivity from 0.6977 to 0.7662 while maintaining macro-average specificity at "
         "0.7515 — a configuration substantially better suited to a screening or triage context, in which "
         "false negatives carry greater clinical cost than false positives. Second, selecting thresholds to "
         "maximize F1 instead raises the macro-average F1 from 0.2796 to 0.3640, illustrating that the "
         "apparently modest F1 values obtained at τ = 0.5 reflect threshold placement rather than a "
         "deficiency in ranking capacity. Third, the validation-derived thresholds cluster around 0.44 on "
         "average and remain within a moderate band across classes (0.14–0.63), and the macro-average Brier "
         "score of 0.1577 indicates reasonable, though not perfect, probability calibration; the classes with "
         "the largest Brier scores (Infiltration, 0.2933; Atelectasis, 0.2396) are precisely those with the "
         "weakest discriminative performance, consistent with label-noise-driven uncertainty rather than "
         "systematic miscalibration."),
        ("Precision–recall curves and reliability diagrams for all 15 classes are presented in Figure 6. "
         "The precision–recall curves confirm that the model operates well above the prevalence baseline "
         "for every class, and the reliability diagrams show predicted probabilities tracking observed "
         "frequencies closely in the mid-probability range, with mild over-confidence at the extremes for the "
         "rarest classes."),
    ]:
        p = clone_par_after(anchor, pars[65], t)
        anchor = p._element

    figp = insert_picture_par(doc, anchor, pars[89], FIGDIR / 'fig6_pr_calibration.png', width_in=6.2)
    anchor = figp._element
    capf6 = clone_par_after(anchor, pars[89],
                            'Fig 6. (a) Per-class precision–recall curves and (b) calibration/reliability '
                            'diagrams on the test set.')
    anchor = capf6._element
    print('✓ Bölüm 3.2 (threshold kalibrasyonu) + Tablo 3 + Fig 6 eklendi')

    # 6) YENİ Bölümler 3.4 / 3.5 / 3.6 — TTA tartışmasından (P074) sonra
    anchor = pars[74]._element

    h34 = clone_par_after(anchor, pars[62], '3.4 Ablation Study')
    anchor = h34._element
    p = clone_par_after(anchor, pars[63],
                        'Table 5 reports the eight-configuration ablation study described in Section 2.5. All '
                        'eight runs share an identical patient-level split, preprocessing pipeline, optimizer '
                        'configuration, random seed and 10-epoch budget, including the proposed model itself; only '
                        'the factor named in each row differs. The proposed model at its full 18-epoch budget is '
                        'appended in the final row for reference.')
    anchor = p._element
    tbl5 = make_table(doc, TBL_ABLATION, style_src, anchor, font_pt=7.5,
                      left_cols=LEFT_COLS['ablation'])
    set_table_widths(tbl5, COL_WIDTHS['ablation'])
    anchor = tbl5._element
    cap5 = clone_par_after(anchor, pars[64],
                           'Table 5. Controlled ablation study. The first eight rows were all trained for 10 '
                           'epochs under identical settings, varying only the stated factor; the proposed model '
                           'appears as the epoch-matched reference in the first row. The final row reports the same '
                           'configuration at its complete 18-epoch budget; the 0.0014 macro-AUC difference between '
                           'these two rows sets an empirical scale for run-to-run variability.')
    anchor = cap5._element

    for t in [
        ("The imaging modality dominates. Under an identical 10-epoch budget the image-only configuration "
         "reaches a macro-AUC of 0.8307 against 0.8323 for the full gated multimodal model, so adding age, sex "
         "and projection metadata is worth a nominal 0.0016. That figure has to be read against the run-to-run "
         "variability of the pipeline itself: the identical gated configuration scores 0.8323 at 10 epochs and "
         "0.8309 at 18 epochs, a spread of 0.0014. The apparent metadata contribution is therefore of the same "
         "order as the noise floor of a single training run, and we do not claim it as a real improvement. We "
         "report this explicitly rather than attributing performance to multimodality."),
        ("The model is not exploiting demographic shortcuts. The metadata-only configuration attains a macro-AUC "
         "of 0.6124 — above chance, reflecting genuine epidemiological association between demographics and "
         "pathology prevalence, but far below any image-based configuration. This directly addresses the concern "
         "that a multimodal model might learn demographic or acquisition-related shortcuts in place of "
         "radiographic evidence: the 0.22 AUC gap between metadata-only and image-only performance establishes "
         "that the discriminative signal originates predominantly from the radiograph itself."),
        ("Fusion mechanism complexity is not rewarded. Simple concatenation (0.8320) lands within 0.0003 macro-AUC "
         "of learned gating (0.8323) — an order of magnitude below the 0.0014 variability estimate above — "
         "while genuine multi-head self-attention performs substantially worse (0.8119, F1 0.2334) and "
         "additionally proved numerically unstable: under float16 mixed precision the self-attention "
         "configuration diverged to a non-finite loss at epoch 9 of 10, and the reported figures were obtained "
         "from the last stable checkpoint (epoch 6). This is an intelligible result rather than an anomaly "
         "— dot-product self-attention over a two-token sequence has very little structure to exploit, "
         "while adding parameters and numerical fragility. The gating mechanism was therefore retained not "
         "because it outperforms concatenation, but because it provides interpretable per-sample modality "
         "weights at negligible cost and without the instability of the attention variant."),
        ("Correct positive-class weighting is decisive. Substituting the naive sum-normalized weighting scheme "
         "for the standard negative-to-positive ratio reduces macro-AUC only modestly (0.8112 vs. 0.8323) but "
         "collapses macro-average sensitivity from 0.6957 to 0.1920 and macro-average F1 from 0.2740 to 0.1431 "
         "— an approximately 3.6-fold reduction in the proportion of true positives detected at the default "
         "threshold. Because the naive scheme assigns positive-class weights below unity to the high-prevalence "
         "classes, it actively discourages positive predictions for exactly those categories, suppressing their "
         "predicted probabilities below τ = 0.5 while leaving the ranking (and hence AUC) largely intact. "
         "This ablation isolates a failure mode that is invisible to AUC-only reporting."),
        ("Focal loss and augmentation contribute modestly and in different ways. Removing focal loss slightly "
         "reduces macro-AUC (0.8257 vs. 0.8323) yet produces the highest macro-F1 of any configuration (0.3026), "
         "indicating that, once positive-class weighting is correctly specified, focal loss does not provide an "
         "additional advantage in the precision–recall regime. Removing augmentation reduces macro-AUC by a "
         "somewhat larger margin (0.8193), but its effect on training dynamics is more pronounced than this figure "
         "suggests: without augmentation the train–validation AUC gap grows monotonically after epoch 6 "
         "(reaching 0.141 by epoch 10, with training AUC 0.937 against validation AUC 0.796), whereas augmented "
         "configurations maintain a near-zero or negative gap throughout. Augmentation therefore functions "
         "primarily as a stabilizer against overfitting rather than as a direct contributor to peak test "
         "performance."),
    ]:
        p = clone_par_after(anchor, pars[65], t)
        anchor = p._element

    # 3.5 profiling
    h35 = clone_par_after(anchor, pars[62], '3.5 Computational Profiling')
    anchor = h35._element
    p = clone_par_after(anchor, pars[63],
                        'Table 6 reports measured parameter counts, computational cost, memory footprint and '
                        'inference latency for the proposed model and relevant baselines, obtained under the '
                        'protocol described in Section 2.5.')
    anchor = p._element
    tbl6 = make_table(doc, TBL_PROFILE, style_src, anchor, font_pt=7)
    set_table_widths(tbl6, COL_WIDTHS['profile'])
    anchor = tbl6._element
    cap6 = clone_par_after(anchor, pars[64],
                           'Table 6. Measured computational characteristics at 300 × 300 input resolution '
                           'with a 15-class output head. GFLOPs are reported as 2 × multiply–accumulate '
                           'operations for a single image. Latency values were measured on a single '
                           'consumer-grade GPU and should be interpreted comparatively rather than as absolute '
                           'deployment figures.')
    anchor = cap6._element
    p = clone_par_after(anchor, pars[65],
                        'These measurements require an explicit correction to a claim made in the earlier version '
                        'of this work. EfficientNet-B3 is not lighter than DenseNet-121 in parameter count: the '
                        'image-only B3 backbone with a 15-class head comprises 11.65 M parameters against 6.97 M '
                        'for DenseNet-121, and the full multimodal model comprises 12.35 M. The advantage of '
                        'EfficientNet-B3 in this setting is computational rather than parametric — it '
                        'requires 3.86 GFLOPs per image against 10.09 GFLOPs for DenseNet-121, a 2.6-fold '
                        'reduction, which translates into lower measured single-image latency (20.2 ms vs. 24.1 '
                        'ms) despite the larger parameter count. The five-fold TTA ensemble raises single-image '
                        'inference cost to approximately 101 ms, which remains compatible with interactive '
                        'clinical use but represents a five-fold increase in compute for the +0.0032 macro-AUC '
                        'gain quantified in Section 3.3; deployments operating under tight latency budgets may '
                        'reasonably omit TTA.')
    anchor = p._element

    # 3.6 Grad-CAM
    h36 = clone_par_after(anchor, pars[62], '3.6 Model Interpretability: Grad-CAM Analysis')
    anchor = h36._element
    p = clone_par_after(anchor, pars[63],
                        'Figure 7 presents Grad-CAM localization maps for representative test cases spanning '
                        'several pathology classes. In each panel the original radiograph is shown alongside the '
                        'class-specific activation overlay for the predicted pathology.')
    anchor = p._element
    figp7 = insert_picture_par(doc, anchor, pars[89], FIGDIR / 'fig7_gradcam_panel.png', width_in=6.2)
    anchor = figp7._element
    capf7 = clone_par_after(anchor, pars[89],
                            'Fig 7. Grad-CAM localization maps for representative test radiographs. Left: '
                            'original image; right: class activation overlay with the predicted class and '
                            'probability.')
    anchor = capf7._element
    p = clone_par_after(anchor, pars[65],
                        'The activation maps are spatially focused rather than diffuse, and their locations are '
                        'anatomically coherent with the predicted findings: for the Pneumothorax case (predicted '
                        'probability 0.924) activation concentrates on the upper region of the affected '
                        'hemithorax, and for the Effusion case (0.906) on the lower thoracic zone where fluid '
                        'accumulation is expected. This provides case-level visual evidence that the '
                        'network’s decisions are driven by clinically relevant image regions rather than by '
                        'spurious global cues. We stress that this analysis is qualitative and conducted on a '
                        'small sample; a systematic evaluation against expert-annotated bounding boxes is '
                        'required before any localization claim can be made quantitatively, and is identified as '
                        'future work in Section 4.')
    anchor = p._element
    print('✓ Bölümler 3.4 / 3.5 / 3.6 (+ Tablo 5, Tablo 6, Fig 7) eklendi')

    # 7) 3.9 başlığı — ROC paragrafından (P087) önce
    h39 = clone_par_after(pars[86]._element, pars[62], '3.9 ROC Curves and Confusion Matrices')
    print('✓ Bölüm 3.9 başlığı eklendi')

    # 8) yeni kaynaklar — [31]'den (P159) sonra
    anchor = pars[159]._element
    for ref in NEW_REFS:
        p = clone_par_after(anchor, pars[159], ref)
        anchor = p._element
    print(f'✓ {len(NEW_REFS)} yeni kaynak eklendi ([32]-[37])')

    # 9) figür görsellerini değiştir
    ok2 = replace_image(doc, pars[23], FIGDIR / 'fig2_architecture.png')
    ok8 = replace_image(doc, pars[88], FIGDIR / 'roc_curves.png')
    ok9 = replace_image(doc, pars[92], FIGDIR / 'confusion_matrices.png')
    print(f'✓ figür görselleri değiştirildi: Fig2={ok2} ROC={ok8} Confusion={ok9}')

    # 10) yazım geçişi: orijinalden gelen "…and" boşluk hataları (tablolar dahil)
    from docx.text.paragraph import Paragraph as _P
    fixed = 0
    for p in doc.paragraphs:
        fixed += fix_missing_spaces(p)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    fixed += fix_missing_spaces(p)
    print(f'✓ {fixed} adet eksik boşluk düzeltildi ("exposureand" → "exposure and" vb.)')

    doc.save(DST)
    print(f'\n✅ KAYDEDİLDİ: {DST}')


if __name__ == '__main__':
    main()
