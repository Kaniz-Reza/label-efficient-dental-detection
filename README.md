# Label-Efficient Dental Pathology Detection

Self-supervised pretraining and multi-object tracking for label-efficient object detection on
dental panoramic radiographs (OPGs). Compares three fully-supervised YOLO detectors against
four self-supervised learning (SSL) backbones — SimCLR, BYOL, I-JEPA, and DINOv3 — fine-tuned
at a 20% label budget, then deploys the best detector in a ByteTrack tracking pipeline.

Bounding-box annotation for dental radiographs is expensive, requiring expert clinical review.
This project investigates how much of that annotation cost can be saved by pretraining on
unlabelled images first, while still recovering strong detection performance.

## Dataset

- **Dental OPG Object Detection Dataset** — 10,000 panoramic dental radiographs, 24,754
  annotated bounding-box instances across four classes: **Cavities, Damage, Infection, Wisdom
  teeth**.
- Source: [Mendeley Data, DOI 10.17632/wxv6h9p39g.1](https://doi.org/10.17632/wxv6h9p39g.1)
- Split with a leakage-safe protocol: images are partitioned once, at image level, with a fixed
  seed — 10% held out for final testing, 10% for validation, and the remaining 80% used as the
  self-supervised pretraining pool for Part B.

## Part A — Fully Supervised Object Detection

Three YOLO-family detectors trained under full supervision (100% labels, 640×640, batch 16,
50 epochs):

| Model | Precision | Recall | mAP50 | mAP50-95 | F1 | FPS |
|---|---|---|---|---|---|---|
| YOLOv10-m | 0.621 | 0.579 | 0.477 | 0.221 | 0.599 | 52.2 |
| **YOLOv12-m (selected)** | **0.655** | 0.572 | **0.481** | **0.226** | **0.611** | 24.4 |
| YOLOv26-m | 0.621 | 0.550 | 0.461 | 0.213 | 0.583 | 39.7 |

YOLOv12-m achieved the best detection accuracy on every metric and was selected as the Part B
backbone, at the cost of being the slowest of the three (24.4 FPS — roughly half YOLOv10-m's
throughput). YOLOv10-m is the stronger choice where inference latency matters more than peak
accuracy.

**Error analysis** (consistent across all three architectures): Infection was the hardest class
to detect (60–67% false-negative rate); Wisdom was the easiest (7.6–8.8%). Misclassifications
were rare overall (~1% of instances) and concentrated almost entirely in Cavities↔Damage
confusion — these two conditions appear visually similar to the model.

## Part B — Self-Supervised Pretraining and Backbone Transfer

Four SSL methods pretrain a backbone on the unlabelled 80% image pool, which is then
fine-tuned into the Part A-selected YOLOv12-m architecture at a fixed 20% label budget, and
compared against random-init and COCO-pretrained baselines:

| Initialisation | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Random-init | 0.4448 | 0.5229 | 0.4448 | 0.1916 |
| I-JEPA | 0.4335 | 0.5274 | 0.4487 | 0.1941 |
| BYOL | 0.4604 | **0.5293** | 0.4641 | 0.2019 |
| SimCLR | 0.4847 | 0.5183 | 0.4699 | 0.2050 |
| DINOv3 (domain-adaptive) | 0.4704 | 0.5318 | 0.4679 | 0.2054 |
| **COCO-pretrained** | **0.5045** | 0.5187 | **0.4796** | **0.2073** |

SimCLR and a domain-adaptively continued DINOv3 both closely approach COCO-pretrained
transfer learning at this label budget. BYOL achieves the highest recall of any tested
configuration. I-JEPA, trained under a compute-constrained schedule (a fraction of its
published training scale), shows measurable partial representational collapse and performs
close to random initialisation — a diagnosed negative result rather than an implementation
failure, verified via embedding-space diagnostics (t-SNE, nearest-neighbor retrieval, and
cosine-similarity analysis) included in its notebook.

### The four SSL methods

| Method | Family | Key mechanism |
|---|---|---|
| **SimCLR** | Contrastive | Two augmented views, NT-Xent loss, large-batch sensitivity |
| **BYOL** | Self-distillation (negative-free) | Online network predicts EMA target network; no negative pairs |
| **I-JEPA** | Latent prediction (masked) | ViT encoder predicts masked-region representations in latent space, not pixels |
| **DINOv3** | Self-distillation (ViT foundation) | Student matches teacher across multi-crop views, Gram anchoring for dense features |

## Bonus — Label-Efficiency Ablation

A sweep of DINOv3 vs. COCO initialisation across 10–50% label budgets, evaluated on the same
held-out test set. DINOv3 reaches within about 1% of COCO's performance at matched label
budgets by the 40–50% range, suggesting SSL pretraining narrows the gap with supervised
transfer learning as more labels become available.

## Tracking

The best Part B detector (DINOv3-initialised YOLOv12-m) drives a ByteTrack multi-object
tracking pipeline, evaluated on a synthetic video built from real annotated test-set
radiographs with exact ground truth:

| Metric | Value |
|---|---|
| MOTA | 0.047 |
| MOTP | 0.300 |
| IDF1 | 0.137 |
| Identity switches | 0 |
| Mostly tracked | 5 / 45 objects |
| Fragmentations | 22 |

Zero identity switches were observed throughout — whenever the tracker had a detection to
work with, it never confused one object's identity with another's. The limiting factor was
detector recall under viewpoint shift (the panning camera window rarely matches the detector's
original training-time framing), not tracker association quality — a useful illustration of how
strongly tracking-by-detection depends on upstream detector robustness.

## Project Structure

```text
.
├── requirements.txt
├── .gitignore
├── partA/
│   ├── 01-dental-eda.ipynb       Dataset exploration, leakage-safe split, augmentation design
│   ├── 02-yolov10.ipynb          YOLOv10-m: training, evaluation, error analysis
│   ├── 03-yolov12.ipynb          YOLOv12-m: training, evaluation, error analysis
│   └── 04-yolov26.ipynb          YOLOv26-m: training, evaluation, error analysis
├── partB/
│   ├── 00-data-partition.ipynb              Leakage-safe 10/10/80 partition for Part B
│   ├── partB-01a-simclr-pretrain.ipynb      SimCLR self-supervised pretraining
│   ├── partB-01b-simclr-detect.ipynb        SimCLR backbone → YOLOv12-m fine-tune @ 20% labels
│   ├── partB-02a-byol-pretrain.ipynb        BYOL self-supervised pretraining
│   ├── partB-02b-byol-detect.ipynb          BYOL backbone → YOLOv12-m fine-tune @ 20% labels
│   ├── partB-03a-ijepa-pretrain.ipynb       I-JEPA self-supervised pretraining
│   ├── partB-03b-ijepa-detect.ipynb         I-JEPA backbone → YOLOv12-m fine-tune @ 20% labels
│   ├── partB-04a-dinov3-pretrain.ipynb      DINOv3 domain-adaptive pretraining
│   ├── partB-04b-dinov3-detect.ipynb        DINOv3 backbone → YOLOv12-m fine-tune @ 20% labels
│   ├── partB-05-tracking.ipynb              ByteTrack multi-object tracking on video
│   ├── partB-bonus-label-efficiency.ipynb   Consolidated bonus ablation (10–50% labels)
│   └── bonus-runs/                          Individual per-run bonus notebooks
├── results/                        Output figures, checkpoint notes, tracking frames
└── report/                         Full experimental report (Part A + Part B)
```

## Methodology Summary

1. **Leakage-safe partitioning** — 10% test / 10% validation / 80% SSL pool, split once at
   image level with a fixed seed; every notebook verifies the test and validation sets are
   disjoint from the SSL pretraining pool by filename intersection.
2. **SSL pretraining** — each method pretrains a backbone on the unlabelled 80% pool for 30
   epochs, with representation diagnostics (embedding visualization, nearest-neighbor
   retrieval) logged before any labels are touched.
3. **Backbone transfer** — the pretrained backbone is loaded into the Part A-selected YOLOv12-m
   architecture and fine-tuned for 50 epochs on a fixed, seeded 20% label subset drawn from the
   SSL pool.
4. **Fair-comparison protocol** — resolution, batch size, epoch budget, optimizer, augmentation
   policy, and confidence/NMS thresholds are held identical across every SSL method and
   baseline; only backbone initialization and label fraction vary.
5. **Evaluation** — precision, recall, mAP50, mAP50-95, and IoU-matched false-negative /
   false-positive / misclassification error analysis on the held-out test split.

## Technologies

- Python, PyTorch
- [Ultralytics](https://docs.ultralytics.com) (YOLOv10/v12/v26, ByteTrack)
- [timm](https://github.com/huggingface/pytorch-image-models) (Vision Transformer backbones)
- [Transformers](https://github.com/huggingface/transformers) (DINOv3)
- Albumentations (augmentation pipeline)
- scikit-learn (t-SNE diagnostics, stratified splitting)
- motmetrics (MOT evaluation)
- OpenCV, Matplotlib, Pandas, NumPy

All notebooks were developed and run on Kaggle (T4 GPU).

## Setup

```bash
pip install -r requirements.txt
```

Each notebook is self-contained and can be run independently on Kaggle or any Jupyter
environment with GPU access; SSL pretraining and YOLO training notebooks expect a single
GPU session (T4-class or better).

## Full Report

See `report/Dental-Detection-Report.pdf` for the complete write-up: methodology, per-class
breakdowns, embedding diagnostics for every SSL method, the full tracking analysis, and
discussion connecting each method's mechanism to its observed performance.


 
