# Arbiter

Automated waste sorting for Raspberry Pi. A camera classifies items as recyclable, waste, or empty; a servo-driven lever physically routes them.

---

## Demo Video

[Demo video](https://drive.google.com/file/d/1hrgxE8dlGoFcHhTaqKrMwUdue5Joh928/view?usp=sharing)

## Physical Config:

<img width="2160" height="2880" alt="image" src="https://github.com/user-attachments/assets/34bbe5cc-3414-43e6-a4ea-261281ac27b2" />


---

## What this is

Arbiter is a physical sorting machine, not a software package. An item placed on the platform is detected by the camera, classified by a fine-tuned MobileNetV3-small model, and sorted by a servo lever — no button press required. The classifier distinguishes three states: **recycle**, **waste**, and **empty** (nothing present). The `empty` class is the presence-detection mechanism; the pipeline polls continuously and only triggers the servo when something is actually there.

This is a hardware project first. It cannot be cloned and run by a stranger — it requires a Raspberry Pi 4, Pi Camera Module 3, a DS3218MG servo, and a physical mounting rig. The ML training pipeline runs on any machine; the inference + servo pipeline runs only on the Pi.

---

## System overview

```
┌─────────────────────────────────────────────────────────┐
│  Raspberry Pi 4                                         │
│                                                         │
│  Picamera3 (1280×720, BGR888, AWB=Constant)             │
│      │                                                  │
│      ▼  every 2 s                                       │
│  WasteClassifier.predict()  ──► "empty" → keep polling  │
│      │                                                  │
│      │ non-empty detected                               │
│      ▼                                                  │
│  cam.autofocus_cycle()  (Pi Camera Module 3 AF)         │
│      │                                                  │
│      ▼                                                  │
│  WasteClassifier.predict()  ──► "recycle" or "waste"    │
│      │                                                  │
│      ▼                                                  │
│  gpiozero Servo (GPIO 18, DS3218MG)                     │
│    glide → hold 2 s → glide back to mid → detach        │
└─────────────────────────────────────────────────────────┘
```

`WasteClassifier` lives in `src/infer.py` and is imported directly by the pipeline. Training runs separately on a workstation; the resulting `models/best.pt` checkpoint is deployed to the Pi.

---

## Hardware bill of materials

| Component | Notes |
|---|---|
| Raspberry Pi 4 (2 GB+) | Primary compute |
| Pi Camera Module 3 | Required — Module 3 specifically for hardware autofocus |
| DS3218MG servo | 20 kg·cm torque; driven via GPIO 18 |
| 5 V / 3 A power supply | Servo draws significant current under load |
| Physical lever + platform | Custom-built; not documented here |

GPIO 17 (`ir_sensor_pin` in config) is reserved for an IR break-beam sensor but is not currently wired into the pipeline — presence detection is handled by the `empty` class.

---

## ML approach

**Architecture:** MobileNetV3-small (`mobilenetv3_small_100` via `timm`), ImageNet pretrained. The default classifier is replaced with:

```
Dropout(p=0.3) → Linear(in_features, 3)
```

**Classes:**

| Label | TrashNet source classes |
|---|---|
| `recycle` (1) | glass, paper, cardboard, metal, plastic |
| `waste` (0) | trash |
| `empty` (2) | Pi-captured frames with nothing present |

**Training setup** (`configs/config.yaml`):

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0003 |
| LR schedule | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) |
| Batch size | 32 |
| Epochs | 30 |
| Confidence threshold | 0.35 |
| Seed | 134 |

**Inference logic** — not a plain argmax. The model returns three class probabilities; the decision rule is:

1. If `argmax == empty` → return `empty`.
2. Else if `waste_prob ≥ 0.35` → return `waste`.
3. Else → return `recycle`.

The threshold on waste makes the system conservative: anything with a 35 %+ waste probability is sorted as waste. This prevents recyclable-stream contamination at the cost of occasionally discarding borderline recyclables.

**Metrics:**

```
Best val accuracy: 0.99

Per-class (test set):
              precision  recall  f1-score  
waste         0.93       0.95    0.94      
recycle       0.90       0.95    0.92     
empty         0.91       1.00    0.95     
```

---

## Dataset

Training data is a mix of [TrashNet](https://github.com/garythung/trashnet) and Pi-captured images taken with the same camera and lighting setup used at inference time.

**HuggingFace dataset (Pi-captured images):**
`https://huggingface.co/datasets/aaryavlal/arbiter-mini`

**Composition and per-sample weights:**

| Source | Classes covered | Sample weight |
|---|---|---|
| TrashNet | glass, paper, cardboard, metal, plastic, trash | 1.0 |
| Pi-captured | cardboard | 20.0 |
| Pi-captured | paper | 15.0 |
| Pi-captured | trash | 10.0 |
| Pi-captured | empty (blank platform) | 1.0 |

Pi-captured images are drawn more frequently during training via `WeightedRandomSampler`. The weights were set to reflect the domain gap: the Pi camera's colour profile differs from TrashNet's web-scraped images, so in-distribution Pi data needs higher representation.

Glass, metal, and plastic have **no Pi-captured images** — those classes rely entirely on TrashNet. Domain gap is real for those classes.

---

## Camera pipeline: why AWB is locked

The Pi Camera Module 3's auto white balance (AWB) adapts per scene. During early data collection, paper and trash were captured in separate sessions; AWB converged to different colour gains for each. This created a class-correlated colour confound: the camera background in paper captures had a systematically different colour profile than in trash captures, giving the model a cheap shortcut that would not generalise to inference time.

`scripts/audit_color_distributions.py` was written to measure this. It computes per-channel means and corner-patch R/G and B/G ratios for Pi-captured images, reports Cohen's d between paper and trash captures, and runs a 5-fold logistic regression on colour features alone. If colour statistics can separate classes with >80% CV accuracy, or corner Cohen's d > 0.8, the script flags the captures as confounded.

The fix: lock `AwbMode` to `controls.AwbModeEnum.Fluorescent` in every capture script **and** in the inference pipeline, so training and inference images share the same colour response.

---

## Repository structure

```
Arbiter-/
├── configs/
│   └── config.yaml              # all hyperparameters and hardware pins
├── src/
│   ├── dataset.py               # TrashDataset, label mapping, per-sample weights, dataloaders
│   ├── train.py                 # training entry point; saves best.pt to models/
│   ├── evaluate.py              # standalone test-set evaluation (see known limitations)
│   └── infer.py                 # WasteClassifier — used by the pipeline and standalone
├── pipeline/
│   ├── main.py                  # autonomous pipeline: polls, detects, autofocuses, sorts
│   └── main_train.py            # manual pipeline: press Enter to capture and classify
├── scripts/
│   ├── capture-scripts/         # per-class Pi capture tools with live MJPEG preview
│   │   ├── capturepaper.py
│   │   ├── capturetrash.py
│   │   ├── capturecardboard.py
│   │   ├── captureglass.py
│   │   ├── capturemetal.py
│   │   ├── captureplastic.py
│   │   └── captureempty.py
│   ├── audit_color_distributions.py   # AWB confound detector (Cohen's d + logistic regression)
│   └── import_taco.py           # TACO dataset import utility
├── models/
│   └── best.pt                  # trained checkpoint (gitignored)
├── data/
│   └── raw/                     # training images (gitignored)
└── requirements.txt
```

---

## Reproducing this

### ML only (any machine)

1. Download [TrashNet](https://github.com/garythung/trashnet) and extract into `data/raw/` so subdirectories are named `glass/`, `paper/`, `cardboard/`, `metal/`, `plastic/`, `trash/`.
2. Optionally add Pi-captured images from the HuggingFace dataset (link above) into the same structure. Add an `empty/` subdirectory with blank-platform captures.
3. Install dependencies:
   ```bash
   pip install torch torchvision timm Pillow PyYAML scikit-learn opencv-python python-dotenv
   ```
4. Train:
   ```bash
   cd src && python train.py --config ../configs/config.yaml
   ```
   The best checkpoint saves to `models/best.pt`. Per-class metrics print to stdout at the end.

5. Classify a single image:
   ```bash
   cd src && python infer.py ../path/to/image.jpg --config ../configs/config.yaml
   ```

### Full hardware reproduction

This is a build, not an install. You need the BOM above, a physical sorting rig, and a Raspberry Pi running Raspberry Pi OS. Install Pi-specific dependencies:

```bash
pip install picamera2 gpiozero
# libcamera is a system package — install via apt, not pip
```

Run the autonomous pipeline:

```bash
python pipeline/main.py --config configs/config.yaml
```

Run the manual pipeline (press Enter to classify, useful for testing):

```bash
python pipeline/main_train.py --config configs/config.yaml
```

**Collecting new Pi-captured images:** each script in `scripts/capture-scripts/` starts a live MJPEG preview server. Open `http://<PI_IP>:8000` in a browser to see the feed, then press Enter in the terminal to save a frame. Images save to `data/raw/<class>/`.

---

## Known limitations

- **Glass, metal, plastic have no Pi-captured training data.** Those classes come entirely from TrashNet (web-scraped photos). Classification accuracy for those materials at inference time is unverified on real Pi camera images.
- **Single item at a time.** The pipeline assumes one object occupies the frame. Multiple overlapping items are not handled.
- **AWB is locked to Fluorescent.** Performance will degrade in significantly different lighting conditions (e.g. outdoor, incandescent, or direct sunlight). The lock is the fix for training consistency, but it also constrains deployment environment.
- **`evaluate.py` does not load checkpoints correctly.** The custom classifier head (`Sequential(Dropout, Linear)`) differs from the default timm head. `evaluate.py` creates a vanilla timm model and calls `load_state_dict`, which will fail with a key mismatch. Use the test-set report that prints at the end of `train.py` instead.
- **IR sensor is wired but not used.** `configs/config.yaml` reserves GPIO 17 for a break-beam sensor. The current pipeline detects presence via the `empty` class, not a physical sensor.

---

## Roadmap

- Wire in the IR break-beam sensor as a hardware trigger to replace polling
- Capture Pi-native training images for glass, metal, and plastic
- Persist test-set metrics to a file rather than relying on stdout
- Evaluate under different lighting conditions to characterise the AWB constraint

---

## Acknowledgments

- [TrashNet](https://github.com/garythung/trashnet) — Gary Thung and Mindy Yang
