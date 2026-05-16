# Arbiter

Automated waste sorting system for the Raspberry Pi. Place an item on the platform, classify it as **recyclable** or **waste** using MobileNetV3-small, and physically sort it with a servo-driven lever.

## Setup

```bash
pip install -r requirements.txt
```

> `RPi.GPIO` and `picamera2` are only needed on the Pi. Training works without them.

## Training (on any machine)

1. Download [TrashNet](https://github.com/garythung/trashnet) and extract into `data/raw/`
2. Preprocess into `data/processed/` (maintain class subdirectories)
3. Train:

```bash
cd src && python train.py --config ../configs/config.yaml
```

4. Evaluate:

```bash
cd src && python evaluate.py --config ../configs/config.yaml
```

## Inference

```bash
cd src && python infer.py path/to/image.jpg
```

## Pipeline (on Raspberry Pi)

```bash
python pipeline/main.py --config configs/config.yaml
```

Use `--dry-run` to test without GPIO hardware.

## Architecture

- **Binary classification:** recycle vs. waste
- **Model:** MobileNetV3-small (edge-optimized for Pi CPU)
- **Bias:** High recall on recycle class via class-weighted loss — missing a recyclable is worse than a false positive
- **Training/inference decoupled:** train on a workstation, deploy checkpoint to Pi
