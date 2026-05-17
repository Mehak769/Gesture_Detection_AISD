# Gesture_Detection_AISD
# Gesture Detection AISD Project
**Deggendorf Institute of Technology**
Course: AI for Smart Devices (AISD)
Team: aisd_user12

---

## Project Overview
This project implements a real-time hand gesture detection system using a Raspberry Pi 5 with an IMX500 AI Camera. The system detects three hand gestures:
- ✋ `open_hand` — Open hand with fingers spread
- 👍 `thumbs_up` — Thumb pointing up
- 👎 `thumbs_down` — Thumb pointing down

---

## Workflow

```
Capture Images → Label Data → Train YOLO Model → Export to IMX → Run on Raspberry Pi
```

---

## Project Structure

```
Gesture_Detection_AISD/
├── dataset/                  # Raw captured images
├── my_training_data/         # Labeled training data
│   ├── images/               # Image files
│   ├── labels/               # YOLO annotation files
│   ├── classes.txt           # Class names
│   └── notes.json            # Label metadata
├── network.rpk               # Final model for Raspberry Pi camera
├── collect_images.py         # Script to capture images
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## Step 1 — Data Collection

Images were captured using the Raspberry Pi IMX500 AI Camera:

```bash
python collect_images.py
```

- 45 images collected in total
- 15 images per class (open_hand, thumbs_up, thumbs_down)

---

## Step 2 — Data Labeling

Images were labeled using **Label Studio**:

```bash
pip install label-studio
label-studio
```

- Created bounding boxes around hands
- Labeled each image with the correct gesture class
- Exported in **YOLO with Images** format

---

## Step 3 — Data Preparation (Linux Workstation)

SSH into the Linux workstation:

```bash
ssh aisd_user12@10.1.65.207
```

Prepare training data:

```bash
cd yolo-uv
uv run python prepare_training_data.py
```

This splits data into:
- 80% training (40 images)
- 20% validation (5 images)

And creates `yolo_config.yaml`:

```yaml
train: /home/aisd_user12/yolo-uv/split/train/images
val: /home/aisd_user12/yolo-uv/split/val/images
nc: 3
names:
- open_hand
- thumbs_up
- thumbs_down
```

---

## Step 4 — Model Training (Linux Workstation)

Train the YOLO11n model:

```bash
uv run python train.py
```

Training settings:
- Model: `yolo11n.pt`
- Epochs: 100
- Image size: 640
- Batch size: 16

Results achieved:
| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| all | 0.995 | 0.970 |
| open_hand | 0.995 | 0.995 |
| thumbs_up | 0.995 | 0.970 |
| thumbs_down | 0.995 | 0.945 |

Output: `runs/detect/train-4/weights/best.pt`

---

## Step 5 — Export to IMX Format (Linux Workstation)

Install export tools:

```bash
uv pip install "edge-mdt-cl[torch]" model_compression_toolkit mct-quantizers
```

Export model:

```bash
uv run python yolo_export.py \
  --init_model runs/detect/train-4/weights/best.pt \
  --export_format imx \
  --export_only \
  --int8_weights
```

Output: `runs/detect/train-4/weights/best_imx_model/packerOut.zip`

---

## Step 6 — Convert to RPK (Raspberry Pi)

Install IMX tools:

```bash
sudo apt install imx500-all
```

Convert to RPK:

```bash
imx500-package -i packerOut.zip -o ./
```

Output: `network.rpk`

---

## Step 7 — Run Gesture Detection (Raspberry Pi)

```bash
cd /home/pi/Desktop/Project_AISD/Gesture_Detection_AISD
source .venv/bin/activate
python /home/pi/picamera2/examples/imx500/imx500_object_detection_demo.py \
  --model network.rpk \
  --labels my_training_data/classes.txt \
  --fps 25 \
  --bbox-normalization \
  --bbox-order xy
```

---

## Results

The system successfully detects all 3 gestures in real-time:

| Gesture | Confidence |
|---------|------------|
| thumbs_up | ~0.56 |
| thumbs_down | ~0.59 |
| open_hand | ~0.56 |

---

## Hardware Used
- Raspberry Pi 5 (8GB)
- Raspberry Pi AI Camera (IMX500)

## Software Used
- Python 3.10 / 3.11
- Ultralytics YOLO 8.4.51
- Label Studio
- picamera2
- PyTorch 2.4.1
- Linux Workstation (NVIDIA RTX A5000)

---

## References
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com)
- [Raspberry Pi IMX500 Guide](https://docs.ultralytics.com/integrations/sony-imx500/)
- [Label Studio](https://labelstud.io/)
- Project Guidance — Prof. Dr. Thomas Ewender, DIT