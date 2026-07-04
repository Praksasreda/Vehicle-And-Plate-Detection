# Vehicle & License Plate Detection

Real-time vehicle detection and license plate recognition from a live camera feed. Uses a two-stage YOLO pipeline — general object detection to isolate vehicles, followed by a dedicated license plate model — with EasyOCR for character recognition.

Built as a portfolio project exploring computer vision and real-time inference.

---

## How It Works

```
Camera → YOLOv8 (vehicle detection) → license_plate_detector.pt (plate localization) → EasyOCR (OCR) → terminal output
```

1. YOLOv8n detects vehicles in each frame (class ID 2 = car)
2. Only cars occupying >30% of frame width are processed (proximity filter)
3. A second YOLO model localizes the license plate within the cropped car region
4. EasyOCR reads the plate; results above 95% confidence and exactly 7 characters are printed

---

## Requirements

- Python 3.9+
- USB or built-in webcam
- `license_plate_detector.pt` — custom-trained YOLO model (not included, see Notes)

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Praksasreda/Vehicle-And-Plate-Detection
cd Vehicle-And-Plate-Detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> First run downloads YOLOv8n weights (~6MB) and EasyOCR models automatically.

**3. Add the plate detection model**

Place `license_plate_detector.pt` in the project root. See Notes for sourcing or training your own.

**4. Run**
```bash
python main.py
```

Press `Q` to quit.

---

## Configuration

At the top of `main.py`:

```python
kamera1 = Kamera(1)          # camera index (0 = built-in, 1 = USB)
```

Inside the detection loop:

```python
sirina_slike * 0.3           # proximity threshold — car must fill 30% of frame width
zaupanje >= 0.95             # OCR confidence threshold
len(tekst) == 7              # expected plate length (adjust for your region)
```

To find available camera indexes:
```bash
python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i} available')
    cap.release()
"
```

---

## Stack

| Component | Library |
|-----------|---------|
| Camera capture | OpenCV + threading |
| Vehicle detection | YOLOv8n (`ultralytics`) |
| Plate localization | Custom YOLO model |
| OCR | EasyOCR |
| Inference backend | PyTorch |

---

## Notes

- `license_plate_detector.pt` is not included in this repository. You can train your own using the [Ultralytics docs](https://docs.ultralytics.com/modes/train/) or use a pre-trained model from [Roboflow Universe](https://universe.roboflow.com/).
- OCR allowlist is restricted to alphanumeric characters; adjust `allowlist` in `reader.readtext()` for non-Latin plates.
- Plate length filter (`len(tekst) == 7`) is calibrated for Slovenian plates — modify as needed.

---

## Roadmap

- [ ] Logging detections to file or database
- [ ] Multi-camera support
- [ ] On-screen bounding box label with recognized plate text
- [ ] Support for additional plate formats / regions

---

## License

MIT
