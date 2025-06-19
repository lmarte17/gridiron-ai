# Gridiron AI

Gridiron AI is an experimental project for classifying American football plays from All‑22 video. The code collects video frames from YouTube, extracts detailed formation features using a fine‑tuned YOLO detector, and trains a multimodal neural network that combines those features with image data to predict whether a play is a run or a pass.

## Project Goals

1. **Data Collection** – Download All‑22 footage from YouTube, capture screenshots at specific timestamps, and save annotations for each frame.
2. **Feature Extraction** – Detect player positions and compute rich formation features such as personnel grouping, formation width, backfield depth, and defensive alignment.
3. **Label Transfer** – Assign run/pass labels to unlabeled frames by measuring similarity between formation feature vectors.
4. **Play Prediction Model** – Train a neural network that fuses EfficientNet image features with encoded formation features to classify plays and predict formation attributes.

## Repository Structure

- `src/data_collection.py` – Tools to download YouTube videos with `yt_dlp`, read timestamp CSVs, and export annotated screenshots.
- `src/feature_extraction.py` – Loads a YOLO model and converts detections into formation features (personnel counts, trips alignment, bunch sets, etc.).
- `src/label_transfer.py` – Computes similarity between feature vectors to transfer labels from a labeled set to unlabeled data.
- `src/models.py` – Defines the `MultiModalPlayPredictor` network which merges EfficientNet visual features with formation features via attention and outputs run/pass predictions along with personnel and formation width estimates.
- `data/processed/` – CSV files containing extracted features and metadata for training, validation, testing, and YouTube screenshots.
- `models/` – Pretrained weights for the detection model and a saved play predictor checkpoint.
- `notebooks/` – Example notebooks for fine‑tuning the detection model, exploring the dataset, and training the prediction model.

## Usage

This repository does not include a full training pipeline, but the core components are available:

1. **Collect Data**
   ```bash
   python src/data_collection.py
   ```
   Customize the script to point to your timestamp CSVs and YouTube links.

2. **Extract Features**
   ```bash
   python -m src.feature_extraction
   ```
   This will generate CSV rows with formation features for each image.

3. **Transfer Labels**
   ```bash
   python -m src.label_transfer
   ```
   Use formation similarity to assign run/pass labels to new data.

4. **Train or Evaluate Models**
   A pretrained `best_play_predictor.pth` is provided. Model code lives in `src/models.py` and can be adapted for your experiments.

## Requirements

The project relies on PyTorch, `timm`, `ultralytics` (for YOLO), `opencv-python`, and `yt-dlp`. Install dependencies with:

```bash
pip install torch torchvision timm ultralytics opencv-python yt-dlp
```

## Acknowledgements

This repository is a proof of concept for combining computer vision and structured formation features to understand American football strategy. The code and data are provided as research examples and may require additional work for a production system.
