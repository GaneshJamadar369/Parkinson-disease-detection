# Parkinson's Disease Detection

This repository contains a Parkinson's disease voice detection model built with Python and scikit-learn.

## Structure
- `dataset/` - audio dataset with `healthy` and `parkinsons` folders
- `model/` - training, prediction, and evaluation scripts
- `model/features.py` - feature extraction
- `model/train.py` - train and save the model
- `model/predict.py` - realtime or file-based prediction
- `model/evaluation.py` - train/test evaluation and ROC curve generation
- `model/test_model.py` - full dataset evaluation script
- `model/test_samples.py` - random sample prediction tester

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   cd model
   python train.py
   ```
3. Predict from a voice file:
   ```bash
   python predict.py ../dataset/healthy/healthy_000.wav
   ```
4. Run the evaluation:
   ```bash
   python evaluation.py
   ```
5. Use realtime recording:
   ```bash
   python predict.py
   ```

## Notes
- `model/predict.py` accepts an optional audio file path.
- The model should be retrained after changing feature extraction or data.
