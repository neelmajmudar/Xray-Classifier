# 🩻 Xray Classifier

## About
Xray Classifier is a deep learning-powered tool for detecting abnormalities in chest X-ray images. Built with convolutional neural networks (CNNs) and trained on labeled medical datasets, the model can help automate the identification of conditions such as pneumonia, fibrosis, or other thoracic pathologies.

The goal is to assist radiologists and healthcare professionals by providing fast, preliminary image classification that can prioritize or flag cases requiring closer inspection.

## Features

- **CNN-based Image Classifier**  
  Utilizes a Convolutional Neural Network architecture (e.g., ResNet or VGG) for robust image classification.

- **Binary or Multi-class Support**  
  Depending on dataset labeling, supports classification of normal vs. abnormal or multiple disease categories.

- **Transfer Learning Capable**  
  Easily fine-tune pre-trained ImageNet models for X-ray classification using transfer learning.

- **Real-time Inference**  
  Script-based or notebook-based prediction for rapid testing of new X-ray images.

- **Evaluation and Metrics**  
  Outputs accuracy, precision, recall, F1-score, and confusion matrices. Includes visualization for ROC/AUC and model performance over epochs.

- **Visualization Tools**  
  Grad-CAM and saliency map support to visualize which areas of the X-ray the model focuses on for decision-making.

- **Extensible Pipeline**  
  Modular codebase for easy experimentation with different models, optimizers, loss functions, and data augmentations.

## Project Overview

Xray Classifier follows a structured ML pipeline:

### 📁 Data Preparation
- Input: Chest X-ray datasets (e.g., [NIH ChestX-ray14](https://www.nature.com/articles/sdata201711).
- Preprocessing includes resizing, grayscale normalization, and augmentation (horizontal flip, random crop).
- Dataset split into train/validation/test using `split_data.py`.

### 🧠 Model Training
- Defined in `train.py`. Configurable architecture (ResNet18, DenseNet121, etc.).
- Uses standard PyTorch training loop with checkpointing and early stopping.
- Outputs training/validation accuracy and loss curves for analysis.

### 📸 Prediction
- `predict.py` enables classification on a single image or batch of X-rays.
- Outputs predicted class and confidence score.

### 📊 Evaluation
- `evaluate.py` runs the model on a test set and generates:
  - Confusion matrix
  - Classification report
  - ROC/AUC curve
- Helps measure generalization performance and overfitting.

### 🖼️ Grad-CAM Visualization
- Generate Grad-CAM heatmaps to visualize which regions of the X-ray influenced the model's decision.
- Useful for interpretability in medical settings.

---

App Demo:
![XrayPredictionApp](https://github.com/neelmajmudar/Xray-Classifier/assets/142572400/993cf248-74e8-4cd1-9121-3a5c278463d5)

## Example Usage

```bash
# Clone and install
git clone https://github.com/neelmajmudar/Xray-Classifier.git
cd Xray-Classifier
pip install -r requirements.txt

# Train the model
python train.py --data_dir ./data --epochs 20 --model resnet18

# Run evaluation
python evaluate.py --model_path ./models/resnet18_best.pth

# Predict a new image
python predict.py --image ./samples/patient_xray.jpg --model ./models/resnet18_best.pth


