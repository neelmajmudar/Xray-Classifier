# 🩻 Xray Classifier

## About

Xray Classifier is a deep learning-powered tool for detecting abnormalities in chest X-ray images. Built with Convolutional Neural Networks (CNNs) and trained on labeled medical datasets, this model helps automate the identification of pathologies such as pneumonia, COVID-19, or None.

It aims to support radiologists and healthcare professionals by providing rapid image classification that can assist in triaging or flagging critical cases.

## Features

- **CNN-based Image Classifier**  
  Uses a convolutional neural network (CNN) architecture implemented in `XrayClassifier.py` for robust and efficient image classification.

- **Pretrained Model Support**  
  Easily extendable to integrate pretrained models using transfer learning techniques.

- **Real-time Inference**  
  Classify X-ray images directly through a browser interface powered by Streamlit via `XrayApp.py`.

- **Model Performance Metrics**  
  Training/validation accuracy and loss visualized through:
  - `Training&ValidationAccuracy.png`
  - `Training&ValidationLoss.png`
  - `ConfusionMatrixActualvPredicted.png`

- **Testing Output Visualization**  
  Includes `testpredict.png` as a sample model inference result.

- **Modular and Extensible**  
  Codebase is designed to easily modify model structure, optimizer, or add support for Grad-CAM visualizations and multi-class classification.

## File Overview

| File                               | Description                                               |
|------------------------------------|-----------------------------------------------------------|
| `XrayClassifier.py`                | Core training and classification logic                    |
| `XrayApp.py`                       | Streamlit-based app for uploading and classifying images  |
| `testpredict.png`                  | Sample output prediction visualization                    |
| `Training&ValidationAccuracy.png`  | Accuracy curves over training epochs                      |
| `Training&ValidationLoss.png`      | Loss curves over training epochs                          |
| `ConfusionMatrixActualvPredicted.png` | Confusion matrix visualizing prediction performance    |
| `README.md`                        | Project documentation                                     |
| `LICENSE`                          | License (MIT)                                             |


App Demo:
![XrayPredictionApp](https://github.com/neelmajmudar/Xray-Classifier/assets/142572400/993cf248-74e8-4cd1-9121-3a5c278463d5)

---

## Technologies Used

- **Deep Learning**: Keras, TensorFlow  
- **Data Processing**: Python, NumPy, OpenCV, PIL  
- **Visualization**: Matplotlib, Seaborn  
- **Web Interface**: Streamlit  
- **Model Evaluation**: Accuracy, Loss, Confusion Matrix

## Future Improvements

- Add Grad-CAM support for better visual interpretability  
- Expand dataset support (e.g., COVIDx, CheXpert)  
- Integrate multi-label classification  
- Add export option for DICOM overlays  
- Deploy as a hosted diagnostic aid platform

## Example Usage

### 1. Clone the Repository

```bash
git clone https://github.com/neelmajmudar/Xray-Classifier.git
cd Xray-Classifier

pip install -r requirements.txt

streamlit run XrayApp.py
```




