# 🌿 Plant Disease Detection using ResNet50 (Deep Learning)

This project implements a **deep learning–based system for multi-class plant disease classification** using a ResNet50 Convolutional Neural Network (CNN).  
It demonstrates an end-to-end machine learning workflow including dataset acquisition, preprocessing, model training, evaluation, and deployment.

---

## 🚀 Live Demo

A real-time prediction web application is deployed using **Gradio** on Hugging Face Spaces:

👉 https://huggingface.co/spaces/YashDawange21/plant-disease-detector

---

## 📌 Project Highlights

- Image-based multi-class plant disease classification  
- Transfer learning using a pretrained ResNet50 model  
- Handling class imbalance using weighted sampling and loss weighting  
- Model evaluation using multiple classification metrics  
- Deployment of the trained model for real-time inference  

---

## 🧠 Model Architecture

- **Model:** ResNet50  
- **Framework:** PyTorch  
- **Pretrained on:** ImageNet  
- **Input Image Size:** 224 × 224  
- **Number of Classes:** 38 plant disease categories  

---

## 🧪 Dataset

- **Dataset Name:** PlantVillage  
- **Source:** Kaggle  
- **Access Method:** Kaggle API (`kaggle.json`)  
- **Image Type:** Color leaf images  

### Dataset Characteristics
- Controlled lighting conditions  
- Uniform background  
- Centered leaf images  

---

## 🔧 Training Methodology

- Dataset loaded using `torchvision.datasets.ImageFolder`  
- Transfer learning with a pretrained ResNet50 backbone  
- **Class imbalance handling:**
  - `WeightedRandomSampler`
  - Class-weighted `CrossEntropyLoss`  
- Training strategy:
  - Freezing backbone layers initially  
  - Fine-tuning classifier layers  
- Optimization techniques:
  - AdamW optimizer  
  - Learning rate scheduling  
  - Early stopping to prevent overfitting  

---

## 📊 Model Evaluation (Validation Set)

| Metric | Value |
|------|------|
| Accuracy | **98.93%** |
| Precision (weighted) | **98.99%** |
| Recall (weighted) | **98.93%** |
| F1-score (weighted) | **98.94%** |
| Top-3 Accuracy | **99.89%** |

> Metrics computed on a held-out validation dataset.

---

## 📈 Evaluation Analysis

- Confusion matrix used for class-wise performance analysis  
- Higher confusion observed between visually similar disease categories  
- High Top-3 accuracy indicates strong ranking capability even when Top-1 prediction is incorrect  

---

## ⚠️ Limitations

- The PlantVillage dataset consists of **clean, lab-controlled images**  
- Real-world images may differ due to:
  - Background clutter  
  - Lighting variations  
  - Camera angle and image quality  

This highlights the challenge of **domain shift** in real-world deployment.

---

## 🛠 Tech Stack

{Python, PyTorch, Torchvision, NumPy, Scikit-learn, Matplotlib, PIL, Gradio, Kaggle API}

---
