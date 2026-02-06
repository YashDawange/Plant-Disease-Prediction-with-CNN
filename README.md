# 🌿 Plant Disease Detection using Deep Learning (ResNet)

This project implements a deep learning–based system for classifying plant leaf diseases using a ResNet Convolutional Neural Network (CNN).  
It demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and analysis.

---

## 🚀 Live Demo

An interactive web application for real-time prediction is deployed on Hugging Face:

👉 https://huggingface.co/spaces/YashDawange21/plant-disease-detector

---

## 📌 Project Overview

- Image-based plant disease classification  
- ResNet-based CNN trained on the PlantVillage dataset  
- Handling class imbalance using weighted sampling and loss weighting  
- Extensive evaluation using multiple performance metrics  
- Analysis of real-world generalization challenges  

---

## 🧠 Model Details

- Architecture: ResNet50  
- Framework: PyTorch  
- Pretraining: ImageNet  
- Number of Classes: 38 plant disease categories  
- Input Size: 224 × 224  

---

## 🧪 Dataset

- Dataset: PlantVillage (Kaggle)
- Image Type: Color leaf images
- Dataset Characteristics:
  - Controlled lighting
  - Clean background
  - Centered leaf images

---

## 🔧 Training Strategy

- Data augmentation to improve robustness  
- Class imbalance handled using:
  - WeightedRandomSampler
  - Class-weighted CrossEntropy loss  
- Two-stage fine-tuning:
  - Initial freezing of backbone
  - Gradual unfreezing of deeper layers  
- Label smoothing to reduce overconfidence  
- Learning rate scheduling and early stopping  

---

## 📊 Model Performance (Validation Set)

| Metric | Value |
|------|------|
| Accuracy | 98.93 % |
| Precision (weighted) | 98.99 % |
| Recall (weighted) | 98.93 % |
| F1-score (weighted) | 98.94 % |
| Top-3 Accuracy | 99.89 % |

> Metrics computed on a held-out validation dataset.

---

## 📈 Evaluation & Analysis

- Confusion matrix used to analyze class-wise performance  
- Visually similar diseases show higher confusion  
- Top-3 accuracy is significantly higher than top-1 accuracy, indicating effective ranking of predictions  

---

## ⚠️ Limitations

The model was trained on a controlled dataset.  
Performance on real-world images may vary due to:

- Background clutter  
- Lighting variations  
- Camera angle and quality  

This highlights the importance of dataset diversity and domain adaptation.

## 🛠 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 👨‍💻 Author

**Yash Dawange**  
Biomedical Engineering | AI & Machine Learning Enthusiast
