# 🧠 Brain Tumor Detection using Deep Learning

An AI-powered medical imaging system that detects brain tumors from MRI scans using Convolutional Neural Networks (CNN) and visualizes predictions using Grad-CAM heatmaps.

---

## 🚀 Project Overview

This project is designed to assist in the early detection of brain tumors using MRI images. The system uses a deep learning model trained on MRI data to classify whether a tumor is present or not.

The application provides:

* Tumor / No Tumor prediction
* Confidence score
* Heatmap visualization (Grad-CAM)
* Tumor region highlighting

---

## 🎯 Features

* 🧠 Brain Tumor Detection (Binary Classification)
* 🔥 Grad-CAM Heatmap Visualization
* 📍 Tumor Region Highlighting
* 📊 Confidence Score Display
* 🧑‍⚕️ Patient Information Input
* 🌐 Interactive Web App using Streamlit

---

## 🖥️ Demo

Upload an MRI image and get:

* Prediction: Tumor / No Tumor
* Confidence Score
* Heatmap
* Tumor outline

---

## 📂 Dataset

You can use:

* Kaggle Brain MRI Dataset
* BRATS Dataset

Images should be organized as:

data/
├── tumor/
└── no_tumor/

---

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) with:

* Convolution Layers
* ReLU Activation
* MaxPooling
* Global Average Pooling
* Fully Connected Layer

Output:

* Class 0 → No Tumor
* Class 1 → Tumor

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/brain-tumor-ai.git
cd brain-tumor-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🏋️ Training the Model

To train the model:

```bash
python train.py
```

This will generate:

```
model.pth
```

Move it to:

```
saved_models/model.pth
```

---

## 📊 Results

* Accuracy: ~85-95% (depending on dataset)
* Fast inference
* Visual explanation with Grad-CAM

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **NOT a medical diagnosis tool**.

---

## 📌 Future Improvements

* 3D MRI processing (.nii files)
* Multi-class tumor classification
* Better segmentation (U-Net)
* Deploy on cloud (AWS / HuggingFace)

---

## 👨‍💻 Author

Jitesh Sharma
B.Tech Student | AI/ML Enthusiast

---

## ⭐ If you like this project, give it a star!
