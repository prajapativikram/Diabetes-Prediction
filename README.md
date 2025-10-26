# 🩺 Diabetes Prediction using Machine Learning

![GitHub Repo stars](https://img.shields.io/github/stars/prajapativikram/Diabetes-Prediction?style=for-the-badge&color=brightgreen)
![GitHub forks](https://img.shields.io/github/forks/prajapativikram/Diabetes-Prediction?style=for-the-badge&color=blue)
![License](https://img.shields.io/github/license/prajapativikram/Diabetes-Prediction?style=for-the-badge&color=orange)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Model-Machine%20Learning-red?style=for-the-badge)

---

> 🧠 A predictive machine learning model that analyzes medical data to determine the likelihood of **diabetes in patients** using algorithms like Logistic Regression, SVM, and Random Forest.

---

## 📊 Project Overview

This project aims to **predict whether a person has diabetes** based on key medical parameters such as glucose level, BMI, insulin, age, and blood pressure.  
It uses a **supervised learning approach** trained on the **Pima Indians Diabetes Dataset**.

---
## 📍 Live App  
Access the live demo of this project here:  
[**Diabetes Prediction - Hugging Face Space**](https://huggingface.co/spaces/vikram8651/Diabetes-Prediction)  

---
## 🚀 Features

- 📈 Trained using multiple ML models  
- 🧮 Data preprocessing & feature scaling  
- 🤖 Model comparison & performance visualization  
- 💾 Model saving using `pickle`  
- 🧰 Clean, modular, and reproducible code  
- 🌐 Deployed using  **Hugging Face** 

---

## 🧠 Algorithms Used

| Algorithm | Description |
|------------|--------------|
| **Logistic Regression** | Baseline model for binary classification |
| **Random Forest Classifier** | Ensemble learning model for better accuracy |
| **Support Vector Machine (SVM)** | Robust classifier for nonlinear boundaries |
| **K-Nearest Neighbors (KNN)** | Simple distance-based learning |

---

## 📂 Dataset Information

- **Dataset Source:** Pima Indians Diabetes Dataset (UCI Repository)  
- **Features:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

- **Target:**  
  - `1` → Diabetic  
  - `0` → Non-Diabetic

---

## ⚙️ Installation & Setup

Clone this repository and set up the environment:

```bash
# 1️⃣ Clone the repo
git clone https://github.com/prajapativikram/Diabetes-Prediction.git

# 2️⃣ Navigate to the project directory
cd Diabetes-Prediction

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Jupyter Notebook or main script
jupyter notebook Diabetes_Prediction.ipynb
# or
python main.py
