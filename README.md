# 🏥 **Disease Prediction System**
## 📌 **Overview**
This is a **Disease Prediction System**, a machine learning-based application designed to predict whether a person has **Diabetes**, **Heart Disease**, or **Parkinson’s Disease** based on user input. Built using **Streamlit** for the frontend and various **ML models** for prediction, this system provides quick and accurate results for healthcare predictions.

## 📁 **Project Structure**
```plaintext
 disease_prediction/
│── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── parkinsons.csv
│── training_modules/
│   ├── diabetes_model.sav
│   ├── heart_model.sav
│   ├── parkinson.sav
│── web.py
│── README.md
|── requirements.txt
```

## 🛠️ **Technologies Used**
  - Python 🐍
  - Streamlit (for UI)
  - Scikit-learn (for model training)
  - Pandas & NumPy (for data processing)
  - Pickle (for model storage)
  ## ⚡ **Features**
  - ✔ Predicts Diabetes, Heart Disease, and Parkinson’s Disease
  - ✔ User-friendly Streamlit interface
  - ✔ Models trained on real-world datasets
  - ✔ Fast and accurate results

## 🚀 **How to Run**
1️⃣ Clone the repository:
  git clone https://github.com/Ayush1665/disease_prediction.git

2️⃣ Install dependencies:
  pip install -r requirements.txt

3️⃣ Run the application:
  streamlit run web.py

4️⃣ Enter the required details and get predictions!
