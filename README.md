❤️ Heart Disease Risk Assessment Web Application

A clinically inspired Machine Learning–powered web application for assessing heart disease risk using a highly accurate XGBoost model (94.6%).
Designed for clinicians, researchers, and patients, the system evaluates 13 clinical parameters and provides instant predictions, parameter-level analysis, and personalized recommendations.

📌 Project Highlights

🔥 94.6% accurate XGBoost model for heart disease prediction

⚕️ Real-time clinical parameter analysis

📊 Feature importance visualization

🎨 Professional, medical-grade UI/UX built with Streamlit

🧠 Evidence-based medical recommendations

📈 Adjustable decision threshold for clinical flexibility

🚀 Deployable through Streamlit Cloud, Heroku, or Docker

🧠 Model Overview

Machine Learning Algorithm: XGBoost
Training Dataset: Cleveland Heart Disease Dataset (UCI)
Features Used: 13 standard cardiac risk parameters
Target: 0 = No Heart Disease, 1 = Heart Disease

✔️ Model Performance
Metric	Value
Accuracy	94.6%
Precision	88.4%
Recall	90.6%
F1 Score	89.4%
ROC-AUC	0.982
Specificity	95.9%

🔝 Top 5 Feature Importances

Major Vessels (ca) – 16.16%

Max Heart Rate (thalach) – 15.97%

Thalassemia (thal) – 11.55%

Exercise-Induced Angina (exang) – 11.15%

Age – 10.32%

🧬 Clinical Parameters Analyzed
Demographics

Age

Sex

Vital Signs

Resting Blood Pressure

Serum Cholesterol

Fasting Blood Sugar

Symptoms

Chest Pain Type

Exercise-Induced Angina

Diagnostic Test Results

Max Heart Rate

Major Vessels (ca)

ST Depression (oldpeak)

Resting ECG

ST Slope

Thalassemia

🖥️ Application Features
1️⃣ Intuitive Input Form

Logical grouping (Demographics, Symptoms, Test Results)

Tooltips and data validation

2️⃣ Real-Time Parameter Analysis

Each parameter is flagged as:

🟢 Normal

🟡 Warning

🔴 Danger

3️⃣ Risk Prediction Engine

Heart disease probability (%)

Risk categories: Very Low → Very High

Adjustable decision threshold (default: 62.7%)

4️⃣ Personalized Clinical Recommendations

Lifestyle guidance

Medical follow-up suggestions

Emergency warnings

5️⃣ Model Insights

Feature importance chart

Clinical interpretation support

📁 Project Structure
heart-disease-predictor/
│
├── app/
│   ├── heart_disease_app.py
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── feature_order.pkl
│
├── data/
│   ├── cleveland_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── models/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── requirements.txt
└── README.md

🛠️ Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Add Model Files to /app

Ensure the following files exist:

xgboost_model.pkl

scaler.pkl

feature_order.pkl

5. Run the Application
streamlit run app/heart_disease_xgboost_app.py


App will open at:
👉 http://localhost:8501

🚀 Deployment Options
Streamlit Cloud (Recommended)

Push project to GitHub

Go to https://share.streamlit.io

Connect repo → Deploy

Deploy on Heroku
pip freeze > requirements.txt
echo "web: streamlit run app/heart_disease_xgboost_app.py --server.port $PORT" > Procfile
heroku create heart-disease-app
git push heroku main

Run with Docker
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/heart_disease_xgboost_app.py"]

⚠️ Medical Disclaimer

This tool is NOT a diagnostic device.
It is intended for educational and clinical decision-support purposes only.

Seek immediate medical attention for:

Severe chest pain

Pain radiating to arm/neck/jaw

Shortness of breath

Fainting

Heavy sweating + chest pressure

Always consult licensed healthcare professionals for medical advice.

🔮 Future Enhancements

🔤 Multi-language support

📱 Mobile application

⏳ Patient history tracking

📡 FHIR-based EHR integration

🧬 Genetic risk factor support

🧠 Improved model training pipeline

📊 More advanced visualizations

🔒 Enhanced data security


📚 References

XGBoost Documentation

Streamlit Documentation

Scikit-learn Documentation

👤 Author

Aniket Paswan
🔗 GitHub: https://github.com/Anikk02