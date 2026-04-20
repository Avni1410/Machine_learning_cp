# 🌍 FinAccess AI — Financial Inclusion Prediction System

An end-to-end Machine Learning project that predicts financial inclusion status of individuals across African countries using socio-economic and demographic data.

The system includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple ML models (Logistic Regression, XGBoost, LightGBM, CatBoost)
- Stacking Ensemble
- Streamlit Web App for real-time predictions

---

## 🚀 Project Structure
financial-inclusion-africa-ml-zindi/
│
├── data/ # Raw datasets
├── notebooks/ # EDA, modeling, tuning
├── src/ # Core ML pipeline code
├── artifacts/ # Saved trained models
├── app.py # Streamlit application
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Installation

### 1. Clone repository
```bash
git clone https://github.com/Avni1410/Machine_learning_cp.git
cd finaccess-ai

2. Create virtual environment

python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies

pip install -r requirements.txt


Run Exploratory Data Analysis

python notebooks/01_EDA.py

Train Models

python notebooks/03_modeling.py

Run Streamlit App

streamlit run app.py

http://localhost:8501


---

# 🚨 IMPORTANT FINAL STEP

Add dependencies file if missing:

```bash
pip freeze > requirements.txt
