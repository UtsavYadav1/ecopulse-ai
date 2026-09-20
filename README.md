# ⚡ EcoPulse AI

### AI-Based Energy Consumption Prediction & Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SDG 7](https://img.shields.io/badge/SDG-7%20Clean%20Energy-yellow)](https://sdgs.un.org/goals/goal7)
[![SDG 13](https://img.shields.io/badge/SDG-13%20Climate%20Action-darkgreen)](https://sdgs.un.org/goals/goal13)

---

## 📋 Project Overview

EcoPulse AI is a machine learning system that predicts residential appliance energy consumption and detects anomalous consumption patterns in real-time. It combines traditional ML regression models with Google Gemini AI to generate human-readable explanations and actionable energy-saving recommendations.

Built for a sustainability-focused AI internship, this project demonstrates practical AI/ML engineering applied to a real-world environmental problem.

---

## 🌍 Problem Statement

### Why Energy Consumption Matters

- Buildings account for **~40% of global energy consumption** and **~36% of CO₂ emissions** (IEA, 2023).
- Residential appliances are a major contributor to household energy bills and carbon footprints.
- Most households lack real-time visibility into **unusual** energy consumption patterns.
- Undetected anomalies (e.g., appliances left on unnecessarily, HVAC inefficiency) lead to wasted energy and avoidable emissions.

### Solution

EcoPulse AI addresses this by:
1. **Predicting** expected appliance energy consumption using ML regression
2. **Detecting anomalies** where actual consumption deviates significantly from predictions
3. **Explaining** anomalies in plain language via Gemini AI
4. **Recommending** practical energy-saving actions

---

## 🤖 How AI Is Used

| Component | Technology | Role |
|-----------|------------|------|
| Prediction | Linear Regression (baseline) | Predict energy consumption from environmental features |
| Prediction | Random Forest Regressor | Non-linear prediction model |
| Model selection | Scikit-learn + RMSE metric | Automatically select the better model |
| Anomaly detection | Statistical residual analysis | Flag unusual consumption deviations |
| Explanation | Google Gemini 1.5 Flash | Generate human-readable explanations and recommendations |

> **Important:** Gemini does NOT perform the prediction. All ML predictions are done by Scikit-learn models trained on the actual dataset.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI ML Repository |
| URL | https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |
| Observations | 19,735 rows |
| Period | Jan 11, 2016 – May 27, 2016 |
| Frequency | Every 10 minutes |
| Location | Low-energy house, Stambruges, Belgium |
| Target | `Appliances` — energy consumption (Wh) |

### Key Features

| Feature | Description |
|---------|-------------|
| `T1`–`T9` | Indoor temperature per room (°C) |
| `RH_1`–`RH_9` | Indoor relative humidity per room (%) |
| `T_out` | Outdoor temperature (°C) |
| `RH_out` | Outdoor humidity (%) |
| `Windspeed` | Wind speed (m/s) |
| `Visibility` | Visibility (km) |
| `Tdewpoint` | Dew point temperature (°C) |
| `lights` | Energy from lights (Wh) |
| `date` | Timestamp → `hour`, `day`, `day_of_week`, `month`, `is_weekend` |

---

## 🔬 Machine Learning Approach

### Data Preprocessing

- `date` parsed as datetime and used for temporal feature extraction
- Chronological 80/20 train/test split (no random shuffling)
- `rv1`, `rv2` (random noise variables) dropped
- No missing values in this dataset; median imputation as safety net
- Features aligned to avoid target leakage

### Why Chronological Splitting?

This is a **time-series dataset**. Random splitting would allow the model to see future data during training, leading to:
- Artificially inflated test metrics
- Unreliable generalisation estimates
- Data leakage

Chronological splitting (first 80% → train, last 20% → test) respects the temporal ordering and gives an honest evaluation.

### Feature Engineering

From the `date` column:

| Feature | Rationale |
|---------|-----------|
| `hour` | Intra-day consumption patterns |
| `day` | Day-of-month variation |
| `day_of_week` | Weekday vs weekend behaviour |
| `month` | Seasonal patterns |
| `year` | Inter-year baseline |
| `is_weekend` | Behavioural shift on weekends |

### Models Trained

| Model | Type | Description |
|-------|------|-------------|
| Linear Regression | Baseline | Establishes a simple linear relationship between features and target |
| Random Forest Regressor | Non-linear ensemble | Captures complex, non-linear interactions; 100 trees |

### Model Comparison

> **Note:** Results below are computed on actual held-out test data. No values are fabricated.

| Model | MAE (Wh) | RMSE (Wh) | R² |
|-------|----------|-----------|-----|
| Linear Regression | **49.88** | **85.93** | **0.1092** |
| Random Forest | 130.70 | 166.13 | -2.3301 |

**Selected model: Linear Regression** (lower RMSE on test set)

> **Note on Random Forest:** The negative R² for Random Forest on the test period indicates the model failed to generalise to the held-out period. This is an honest result — Random Forest overfitted on training data. This is precisely why we evaluate both models on a true temporal hold-out set rather than cross-validation with random splits.

**Selection criterion:** Lowest RMSE on the test set.  
RMSE is preferred over MAE because it penalises large errors more heavily — important when large deviations signal anomalies.

---

## 🚨 Anomaly Detection

### Methodology

After training, residuals on the **training set** are used to establish the anomaly threshold:

```
threshold = mean(|residual_train|) + 2 × std(|residual_train|)
```

This is statistically interpretable: observations whose absolute prediction error exceeds the typical training error by more than 2 standard deviations are flagged as anomalies.

### Severity Levels

| Severity | Condition |
|----------|-----------|
| Normal | `|residual| ≤ threshold` |
| Moderate | `threshold < |residual| ≤ 1.5 × threshold` |
| High | `|residual| > 1.5 × threshold` |

### ⚠️ Important Disclaimer

An anomaly means the observed consumption is **unusually different from the model's expectation**. It does **NOT** automatically prove equipment failure, sensor malfunction, or any specific physical cause.

---

## ✨ Gemini AI Integration

Gemini is used **only** for natural-language explanation — not for prediction.

**Flow:**
```
ML Prediction → Residual → Anomaly Detection → Structured Data → Gemini → Explanation
```

Gemini receives:
- Actual vs predicted consumption
- Residual magnitude and severity
- Time of observation (hour, day of week)
- Environmental context (temperature, humidity, wind speed)

Gemini returns:
- Short explanation of the deviation
- Possible contributing factors (clearly framed as possibilities)
- 2–4 practical energy-saving recommendations
- Sustainability impact statement

---

## 🏗️ System Architecture

```
ecopulse-ai/
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
│
├── src/
│   ├── data_loader.py        # Dataset download and loading
│   ├── preprocessing.py      # Data cleaning and splitting
│   ├── feature_engineering.py # Feature creation
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Metrics computation
│   ├── anomaly_detection.py  # Residual-based anomaly detection
│   └── gemini_service.py     # Google Gemini API integration
│
├── templates/
│   ├── index.html            # Dashboard
│   └── result.html           # Prediction result page
│
├── static/
│   ├── css/style.css         # Stylesheet
│   ├── js/script.js          # Frontend JS
│   └── plots/                # Generated visualizations
│
├── models/                   # Trained model artifacts
│   ├── final_model.joblib
│   ├── feature_columns.joblib
│   ├── metrics.json
│   └── anomaly_threshold.json
│
├── data/                     # Dataset (downloaded at runtime)
│   └── README.md
│
└── notebooks/
    └── exploration.ipynb     # EDA notebook
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| ML Framework | Scikit-learn |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Model Serialisation | Joblib |
| AI Explanation | Google Gemini 1.5 Flash |
| Frontend | HTML5, CSS3, Vanilla JS |
| Dataset Source | UCI ML Repository |

---

## 🌱 SDG Alignment

### SDG 7 — Affordable and Clean Energy
EcoPulse AI helps building occupants understand and reduce energy consumption, directly contributing to energy efficiency — a key pillar of SDG 7.

### SDG 13 — Climate Action
By detecting and explaining energy anomalies, EcoPulse AI helps reduce unnecessary energy waste, lowering the carbon footprint of residential buildings.

---

## 🛡️ Responsible AI

| Principle | Implementation |
|-----------|---------------|
| **Transparency** | All model metrics are calculated on real data and displayed in the dashboard. No results are fabricated. |
| **Human oversight** | All recommendations are advisory. Human judgment is required for any action. |
| **Privacy** | No personally identifiable information is used or collected. |
| **Limitations** | Dataset represents a single low-energy house in Belgium (Jan–May 2016). Performance may differ in other contexts. |
| **Honest AI** | Gemini explanations are clearly framed as possible interpretations, not verified physical diagnoses. |
| **Scope clarity** | Anomaly detection flags statistical deviations — it does not diagnose equipment failures. |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ecopulse-ai.git
cd ecopulse-ai
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_actual_api_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

> **Note:** The application runs without a Gemini API key — AI explanations will be disabled but all ML features work normally.

---

## 🎓 How to Train

```bash
python src/train.py
```

This will:
1. Download the dataset automatically (~4 MB)
2. Clean and preprocess data
3. Train Linear Regression and Random Forest models
4. Evaluate both on the test set
5. Select and save the best model
6. Run anomaly detection
7. Generate all evaluation plots

Expected runtime: ~2–5 minutes depending on hardware.

---

## ▶️ How to Run Flask

```bash
python app.py
```

Then open: **http://localhost:5000**

---

## 📋 Example Workflow

```
1. python src/train.py          # Train models, generate plots
2. python app.py                # Start Flask app
3. Open http://localhost:5000   # View dashboard with real metrics
4. Fill in the prediction form  # Enter sensor values
5. Submit → view prediction     # See predicted energy and anomaly status
6. (If anomaly) Gemini explains # Get AI-powered explanation
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Redirect to dashboard |
| GET | `/dashboard` | Main dashboard |
| POST | `/predict` | Predict energy consumption |
| POST | `/explain` | Get Gemini explanation (JSON) |
| GET | `/api/metrics` | Model metrics (JSON) |
| GET | `/api/anomalies` | Anomaly records (JSON) |

---

## ⚠️ Limitations

- Trained on a single building in Belgium (Jan–May 2016) — not representative of all climates or building types.
- No real-time sensor integration; predictions use manually entered values.
- Gemini explanations are based only on the supplied data, not physical inspection.
- Anomaly detection is statistical and may produce false positives/negatives.
- Model performance may degrade outside the training distribution.

---

## 🔮 Future Improvements

- [ ] Live sensor data integration via MQTT or REST API
- [ ] LSTM-based time-series models for sequential pattern learning
- [ ] Multi-building support and user accounts
- [ ] Automated retraining pipeline with new data
- [ ] Mobile-responsive PWA version
- [ ] Carbon footprint calculator integrated with energy predictions
- [ ] Alerting system for high-severity anomalies

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📖 Dataset Attribution

> Candanedo, L., Feldheim, V., & Deramaix, D. (2017).  
> *Appliances energy prediction*.  
> UCI Machine Learning Repository.  
> https://doi.org/10.24432/C5VC8G

---

*EcoPulse AI — Powering sustainability through machine intelligence* ⚡🌱
