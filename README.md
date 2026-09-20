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

EcoPulse AI is a machine learning system for historical energy consumption analysis, appliance energy prediction, and anomaly detection. It combines traditional Scikit-learn regression models with Google Gemini AI to generate human-readable explanations and actionable energy-saving recommendations.

Built for a sustainability-focused AI internship, this project demonstrates practical AI/ML engineering applied to a real-world environmental problem.

---

## 🌍 Problem Statement

### Why Energy Consumption Matters

- Buildings account for **~40% of global energy consumption** and **~36% of CO₂ emissions** (IEA, 2023).
- Residential appliances are a major contributor to household energy bills and carbon footprints.
- Most households lack automated visibility into **unusual** energy consumption patterns.
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

### Features (31 Model Input Features)

The model trains on **31 continuous and temporal features** (excluding the target `Appliances` and raw `date`):

| Category | Features | Count | Description |
|----------|----------|:-----:|-------------|
| **Lighting** | `lights` | 1 | Light fixtures energy consumption in the house (Wh) |
| **Indoor Temperatures** | `T1`–`T9` | 9 | Temperatures (°C): kitchen (`T1`), living room (`T2`), laundry (`T3`), office (`T4`), bathroom (`T5`), outside north facade (`T6`), ironing room (`T7`), teenager room (`T8`), parents room (`T9`) |
| **Indoor Humidity** | `RH_1`–`RH_9` | 9 | Relative humidity (%) measured across the 9 respective household zones |
| **Weather Station** | `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint` | 6 | External meteorological variables from Chièvres weather station: outdoor temperature (°C), barometric pressure (mm Hg), outdoor humidity (%), wind speed (m/s), visibility (km), dew point temperature (°C) |
| **Engineered Datetime** | `hour`, `day`, `day_of_week`, `month`, `year`, `is_weekend` | 6 | Derived from `date`: hour of day (0–23), day of month (1–31), day of week (0=Mon…6=Sun), month (1–12), year, and weekend binary flag (1=weekend, 0=weekday) |
| **Target Variable** | `Appliances` | 1 | Appliance energy consumption (Wh) — *predicted variable (y), excluded from feature matrix (X)* |
| **Excluded Noise** | `rv1`, `rv2` | 2 | Random dummy variables from original dataset — *dropped during preprocessing* |

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

### Methodology & Threshold Calculation

Anomaly detection is performed by evaluating the prediction residual:

$$\text{residual} = \text{actual\_energy} - \text{predicted\_energy}$$
$$\text{absolute\_error} = |\text{residual}|$$

To prevent temporal leakage, the anomaly threshold is derived strictly from the **training set** ($N = 15,788$) residual distribution:

$$\text{threshold} = \text{mean}(|\text{residual}_{\text{train}}|) + 2 \times \text{std}(|\text{residual}_{\text{train}}|)$$

From the training run on the UCI dataset:
- $\text{mean}(|\text{residual}_{\text{train}}|) = 54.25\text{ Wh}$
- $\text{std}(|\text{residual}_{\text{train}}|) = 78.51\text{ Wh}$
- $\mathbf{\text{threshold} = 54.25 + 2 \times 78.51 = 211.27\text{ Wh}}$
- Moderate/High boundary ($1.5 \times \text{threshold}$) = **316.90 Wh**

### Severity Classification Table

| Severity Level | Anomaly Flag | Code Condition | Threshold Bounds (Wh) | Test Set Count ($N=3,947$) | Description |
|---|:---:|---|---|:---:|---|
| **Normal** | `False` | `abs(residual) <= threshold` | `abs(residual) <= 211.27` | 3,807 (96.45%) | Consumption deviation is within typical expected variation ($2\sigma$) |
| **Moderate** | `True` | `threshold < abs(residual) <= 1.5 * threshold` | `211.27 < abs(residual) <= 316.90` | 60 (1.52%) | Noticeable consumption deviation exceeding statistical expectation |
| **High** | `True` | `abs(residual) > 1.5 * threshold` | `abs(residual) > 316.90` | 80 (2.03%) | Severe consumption deviation exceeding 1.5× the anomaly cutoff |

**Summary of Anomaly Results:**
- **Total Test Predictions:** 3,947
- **Total Anomalies Detected:** 140 (3.55%)
- **Moderate Anomalies:** 60 (1.52%)
- **High Anomalies:** 80 (2.03%)

### ⚠️ Important Disclaimer

An anomaly means the observed consumption is **unusually different from the statistical model expectation**. It does **NOT** prove equipment malfunction, physical appliance failure, or electrical fault.

---

## ✨ Gemini AI Integration

EcoPulse AI uses Google Gemini **strictly for natural-language explanation and advisory recommendations** — never for numerical prediction.

**Architecture Flow:**
```
ML Prediction (Linear Regression)
       ↓
Residual Calculation (Actual - Predicted)
       ↓
Statistical Anomaly Detection (Threshold: 211.27 Wh)
       ↓
Structured Context (Residual, Severity, Sensor Values, Time)
       ↓
Google Gemini 1.5 Flash (google-generativeai SDK)
       ↓
Natural-Language Explanation & Energy-Saving Recommendations
```

### Implementation Details

- **Model:** `gemini-1.5-flash` (via `genai.GenerativeModel("gemini-1.5-flash")` in `src/gemini_service.py`)
- **Package:** `google-generativeai>=0.7.0` (configured in `requirements.txt`)
- **Authentication:** Read via `GEMINI_API_KEY` environment variable (never committed)
- **Graceful Fallback:** If `GEMINI_API_KEY` is missing or invalid, the application runs normally and returns an informative fallback message.

### Structured Input to Gemini

When an anomaly is flagged (`Moderate` or `High`), the system sends the following structured parameters:
- **Consumption Data:** `actual_energy` (Wh), `predicted_energy` (Wh), `residual` (Wh), `severity` ("Moderate" / "High")
- **Temporal Context:** `hour` (0–23), `day_of_week` (e.g., "Monday")
- **Indoor Climate:** `T1` (Indoor temperature, °C), `RH_1` (Indoor humidity, %)
- **Outdoor Meteorological:** `T_out` (°C), `RH_out` (%), `Windspeed` (m/s), `Visibility` (km), `Tdewpoint` (°C)
- **Sub-metering:** `lights` (Lighting energy, Wh)

### Gemini Output Structure

Gemini is strictly prompted to return:
1. **Short Explanation (2–3 sentences):** Contextual assessment of why consumption deviated from the model baseline.
2. **Possible Contributing Factors (2–4 bullets):** Environmental or behavioural hypotheses based *strictly* on provided features, explicitly disclaiming physical diagnosis.
3. **Energy-Saving Recommendations (2–4 bullets):** Concrete, actionable tips tailored to the time of day and environmental context.
4. **Sustainability Impact (1 sentence):** Concise SDG-aligned impact statement.

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
git clone https://github.com/UtsavYadav1/ecopulse-ai.git
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
