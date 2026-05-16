# 🏏 IPL Match Win Probability Predictor

> **Predict the next big win!** Machine learning-powered win probability predictions for IPL second-innings matches.

---

## 🎯 Overview

This project provides a complete machine learning pipeline to predict win probabilities for Indian Premier League (IPL) matches during second innings. The system analyzes ball-by-ball cricket data and provides real-time probability predictions for both teams, enabling data-driven match analysis.

**Key Capability:** Simulate second-innings match scenarios and get instant win probability forecasts!

---

## ✨ Features

- 🤖 **Machine Learning Model** - Logistic Regression trained on historical IPL data
- 📊 **Interactive Dashboard** - Beautiful Streamlit web interface for predictions
- ⚡ **Real-Time Analysis** - Ball-by-ball probability updates
- 📈 **Reproducible Pipeline** - Easily retrain the model with new data
- 📦 **Pre-trained Model** - Ready-to-use artifact included

---

## 🚀 Quick Start

### Run the Prediction App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Train the Model

```bash
python train_model.py
```

---

## 📥 Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Setup

1. **Clone the project**
   ```bash
   git clone https://github.com/RajHarsh03/IPL_Win_Probability_Predictor
   cd IPL_Win_Probability_Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **That's it!** You're ready to use the app or train the model.

---

## 💡 Usage

### Using the Web App

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Adjust match parameters:
   - Current runs
   - Current wickets
   - Balls faced
   - Target runs
   - Team selection

3. View real-time win probability for both teams!

### Retraining the Model

To train a fresh model with your own data:

```bash
python train_model.py
```

**What it does:**
- Processes raw IPL match data from `Dataset/`
- Engineers ball-by-ball second-innings features
- Trains a Logistic Regression classifier
- Saves the trained model and metrics to `artifacts/`

---

## 📁 Project Structure

```
IPL_Win_Probability_Predictor/
├── app.py                                    # Streamlit web application
├── train_model.py                           # Model training pipeline
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── Procfile                                 # Deployment configuration
├── artifacts/
│   ├── ipl_win_probability_model.joblib    # Trained ML model
│   └── ipl_win_probability_model.metrics.json
├── Dataset/
│   ├── matches.csv                          # IPL match metadata
│   └── deliveries.csv                       # Ball-by-ball delivery data
└── IPL_Win_Probability_Predictor_Solution.ipynb  # Jupyter notebook
```

---

## 🧠 Model Details

- **Algorithm:** Logistic Regression
- **Input Features:** Ball-by-ball cricket statistics (runs, wickets, required rate, etc.)
- **Target:** Win probability for the batting team (second innings)
- **Output:** Probability score (0-1) for both teams

**Model Artifacts:**
- Pre-trained model saved as `joblib` file for fast loading
- Metric evaluations stored in JSON format

---

## 📊 Dataset

The project uses IPL cricket data:

| File | Description |
|------|-------------|
| `matches.csv` | High-level match information (teams, venues, winners, etc.) |
| `deliveries.csv` | Ball-by-ball data (batsman, bowler, runs, wickets, etc.) |

**Data Source:** Cleaned IPL dataset from public cricket databases

---

## 🛠️ Technologies Used

- **Python 3** - Core programming language
- **Streamlit** - Interactive web framework
- **Scikit-learn** - Machine learning models
- **Pandas & NumPy** - Data processing
- **Joblib** - Model serialization

---

## 📝 Notes

- The model is specifically trained for **second-innings predictions**
- Predictions improve with more historical data
- Real-time accuracy depends on input feature quality

---

*Happy predictions! 🎉*
