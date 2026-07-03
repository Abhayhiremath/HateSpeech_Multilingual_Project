# Multilingual Harmful Speech Detection System

A machine learning application that detects harmful speech across **English, Hindi, Kannada, Marathi, Arabic, and French** using NLP and machine learning.

## Features

* Multilingual harmful speech detection
* Real-time text prediction
* Streamlit-based interactive UI
* Text preprocessing and feature extraction
* Binary classification (Harmful / Non-Harmful)

## Tech Stack

* Python
* Streamlit
* scikit-learn
* Pandas
* NumPy
* HTML/CSS (for UI customization)

## How It Works

1. Enter text in any supported language.
2. The text is preprocessed and converted into model features.
3. The trained model predicts whether the text is **Harmful** or **Non-Harmful**.
4. The prediction is displayed instantly through the Streamlit interface.

## Future Improvements

* Add more language support
* Improve accuracy with transformer-based models
* Deploy on Streamlit Cloud or AWS
* Add explainable AI (LIME/SHAP) for prediction insights



# HateSpeech Detection (Binary)

## Quickstart
```bash
# 1) prepare data
python prepare_data.py

# 2) train classical models
python training/train_svm.py --type svm --train data/labeled/train.csv --val data/labeled/val.csv --out models/traditional
python training/train_svm.py --type logreg --train data/labeled/train.csv --val data/labeled/val.csv --out models/traditional

# 3) train DL baselines
python training/train_lstm.py --train data/labeled/train.csv --val data/labeled/val.csv --out models/deep_learning
python training/train_gru.py  --train data/labeled/train.csv --val data/labeled/val.csv --out models/deep_learning

# 4) train XLM-R
python training/train_xlmr.py --train data/labeled/train.csv --val data/labeled/val.csv --out models/transformer --epochs 2

# 5) evaluate a sklearn model
python training/evaluate.py --model models/traditional/svm.pkl --test data/labeled/test.csv --out evaluation/metrics_report.json

# 6) run Streamlit
cd streamlit_app
streamlit run app.py
