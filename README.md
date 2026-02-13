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
