import joblib
import torch
import os
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

PROJECT_ROOT = os.path.abspath(os.getcwd())

MODEL_PATHS = {
    "svm.pkl": "models/traditional/svm.pkl",
    "logistic.pkl": "models/traditional/logistic.pkl",
    "lstm_model.pt": "models/deep_learning/lstm_model.pt",
    "gru_model.pt": "models/deep_learning/gru_model.pt",
    "tfidf_dl.pkl": "models/deep_learning/tfidf_dl.pkl",
    "xlmr_model.pt": "models/transformer/xlmr_model.pt",
}


def check_file(path):
    full = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(full):
        print(f"✔ FOUND: {path}")
        return full
    else:
        print(f"✘ MISSING: {path}")
        return None


def test_sklearn_model(path):
    try:
        model = joblib.load(path)
        print("   ✔ Loaded successfully")
        print(f"   ✔ Model type: {type(model)}")
    except Exception as e:
        print("   ✘ Error loading:", e)


def test_tfidf_dl(path):
    try:
        tfidf = joblib.load(path)
        print("   ✔ TF-IDF Loaded, features =", len(tfidf.get_feature_names_out()))
    except Exception as e:
        print("   ✘ Error loading TF-IDF:", e)


def test_torch_model(path):
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            print("   ✔ Torch model appears trained")
        else:
            print("   ⚠ Torch file exists but structure unknown")
    except Exception as e:
        print("   ✘ Error loading torch model:", e)


def test_xlmr(path):
    try:
        model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        print("   ✔ XLM-R model weights loaded")
    except Exception as e:
        print("   ✘ Error loading XLM-R:", e)


print("\n==============================")
print("🔍 MODEL HEALTH CHECK STARTED")
print("==============================\n")

for name, relative_path in MODEL_PATHS.items():
    print(f"\nChecking: {name}")
    full_path = check_file(relative_path)
    if not full_path:
        continue

    if name.endswith(".pkl") and name != "tfidf_dl.pkl":
        test_sklearn_model(full_path)

    elif name == "tfidf_dl.pkl":
        test_tfidf_dl(full_path)

    elif name in ["lstm_model.pt", "gru_model.pt"]:
        test_torch_model(full_path)

    elif name == "xlmr_model.pt":
        test_xlmr(full_path)

print("\n==============================")
print("🏁 HEALTH CHECK COMPLETE")
print("==============================")
