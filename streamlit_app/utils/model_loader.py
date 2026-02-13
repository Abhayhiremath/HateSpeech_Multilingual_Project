# streamlit_app/utils/model_loader.py

import os
import joblib
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# Project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATHS = {
    "svm.pkl": os.path.join(PROJECT_ROOT, "models/traditional/svm.pkl"),
    "logistic.pkl": os.path.join(PROJECT_ROOT, "models/traditional/logistic.pkl"),

    "lstm_model.pt": os.path.join(PROJECT_ROOT, "models/deep_learning/lstm_model.pt"),
    "gru_model.pt": os.path.join(PROJECT_ROOT, "models/deep_learning/gru_model.pt"),
    "tfidf_dl.pkl": os.path.join(PROJECT_ROOT, "models/deep_learning/tfidf_dl.pkl"),

    # IMPORTANT: Load the *directory*, not the .pt file
    "xlmr_model.pt": os.path.join(PROJECT_ROOT, "models/transformer"),
}

def load_any_model(name):
    path = MODEL_PATHS[name]

    # ======================================================================
    # 1️⃣ TRADITIONAL MODELS (SVM / LOGISTIC REGRESSION)
    # ======================================================================
    if name.endswith(".pkl") and ("lstm" not in name and "gru" not in name):
        return joblib.load(path)

    # ======================================================================
    # 2️⃣ LSTM DEEP LEARNING MODEL
    # ======================================================================
    if "lstm" in name:
        from training.train_lstm import LSTMClassifier
        tfidf = joblib.load(MODEL_PATHS["tfidf_dl.pkl"])
        in_dim = len(tfidf.get_feature_names_out())
        model = LSTMClassifier(in_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    # ======================================================================
    # 3️⃣ GRU DEEP LEARNING MODEL
    # ======================================================================
    if "gru" in name:
        from training.train_gru import GRUClassifier
        tfidf = joblib.load(MODEL_PATHS["tfidf_dl.pkl"])
        in_dim = len(tfidf.get_feature_names_out())
        model = GRUClassifier(in_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    # ======================================================================
    # 4️⃣ TRANSFORMER: XLM-RoBERTa (LOAD HF FOLDER — NO META)
    # ======================================================================
    if "xlmr" in name:
        model_dir = path   # this is now the directory, NOT a .pt file

        # Load the HuggingFace model fully
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True
        )
        tokenizer = XLMRobertaTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )

        # EXTRA SAFETY: remove any accidental meta tensors
        for param_name, param in model.named_parameters():
            if param.device.type == "meta":
                model._parameters[param_name] = torch.nn.Parameter(
                    torch.zeros_like(param, device="cpu")
                )

        model.to("cpu")
        model.eval()

        return (model, tokenizer)

    return None
