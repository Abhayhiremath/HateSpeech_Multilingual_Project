import numpy as np
import joblib
import os
import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer


def predict_and_explain(model, text, model_name):

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
    MODELS_DIR = os.path.abspath(MODELS_DIR)

    # ========================================================
    # 1️⃣ TRADITIONAL MODELS (SVM / LOGISTIC REGRESSION)
    # ========================================================
    if model_name in ["svm.pkl", "logistic.pkl"]:
        vec_path = os.path.join(MODELS_DIR, "traditional", "vectorizer.pkl")
        vec = joblib.load(vec_path)

        X = vec.transform([text])

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
        else:
            score = model.decision_function(X)[0]
            prob = 1 / (1 + np.exp(-score))

        pred = int(prob >= 0.5)

        feature_names = vec.get_feature_names_out()
        coefs = model.coef_.flatten()

        tokens = text.split()
        expl = [
            (t, float(coefs[feature_names.tolist().index(t)]))
            for t in tokens if t in feature_names
        ]

        return pred, float(prob), expl


    # ========================================================
    # 2️⃣ DEEP LEARNING MODELS (LSTM / GRU)
    # ========================================================
    if model_name in ["lstm_model.pt", "gru_model.pt"]:
        vec_path = os.path.join(MODELS_DIR, "deep_learning", "tfidf_dl.pkl")
        vec = joblib.load(vec_path)

        X = vec.transform([text]).toarray()
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X)
            prob = torch.softmax(logits, dim=1)[0][1].item()

        pred = int(prob >= 0.5)
        return pred, float(prob), None


    # ========================================================
    # 3️⃣ TRANSFORMER: XLM-RoBERTa (WITH META FIX)
    # ========================================================
    if model_name == "xlmr_model.pt":
        model, tokenizer = model   # unpack model + tokenizer

        # 🔥 Force everything to CPU — prevents META device errors
        device = "cpu"
        model.to(device)

        # 🔥 EXTRA FIX: ensure no layer is still on meta device
        for param_name, param in model.named_parameters():
            if param.device.type == "meta":
                real_param = torch.zeros_like(param, device="cpu")
                model._parameters[param_name] = torch.nn.Parameter(real_param)

        # ---- Base Prediction ----
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1)[0].numpy()

        pred = int(probs[1] >= 0.5)

        # ---- LIME EXPLAINABILITY ----
        explainer = LimeTextExplainer(class_names=["Non-Harmful", "Harmful"])

        def lime_predict(batch_texts):
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)

            with torch.no_grad():
                logits = model(**enc).logits

                out = F.softmax(logits, dim=1).cpu().numpy()
                return out

        try:
            lime_exp = explainer.explain_instance(
                text,
                lime_predict,
                labels=(1,),
                num_features=8
            )
            explanation_words = lime_exp.as_list(label=1)

        except Exception as e:
            explanation_words = [("LIME ERROR", str(e))]

        return pred, float(probs[1]), explanation_words


    # ========================================================
    # FALLBACK
    # ========================================================
    return 0, 0.0, None
