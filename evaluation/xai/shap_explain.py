# evaluation/xai/shap_explain.py
import shap
def shap_explain_sklearn(pipeline, texts):
    explainer = shap.Explainer(pipeline.predict_proba, algorithm="permutation")
    shap_values = explainer(texts)
    return shap_values
