# evaluation/xai/lime_explain.py
from lime.lime_text import LimeTextExplainer

def explain_sklearn(pipeline, text: str, num_features=8):
    explainer = LimeTextExplainer(class_names=["non-harmful","harmful"])
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=num_features)
    return exp.as_list()
