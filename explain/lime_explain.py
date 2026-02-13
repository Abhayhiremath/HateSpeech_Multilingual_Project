import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/transformer"   # <-- change if needed

model = XLMRobertaForSequenceClassification.from_pretrained(model_path).to(DEVICE)
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)

# Prediction function that LIME needs
def predict_proba(texts):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

# LIME explain function
def explain_text(text):
    explainer = LimeTextExplainer(class_names=["Not Hate", "Hate"])

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,         # Show top 10 influential words
        labels=(1,)              # Explain class 1 (hate speech)
    )

    print("\n=============== LIME EXPLANATION ===============")
    print("Input text:", text, "\n")
    print("Prediction probabilities:", predict_proba([text])[0])
    print("\nHighlighted words towards 'HATE' class:\n")
    print(exp.as_list(label=1))

    # Save HTML visualization
    html_path = "lime_output.html"
    exp.save_to_file(html_path)
    print(f"\nHTML visualization saved to: {html_path}")

    return exp
