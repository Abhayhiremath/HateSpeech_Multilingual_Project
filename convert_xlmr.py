import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaTokenizer
import os

BASE = "models/transformer"
PT_MODEL = os.path.join(BASE, "xlmr_model.pt")

print("Loading your .pt trained model...")
state = torch.load(PT_MODEL, map_location="cpu")

print("Creating new HF config...")
config = XLMRobertaConfig.from_pretrained("xlm-roberta-base", num_labels=2)

print("Loading base XLM-R architecture...")
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    config=config
)

print("Applying your trained weights...")
model.load_state_dict(state)

print("Saving HuggingFace model...")
model.save_pretrained(BASE)

print("Saving tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
tokenizer.save_pretrained(BASE)

print("\n✨ MODEL CONVERSION SUCCESS! Now transformer folder has:")
print(" - pytorch_model.bin")
print(" - config.json")
print(" - tokenizer files")
