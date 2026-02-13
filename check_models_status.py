import os

# List of languages you trained
LANGUAGES = ["arabic", "spanish"]   # add more like: "hindi", "kannada", etc.

# Models expected per language
EXPECTED_MODELS = [
    "svm.pkl",
    "logistic.pkl",
    "lstm_best.pt",
    "gru_best.pt",
    "xlmr_best.pt",
]

def check_models():
    print("\n==============================")
    print("   MODEL TRAINING STATUS")
    print("==============================\n")

    for lang in LANGUAGES:
        folder = f"models/{lang}_models"
        print(f"\n📌 Checking: {folder}\n")

        if not os.path.exists(folder):
            print(f"❌ Folder not found → {folder}\n")
            continue

        trained = []
        pending = []

        for model in EXPECTED_MODELS:
            path = os.path.join(folder, model)
            if os.path.exists(path):
                trained.append(model)
            else:
                pending.append(model)

        print("✔ Trained Models:")
        for m in trained:
            print(f"   ✅ {m}")

        print("\n⏳ Pending Models:")
        for m in pending:
            print(f"   ❌ {m}")

        print("\n----------------------------------")

if __name__ == "__main__":
    check_models()
