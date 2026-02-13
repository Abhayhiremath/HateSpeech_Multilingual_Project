# training/train_kannada_nb.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

TRAIN = r"D:\HateSpeech_Multilingual_Project\data\kannada\train.csv"
VAL   = r"D:\HateSpeech_Multilingual_Project\data\kannada\val.csv"
OUT   = r"D:\HateSpeech_Multilingual_Project\models\kannada"
os.makedirs(OUT, exist_ok=True)

def run():
    tr = pd.read_csv(TRAIN)
    va = pd.read_csv(VAL)

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    Xtr = vec.fit_transform(tr["text"])
    Xva = vec.transform(va["text"])

    clf = MultinomialNB()
    clf.fit(Xtr, tr["label"])

    preds = clf.predict(Xva)
    print("\n[Kannada NB] Validation Report:\n")
    print(classification_report(va["label"], preds))

    joblib.dump(vec, os.path.join(OUT, "kannada_nb_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(OUT, "kannada_nb.pkl"))
    print("✔ Kannada NB model saved to", OUT)

if __name__ == "__main__":
    run()
