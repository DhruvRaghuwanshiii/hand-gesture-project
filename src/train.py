import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH = "data/gestures.csv"
MODEL_OUT = "models/gesture_classifier.pkl"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing {CSV_PATH}. Collect gestures first using collect_data.py.")

df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["label"]).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("\n=== Model Evaluation ===")
print(classification_report(y_test, preds))

joblib.dump(clf, MODEL_OUT)
print(f"\n[SUCCESS] Model successfully saved to {MODEL_OUT}")