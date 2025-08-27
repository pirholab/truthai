import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from infer import predict

df = pd.read_csv("data/processed/test.csv").sample(1000, random_state=1)
y_true = df.label.tolist()
y_pred = []
for t in df.text.tolist():
    y_pred.append(1 if predict(t)["label"]=="REAL" else 0)

print(classification_report(y_true, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
try:
    # Optional: average confidence for AUC proxy
    import numpy as np
    print("Mean confidence:", float(np.mean([predict(t)["confidence"] for t in df.text])))
except:
    pass
