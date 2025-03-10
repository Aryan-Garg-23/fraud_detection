import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv(r"C:\Users\Mehak\Downloads\creditcard.csv\creditcard.csv") 

print("ok")

X = df.drop("Class", axis=1)
y = df["Class"]


X["Amount"] = (X["Amount"] - X["Amount"].mean()) / X["Amount"].std()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train_res, y_train_res)


joblib.dump(model, "fraud_model.pkl")


y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))