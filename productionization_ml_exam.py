import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

plt.rcParams['figure.figsize'] = (6, 4)
np.random.seed(42)

app = FastAPI()

# 1. Load the BankLoan training dataset.
df = pd.read_csv('BANK LOAN.csv')
print(df.head())
test_df = pd.read_csv('BANK LOAN_TEST.csv')
print(test_df.head())

#convert age into category and create dummies
df['AGE'] = df['AGE'].astype('category')
df = pd.get_dummies(df, columns=['AGE'], drop_first=True)
test_df['AGE'] = test_df['AGE'].astype('category')
test_df = pd.get_dummies(test_df, columns=['AGE'], drop_first=True)
print(df.head())
print(test_df.head())

# 2. Use Defaulter as the dependent (target) variable.
y_train = df['DEFAULTER']
y_test = test_df['DEFAULTER']

# 3. Use all variables except SN as independent variables.
X_train = df.drop(columns=['SN', 'DEFAULTER'])
X_test = test_df.drop(columns=['SN', 'DEFAULTER'])

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 4. Develop a Random Forest classification model using 500 trees.
rf_model = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print('OOB Score (baseline RF):', round(rf_model.oob_score_, 3))

# 5. Load the BankLoan_Test dataset and estimate the probability of default.
y_pred = rf_model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Estimate the probability of default
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print("Predicted probabilities of default:\n", y_pred_proba)

# 6. Display the predicted probability in the Excel worksheet.
# 7. Create a FastAPI endpoint (/predict) to return the predicted probability of default.

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.get("/predict")
def campaign_analysis():

    result_df = test_df.drop(columns=['DEFAULTER', 'AGE_2', 'AGE_3'])
    result_df['predicted'] = y_pred_proba
    result_df['predicted_rounded'] = np.round(y_pred_proba, 0)
    # The prediction is calculated once, when the script is run, and returned here.
    return result_df.to_dict(orient='records')