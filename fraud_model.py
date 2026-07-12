
# ==========================================
# CREDIT CARD FRAUD DETECTION
# PART 1 - LOAD DATASET
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

print("="*60)
print("Loading Dataset...")
print("="*60)

train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest_small.csv")

print("Train Shape :", train.shape)
print("Test Shape  :", test.shape)

# Remove unwanted column
if "Unnamed: 0" in train.columns:
    train.drop("Unnamed: 0", axis=1, inplace=True)

if "Unnamed: 0" in test.columns:
    test.drop("Unnamed: 0", axis=1, inplace=True)

# Remove missing values
train.dropna(inplace=True)
test.dropna(inplace=True)

# Remove duplicates
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

print("\nDataset Loaded Successfully")

print(train.head())

# ==========================================
# PART 2 - PREPROCESSING
# ==========================================

from sklearn.preprocessing import LabelEncoder
import joblib

print("\nEncoding Data...")

# Create Hour column
train["trans_date_trans_time"] = pd.to_datetime(train["trans_date_trans_time"])
test["trans_date_trans_time"] = pd.to_datetime(test["trans_date_trans_time"])

train["Hour"] = train["trans_date_trans_time"].dt.hour
test["Hour"] = test["trans_date_trans_time"].dt.hour

# Drop unnecessary columns
drop_cols = [
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "dob"
]

for col in drop_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)

    if col in test.columns:
        test.drop(col, axis=1, inplace=True)

encoders = {}

cat_cols = train.select_dtypes(include="object").columns

for col in cat_cols:

    le = LabelEncoder()

    all_values = pd.concat([train[col], test[col]]).astype(str)

    le.fit(all_values)

    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

    encoders[col] = le

joblib.dump(encoders, "label_encoders.pkl")

print("Encoding Completed")

X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# PART 3 - TRAIN MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest Completed")

print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

print("Decision Tree Completed")

rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nRandom Forest Accuracy :", rf_accuracy)
print("Decision Tree Accuracy :", dt_accuracy)

if rf_accuracy >= dt_accuracy:
    best_model = rf_model
else:
    best_model = dt_model

joblib.dump(best_model, "fraud_model.pkl")

print("\nModel Saved Successfully")

# ==========================================================
# PART 4 - MODEL EVALUATION
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import cross_val_score
from scipy.stats import f_oneway

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*60)
print("PART 4 : MODEL EVALUATION")
print("="*60)

# -----------------------------
# METRICS
# -----------------------------

accuracy = accuracy_score(y_test, rf_pred)
precision = precision_score(y_test, rf_pred)
recall = recall_score(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

print("\nClassification Report\n")
print(classification_report(y_test, rf_pred))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# ROC CURVE
# -----------------------------

rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

auc = roc_auc_score(y_test, rf_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="AUC = %.4f" % auc)
plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

print("\nROC AUC :", auc)

# -----------------------------
# CROSS VALIDATION
# -----------------------------

cv = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores")
print(cv)

print("Average Accuracy :", cv.mean())

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.show()

# -----------------------------
# F TEST
# -----------------------------

fraud_amt = train[train["is_fraud"]==1]["amt"]
genuine_amt = train[train["is_fraud"]==0]["amt"]

F, P = f_oneway(fraud_amt, genuine_amt)

print("\nF-Test")
print("F Value :", F)
print("P Value :", P)

if P < 0.05:
    print("Statistically Significant")
else:
    print("Not Statistically Significant")

# -----------------------------
# FRAUD VS GENUINE
# -----------------------------

plt.figure(figsize=(6,5))

sns.countplot(x=train["is_fraud"])

# ==========================================
# CREDIT CARD FRAUD DETECTION
# PART 1 - LOAD DATASET
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

print("="*60)
print("Loading Dataset...")
print("="*60)

train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

print("Train Shape :", train.shape)
print("Test Shape  :", test.shape)

# Remove unwanted column
if "Unnamed: 0" in train.columns:
    train.drop("Unnamed: 0", axis=1, inplace=True)

if "Unnamed: 0" in test.columns:
    test.drop("Unnamed: 0", axis=1, inplace=True)

# Remove missing values
train.dropna(inplace=True)
test.dropna(inplace=True)

# Remove duplicates
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

print("\nDataset Loaded Successfully")

print(train.head())

# ==========================================
# PART 2 - PREPROCESSING
# ==========================================

from sklearn.preprocessing import LabelEncoder
import joblib

print("\nEncoding Data...")

# Create Hour column
train["trans_date_trans_time"] = pd.to_datetime(train["trans_date_trans_time"])
test["trans_date_trans_time"] = pd.to_datetime(test["trans_date_trans_time"])

train["Hour"] = train["trans_date_trans_time"].dt.hour
test["Hour"] = test["trans_date_trans_time"].dt.hour

# Drop unnecessary columns
drop_cols = [
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "dob"
]

for col in drop_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)

    if col in test.columns:
        test.drop(col, axis=1, inplace=True)

encoders = {}

cat_cols = train.select_dtypes(include="object").columns

for col in cat_cols:

    le = LabelEncoder()

    all_values = pd.concat([train[col], test[col]]).astype(str)

    le.fit(all_values)

    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

    encoders[col] = le

joblib.dump(encoders, "label_encoders.pkl")

print("Encoding Completed")

X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# PART 3 - TRAIN MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest Completed")

print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

print("Decision Tree Completed")

rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nRandom Forest Accuracy :", rf_accuracy)
print("Decision Tree Accuracy :", dt_accuracy)

if rf_accuracy >= dt_accuracy:
    best_model = rf_model
else:
    best_model = dt_model

joblib.dump(best_model, "fraud_model.pkl")

print("\nModel Saved Successfully")

# ==========================================================
# PART 4 - MODEL EVALUATION
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import cross_val_score
from scipy.stats import f_oneway

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*60)
print("PART 4 : MODEL EVALUATION")
print("="*60)

# -----------------------------
# METRICS
# -----------------------------

accuracy = accuracy_score(y_test, rf_pred)
precision = precision_score(y_test, rf_pred)
recall = recall_score(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

print("\nClassification Report\n")
print(classification_report(y_test, rf_pred))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# ROC CURVE
# -----------------------------

rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

auc = roc_auc_score(y_test, rf_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="AUC = %.4f" % auc)
plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

print("\nROC AUC :", auc)

# -----------------------------
# CROSS VALIDATION
# -----------------------------

cv = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores")
print(cv)

print("Average Accuracy :", cv.mean())

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.show()

# -----------------------------
# F TEST
# -----------------------------

fraud_amt = train[train["is_fraud"]==1]["amt"]
genuine_amt = train[train["is_fraud"]==0]["amt"]

F, P = f_oneway(fraud_amt, genuine_amt)

print("\nF-Test")
print("F Value :", F)
print("P Value :", P)

if P < 0.05:
    print("Statistically Significant")
else:
    print("Not Statistically Significant")

# -----------------------------
# FRAUD VS GENUINE
# -----------------------------

plt.figure(figsize=(6,5))

sns.countplot(x=train["is_fraud"])

