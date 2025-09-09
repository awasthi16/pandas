# Telecom Customer Churn Prediction
# Jupyter-friendly script (use with VSCode/Notebook where `# %%` separates cells)
# Dataset expected: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the working directory

# %%
"""
1. Objective
   - Build a classification model to predict customer churn for a telecom dataset.
   - Full ML workflow: EDA -> Preprocessing -> Modeling -> Evaluation -> Save model

2. Libraries used:
   - pandas, numpy, matplotlib, seaborn
   - scikit-learn (preprocessing, model_selection, metrics, ensemble, linear_model)
   - joblib

3. Notes / Hints used in this notebook
   - LabelEncoder for binary features (Yes/No)
   - One-Hot (pd.get_dummies) for multi-class categoricals
   - StandardScaler for scaling numeric features when using Logistic Regression
   - RandomForest feature importance for visualizing key features
   - Save trained model + preprocessing objects using joblib
"""

# %%
# Imports
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
import joblib

# %%
# 1) Load data
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'  # change this if your filename differs
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download and place it in the working directory.")

df = pd.read_csv(DATA_PATH)
print('Dataset shape:', df.shape)

# %%
# Quick peek
print(df.columns.tolist())
print(df.head())

# %%
# 2) Basic EDA
print('\n--- Info ---')
df.info()

print('\n--- Missing values per column ---')
print(df.isnull().sum())

# 'TotalCharges' sometimes loads as object; convert it to numeric (coerce errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('\nMissing after conversion:', df['TotalCharges'].isnull().sum())

# If few missing values in TotalCharges, drop them
print('\nChurn value counts:')
print(df['Churn'].value_counts())

# %%
# Visualize churn distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn distribution')
plt.show()

# %%
# 3) Preprocessing
# Drop customerID as it's an identifier
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Fill or drop NaNs in TotalCharges
if df['TotalCharges'].isnull().sum() > 0:
    # usually only a few rows — we drop them
    df = df.dropna(subset=['TotalCharges'])

# Binary columns with 'Yes'/'No' -> use LabelEncoder
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
print('Binary categorical columns:', binary_cols)

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# For remaining categorical columns (multi-class), use one-hot encoding
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print('Multi-class categorical columns to One-Hot encode:', cat_cols)

# One-hot encode and avoid dummy trap by drop_first=False (we'll scale or regularize models)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print('Shape after encoding:', df.shape)

# %%
# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

# %%
# Optional: Scale numeric features for models like LogisticRegression
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# We'll scale only continuous variables, but numeric includes encodings too; pick columns that are truly continuous
continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

# %%
# 4) Modeling: Try Logistic Regression, RandomForest, GradientBoosting
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    # For logistic, use scaled data. For tree-based, use original (scaling not required)
    if name == 'LogisticRegression':
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probas = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probas)

    results[name] = {'model': model, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}

# %%
# Show results
results_df = pd.DataFrame(results).T.sort_values(by='roc_auc', ascending=False)
print(results_df[['accuracy','precision','recall','f1','roc_auc']])

# %%
# Print classification report for the best model (by ROC AUC)
best_model_name = results_df.index[0]
best_model = results[best_model_name]['model']
print('\nBest model:', best_model_name)

if best_model_name == 'LogisticRegression':
    preds = best_model.predict(X_test_scaled)
else:
    preds = best_model.predict(X_test)

print('\nClassification report:')
print(classification_report(y_test, preds))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
# 5) Feature importance (RandomForest)
# If RandomForest is present, plot feature importances
if 'RandomForest' in results:
    rf = results['RandomForest']['model']
    importances = rf.feature_importances_
    fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:30]

    plt.figure(figsize=(8,10))
    fi.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title('Top 30 Feature Importances (Random Forest)')
    plt.show()

# %%
# 6) Save the best model + preprocessing objects
os.makedirs('models', exist_ok=True)

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')
# Save full pipeline model (we'll save the best_model object)
joblib.dump(best_model, f'models/{best_model_name}.joblib')

print('Saved scaler and model to /models')

# %%
# 7) Optional: Hyperparameter tuning example for Random Forest (quick grid)
# (uncomment to run — will take longer)
"""
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
clf = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
clf.fit(X_train, y_train)
print('Best RF params:', clf.best_params_)
print('Best RF ROC AUC:', clf.best_score_)
joblib.dump(clf.best_estimator_, 'models/rf_best.joblib')
"""

# %%
# 8) How to load model and scaler later for deployment
print('\nExample to load model and scaler:')
print("scaler = joblib.load('models/scaler.joblib')")
print(f"model = joblib.load('models/{best_model_name}.joblib')")

# End of notebook/script
print('\nDone')
