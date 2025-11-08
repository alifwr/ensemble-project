# 🚀 Quick Start Guide - UNSW-NB15 Classification

## 📊 Dataset Overview
- **Training samples**: 82,332
- **Testing samples**: 175,341
- **Features**: 44 (numerical + categorical)
- **Classes**: Binary (Normal/Attack) + 10 multi-class attack types
- **Data quality**: Excellent (no missing values)

## 🎯 Recommended Classification Methods (Priority Order)

### 1. 🥇 XGBoost/LightGBM (HIGHEST PRIORITY)
**Why**: State-of-the-art performance, handles imbalanced data excellently

```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Quick implementation
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,  # Adjust for class imbalance
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

### 2. 🥈 Random Forest (HIGH PRIORITY)
**Why**: Robust, easy to implement, no feature scaling needed

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',  # Handle imbalanced data
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. 🥉 Ensemble Methods (PRODUCTION)
**Why**: Combines strengths of multiple models

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Create ensemble
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)
lgb = LGBMClassifier(n_estimators=100, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lgb', lgb)],
    voting='soft'
)
ensemble.fit(X_train, y_train)
```

## 🔧 Essential Preprocessing Steps

### 1. Load Data
```python
import pandas as pd
import numpy as np

# Load datasets
df_train = pd.read_csv('../data/UNSW_NB15_training-set.csv')
df_test = pd.read_csv('../data/UNSW_NB15_testing-set.csv')
```

### 2. Encode Categorical Features
```python
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['proto', 'service', 'state']

le = LabelEncoder()
for col in categorical_cols:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
```

### 3. Prepare Features and Target
```python
# For binary classification
X_train = df_train.drop(['id', 'label', 'attack_cat'], axis=1)
y_train = df_train['label']

X_test = df_test.drop(['id', 'label', 'attack_cat'], axis=1)
y_test = df_test['label']

# For multi-class classification, use 'attack_cat' instead of 'label'
```

### 4. Feature Scaling (Optional - only for SVM/Neural Networks)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 📈 Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")  # MOST IMPORTANT
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## 🎨 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance (for tree-based models)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## ⚡ Complete Workflow Example

```python
# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load data
df_train = pd.read_csv('../data/UNSW_NB15_training-set.csv')
df_test = pd.read_csv('../data/UNSW_NB15_testing-set.csv')

# 3. Encode categorical features
categorical_cols = ['proto', 'service', 'state']
le = LabelEncoder()
for col in categorical_cols:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# 4. Prepare X and y
X_train = df_train.drop(['id', 'label', 'attack_cat'], axis=1)
y_train = df_train['label']
X_test = df_test.drop(['id', 'label', 'attack_cat'], axis=1)
y_test = df_test['label']

# 5. Train model
model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## 📝 Important Notes

### ⚠️ For Security Applications
- **Prioritize RECALL over Precision** (better to have false alarms than miss attacks)
- Use `class_weight='balanced'` to handle class imbalance
- Consider SMOTE for severe imbalance in multi-class

### 🔍 Handling Class Imbalance
```python
# Method 1: Class weights
model = XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))

# Method 2: SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### 🎯 Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## 📊 Expected Performance

| Method | Expected Accuracy | Training Time |
|--------|------------------|---------------|
| Random Forest | 85-90% | 1-2 hours |
| XGBoost | 90-95% | 2-4 hours |
| Ensemble | 92-97% | 4-6 hours |

## 📚 Required Libraries

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn imbalanced-learn
```

## 🎓 Next Steps

1. ✅ Run the complete workflow example
2. ✅ Start with Random Forest as baseline
3. ✅ Implement XGBoost for best performance
4. ✅ Try ensemble methods for production
5. ✅ Tune hyperparameters
6. ✅ Analyze feature importance
7. ✅ Compare all models

---

**Good luck with your classification task! 🚀**
