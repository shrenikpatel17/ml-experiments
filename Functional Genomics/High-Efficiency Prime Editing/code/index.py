import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import resample  
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette('muted')

# Load data
print("📊 Loading and processing data...")
file_path = "Supplemental_Table_2.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# Filter and clean
df = df[df['included_in_analysis'] == 'yes'].copy()

def gc_content(seq):
    if pd.isna(seq) or len(seq) == 0:
        return 0
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq)

df['gc_content'] = df['Protospacer'].apply(gc_content)
df = df.dropna(subset=['Edit_length', 'epegRNA_type', 'gc_content', 'Z_score_d28_avg'])
df['dropout'] = df['Z_score_d28_avg'] < -2

n_total = len(df)
n_dropout = df['dropout'].sum()
print(f"Total samples: {n_total}")
print(f"Dropout samples: {n_dropout} ({n_dropout/n_total:.1%})")
print(f"Non-dropout samples: {n_total-n_dropout} ({1-n_dropout/n_total:.1%})")

# Encode and prepare features
df = pd.get_dummies(df, columns=['epegRNA_type'], drop_first=True)

feature_cols = ['Edit_length', 'gc_content'] + [col for col in df.columns if col.startswith('epegRNA_type_')]
X = df[feature_cols]
y = df['dropout']

# Balance the classes
print("\n⚖️ Balancing dataset...")
df_majority = df[df.dropout == False]
df_minority = df[df.dropout == True]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X_bal = df_balanced[feature_cols]
y_bal = df_balanced['dropout']

# Train-test split and model training
print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_bal, y_bal, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] 

print("\n📊 Classification Report (Balanced Training):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"])
print("\n🧩 Confusion Matrix:")
print(cm_df)

# Try alternative models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

print("🚀 Trying Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]
print("Logistic Regression Report:")
print(classification_report(y_test, log_pred))
print("AUC:", roc_auc_score(y_test, log_prob))

print("🚀 Trying Gradient Boosting...")
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_prob = gb_model.predict_proba(X_test)[:, 1]
print("Gradient Boosting Report:")
print(classification_report(y_test, gb_pred))
print("AUC:", roc_auc_score(y_test, gb_prob))

try:
    from xgboost import XGBClassifier
    print("🚀 Trying XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    print("XGBoost Report:")
    print(classification_report(y_test, xgb_pred))
    print("AUC:", roc_auc_score(y_test, xgb_prob))
except ImportError:
    print("⚠️ XGBoost not installed. Skipping...")


# Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [feature_cols[i] for i in indices]
importance_values = [importances[i] for i in indices]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(indices)), importance_values, align='center', color=colors[0], alpha=0.8)
plt.yticks(range(len(indices)), [feature_names[i] for i in range(len(indices))])
plt.xlabel('Relative Importance')
plt.title('Feature Importance for Dropout Prediction', fontsize=14, fontweight='bold')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
             ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Enhanced confusion matrix plot
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Predicted\nNo Dropout", "Predicted\nDropout"], 
            yticklabels=["Actual\nNo Dropout", "Actual\nDropout"])
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color=colors[1], lw=2, label=f'ROC curve (AUC = {roc_auc:.6f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Class Distribution Before/After Balancing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before balancing
counts_before = df['dropout'].value_counts()
ax1.pie(counts_before, labels=['No Dropout', 'Dropout'], autopct='%1.1f%%', 
        colors=[colors[0], colors[1]], explode=(0, 0.1), startangle=90, shadow=True)
ax1.set_title('Original Class Distribution', fontsize=14, fontweight='bold')

# After balancing
counts_after = df_balanced['dropout'].value_counts()
ax2.pie(counts_after, labels=['No Dropout', 'Dropout'], autopct='%1.1f%%',
        colors=[colors[0], colors[1]], explode=(0, 0.1), startangle=90, shadow=True)
ax2.set_title('Balanced Class Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Analysis complete! All figures saved as PNG files.")

from sklearn.metrics import precision_score, recall_score, f1_score

metrics = []

# Random Forest
metrics.append({
    'Model': 'Random Forest',
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_prob)
})

# Logistic Regression
metrics.append({
    'Model': 'Logistic Regression',
    'Precision': precision_score(y_test, log_pred),
    'Recall': recall_score(y_test, log_pred),
    'F1': f1_score(y_test, log_pred),
    'AUC': roc_auc_score(y_test, log_prob)
})

# Gradient Boosting
metrics.append({
    'Model': 'Gradient Boosting',
    'Precision': precision_score(y_test, gb_pred),
    'Recall': recall_score(y_test, gb_pred),
    'F1': f1_score(y_test, gb_pred),
    'AUC': roc_auc_score(y_test, gb_prob)
})

try:
    metrics.append({
        'Model': 'XGBoost',
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1': f1_score(y_test, xgb_pred),
        'AUC': roc_auc_score(y_test, xgb_prob)
    })
except NameError:
    pass

metric_df = pd.DataFrame(metrics)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
metric_df.set_index('Model')[['Precision', 'Recall', 'F1', 'AUC']].plot(kind='bar', ax=ax)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
