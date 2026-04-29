import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

def main():
    print("=" * 60)
    print("PHASE 7: Binary XGBoost — Crime vs. No-Crime")
    print("=" * 60)

    # ── 1. Load preprocessed binary data ───────────────────────────────
    print("\nLoading sf_crime_binary_preprocessed.csv...")
    df = pd.read_csv('sf_crime_binary_preprocessed.csv')
    print(f"Dataset shape: {df.shape}")

    # ── 2. Define features ─────────────────────────────────────────────
    numeric_features = ['X', 'Y', 'Hour', 'Month', 'Year']
    cyclical_features = ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
    binary_features = ['Is_Weekend']
    district_cols = [c for c in df.columns if c.startswith('PdDistrict_')]
    day_cols = [c for c in df.columns if c.startswith('DayOfWeek_')]

    feature_cols = numeric_features + cyclical_features + binary_features + district_cols + day_cols
    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df['Is_Crime'].values
    print(f"Class distribution: Crime={y.sum():,}, No-Crime={(y==0).sum():,}")

    # ── 3. 80/20 stratified split ──────────────────────────────────────
    print("\nSplitting 80/20 (stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} | Validation: {X_val.shape[0]:,}")

    # ── 4. Train binary XGBoost ────────────────────────────────────────
    print("\nTraining XGBClassifier (binary:logistic)...")
    start = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )
    xgb.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f} seconds.")

    # ── 5. Evaluate ────────────────────────────────────────────────────
    y_pred = xgb.predict(X_val)
    y_proba = xgb.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    print(f"\n{'='*60}")
    print(f"  ACCURACY:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-SCORE:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"{'='*60}")

    # ── 6. Classification report ───────────────────────────────────────
    report = classification_report(
        y_val, y_pred,
        target_names=['No-Crime', 'Crime'],
        zero_division=0
    )
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # ── 7. Confusion Matrix ────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=',d', cmap='RdYlGn_r',
        xticklabels=['No-Crime', 'Crime'],
        yticklabels=['No-Crime', 'Crime'],
        annot_kws={'size': 16}
    )
    plt.title('Binary Confusion Matrix — Crime vs. No-Crime', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_binary.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix_binary.png'")

    # ── 8. Feature Importance ──────────────────────────────────────────
    importances = xgb.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    feat_imp.head(15).plot(kind='barh', color='crimson')
    plt.title('Top 15 Feature Importances — Binary XGBoost')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_binary.png', dpi=150)
    plt.close()
    print("Feature importance saved to 'feature_importance_binary.png'")

    # ── 9. Save results ───────────────────────────────────────────────
    with open('binary_xgb_results.txt', 'w') as f:
        f.write(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n\n")
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report + "\n")
        f.write("=== CONFUSION MATRIX ===\n")
        cm_df = pd.DataFrame(cm, index=['No-Crime', 'Crime'], columns=['No-Crime', 'Crime'])
        f.write(cm_df.to_string() + "\n\n")
        f.write("=== TOP 15 FEATURE IMPORTANCES ===\n")
        f.write(feat_imp.head(15).to_string() + "\n")
    print("Results saved to 'binary_xgb_results.txt'")

    # ── 10. Save model & feature list ──────────────────────────────────
    joblib.dump(xgb, 'xgb_binary_model.pkl')
    joblib.dump(feature_cols, 'binary_feature_cols.pkl')
    print("Model saved to 'xgb_binary_model.pkl'")
    print("\nPhase 7: Binary classification complete!")


if __name__ == '__main__':
    main()
