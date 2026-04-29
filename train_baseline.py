import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

def main():
    # ── 1. Load grouped preprocessed data ──────────────────────────────
    print("Loading sf_crime_data_grouped.csv...")
    df = pd.read_csv('sf_crime_data_grouped.csv')
    print(f"Dataset shape: {df.shape}")

    # ── 2. Define feature columns ──────────────────────────────────────
    numeric_features = ['X', 'Y', 'Hour', 'Month', 'Year']
    district_cols = [c for c in df.columns if c.startswith('PdDistrict_')]
    day_cols = [c for c in df.columns if c.startswith('DayOfWeek_')]
    feature_cols = numeric_features + district_cols + day_cols
    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df['Category_Grouped_Encoded'].values

    # ── 3. Load label encoder ──────────────────────────────────────────
    le = joblib.load('category_grouped_label_encoder.pkl')
    print(f"Target classes ({len(le.classes_)}): {list(le.classes_)}")

    # ── 4. 80/20 stratified split ──────────────────────────────────────
    print("Splitting data 80/20 (stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} | Validation: {X_val.shape[0]:,}")

    # ── 5. Train Random Forest (max_depth=20) ──────────────────────────
    print("\nTraining RandomForestClassifier (max_depth=20, n_jobs=-1)...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f} seconds.")

    # ── 6. Evaluate ────────────────────────────────────────────────────
    print("Evaluating on validation set...")
    y_pred = rf.predict(X_val)
    accuracy = np.mean(y_pred == y_val)
    print(f"\n{'='*60}")
    print(f"OVERALL VALIDATION ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # ── 7. Full classification report ──────────────────────────────────
    report = classification_report(
        y_val, y_pred,
        target_names=le.classes_,
        zero_division=0
    )
    print("\n=== CLASSIFICATION REPORT (6 Groups) ===")
    print(report)

    # ── 8. Confusion matrix ────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=',d', cmap='Blues',
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title('Confusion Matrix — 6 Consolidated Crime Groups (max_depth=20)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_6groups.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix_6groups.png'")

    # ── 9. Save results to file ────────────────────────────────────────
    with open('baseline_v2_results.txt', 'w') as f:
        f.write(f"Overall Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("=== CLASSIFICATION REPORT (6 Groups) ===\n")
        f.write(report + "\n")
        f.write("=== CONFUSION MATRIX ===\n")
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        f.write(cm_df.to_string() + "\n")
    print("Full results saved to 'baseline_v2_results.txt'")

    # ── 10. Save model ─────────────────────────────────────────────────
    joblib.dump(rf, 'rf_grouped_model.pkl')
    print("Model saved to 'rf_grouped_model.pkl'")
    print("\nPhase 4 complete!")


if __name__ == '__main__':
    main()
