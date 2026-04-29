import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("PHASE 7: Binary Preprocessing")
    print("=" * 60)

    # ── 1. Load binary raw data ────────────────────────────────────────
    print("\nLoading sf_crime_binary_raw.csv...")
    df = pd.read_csv('sf_crime_binary_raw.csv')
    print(f"Shape: {df.shape}")
    print(f"Class balance:\n{df['Is_Crime'].value_counts()}")

    # ── 2. Feature Engineering ─────────────────────────────────────────
    print("\nEngineering features...")

    # Cyclical encoding
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Is_Weekend
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)

    # One-Hot: PdDistrict
    district_dummies = pd.get_dummies(df['PdDistrict'], prefix='PdDistrict')
    df = pd.concat([df, district_dummies], axis=1)

    # One-Hot: DayOfWeek
    day_dummies = pd.get_dummies(df['DayOfWeek'], prefix='DayOfWeek')
    df = pd.concat([df, day_dummies], axis=1)

    # ── 3. Save ────────────────────────────────────────────────────────
    output = 'sf_crime_binary_preprocessed.csv'
    print(f"Saving to '{output}'...")
    df.to_csv(output, index=False)
    print(f"Final shape: {df.shape}")
    print("Binary preprocessing complete!")


if __name__ == '__main__':
    main()
