import numpy as np
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap

def main():
    print("=" * 60)
    print("BINARY CRIME RISK MAP")
    print("Scenario: Saturday at 11:00 PM")
    print("=" * 60)

    # ── 1. Load model and feature list ─────────────────────────────────
    print("\nLoading binary model...")
    xgb = joblib.load('xgb_binary_model.pkl')
    feature_cols = joblib.load('binary_feature_cols.pkl')

    # ── 2. Build coordinate grid over SF ───────────────────────────────
    print("Building coordinate grid...")
    x_range = np.linspace(-122.51, -122.37, 100)   # Longitude
    y_range = np.linspace(37.70, 37.82, 80)         # Latitude
    xx, yy = np.meshgrid(x_range, y_range)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_points = coords.shape[0]
    print(f"Grid: {len(x_range)} × {len(y_range)} = {n_points} points")

    # ── 3. Scenario: Saturday 11:00 PM ─────────────────────────────────
    hour = 23
    month = 6
    year = 2015

    grid_df = pd.DataFrame(index=range(n_points))
    grid_df['X'] = coords[:, 0]
    grid_df['Y'] = coords[:, 1]
    grid_df['Hour'] = hour
    grid_df['Month'] = month
    grid_df['Year'] = year

    # Cyclical features
    grid_df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    grid_df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    grid_df['Month_sin'] = np.sin(2 * np.pi * month / 12)
    grid_df['Month_cos'] = np.cos(2 * np.pi * month / 12)

    # Is_Weekend
    grid_df['Is_Weekend'] = 1

    # One-hot PdDistrict: all zeros
    for col in feature_cols:
        if col.startswith('PdDistrict_') and col not in grid_df.columns:
            grid_df[col] = 0

    # One-hot DayOfWeek: Saturday = 1
    for col in feature_cols:
        if col.startswith('DayOfWeek_') and col not in grid_df.columns:
            grid_df[col] = 0
    if 'DayOfWeek_Saturday' in feature_cols:
        grid_df['DayOfWeek_Saturday'] = 1

    X_grid = grid_df[feature_cols].values

    # ── 4. Predict crime probability ───────────────────────────────────
    print("Predicting crime probability across grid...")
    crime_proba = xgb.predict_proba(X_grid)[:, 1]  # P(Is_Crime=1)

    print(f"Crime probability range: [{crime_proba.min()*100:.1f}%, {crime_proba.max()*100:.1f}%]")
    print(f"Mean crime probability:  {crime_proba.mean()*100:.1f}%")

    # ── 5. Save probability grid ───────────────────────────────────────
    result_df = pd.DataFrame({
        'Longitude': coords[:, 0],
        'Latitude': coords[:, 1],
        'Crime_Probability': crime_proba,
        'Crime_Probability_Pct': (crime_proba * 100).round(2),
    })
    result_df.to_csv('binary_risk_grid_saturday_11pm.csv', index=False)
    print("Grid saved to 'binary_risk_grid_saturday_11pm.csv'")

    # ── 6. Generate Folium Heatmap ─────────────────────────────────────
    print("\nGenerating interactive heatmap...")
    sf_center = [37.76, -122.44]
    m = folium.Map(location=sf_center, zoom_start=13, tiles='CartoDB dark_matter')

    # Heatmap data: [lat, lon, intensity]
    heat_data = []
    for i in range(n_points):
        lat = coords[i, 1]
        lon = coords[i, 0]
        intensity = float(crime_proba[i])
        heat_data.append([lat, lon, intensity])

    HeatMap(
        heat_data,
        radius=12,
        blur=18,
        max_zoom=15,
        gradient={
            '0.0': '#00ff00',
            '0.3': '#adff2f',
            '0.5': '#ffff00',
            '0.7': '#ff8c00',
            '0.85': '#ff4500',
            '1.0': '#ff0000'
        }
    ).add_to(m)

    # Title overlay
    title_html = '''
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:9999; font-size:18px; font-weight:bold; color:white;
                background:rgba(0,0,0,0.75); padding:12px 24px; border-radius:10px;
                font-family:Arial,sans-serif;">
        🟢🟡🔴 Predicted Crime Probability (0–100%) — Saturday 11:00 PM
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    output_map = 'sf_prediction_map.html'
    m.save(output_map)
    print(f"Interactive heatmap saved to '{output_map}'")
    print("\nBinary risk map generation complete!")


if __name__ == '__main__':
    main()
