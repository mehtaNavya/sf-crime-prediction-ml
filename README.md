## Crime Prediction Heatmap for San Francisco Using Machine Learning
A machine learning pipeline that predicts crime occurrence and category in San Francisco using 1.75M SFPD historical records, achieving **97.07% binary classification accuracy** with XGBoost and visualizing predicted crime risk as an interactive geospatial heatmap.


## Results
 Model                       Task              Accuracy    AUC 

 XGBoost (Binary)            Crime / No-Crime  97.07%      0.9911 
 Random Forest (6-class)     Crime Category    43.24%      — 
 Weighted XGBoost (6-class)  Crime Category    31.91%      — 

Key finding: Police district is the strongest predictor of crime, accounting for 87% of feature importance. Location matters far more than time.


## Project Structure

```
sf-crime-prediction-ml/
│
├── preprocess.py                        # Data cleaning and feature engineering
├── train_binary.py                      # Binary XGBoost classifier (crime vs no-crime)
├── train_baseline.py                    # Random Forest multi-class baseline
├── generate_risk_map.py                 # Geospatial crime risk heatmap generation
│
├── binary_xgb_results.txt               # Binary model performance metrics
├── baseline_v2_results.txt              # Random Forest results
├── weighted_xgb_results.txt             # Weighted XGBoost results
│
├── confusion_matrix_xgboost.png         # Binary model confusion matrix
├── confusion_matrix_top5.png            # Random Forest confusion matrix
├── confusion_matrix_weighted_xgb.png    # Weighted XGBoost confusion matrix
│
├── sf_prediction_map.html               # Interactive crime risk heatmap (open in browser)
├── .gitignore
└── README.md
```


## Features Used

 Feature               Description 

 X, Y                  GPS longitude and latitude 
 Hour, Month, Year     Temporal features 
 Hour_sin, Hour_cos    Cyclical encoding of hour 
 Month_sin, Month_cos  Cyclical encoding of month 
 Is_Weekend            Binary flag for Saturday/Sunday 
 PdDistrict (OHE)      One-hot encoded police district (10 districts) 
 DayOfWeek (OHE)       One-hot encoded day of week 
 

## Models

### Binary Classification (XGBoost)
Predicts whether a crime will occur at a given location and time.
- 300 estimators, max_depth=10, learning_rate=0.1
- Trained on 1.4M samples, validated on 351,193 samples
- **Accuracy: 97.07% | AUC: 0.9911 | Crime Recall: 1.00**

### Multi-Class Classification
Predicts which of 6 crime categories an incident belongs to.

 Category      Examples                    Weight (Weighted XGB) 

 PROPERTY      Theft, Burglary, Car theft  1.0× 
 PUBLIC ORDER  Vandalism, Trespassing      1.2× 
 VIOLENT       Assault, Robbery            **5.0×** 
 SUBSTANCE     Drug offenses, DUI          1.5× 
 FINANCIAL     Fraud, Forgery              **4.0×** 
 OTHER         Suspicious activity         1.0× 

Weighted XGBoost improved VIOLENT crime recall from **27% → 79%** by applying cost-sensitive learning.

### Geospatial Risk Map
Crime probability predicted across 8,000 grid points covering San Francisco for a Saturday at 11 PM scenario. Peak risk zone: **Mission/SoMa district at 5.91%** predicted probability.


## Dataset

The dataset used is the **San Francisco Police Department (SFPD) Incident Reports** dataset, publicly available at:  
[https://data.sfgov.org](https://data.sfgov.org)


## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-yellow)
![Folium](https://img.shields.io/badge/Folium-0.14-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue)

