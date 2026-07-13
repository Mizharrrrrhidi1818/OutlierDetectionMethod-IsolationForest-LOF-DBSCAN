# 🚀 Outlier Detection: Space Shuttle Telemetry
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This notebook analyzes the Space Shuttle Telemetry dataset. The goal is to identify fault conditions (anomalies) in the attitude control system sensor readings. 

- **Dataset:** `train_shuttle.csv` (43,499 instances, 9 features)
- **Target:** Class 1 is "Normal". Classes 2–7 are "Outliers".
- **Outlier Ratio:** ~21.6%

## 🛠️ Methods Applied
1. **Isolation Forest:** Tree-based method. Best for high-dimensional global anomalies.
2. **Local Outlier Factor (LOF):** Density-based method. Good for local deviations.
3. **DBSCAN:** Clustering-based method. Flags low-density points as noise.

## 📊 Expected Results
We will evaluate each method using:
- Accuracy, Precision, Recall, and F1-Score
- Confusion Matrices
- PCA Visualizations (2D scatter plots)

## 📂 Notebook Structure
1. **Data Loading & Preprocessing:** Scaling and binary label creation.
2. **Model Training:** Applying the three outlier detection algorithms.
3. **Evaluation:** Metrics and visual comparison.
4. **Conclusion:** Summary of the best-performing method.
