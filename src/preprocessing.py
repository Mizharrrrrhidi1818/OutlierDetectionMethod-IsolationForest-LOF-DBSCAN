import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath: str):
    """Loads shuttle data, creates binary labels, and scales features."""
    df = pd.read_csv(filepath)
    
    # Assuming the last column is the target
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]
    
    X = df[feature_cols].values
    y_true = df[target_col].values
    
    # Binary labels: 0 for Normal (Class 1), 1 for Outlier (Classes 2-7)
    y_binary = np.where(y_true == 1, 0, 1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_binary, df.shape[0], np.sum(y_binary == 1)
