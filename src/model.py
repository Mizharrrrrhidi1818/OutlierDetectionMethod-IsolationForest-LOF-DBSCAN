from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_isolation_forest(X, outlier_ratio):
    model = IsolationForest(n_estimators=100, contamination=outlier_ratio, random_state=42, n_jobs=-1)
    preds = model.fit_predict(X)
    return np.where(preds == -1, 1, 0)

def train_lof(X, outlier_ratio):
    model = LocalOutlierFactor(n_neighbors=20, contamination=outlier_ratio, novelty=False)
    preds = model.fit_predict(X)
    return np.where(preds == -1, 1, 0)

def train_dbscan(X):
    # Heuristic parameters for scaled shuttle data
    model = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1)
    preds = model.fit_predict(X)
    return np.where(preds == -1, 1, 0)

def evaluate_model(name, y_true, y_pred):
    return {
        'Method': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_true, y_pred, zero_division=0), 4)
    }
