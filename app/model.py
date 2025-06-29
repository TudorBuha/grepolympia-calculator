import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from itertools import product

# Helper to split data
def split_data(df):
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.9 * n)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

def train_and_select_model(df):
    train, val, test = split_data(df)
    X_train = train[["Concentration", "Intuition", "Accuracy"]]
    y_train = train["Score"]
    X_val = val[["Concentration", "Intuition", "Accuracy"]]
    y_val = val["Score"]

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_mse = mean_squared_error(y_val, rf_pred)

    # Polynomial Regression (degree 2)
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    pr = LinearRegression()
    pr.fit(X_train_poly, y_train)
    pr_pred = pr.predict(X_val_poly)
    pr_mse = mean_squared_error(y_val, pr_pred)

    # Select best model
    if pr_mse < rf_mse:
        return ("poly", pr, poly)
    else:
        return ("rf", rf, None)

def generate_distributions(total_points):
    max_attribute = total_points
    distributions = [
        (c, i, a) for c in range(max_attribute + 1)
                  for i in range(max_attribute + 1 - c)
                  for a in [total_points - c - i]
                  if a >= 0
    ]
    return distributions

def predict_best_distribution(model_tuple, total_points, df=None):
    model_type, model, poly = model_tuple
    candidates = generate_distributions(total_points)
    X_pred = pd.DataFrame(candidates, columns=["Concentration", "Intuition", "Accuracy"])
    if model_type == "poly":
        X_pred_poly = poly.transform(X_pred)
        scores = model.predict(X_pred_poly)
    else:
        scores = model.predict(X_pred)
    best_index = np.argmax(scores)
    best_dist = X_pred.iloc[best_index].to_dict()
    best_score = scores[best_index]
    # Extrapolation warning
    extrapolation = False
    if df is not None and total_points > df[["Concentration", "Intuition", "Accuracy"]].sum(axis=1).max():
        extrapolation = True
    return best_dist, round(float(best_score), 2), extrapolation
