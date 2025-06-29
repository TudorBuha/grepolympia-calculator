import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import product

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

def train_model(df):
    X = df[["Concentration", "Intuition", "Accuracy"]]
    y = df["Score"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def generate_distributions(total_points):
    max_attribute = total_points
    distributions = [
        (c, i, a) for c in range(max_attribute + 1)
                  for i in range(max_attribute + 1 - c)
                  for a in [total_points - c - i]
                  if a >= 0
    ]
    return distributions

def predict_best_distribution(model, total_points):
    candidates = generate_distributions(total_points)
    X_pred = pd.DataFrame(candidates, columns=["Concentration", "Intuition", "Accuracy"])
    scores = model.predict(X_pred)
    best_index = np.argmax(scores)
    best_dist = X_pred.iloc[best_index].to_dict()
    best_score = scores[best_index]
    return best_dist, round(float(best_score), 2)
