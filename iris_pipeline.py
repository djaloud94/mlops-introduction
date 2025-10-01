# iris_pipeline.py
import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y

def explore_data():
    X, y = load_data()
    print("✅ Iris dataset loaded successfully")
    print("Shape:", X.shape)
    print("First 5 rows:\n", X.head())
    print("Target distribution:\n", y.value_counts())

if __name__ == "__main__":
    explore_data()
