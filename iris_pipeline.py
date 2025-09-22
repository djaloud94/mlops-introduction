# iris_pipeline.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def load_and_explore():
    """Load the Iris dataset and return as a DataFrame"""
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("First 5 rows of the dataset:")
    print(df.head())
    return iris, df

def train_model(iris):
    """Train a simple Logistic Regression classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained with accuracy: {acc:.2f}")
    return model, acc

if __name__ == "__main__":
    iris, df = load_and_explore()
    model, acc = train_model(iris)
