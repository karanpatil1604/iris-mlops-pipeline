import os
import pytest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="module")
def iris_data():
    file_path = "data/iris.csv"
    if not os.path.exists(file_path):
        pytest.skip("Iris dataset not found. Skipping tests.")
    data = pd.read_csv(file_path)
    return data

def test_no_nulls(iris_data):
    assert not iris_data.isnull().values.any(), "Dataset contains null values"

def test_model_accuracy(iris_data):
    train, test = train_test_split(iris_data, test_size=0.4, stratify=iris_data['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train['species']
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']

    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9, f"Model accuracy too low: {accuracy}"
