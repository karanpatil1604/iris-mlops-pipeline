import os
import unittest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestIrisModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_path = "data/iris.csv"
        if not os.path.exists(cls.file_path):
            raise unittest.SkipTest("Iris dataset not found. Skipping tests.")
        cls.data = pd.read_csv(cls.file_path)

    def test_no_nulls(self):
        self.assertFalse(
            self.data.isnull().values.any(),
            "Dataset contains null values"
        )

    def test_model_accuracy(self):
        train, test = train_test_split(
            self.data,
            test_size=0.4,
            stratify=self.data['species'],
            random_state=42
        )

        X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y_train = train['species']
        X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y_test = test['species']

        model = DecisionTreeClassifier(max_depth=3, random_state=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.9, f"Model accuracy too low: {accuracy}")

if __name__ == "__main__":
    unittest.main()

