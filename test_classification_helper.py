import unittest

from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)

from classification_helper import (
    assemble,
    feature_importance,
    perform,
    setup_spark,
)


class ClassificationHelperTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = setup_spark("ClassificationHelperTest")
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_assemble_and_perform(self):
        train = self.spark.createDataFrame(
            [
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (1.1, 1.0, 1.0),
            ],
            ["x", "y", "label"],
        )
        test = self.spark.createDataFrame(
            [
                (0.2, 0.0, 0.0),
                (0.9, 1.0, 1.0),
            ],
            ["x", "y", "label"],
        )

        train, test = assemble(train, test, ["x", "y"])

        self.assertIn("features", train.columns)
        self.assertIn("features", test.columns)

        models = {
            "Logistic Regression": LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=20,
            ),
            "Decision Tree": DecisionTreeClassifier(
                featuresCol="features",
                labelCol="label",
                maxDepth=2,
                seed=42,
            ),
        }

        results = perform(
            train,
            test,
            models,
            ["accuracy", "f1"],
            "label",
        )

        self.assertEqual(set(results), set(models))
        for result in results.values():
            self.assertIn("predictions", result)
            self.assertIn("accuracy", result)
            self.assertIn("f1", result)
            self.assertEqual(result["predictions"].count(), 2)

    def test_feature_importance(self):
        train = self.spark.createDataFrame(
            [
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (1.1, 1.0, 1.0),
            ],
            ["x", "y", "label"],
        )

        train, _ = assemble(train, train, ["x", "y"])
        model = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=5,
            maxDepth=2,
            seed=42,
        ).fit(train)

        importance = feature_importance(model, train)

        self.assertEqual(list(importance.columns), ["feature", "importance"])
        self.assertEqual(len(importance), 2)
        self.assertEqual(set(importance["feature"]), {"x", "y"})
        self.assertTrue((importance["importance"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
