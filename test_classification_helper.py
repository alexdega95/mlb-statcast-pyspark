import unittest
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)

from classification_helper import (
    add_strike_zone,
    assemble,
    check_label_coverage,
    classification_report,
    evaluate_models,
    evaluate_predictions,
    feature_importance,
    perform,
    plot_feature_importance,
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

        results, fit_models = perform(
            train,
            test,
            models,
            ["accuracy", "f1"],
            "label",
        )

        self.assertEqual(set(results), set(models))
        self.assertEqual(set(fit_models), set(models))
        for result in results.values():
            self.assertIn("predictions", result)
            self.assertIn("accuracy", result)
            self.assertIn("f1", result)
            self.assertEqual(result["predictions"].count(), 2)
        test_results = evaluate_models(test, fit_models, ["accuracy", "f1"])
        for name in models:
            self.assertEqual(test_results[name]["f1"], results[name]["f1"])
            results[name]["predictions"].unpersist()
            test_results[name]["predictions"].unpersist()

    def test_evaluate_models_does_not_fit(self):
        predictions = self.spark.createDataFrame(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
            ["target", "prediction"],
        )
        fitted_model = Mock()
        fitted_model.transform.return_value = predictions
        fitted_model.fit.side_effect = AssertionError("Must not refit")

        results = evaluate_models(
            predictions, {"Existing model": fitted_model}, ["accuracy", "f1"],
            label_col="target",
        )

        fitted_model.fit.assert_not_called()
        fitted_model.transform.assert_called_once_with(predictions)
        self.assertEqual(results["Existing model"]["accuracy"], 0.6667)
        self.assertEqual(results["Existing model"]["f1"], 0.6667)
        results["Existing model"]["predictions"].unpersist()
        with self.assertRaisesRegex(ValueError, "not supported"):
            evaluate_predictions(predictions, ["unsupported"], label_col="target")

    def test_classification_report_includes_predicted_only_classes(self):
        predictions = self.spark.createDataFrame(
            [(0.0, 0.0), (0.0, 1.0)], ["target", "prediction"]
        )

        report, macro_f1 = classification_report(predictions, label_col="target")
        rows = {row.label: row for row in report.collect()}

        self.assertEqual(set(rows), {0.0, 1.0})
        self.assertEqual(rows[0.0].support, 2)
        self.assertEqual(rows[0.0].precision, 1.0)
        self.assertEqual(rows[0.0].recall, 0.5)
        self.assertAlmostEqual(rows[0.0].f1, 2 / 3)
        self.assertEqual(rows[1.0].support, 0)
        self.assertEqual(rows[1.0].precision, 0.0)
        self.assertEqual(rows[1.0].recall, 0.0)
        self.assertEqual(rows[1.0].f1, 0.0)
        self.assertAlmostEqual(macro_f1, 1 / 3)

    def test_check_label_coverage_checks_both_splits(self):
        train = self.spark.createDataFrame([(0.0,), (1.0,)], ["target"])
        known = self.spark.createDataFrame([(1.0,)], ["target"])
        unseen = self.spark.createDataFrame([(2.0,)], ["target"])

        check_label_coverage(train, known, known, "target")
        with self.assertRaisesRegex(ValueError, "validation"):
            check_label_coverage(train, unseen, known, "target")
        with self.assertRaisesRegex(ValueError, "test"):
            check_label_coverage(train, known, unseen, "target")

    def test_add_strike_zone_preserves_missing_measurements(self):
        data = self.spark.createDataFrame(
            [
                (0, 0.0, 2.0, 1.0, 3.0),
                (1, 1.0, 2.0, 1.0, 3.0),
                (2, None, 2.0, 1.0, 3.0),
                (3, 0.0, None, 1.0, 3.0),
                (4, 0.0, 2.0, None, 3.0),
                (5, 0.0, 2.0, 1.0, None),
            ],
            ["id", "plate_x", "plate_z", "sz_bot", "sz_top"],
        )

        rows = add_strike_zone(data).orderBy("id").collect()

        self.assertEqual([row.in_strike_zone for row in rows], [1, 0, None, None, None, None])

    def test_plot_feature_importance_selects_top_features(self):
        importance = pd.DataFrame({
            "feature": ["x", "y", "z"],
            "importance": [0.6, 0.3, 0.1],
        })

        with patch("classification_helper.plt.show") as show:
            axes = plot_feature_importance(importance, top_n=2, title="Example")

        self.assertEqual(axes.get_title(), "Example")
        self.assertEqual([label.get_text() for label in axes.get_yticklabels()], ["y", "x"])
        show.assert_called_once_with()
        plt.close(axes.figure)

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
