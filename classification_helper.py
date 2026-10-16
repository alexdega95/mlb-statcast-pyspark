import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, functions as F


SUPPORTED_METRICS = ("accuracy", "f1", "weightedPrecision", "weightedRecall")


def setup_spark(appName):
    spark = (
        SparkSession.builder
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
        .appName(appName)
        .getOrCreate()
    )
    return spark


def assemble(train, test, features):
    assembler = VectorAssembler(
        inputCols=features,
        outputCol="features",
    )

    assembled_train = assembler.transform(train)
    assembled_test = assembler.transform(test)
    return assembled_train, assembled_test


def evaluate_predictions(predictions, metrics, label_col="label"):
    """Return requested metrics for an existing prediction DataFrame."""
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Metric {metric} not supported")

    results = {}
    for metric in metrics:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric,
        )
        results[metric] = round(evaluator.evaluate(predictions), 4)
    return results


def evaluate_models(data, fitted_models, metrics, label_col="label"):
    """Evaluate named fitted models without training; callers own cached predictions."""
    results = {}
    for name, model in fitted_models.items():
        predictions = model.transform(data).cache()
        results[name] = {
            "predictions": predictions,
            **evaluate_predictions(predictions, metrics, label_col),
        }
    return results


def perform(train, test, models, metrics, labelCol):
    """Fit estimators and return evaluation results and a dictionary of fitted models."""
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Metric {metric} not supported")

    results = {}
    fit_models = {}
    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(name)
        print("=" * 50)
        fit_models[name] = model.fit(train)
        model_results = evaluate_models(
            test, {name: fit_models[name]}, metrics, labelCol
        )
        results[name] = model_results[name]
        for metric in metrics:
            print(f"{metric}: {results[name][metric]}")
    return results, fit_models


def print_results(results):
    print(f"\n{'=' * 50}")
    print("Classification results")
    print("=" * 50)
    for name, result in results.items():
        print(f"\n{name}")
        for metric in SUPPORTED_METRICS:
            if metric in result:
                print(f"{metric}: {result[metric]:.4f}")


def classification_report(predictions, label_col="label"):
    """Return a per-class DataFrame and macro F1, including predicted-only classes."""
    actual = predictions.groupBy(F.col(label_col).alias("label")).agg(
        F.count("*").alias("support"),
        F.sum(
            F.when(F.col(label_col) == F.col("prediction"), 1).otherwise(0)
        ).alias("TP"),
    )
    predicted = predictions.groupBy(F.col("prediction").alias("label")).agg(
        F.count("*").alias("predicted_count")
    )
    per_class = (
        actual.join(predicted, "label", "full_outer")
        .fillna(0, subset=["support", "TP", "predicted_count"])
        .withColumn(
            "precision",
            F.when(
                F.col("predicted_count") > 0,
                F.col("TP") / F.col("predicted_count"),
            ).otherwise(0.0),
        )
        .withColumn(
            "recall",
            F.when(F.col("support") > 0, F.col("TP") / F.col("support"))
            .otherwise(0.0),
        )
        .withColumn(
            "f1",
            F.when(
                (F.col("precision") + F.col("recall")) > 0,
                2 * F.col("precision") * F.col("recall")
                / (F.col("precision") + F.col("recall")),
            ).otherwise(0.0),
        )
        .select("label", "precision", "recall", "f1", "support")
        .orderBy("label")
    )
    macro_f1 = per_class.agg(F.avg("f1")).first()[0]
    return per_class, macro_f1


def check_label_coverage(train, validation, test, label_col):
    """Raise ValueError if an evaluation split contains a class absent from training."""
    training_labels = train.select(label_col).distinct()
    for split_name, split in [("validation", validation), ("test", test)]:
        missing = split.select(label_col).distinct().join(
            training_labels, label_col, "left_anti"
        )
        if missing.limit(1).count():
            raise ValueError(
                f"{split_name} contains a {label_col} class absent from training"
            )


def feature_importance(model, df, features_col="features"):
    if not hasattr(model, "featureImportances"):
        raise TypeError(
            f"{type(model).__name__} does not expose featureImportances"
        )

    attrs = df.schema[features_col].metadata["ml_attr"]["attrs"]
    feature_names = {}
    for attr_type in attrs:
        for attr in attrs[attr_type]:
            feature_names[attr["idx"]] = attr["name"]

    importances = model.featureImportances.toArray()
    importance_df = (
        pd.DataFrame({
            "feature": [feature_names[i] for i in range(len(importances))],
            "importance": importances,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def plot_feature_importance(importance_df, top_n=20, title=None):
    """Plot the highest-ranked features and return the Matplotlib axes."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    top_features = importance_df.head(top_n).sort_values("importance")
    figure, axes = plt.subplots(figsize=(10, 8))
    axes.barh(top_features["feature"], top_features["importance"])
    axes.set_xlabel("Feature Importance")
    axes.set_ylabel("Feature")
    axes.set_title(title or f"Top {top_n} Feature Importances")
    figure.tight_layout()
    plt.show()
    return axes


def add_strike_zone(df):
    """Add geometric zone membership, preserving nulls for missing measurements."""
    ball_radius = 0.12
    half_plate = 17 / 24
    missing = (
        F.col("plate_x").isNull()
        | F.col("plate_z").isNull()
        | F.col("sz_bot").isNull()
        | F.col("sz_top").isNull()
    )
    return df.withColumn(
        "in_strike_zone",
        F.when(missing, F.lit(None).cast("int"))
        .when(
            (F.abs(F.col("plate_x")) <= half_plate + ball_radius)
            & (F.col("plate_z") >= F.col("sz_bot") - ball_radius)
            & (F.col("plate_z") <= F.col("sz_top") + ball_radius),
            1,
        )
        .otherwise(0),
    )
