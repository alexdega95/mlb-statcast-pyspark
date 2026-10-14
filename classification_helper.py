from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

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
        outputCol="features"
    )

    assembled_train = assembler.transform(train)
    assembled_test = assembler.transform(test)
    return assembled_train, assembled_test



def perform(train, test, models, metrics, labelCol):
    #check if requested metrics are supported
    supported_metrics = ["accuracy", "f1", "weightedPrecision","weightedRecall"]
    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported")

    #fit the models and return the requested metrics
    predictions = {}
    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(name)
        print("=" * 50)
        predictions[name] = {}
        fit = model.fit(train)
        model_pred = fit.transform(test).cache()
        predictions[name]["predictions"] = model_pred
        for metric in metrics:
            evaluator = MulticlassClassificationEvaluator(
                labelCol=labelCol,
                predictionCol="prediction",
                metricName=metric
            )
            value = round(evaluator.evaluate(model_pred), 4)
            predictions[name][metric] = value
            print(f"{metric}: {value}")
    return predictions

def print_results(results):
    print(f"\n{'=' * 50}")
    print("Classification results")
    print("=" * 50)
    for name, result in results.items():
        print(f"\n{name}")
        for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
            if metric in result:
                print(f"{metric}: {result[metric]:.4f}")

def feature_importance(model, df, features_col="features"):
    #raise error if the model does not support feature importance
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

    importance_df = pd.DataFrame({
        "feature": [feature_names[i] for i in range(len(importances))],
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df
