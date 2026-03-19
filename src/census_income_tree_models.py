"""Census income prediction using Random Forest and Decision Tree classifiers.

Compares tree-based models for the UCI Census Income prediction task,
using the same preprocessing pipeline as the logistic regression variant.
"""

from __future__ import print_function

import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from preprocessing import clean_census_data, get_preprocessing_stages, COLUMN_NAMES


def evaluate_model(predictions, label_col="Labels"):
    """Compute classification metrics for a set of predictions."""
    metric_names = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
    results = {}

    for metric in metric_names:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName=metric,
        )
        results[metric] = evaluator.evaluate(predictions)

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC",
    )
    results["auc-roc"] = auc_evaluator.evaluate(predictions)

    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: census_income_tree_models.py <train_csv> <test_csv>", file=sys.stderr)
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Census Income - Tree Models") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load and clean data separately
    train_df = spark.read.csv(sys.argv[1], header=False, inferSchema=True).toDF(*COLUMN_NAMES)
    test_df = spark.read.csv(sys.argv[2], header=False, inferSchema=True).toDF(*COLUMN_NAMES)

    train_df = clean_census_data(train_df)
    test_df = clean_census_data(test_df)

    # Combine and re-split to ensure consistent distribution
    full_df = train_df.union(test_df)
    train_df, test_df = full_df.randomSplit([0.8, 0.2], seed=47)

    # Random Forest pipeline (fresh preprocessing stages per pipeline)
    rf = RandomForestClassifier(featuresCol="features", labelCol="Labels", numTrees=100)
    rf_pipeline = Pipeline(stages=get_preprocessing_stages() + [rf])
    rf_model = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)

    # Decision Tree pipeline
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="Labels")
    dt_pipeline = Pipeline(stages=get_preprocessing_stages() + [dt])
    dt_model = dt_pipeline.fit(train_df)
    dt_predictions = dt_model.transform(test_df)

    # Evaluate both models
    print("\n========== Random Forest Evaluation ==========")
    for metric, value in evaluate_model(rf_predictions).items():
        print(f"  {metric}: {value:.4f}")

    print("\n========== Decision Tree Evaluation ==========")
    for metric, value in evaluate_model(dt_predictions).items():
        print(f"  {metric}: {value:.4f}")
    print("===============================================")

    spark.stop()
