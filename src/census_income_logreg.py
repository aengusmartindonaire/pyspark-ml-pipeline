"""Census income prediction using Logistic Regression with cross-validation.

Predicts whether income exceeds $50K/year based on the UCI Census Income
dataset. Uses a full preprocessing pipeline (StringIndexer, OneHotEncoder,
VectorAssembler) with 3-fold cross-validation for hyperparameter tuning.
"""

from __future__ import print_function

import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from preprocessing import clean_census_data, get_preprocessing_stages, COLUMN_NAMES

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: census_income_logreg.py <train_csv> <test_csv>", file=sys.stderr)
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Census Income - Logistic Regression") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load and clean data separately (no leakage from test into train)
    train_df = spark.read.csv(sys.argv[1], header=False, inferSchema=True).toDF(*COLUMN_NAMES)
    test_df = spark.read.csv(sys.argv[2], header=False, inferSchema=True).toDF(*COLUMN_NAMES)

    train_df = clean_census_data(train_df)
    test_df = clean_census_data(test_df)

    # Combine and re-split to ensure consistent distribution
    full_df = train_df.union(test_df)
    train_df, test_df = full_df.randomSplit([0.8, 0.2], seed=47)

    # Build pipeline: preprocessing + logistic regression
    lr = LogisticRegression(featuresCol="features", labelCol="Labels", maxIter=110)
    pipeline = Pipeline(stages=get_preprocessing_stages() + [lr])

    # Cross-validation for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.2]) \
        .build()

    evaluator = BinaryClassificationEvaluator(labelCol="Labels", metricName="areaUnderROC")

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=47,
    )

    # Train with cross-validation (pipeline fit on training data only)
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel

    # Display best hyperparameters
    best_lr = best_model.stages[-1]
    print(f"\nBest regParam: {best_lr.getRegParam()}")
    print(f"Model intercept: {best_lr.intercept:.4f}")

    # Predict on held-out test set
    predictions = cv_model.transform(test_df)

    # Evaluate
    metric_evaluators = {
        "Accuracy": MulticlassClassificationEvaluator(labelCol="Labels", metricName="accuracy"),
        "Precision": MulticlassClassificationEvaluator(labelCol="Labels", metricName="weightedPrecision"),
        "Recall": MulticlassClassificationEvaluator(labelCol="Labels", metricName="weightedRecall"),
        "F1 Score": MulticlassClassificationEvaluator(labelCol="Labels", metricName="f1"),
        "AUC-ROC": evaluator,
    }

    print("\n========== Logistic Regression Evaluation ==========")
    for name, eval_ in metric_evaluators.items():
        print(f"  {name}: {eval_.evaluate(predictions):.4f}")
    print("=====================================================")

    spark.stop()
