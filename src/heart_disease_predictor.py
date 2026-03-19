"""Heart disease risk prediction using Logistic Regression.

Trains a logistic regression model on clinical features from the Framingham
Heart dataset to predict heart disease risk. Reports key risk factors via
model coefficients and evaluates with AUC-ROC, precision, recall, and F1.
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: heart_disease_predictor.py <heart_csv>", file=sys.stderr)
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Heart Disease Prediction") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load data
    df = spark.read.csv(sys.argv[1], header=True, inferSchema=True)

    # Cast columns to appropriate types
    float_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    int_cols = ["restecg", "exang", "slope", "ca", "thal", "target", "sex", "cp", "fbs"]

    for column in float_cols:
        df = df.withColumn(column, col(column).cast(FloatType()))
    for column in int_cols:
        df = df.withColumn(column, col(column).cast(IntegerType()))

    # Build pipeline: feature assembly + logistic regression
    feature_cols = [c for c in df.columns if c != "target"]
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=feature_cols, outputCol="features"),
        LogisticRegression(featuresCol="features", labelCol="target"),
    ])

    # Split and train
    train, test = df.randomSplit([0.8, 0.2], seed=1234)
    pipeline_model = pipeline.fit(train)

    # Extract the trained LR model from the pipeline
    lr_model = pipeline_model.stages[-1]

    # Display risk factors (model coefficients)
    print("\n========== Most Relevant Risk Factors ==========")
    coefficients = lr_model.coefficients.toArray()
    for feature, coeff in zip(feature_cols, coefficients):
        print(f"  {feature}: {coeff:.4f}")
    print(f"  Intercept: {lr_model.intercept:.4f}")

    # Predict and compute risk scores
    predictions = pipeline_model.transform(test)
    extract_risk = udf(lambda x: float(x[1]), FloatType())
    predictions = predictions.withColumn("risk", extract_risk("probability"))

    # Evaluate
    evaluator = BinaryClassificationEvaluator(labelCol="target")
    roc_auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    prediction_rdd = predictions.select("target", "prediction") \
        .rdd.map(lambda x: (float(x[1]), float(x[0])))
    metrics = MulticlassMetrics(prediction_rdd)

    print("\n========== Model Evaluation ==========")
    print(f"  AUC-ROC:   {roc_auc:.4f}")
    print(f"  Precision: {metrics.precision(1.0):.4f}")
    print(f"  Recall:    {metrics.recall(1.0):.4f}")
    print(f"  F1 Score:  {metrics.fMeasure(1.0):.4f}")

    # Sample predictions
    print("\n========== Sample Predictions ==========")
    predictions.select(
        "age", "sex", "cp", "trestbps", "chol", "thalach",
        "oldpeak", "target", "prediction", "risk",
    ).show(20)

    # Confusion matrix
    print("\n========== Confusion Matrix ==========")
    predictions.groupBy("target", "prediction").count() \
        .orderBy("target", "prediction").show()

    spark.stop()
