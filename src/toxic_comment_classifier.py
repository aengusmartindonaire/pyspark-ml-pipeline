"""Multi-label toxic comment classification using TF-IDF and Logistic Regression.

Processes text comments through a TF-IDF pipeline and trains independent binary
logistic regression models for each toxicity category (toxic, severe_toxic,
obscene, threat, insult, identity_hate).
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
import pyspark.sql.functions as F
import pyspark.sql.types as T

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: toxic_comment_classifier.py <train_csv> <test_csv>", file=sys.stderr)
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Toxic Comment Classification") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load data
    train_df = spark.read.csv(sys.argv[1], header=True) \
        .withColumn("comment_text", col("comment_text").cast("string"))
    test_df = spark.read.csv(sys.argv[2], header=True) \
        .withColumn("comment_text", col("comment_text").cast("string"))

    # Identify label columns (everything except id and comment_text)
    label_cols = [c for c in train_df.columns if c not in ("id", "comment_text")]
    for column in label_cols:
        train_df = train_df.withColumn(column, col(column).cast(T.IntegerType()))

    # Build TF-IDF preprocessing pipeline
    tfidf_pipeline = Pipeline(stages=[
        Tokenizer(inputCol="comment_text", outputCol="words"),
        HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000),
        IDF(inputCol="raw_features", outputCol="features"),
    ])

    train_df = train_df.dropna(subset=["comment_text"] + label_cols)
    tfidf_model = tfidf_pipeline.fit(train_df)
    train_processed = tfidf_model.transform(train_df)

    # Preprocess test data with the same fitted pipeline
    test_df = test_df.filter(col("comment_text") != '"') \
        .withColumn("UID", monotonically_increasing_id())
    test_processed = tfidf_model.transform(test_df)
    test_results = test_df.select("UID")

    # Train a separate logistic regression model per label
    extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())

    for label in label_cols:
        lr = LogisticRegression(featuresCol="features", labelCol=label, regParam=0.1)
        lr_model = lr.fit(train_processed)
        predictions = lr_model.transform(test_processed)

        predictions = predictions \
            .withColumn(f"proba_{label}", extract_prob("probability")) \
            .withColumn(f"pred_{label}", col("prediction")) \
            .select("UID", f"proba_{label}", f"pred_{label}")

        test_results = test_results.join(predictions, on=["UID"])

    # Display sample predictions
    print("\n========== Sample Predictions ==========")
    test_results.sample(False, 0.4).show(50)

    spark.stop()
