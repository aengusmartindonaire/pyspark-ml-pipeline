"""Shared preprocessing utilities for UCI Census Income classification.

Provides data cleaning and PySpark ML Pipeline stages used by both the
logistic regression and tree-based classifier scripts.
"""

import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

COLUMN_NAMES = [
    "age", "workClass", "fnlwgt", "education", "educationNum",
    "maritalStatus", "occupation", "relationship", "race", "sex",
    "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "Labels",
]

CATEGORICAL_COLS = [
    "workClass", "maritalStatus", "occupation",
    "relationship", "race", "sex", "nativeCountry",
]

NUMERIC_COLS = ["age", "educationNum", "hoursPerWeek"]


def clean_census_data(df):
    """Apply non-ML transformations to a Census Income DataFrame.

    Trims whitespace, removes irrelevant columns, handles missing values,
    casts types, and encodes the binary income label.
    """
    # Trim whitespace in all columns
    for col_name in df.columns:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))

    # Drop columns not used in modeling
    df = df.drop("fnlwgt", "education", "capitalGain", "capitalLoss")

    # Replace "?" with None, then drop rows with any null
    for col_name in df.columns:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) == "?", None).otherwise(F.col(col_name)),
        )
    df = df.na.drop("any")

    # Cast numeric columns to float
    for col_name in NUMERIC_COLS:
        df = df.withColumn(col_name, F.col(col_name).cast("float"))

    # Encode target: <=50K -> 0, >50K -> 1
    df = df.withColumn(
        "Labels",
        F.when(F.col("Labels").isin(["<=50K", "<=50K."]), 0).otherwise(F.col("Labels")),
    )
    df = df.withColumn(
        "Labels",
        F.when(F.col("Labels").isin([">50K", ">50K."]), 1).otherwise(F.col("Labels")),
    )
    df = df.withColumn("Labels", F.col("Labels").cast("int"))

    return df


def get_preprocessing_stages():
    """Return PySpark ML Pipeline stages for feature engineering.

    Stages: StringIndexer -> OneHotEncoder -> VectorAssembler.
    Designed to be fit on training data only to prevent data leakage.
    """
    indexed_cols = [f"{col}_idx" for col in CATEGORICAL_COLS]
    encoded_cols = [f"{col}_vec" for col in CATEGORICAL_COLS]

    indexer = StringIndexer(
        inputCols=CATEGORICAL_COLS,
        outputCols=indexed_cols,
        handleInvalid="keep",
    )

    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols,
    )

    assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + encoded_cols,
        outputCol="features",
    )

    return [indexer, encoder, assembler]
