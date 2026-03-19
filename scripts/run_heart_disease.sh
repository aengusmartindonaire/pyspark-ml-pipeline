#!/bin/bash
# Heart disease risk prediction — Logistic Regression
#
# Usage:
#   ./scripts/run_heart_disease.sh [heart_csv]
#
# Defaults to data/ directory. For HDFS, pass a full hdfs:// path.

DATA="${1:-data/heart.csv}"

spark-submit \
    --name "Heart Disease Prediction" \
    src/heart_disease_predictor.py "$DATA"
