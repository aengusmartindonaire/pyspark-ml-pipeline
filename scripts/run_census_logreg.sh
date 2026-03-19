#!/bin/bash
# Census income prediction — Logistic Regression with cross-validation
#
# Usage:
#   ./scripts/run_census_logreg.sh [train_csv] [test_csv]
#
# Defaults to data/ directory. For HDFS, pass full hdfs:// paths.

TRAIN_DATA="${1:-data/adult_train.csv}"
TEST_DATA="${2:-data/adult_test.csv}"

spark-submit \
    --name "Census Income - Logistic Regression" \
    --py-files src/preprocessing.py \
    src/census_income_logreg.py "$TRAIN_DATA" "$TEST_DATA"
