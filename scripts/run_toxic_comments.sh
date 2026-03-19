#!/bin/bash
# Toxic comment classification — TF-IDF + Logistic Regression
#
# Usage:
#   ./scripts/run_toxic_comments.sh [train_csv] [test_csv]
#
# Defaults to data/ directory. For HDFS, pass full hdfs:// paths.

TRAIN_DATA="${1:-data/train.csv}"
TEST_DATA="${2:-data/test.csv}"

spark-submit \
    --name "Toxic Comment Classification" \
    src/toxic_comment_classifier.py "$TRAIN_DATA" "$TEST_DATA"
