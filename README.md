# Distributed ML Classification with Apache Spark

End-to-end classification pipelines built with **PySpark ML**, designed to run on a distributed Spark/HDFS cluster. Three distinct datasets, four models, full preprocessing and evaluation.

## Projects

### 1. Toxic Comment Classification (NLP)
Multi-label binary classification on the [Jigsaw Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset (~560K comments). Builds a TF-IDF feature extraction pipeline and trains independent logistic regression models for each toxicity label (toxic, severe_toxic, obscene, threat, insult, identity_hate).

**Key techniques:** Tokenizer, HashingTF, IDF, multi-label logistic regression

### 2. Heart Disease Risk Prediction
Binary classification on the [Framingham Heart Study](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) dataset (1,025 patients). Identifies key clinical risk factors through model coefficients and outputs a per-patient risk score.

**Key techniques:** VectorAssembler, logistic regression, coefficient interpretation, confusion matrix

### 3. Census Income Prediction — Logistic Regression
Binary classification (income >$50K vs <=$50K) on the [UCI Adult Census](https://archive.ics.uci.edu/ml/datasets/adult) dataset (~48K records). Full categorical preprocessing pipeline with **3-fold cross-validation** for hyperparameter tuning.

**Key techniques:** StringIndexer, OneHotEncoder, VectorAssembler, Pipeline API, CrossValidator

### 4. Census Income Prediction — Tree Models
Same task as #3 using Random Forest (100 trees) and Decision Tree classifiers for comparison against the logistic regression baseline.

**Key techniques:** RandomForestClassifier, DecisionTreeClassifier, model comparison

## Results

All metrics below were produced by PySpark ML on a 3-node Spark/HDFS cluster. Raw outputs are in `results/`.

### Heart Disease — Logistic Regression

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8931 |
| Precision | 0.8195 |
| Recall | 0.8934 |
| F1 Score | 0.8549 |

Top risk factors (positive coefficients): **cp** (0.949), **slope** (0.689), **restecg** (0.397).

### Census Income — Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|------|---------|
| Logistic Regression | 0.8138 | 0.8058 | 0.8138 | 0.7896 | **0.8731** |
| Random Forest (100 trees) | 0.8051 | 0.8008 | 0.8051 | 0.7719 | 0.8623 |
| Decision Tree | 0.8107 | 0.8002 | 0.8107 | 0.7877 | 0.6977 |

### Toxic Comment Classification

TF-IDF (10,000 features) + independent logistic regression per label. Produces calibrated probability scores across all six toxicity categories.

See **[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)** for full EDA, visualizations, and detailed result breakdowns.

## Tech Stack
- **Apache Spark** — PySpark ML / MLlib APIs (distributed pipelines)
- **Hadoop HDFS** — distributed data storage
- **pandas / matplotlib / seaborn** — EDA and result visualization notebook
- **Python 3.x**

## Project Structure
```
.
├── src/
│   ├── preprocessing.py               # Shared Census data preprocessing
│   ├── toxic_comment_classifier.py    # Project 1: NLP classification
│   ├── heart_disease_predictor.py     # Project 2: Clinical prediction
│   ├── census_income_logreg.py        # Project 3: LogReg + cross-validation
│   └── census_income_tree_models.py   # Project 4: RF + DT comparison
├── scripts/
│   ├── run_toxic_comments.sh
│   ├── run_heart_disease.sh
│   ├── run_census_logreg.sh
│   └── run_census_trees.sh
├── notebooks/
│   └── analysis.ipynb                 # EDA and Spark result visualizations (no Spark needed)
├── results/
│   ├── part1.csv                      # Toxic comment predictions (Spark output)
│   ├── part2/                         # Heart disease metrics, coefficients, predictions
│   ├── part3/                         # Census LR metrics and coefficients
│   └── part4.csv                      # RF vs DT comparison metrics
├── data/
│   ├── heart.csv                      # Framingham Heart dataset
│   ├── adult_train.csv                # UCI Adult Census (train)
│   └── adult_test.csv                 # UCI Adult Census (test)
├── report/
│   └── project_report.pdf             # Detailed analysis report
├── requirements.txt
└── .gitignore
```

## Setup

### Prerequisites
- Python 3.8+
- Apache Spark 3.x (only needed for `src/` scripts — the notebook runs without Spark)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download Toxic Comments data
The Jigsaw dataset is too large for the repository. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place them in `data/`.

## Usage

Run locally (Spark standalone):
```bash
# Toxic comment classification
./scripts/run_toxic_comments.sh

# Heart disease prediction
./scripts/run_heart_disease.sh

# Census income — logistic regression with cross-validation
./scripts/run_census_logreg.sh

# Census income — random forest & decision tree
./scripts/run_census_trees.sh
```

Run on a Spark cluster with HDFS:
```bash
# Example: pass HDFS paths directly
./scripts/run_heart_disease.sh hdfs://master:9000/data/heart.csv
```

## Design Decisions

- **Pipeline API** — All preprocessing and modeling stages are composed into `pyspark.ml.Pipeline` objects, ensuring transforms are fit on training data only (preventing data leakage).
- **Shared preprocessing** — Census income scripts import from `preprocessing.py` to avoid code duplication between the logistic regression and tree-based model variants.
- **Cross-validation** — The logistic regression census model uses `CrossValidator` with a parameter grid over regularization strength to select the best `regParam`.
- **Cluster-agnostic scripts** — Shell scripts accept data paths as arguments, defaulting to local `data/` for easy testing. Pass `hdfs://` paths for cluster execution.

---
