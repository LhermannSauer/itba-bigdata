"""Model training utilities and baseline experiments for sentiment analysis.

This module provides simple sklearn baseline experiments and logging to
MLflow. It is intended for local experimentation and CI smoke tests.
"""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Paths and constants
DATA_PATH = "/Volumes/workspace/sentiment_analysis/gold"
EXPERIMENT_NAME = "sentiment_baselines"
RANDOM_SEEDS = [69420, 23485, 70001, 10001]
TARGET_COL = "sentiment_label"
TEXT_COL = "clean_text"

# Model save locations
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Ensure a SparkSession is available when running outside Databricks
if "spark" not in globals():
    spark = (
        SparkSession.builder.master("local[1]").appName("itba-bigdata").getOrCreate()
    )

# Load data
print(f"Loading parquet from {DATA_PATH}")
df = spark.read.parquet(str(DATA_PATH))
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
df.printSchema()


def run_experiment(
    model_name,
    classifier,
    params: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    test_size=0.2,
):
    """Run an ML experiment in sklearn and log results to MLflow.

    Parameters
    ----------
    model_name : str
        Human-readable name for the experiment/model.
    classifier
        A scikit-learn compatible estimator with ``fit``/``predict``.
    params : dict
        Hyperparameters to log in MLflow.
    X_train
        Training feature matrix.
    X_test
        Test feature matrix.
    y_train
        Training labels.
    y_test
        Test labels.
    test_size : float
        Fraction of data held out for testing (unused here).

    Returns
    -------
    Tuple[estimator, float]
        The trained model and the F1 score on the test set.

    """
    # ---- MLflow experiment ----
    with mlflow.start_run(run_name=f"{model_name}"):
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ---- Train ----
        model = classifier.fit(X_train, y_train)

        # ---- Predict ----
        y_pred = model.predict(X_test)

        # ---- Metrics (manual or sklearn) ----
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # ---- Log model artifact with signature (required for Unity Catalog) ----
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, artifact_path=model_name, signature=signature)

        print(f"{model_name} | F1 = {f1:.4f}")

        return model, f1


# MODEL DEFINITION
svm_variants = [
    ("SVM_OVR_C01", OneVsRestClassifier(LinearSVC(C=0.1, max_iter=5000)), {"C": 0.1}),
    ("SVM_OVR_C1", OneVsRestClassifier(LinearSVC(C=1.0, max_iter=5000)), {"C": 1.0}),
    ("SVM_OVR_C10", OneVsRestClassifier(LinearSVC(C=10.0, max_iter=5000)), {"C": 10.0}),
    ("SVM_OVR_C50", OneVsRestClassifier(LinearSVC(C=50.0, max_iter=5000)), {"C": 50.0}),
]

lr_variants = [
    (
        "LR_C1",
        LogisticRegression(
            C=1, max_iter=5000, solver="liblinear", class_weight="balanced"
        ),
        {"C": 1},
    ),
    (
        "LR_C5",
        LogisticRegression(
            C=5, max_iter=5000, solver="liblinear", class_weight="balanced"
        ),
        {"C": 5},
    ),
    (
        "LR_C10",
        LogisticRegression(
            C=10, max_iter=5000, solver="liblinear", class_weight="balanced"
        ),
        {"C": 10},
    ),
]

nb_variants = [
    ("NB_alpha1", MultinomialNB(alpha=1.0), {"alpha": 1.0}),
    ("NB_alpha05", MultinomialNB(alpha=0.5), {"alpha": 0.5}),
    ("NB_alpha02", MultinomialNB(alpha=0.2), {"alpha": 0.2}),
    ("NB_alpha2", MultinomialNB(alpha=2.0), {"alpha": 2.0}),
]

all_variants = lr_variants + svm_variants + nb_variants

# Take a sample bc free databricks = bad
df_sample = df.sample(withReplacement=False, fraction=0.4)
df_sample = df_sample.select("clean_text", "sentiment_label")

pdf = df_sample.toPandas()


X_train_text, X_test_text, y_train, y_test = train_test_split(
    pdf["clean_text"], pdf["sentiment_label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

for name, model, params in all_variants:
    print(f"Running experiment on model: {name}")
    run_experiment(name, model, params, X_train, X_test, y_train, y_test)
