# MLflow Experiments & Data Pipeline Explanation

## Project: Sentiment Analysis with Databricks & MLflow

---

## 🏗️ The Data Pipeline (Bronze → Silver → Gold)

This project uses **"Medallion Architecture"** - a standard data engineering pattern for data lakes.

### 1. **Bronze Layer** (Raw Data)
- **Input**: TSV files uploaded to `/Volumes/workspace/sentiment_analysis/raw`
- **Process**: `bronze_data_ingestion.ipynb` reads TSV → adds metadata
- **Output**: Parquet files in `/Volumes/workspace/sentiment_analysis/bronze`
- **Purpose**: "Just dump everything as-is" - preserve raw data
- **MLflow Experiment**: `bronze_data_ingestion` + `bronze_validation`

**What gets logged to MLflow:**
- Row count from source
- Column count
- Distinct reviews & products
- Date ranges
- Null counts per column
- Duplicate detection
- Schema artifacts

### 2. **Silver Layer** (Cleaned Data)
- **Input**: Bronze data
- **Process**: `silver_data_ingestion.ipynb` performs:
  - Remove nulls in required fields
  - Fix data types (star_rating → float, dates → timestamp)
  - Remove duplicates by review_id
  - Filter invalid ratings (keep only 1-5 stars)
- **Output**: Clean data in `/Volumes/workspace/sentiment_analysis/silver`
- **Purpose**: "Clean up the mess" - ensure data quality
- **MLflow Experiment**: `silver_data_ingestion`

**What gets logged to MLflow:**
- Bronze row count
- Silver row count
- Retention ratio (% of data kept after cleaning)
- Invalid rows dropped
- Process timestamp

### 3. **Gold Layer** (ML-Ready Data)
- **Input**: Silver data
- **Process**: `gold_data_ingestion_sa.ipynb` performs:
  - Text cleaning (lowercase, remove HTML, regex cleaning)
  - Create sentiment labels:
    - 1-2 stars → "negative"
    - 3 stars → "neutral"
    - 4-5 stars → "positive"
  - Feature engineering (word count, character count, etc.)
  - Select ML-relevant columns
- **Output**: ML-ready data in `/Volumes/workspace/sentiment_analysis/gold`
- **Purpose**: "Make it perfect for machine learning"
- **MLflow Experiment**: `gold_data_ingestion`

**What gets logged to MLflow:**
- Total rows & columns
- Sentiment label distribution (negative/neutral/positive ratios)
- Average review length
- Missing data ratios
- Completeness score (data quality metric)
- Class balance metrics

### 4. **Model Training** (Uses Gold Data)
- **Input**: Gold data (clean text + sentiment labels)
- **Process**: `sentiment_analysis.ipynb` or `model_train.py`:
  1. Load Gold data
  2. Vectorize text using TF-IDF (50,000 features, unigrams + bigrams)
  3. Split: 80% train, 20% test (stratified)
  4. Train 12 model variants
  5. Evaluate each model
  6. Log everything to MLflow
- **Output**: 12 trained models with metrics
- **MLflow Experiment**: `sentiment_analysis`

**What gets logged to MLflow (per model):**
- Model name
- Hyperparameters (C, alpha, max_iter, etc.)
- F1 Score (macro-averaged) - **PRIMARY METRIC**
- Precision (macro-averaged)
- Recall (macro-averaged)
- Trained model artifacts (saved models)

---

## 🤖 The 12 Models - Detailed Breakdown

**Source**: Lines 80-100 in `src/model_train.py`

### **Group 1: Logistic Regression (3 variants)**

Logistic Regression with One-vs-Rest classification, testing different regularization strengths:

| Model Name | Algorithm | C Value | Description |
|------------|-----------|---------|-------------|
| `LR_C1` | LogisticRegression | C=1 | Lower regularization (more penalty) |
| `LR_C5` | LogisticRegression | C=5 | Medium regularization |
| `LR_C10` | LogisticRegression | C=10 | Higher regularization (less penalty) |

**Common settings:**
- `max_iter=5000`
- `solver="liblinear"`
- `class_weight="balanced"` (handles imbalanced classes)

**Purpose**: Test how much regularization helps with high-dimensional text data.

---

### **Group 2: Linear SVM (4 variants)**

Linear Support Vector Machine with One-vs-Rest, testing a wider range of C values:

| Model Name | Algorithm | C Value | Description |
|------------|-----------|---------|-------------|
| `SVM_OVR_C01` | LinearSVC | C=0.1 | Very strong regularization |
| `SVM_OVR_C1` | LinearSVC | C=1.0 | Medium regularization |
| `SVM_OVR_C10` | LinearSVC | C=10.0 | Weak regularization |
| `SVM_OVR_C50` | LinearSVC | C=50.0 | Very weak regularization |

**Common settings:**
- `max_iter=5000`
- Wrapped in `OneVsRestClassifier` for multi-class

**Purpose**: SVMs often work well for text classification. Testing different regularization to find optimal balance.

---

### **Group 3: Multinomial Naive Bayes (4 variants)**

Naive Bayes with different smoothing parameters:

| Model Name | Algorithm | Alpha Value | Description |
|------------|-----------|-------------|-------------|
| `NB_alpha02` | MultinomialNB | α=0.2 | Minimal smoothing |
| `NB_alpha05` | MultinomialNB | α=0.5 | Less smoothing |
| `NB_alpha1` | MultinomialNB | α=1.0 | Standard Laplace smoothing |
| `NB_alpha2` | MultinomialNB | α=2.0 | More smoothing |

**Purpose**: Naive Bayes is fast and works well with text. Alpha controls smoothing to handle words not seen in training.

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE FLOW                       │
└─────────────────────────────────────────────────────────────┘

📁 Raw TSV Files
   (Your uploaded data: ~15M product reviews)
        ↓
   ┌─────────────────────────────────────┐
   │  bronze_data_ingestion.ipynb        │
   │  - Read TSV with schema             │
   │  - Add metadata (timestamp, source) │
   │  - Log: rows, columns, schema       │
   └─────────────────────────────────────┘
        ↓
📦 BRONZE Layer
   /Volumes/workspace/sentiment_analysis/bronze
   - Raw data preserved
   - ~15M rows (or sampled to 40% for free tier)
   - All original columns
        ↓
   ┌─────────────────────────────────────┐
   │  silver_data_ingestion.ipynb        │
   │  - Remove nulls                     │
   │  - Fix data types                   │
   │  - Remove duplicates                │
   │  - Filter invalid ratings           │
   │  - Log: retention ratio             │
   └─────────────────────────────────────┘
        ↓
🧹 SILVER Layer
   /Volumes/workspace/sentiment_analysis/silver
   - Cleaned data
   - ~10-12M rows (depending on data quality)
   - Standardized column names
        ↓
   ┌─────────────────────────────────────┐
   │  gold_data_ingestion_sa.ipynb       │
   │  - Clean text (lowercase, HTML)     │
   │  - Map stars → sentiment labels     │
   │  - Feature engineering              │
   │  - Log: class distribution          │
   └─────────────────────────────────────┘
        ↓
✨ GOLD Layer
   /Volumes/workspace/sentiment_analysis/gold
   - ML-ready features
   - Columns: review_id, product_id, clean_text, sentiment_label
   - Labels: negative (1-2★), neutral (3★), positive (4-5★)
        ↓
   ┌─────────────────────────────────────┐
   │  sentiment_analysis.ipynb           │
   │  OR model_train.py                  │
   │  - Load gold data                   │
   │  - TF-IDF vectorization             │
   │  - Train 12 models                  │
   │  - Evaluate & log metrics           │
   └─────────────────────────────────────┘
        ↓
🤖 12 TRAINED MODELS
   ┌─────────────────────────────────┐
   │ 3 Logistic Regression variants  │
   │  - LR_C1, LR_C5, LR_C10        │
   ├─────────────────────────────────┤
   │ 4 Linear SVM variants           │
   │  - SVM_OVR_C01, SVM_OVR_C1,    │
   │    SVM_OVR_C10, SVM_OVR_C50    │
   ├─────────────────────────────────┤
   │ 4 Naive Bayes variants          │
   │  - NB_alpha02, NB_alpha05,     │
   │    NB_alpha1, NB_alpha2        │
   └─────────────────────────────────┘
        ↓
📊 MLflow Tracking
   - All 12 runs logged
   - Metrics: F1, Precision, Recall
   - Artifacts: Trained models saved
   - Best model identified
```

---

## 🎯 Why 12 Models? (Hyperparameter Search)

This is called **hyperparameter tuning** or **model selection**. The goal is to:

1. **Test different algorithms**
   - Logistic Regression (linear, probabilistic)
   - SVM (maximum margin classifier)
   - Naive Bayes (probabilistic, assumes feature independence)

2. **Test different hyperparameters**
   - **C** (for LR & SVM): Controls regularization strength
     - Lower C = More regularization = Simpler model
     - Higher C = Less regularization = More complex model
   - **alpha** (for Naive Bayes): Controls smoothing
     - Lower alpha = Less smoothing = Trust training data more
     - Higher alpha = More smoothing = More conservative

3. **Find the best combination**
   - Compare all 12 based on F1 score (macro-averaged)
   - F1 balances precision and recall
   - Macro-average treats all classes equally (good for imbalanced data)

**This is proper MLOps!** Instead of guessing, you systematically test and track everything.

---

## 📊 Your MLflow Experiments

When you look at **Machine Learning → Experiments** in Databricks, you see:

### Experiment: `bronze_data_ingestion`
- **Purpose**: Track raw data ingestion
- **Runs**: 1 per execution
- **Metrics**: rows_read, columns

### Experiment: `bronze_validation`
- **Purpose**: Track data quality checks
- **Runs**: 1 per execution
- **Metrics**: row_count, distinct_reviews, distinct_products, duplicate_reviews

### Experiment: `silver_data_ingestion`
- **Purpose**: Track data cleaning
- **Runs**: 1 per execution
- **Metrics**: retention_ratio, invalid_rows, bronze_rows, silver_rows

### Experiment: `gold_data_ingestion`
- **Purpose**: Track feature engineering
- **Runs**: 1 per execution
- **Metrics**: label_ratio_negative, label_ratio_neutral, label_ratio_positive, completeness_score, avg_review_length

### Experiment: `sentiment_analysis` ⭐ (MAIN EXPERIMENT)
- **Purpose**: Track model training & comparison
- **Runs**: 12 (one per model variant)
- **Metrics per run**:
  - `f1_score` (PRIMARY - used for model selection)
  - `precision`
  - `recall`
- **Parameters per run**:
  - `model_name` (e.g., "LR_C5", "SVM_OVR_C10")
  - Hyperparameters (C, alpha)
- **Artifacts**: Trained model files

---

## 🎓 Key Concepts Explained

### What is MLflow?
- **Experiment tracking system** built into Databricks
- Logs parameters, metrics, and model artifacts automatically
- Allows comparison of multiple runs
- Essential for reproducible ML

### What is Unity Catalog?
- **Enterprise data catalog** for Databricks
- Organizes data in: Catalog → Schema → Tables/Volumes
- Your structure: `workspace.sentiment_analysis.{raw,bronze,silver,gold}`
- Provides governance and access control

### What is the Medallion Architecture?
- **Best practice** for organizing data lakes
- **Bronze** = Raw, immutable data
- **Silver** = Cleaned, deduplicated data
- **Gold** = Business-level, aggregated data (ML-ready)
- Benefits: Data quality, traceability, reprocessing

### What is Hyperparameter Tuning?
- **Systematic search** for best model settings
- Instead of guessing, test multiple configurations
- Track everything with MLflow
- Select best based on objective metric (F1 score)

---

## 💡 How to Use This in Your TP Report

### 1. Architecture Diagram
Use the flow diagram above to explain your data pipeline.

### 2. Experiment Results Table
Export from MLflow:

| Model | F1 Score | Precision | Recall | Best? |
|-------|----------|-----------|--------|-------|
| LR_C1 | 0.XXXX | 0.XXXX | 0.XXXX | |
| LR_C5 | 0.XXXX | 0.XXXX | 0.XXXX | ✓ |
| ... | ... | ... | ... | |

### 3. Key Metrics to Report
- **Data volume**: Rows at each stage (Bronze → Silver → Gold)
- **Data quality**: Retention ratio, duplicate percentage
- **Model performance**: Best F1 score, comparison across algorithms
- **Reproducibility**: All experiments tracked in MLflow

### 4. MLOps Concepts Demonstrated
- ✅ Experiment tracking (MLflow)
- ✅ Data versioning (Medallion architecture)
- ✅ Model registry (Unity Catalog)
- ✅ Hyperparameter tuning (12 variants)
- ✅ Automated pipelines (notebooks)
- ✅ Reproducibility (all parameters logged)

---

## 🚀 Summary

**You have successfully implemented a complete MLOps pipeline!**

1. ✅ **Data Pipeline**: Bronze → Silver → Gold (Medallion Architecture)
2. ✅ **Model Training**: 12 variants with systematic hyperparameter search
3. ✅ **Experiment Tracking**: All metrics logged to MLflow
4. ✅ **Model Registry**: Best model saved to Unity Catalog
5. ✅ **Reproducibility**: Every step tracked and documented

**This is enterprise-grade MLOps**, suitable for production environments!

---

*Generated for: TP - Herramientas para Grandes Volúmenes de Datos*
*Student: nmoccagatta@itba.edu.ar*
*Date: 2025-12-11*
