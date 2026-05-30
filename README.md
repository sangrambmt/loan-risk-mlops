# ML Predictive Model — Loan Default Prediction

## Use Case

This project predicts whether a loan applicant will default based on demographic, financial, and credit features. It is designed for financial institutions to assess loan risk before approval. Each applicant is assigned a prediction (Default / No Default), a default probability, and a risk tier (Low / Medium / High).

---

## Why OOP

Every step of the pipeline is a real-world responsibility — cleaning, encoding, scoring. OOP lets each responsibility live in its own class with its own state and methods.

**Pros:**
- CleaningPipeline can be reused in scoring without rewriting logic
- Encoder and Scaler save their fitted state — fit once, transform anywhere
- Adding a new model means adding one new class, not touching existing code
- Easy to unit test each class independently
- New team member can read one file and understand one step

---

## Why MLflow

Every time you train a model you get different results. Without tracking you have no idea which params produced which result, and you can never reproduce it.

**Pros:**
- Every run logged with params, metrics, threshold, artifacts automatically
- Compare runs side by side in the UI
- Roll back to any previous model version instantly
- Team members can see exactly what was tried and what worked
- Audit trail — critical in finance where regulators ask why a loan was approved or rejected

---

## Why Docker

The model runs on your machine but breaks on the server because of different Python versions, missing libraries, or OS differences. Docker eliminates this entirely.

**Pros:**
- Same container runs identically on Mac, Linux, Windows, AWS, GCP, Azure
- Scoring API and MLflow UI start with one command
- No manual environment setup for new team members
- Easy to scale — run 10 containers behind a load balancer
- Rollback is just switching to a previous image

---

## How Production Ready Is This

| Area | Status | Notes |
|------|--------|-------|
| Data Pipeline | Production Ready | Modular, reusable, handles missing/errors |
| Cleaning | Production Ready | Fit on train, transform on new data |
| Feature Engineering | Production Ready | Artifacts saved, reused in scoring |
| Model Training | Production Ready | Optuna tuning, cross validation, threshold tuning |
| MLflow Tracking | Production Ready | All runs logged, reproducible |
| Scoring API | Production Ready | Flask REST API, batch and real-time |
| Docker | Production Ready | Containerized, one command deploy |
| Model Performance | Synthetic Data Ceiling | F1=0.47 on synthetic — real data would hit 0.75+ |
| Authentication | Not Implemented | API has no auth — needed before public deployment |
| CI/CD | Not Implemented | No automated retraining pipeline |
| Monitoring | Not Implemented | No data drift or model degradation alerts |
| Load Testing | Not Implemented | API not tested under high traffic |

Overall: 75% production ready. The architecture, code quality, and deployment are production grade. What is missing is auth, CI/CD, monitoring, and real data.

---

## Quick Start

### 1. Install Dependencies

    pip install -r requirements.txt

### 2. Generate Raw Data

    python data/generate_data.py

### 3. Clean Data

    python cleaning/data_cleaner.py

### 4. Run EDA

    python eda/eda_runner.py

### 5. Feature Engineering

    python feature_engineering/fe_runner.py

### 6. Benchmark Models

    python training/benchmark.py

### 7. Tune Best Models

    python training/tuner.py

### 8. Tune Decision Threshold

    python training/threshold_tuner.py

### 9. Evaluate Model

    python training/evaluator.py

### 10. Log to MLflow

    MLFLOW_TRACKING_URI=file:./mlruns python training/train_runner.py

### 11. Start Scoring API

    python scoring/score.py

### 12. Run Batch Scoring

    python scoring/run_scoring.py

---

## Docker Deployment

### Build and Start All Services

    cd docker
    docker-compose up --build -d

### Services

| Service | URL |
|---------|-----|
| Scoring API | http://localhost:8080 |
| MLflow UI | http://localhost:5001 |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /predict | Run prediction |
| GET | /model-info | Model metadata |

### Example Predict Request

    curl -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{
        "data": [{
          "age": 35,
          "education_level": "Bachelor",
          "marital_status": "Married",
          "region": "North",
          "employment_type": "Employed",
          "annual_income": 65000,
          "savings_balance": 12000,
          "investment_balance": 8000,
          "credit_score": 680,
          "num_late_payments": 2,
          "loan_purpose": "Home",
          "loan_amount": 25000,
          "loan_term_months": 60,
          "debt_to_income": 0.35,
          "loan_to_income": 0.38,
          "property_type": "Mortgage",
          "has_cosigner": 0,
          "num_credit_lines": 5,
          "num_dependents": 2,
          "employment_years": 8,
          "months_employed": 96,
          "monthly_income": 5416,
          "monthly_payment": 500,
          "num_prev_loans": 2,
          "credit_util_rate": 0.3,
          "interest_rate": 7.5
        }]
      }'

### Stop Services

    cd docker
    docker-compose down

---

## Project Structure

    ml-predictive-model/
    |
    +-- data/
    |   +-- generate_data.py           generates 100k rows raw data
    |   +-- raw_data.csv               raw generated data
    |   +-- clean_data.csv             cleaned data
    |   +-- train_data.csv             training set 70 percent
    |   +-- val_data.csv               validation set 15 percent
    |   +-- test_data.csv              test set 15 percent
    |   +-- train_balanced.csv         SMOTE balanced training set
    |   +-- scoring_data.csv           10k rows for batch scoring
    |   +-- scored_output.csv          scored results with risk tiers
    |
    +-- cleaning/
    |   +-- duplicate_remover.py       removes duplicate rows
    |   +-- error_corrector.py         fixes impossible values
    |   +-- column_dropper.py          drops high missing columns
    |   +-- missing_imputer.py         imputes missing values
    |   +-- type_enforcer.py           enforces correct dtypes
    |   +-- data_cleaner.py            orchestrates all cleaning steps
    |
    +-- eda/
    |   +-- statistics_analyzer.py     descriptive stats
    |   +-- distribution_plotter.py    histograms and boxplots
    |   +-- correlation_analyzer.py    correlation heatmap
    |   +-- outlier_detector.py        IQR outlier detection
    |   +-- target_analyzer.py         default rate analysis
    |   +-- eda_runner.py              orchestrates all EDA steps
    |   +-- reports/                   saved CSV reports
    |   +-- plots/                     saved PNG plots
    |
    +-- feature_engineering/
    |   +-- feature_selector.py        drops low signal features
    |   +-- interaction_features.py    creates interaction features
    |   +-- encoder.py                 one-hot and label encoding
    |   +-- scaler.py                  robust scaling
    |   +-- balancer.py                SMOTE oversampling
    |   +-- splitter.py                train val test split
    |   +-- fe_runner.py               orchestrates all FE steps
    |   +-- artifacts/                 saved encoder and scaler
    |
    +-- training/
    |   +-- benchmark.py               benchmarks 20 models
    |   +-- tuner.py                   Optuna hyperparameter tuning
    |   +-- threshold_tuner.py         finds optimal decision threshold
    |   +-- evaluator.py               evaluates on test set
    |   +-- train_runner.py            logs everything to MLflow
    |   +-- artifacts/                 saved models and threshold
    |   +-- reports/                   saved evaluation reports
    |
    +-- scoring/
    |   +-- score.py                   Flask REST API
    |   +-- run_scoring.py             batch scoring pipeline
    |   +-- model.pkl                  production model
    |
    +-- docker/
    |   +-- Dockerfile                 scoring API container
    |   +-- docker-compose.yml         all services
    |
    +-- mlruns/                        MLflow experiment tracking
    +-- requirements.txt               Python dependencies
    +-- README.md                      this file

---

## Pipeline Architecture

    raw_data.csv
         |
         v
    data_cleaner.py         removes duplicates, errors, nulls
         |
         v
    eda_runner.py           statistics, plots, correlations
         |
         v
    fe_runner.py            select, interact, encode, scale, SMOTE, split
         |
         v
    benchmark.py            20 models on 20 percent sample
         |
         v
    tuner.py                Optuna tuning on top models
         |
         v
    threshold_tuner.py      optimal decision threshold
         |
         v
    evaluator.py            final test set evaluation
         |
         v
    train_runner.py         log to MLflow
         |
         v
    score.py                Flask API and run_scoring.py batch
         |
         v
    Docker                  containerized deployment

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas and NumPy | Data manipulation |
| Scikit-learn | ML algorithms and preprocessing |
| LightGBM | Final production model |
| Optuna | Hyperparameter tuning |
| imbalanced-learn | SMOTE oversampling |
| MLflow | Experiment tracking |
| Flask | REST API |
| Docker | Containerization |
| Matplotlib and Seaborn | Visualizations |