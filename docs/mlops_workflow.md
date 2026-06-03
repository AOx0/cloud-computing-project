# MLOps Workflow

## 1. Data ingestion

The demo uses a synthetic tabular dataset loaded into BigQuery. In a real project, this step can be replaced with data from transactional systems, data lakes, event streams, image buckets, text corpora, or feature stores.

## 2. Data validation

`validate_data_component` checks basic schema and quality conditions:

- table has rows
- label column exists
- numeric feature columns exist
- missing values are reported

For production, replace this with business validation rules, statistical checks, schema contracts, or tools such as TensorFlow Data Validation or Great Expectations.

## 3. Preprocessing

`preprocess_data_component` fills numeric missing values and creates train/test splits. This can later become image resizing, tokenization, embeddings, feature engineering, windowing, or model-specific transformations.

## 4. Training

`train_model_component` trains a simple sklearn logistic regression model. This is only a placeholder that proves the architecture works.

Replace it later with:

- TensorFlow
- PyTorch
- XGBoost
- computer vision
- NLP
- regression
- classification
- forecasting

## 5. Evaluation

`evaluate_model_component` calculates a generic metric and returns a boolean deployment decision.

Current example:

```text
deploy_decision = metric_value >= deploy_threshold
```

For a real business problem, the threshold must reflect business risk. A fraud model, a medical model, a recommendation model, and a churn model should not use the same acceptance rule.

## 6. Conditional deployment

If the metric passes, Google Cloud Pipeline Components upload the model to Vertex AI Model Registry and deploy it to Vertex AI Endpoint. If the metric fails, the pipeline saves the evaluation report and stops before deployment.

## 7. Serving

The serving image is a FastAPI container that follows Vertex AI custom prediction container conventions:

- `/health`
- `/predict`

The current implementation loads `model.joblib`. A real model can replace the predictor class and keep the same deployment architecture.

## 8. Retraining

Cloud Scheduler publishes a weekly Pub/Sub event. Cloud Function receives the event and submits a Vertex AI Pipeline job. The Pub/Sub message can override runtime parameters such as dataset, threshold, metric, image, or endpoint.

