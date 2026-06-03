resource "google_bigquery_dataset" "mlops" {
  dataset_id                 = var.bq_dataset_name
  location                   = "US"
  delete_contents_on_destroy = true
  labels                     = local.labels

  depends_on = [google_project_service.required_apis]
}

resource "google_bigquery_table" "synthetic_training_data" {
  dataset_id          = google_bigquery_dataset.mlops.dataset_id
  table_id            = "synthetic_training_data"
  deletion_protection = false

  schema = jsonencode([
    { name = "feature_1", type = "FLOAT", mode = "NULLABLE" },
    { name = "feature_2", type = "FLOAT", mode = "NULLABLE" },
    { name = "feature_3", type = "FLOAT", mode = "NULLABLE" },
    { name = "feature_4", type = "FLOAT", mode = "NULLABLE" },
    { name = "label", type = "INTEGER", mode = "NULLABLE" }
  ])
}

resource "google_bigquery_table" "model_metrics" {
  dataset_id          = google_bigquery_dataset.mlops.dataset_id
  table_id            = "model_metrics"
  deletion_protection = false

  schema = jsonencode([
    { name = "run_id", type = "STRING", mode = "NULLABLE" },
    { name = "model_display_name", type = "STRING", mode = "NULLABLE" },
    { name = "metric_name", type = "STRING", mode = "NULLABLE" },
    { name = "metric_value", type = "FLOAT", mode = "NULLABLE" },
    { name = "deploy_threshold", type = "FLOAT", mode = "NULLABLE" },
    { name = "deploy_decision", type = "BOOLEAN", mode = "NULLABLE" },
    { name = "created_at", type = "TIMESTAMP", mode = "NULLABLE" }
  ])
}

