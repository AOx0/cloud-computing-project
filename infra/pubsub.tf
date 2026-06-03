resource "google_pubsub_topic" "pipeline_trigger" {
  name   = "mlops-pipeline-trigger"
  labels = local.labels

  depends_on = [google_project_service.required_apis]
}

