output "pipeline_bucket" {
  description = "Cloud Storage bucket used for pipeline root, templates, and model artifacts."
  value       = google_storage_bucket.mlops_artifacts.url
}

output "pipeline_root" {
  description = "Vertex AI Pipelines root path."
  value       = var.pipeline_root
}

output "bigquery_source_table" {
  description = "Example BigQuery source table for synthetic data."
  value       = "${var.project_id}.${google_bigquery_dataset.mlops.dataset_id}.${google_bigquery_table.synthetic_training_data.table_id}"
}

output "artifact_registry_repository" {
  description = "Artifact Registry Docker repository."
  value       = google_artifact_registry_repository.mlops_containers.id
}

output "serving_container_image_uri" {
  description = "Default serving image URI used by the pipeline."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops_containers.repository_id}/mlops-serving:latest"
}

output "pubsub_topic" {
  description = "Pub/Sub topic used to trigger retraining."
  value       = google_pubsub_topic.pipeline_trigger.id
}

output "cloud_function_name" {
  description = "Cloud Function that launches Vertex AI Pipeline jobs."
  value       = google_cloudfunctions2_function.trigger_pipeline.name
}

output "vertex_endpoint_resource_name" {
  description = "Vertex AI Endpoint resource name."
  value       = google_vertex_ai_endpoint.mlops_endpoint.id
}

output "vertex_pipeline_service_account" {
  description = "Service account used by Vertex AI Pipelines."
  value       = google_service_account.vertex_pipeline.email
}

output "cloud_build_service_account" {
  description = "Recommended service account for the Cloud Build GitHub trigger."
  value       = google_service_account.cloud_build.email
}

