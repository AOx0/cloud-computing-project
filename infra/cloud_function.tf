data "archive_file" "trigger_pipeline_source" {
  type        = "zip"
  source_dir  = "${path.module}/../functions/trigger_pipeline"
  output_path = "${path.module}/trigger_pipeline_source.zip"
}

resource "google_storage_bucket_object" "trigger_pipeline_source" {
  name   = "functions/trigger_pipeline/source-${data.archive_file.trigger_pipeline_source.output_md5}.zip"
  bucket = google_storage_bucket.mlops_artifacts.name
  source = data.archive_file.trigger_pipeline_source.output_path
}

resource "google_cloudfunctions2_function" "trigger_pipeline" {
  name        = "mlops-trigger-pipeline"
  location    = var.region
  description = "Starts a Vertex AI Pipeline job when Pub/Sub receives a retraining event."

  build_config {
    runtime     = "python311"
    entry_point = "trigger_pipeline"
    source {
      storage_source {
        bucket = google_storage_bucket.mlops_artifacts.name
        object = google_storage_bucket_object.trigger_pipeline_source.name
      }
    }
  }

  service_config {
    max_instance_count             = 2
    available_memory               = "512M"
    timeout_seconds                = 540
    service_account_email          = google_service_account.cloud_function.email
    all_traffic_on_latest_revision = true

    environment_variables = {
      PROJECT_ID                  = var.project_id
      REGION                      = var.region
      TEMPLATE_PATH               = "gs://${google_storage_bucket.mlops_artifacts.name}/pipeline_templates/mlops_pipeline.json"
      PIPELINE_ROOT               = var.pipeline_root
      BQ_SOURCE_TABLE             = "${var.project_id}.${google_bigquery_dataset.mlops.dataset_id}.${google_bigquery_table.synthetic_training_data.table_id}"
      MODEL_DISPLAY_NAME          = "mlops-placeholder-sklearn"
      ENDPOINT_RESOURCE_NAME      = google_vertex_ai_endpoint.mlops_endpoint.id
      VERTEX_PIPELINE_SA          = google_service_account.vertex_pipeline.email
      SERVING_CONTAINER_IMAGE_URI = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops_containers.repository_id}/mlops-serving:latest"
      DEPLOY_THRESHOLD            = tostring(var.deploy_threshold)
      METRIC_NAME                 = var.metric_name
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.pipeline_trigger.id
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  depends_on = [
    google_project_service.required_apis,
    google_project_iam_member.cloud_function_roles,
    google_service_account_iam_member.function_can_use_vertex_pipeline_sa
  ]
}

