resource "google_cloud_scheduler_job" "weekly_retraining" {
  name        = "mlops-weekly-retraining"
  description = "Weekly trigger for the model-agnostic MLOps pipeline."
  schedule    = var.scheduler_cron
  time_zone   = var.scheduler_time_zone
  region      = var.region

  pubsub_target {
    topic_name = google_pubsub_topic.pipeline_trigger.id
    data = base64encode(jsonencode({
      trigger = "cloud-scheduler"
      reason  = "weekly-retraining"
    }))
  }

  depends_on = [google_project_service.required_apis]
}

