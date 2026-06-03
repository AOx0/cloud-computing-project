variable "project_id" {
  description = "Google Cloud project id."
  type        = string
}

variable "region" {
  description = "Primary GCP region."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Primary GCP zone."
  type        = string
  default     = "us-central1-a"
}

variable "bucket_name" {
  description = "Globally unique Cloud Storage bucket name for pipeline artifacts."
  type        = string
}

variable "bq_dataset_name" {
  description = "BigQuery dataset for training, validation, metrics, and monitoring tables."
  type        = string
  default     = "mlops_dataset"
}

variable "artifact_registry_name" {
  description = "Artifact Registry repository name for training and serving containers."
  type        = string
  default     = "mlops-containers"
}

variable "vertex_endpoint_name" {
  description = "Numeric Vertex AI Endpoint resource id. Terraform provider requires a numeric endpoint name."
  type        = string
  default     = "1000000001"

  validation {
    condition     = can(regex("^[1-9][0-9]{0,9}$", var.vertex_endpoint_name))
    error_message = "vertex_endpoint_name must be numeric, without leading zeros, and at most 10 digits."
  }
}

variable "service_account_name" {
  description = "Service account id used by Vertex AI Pipelines."
  type        = string
  default     = "mlops-vertex-pipeline"
}

variable "pipeline_root" {
  description = "GCS path used as Vertex AI Pipelines root."
  type        = string
}

variable "scheduler_cron" {
  description = "Cloud Scheduler cron expression for periodic retraining."
  type        = string
  default     = "0 3 * * 1"
}

variable "scheduler_time_zone" {
  description = "Time zone for Cloud Scheduler."
  type        = string
  default     = "America/Mexico_City"
}

variable "deploy_threshold" {
  description = "Default metric threshold used by the pipeline trigger."
  type        = number
  default     = 0.75
}

variable "metric_name" {
  description = "Default metric used by the deployment gate."
  type        = string
  default     = "accuracy"
}

