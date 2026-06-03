resource "google_artifact_registry_repository" "mlops_containers" {
  location      = var.region
  repository_id = var.artifact_registry_name
  description   = "Docker images for model-agnostic MLOps training and serving."
  format        = "DOCKER"
  labels        = local.labels

  depends_on = [google_project_service.required_apis]
}

