resource "google_storage_bucket" "mlops_artifacts" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
  labels                      = local.labels

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 60
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

