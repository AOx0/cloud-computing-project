resource "google_service_account" "vertex_pipeline" {
  account_id   = var.service_account_name
  display_name = "MLOps Vertex AI Pipeline Service Account"
  description  = "Runtime identity used by Vertex AI Pipelines components."

  depends_on = [google_project_service.required_apis]
}

resource "google_service_account" "cloud_function" {
  account_id   = "mlops-trigger-function"
  display_name = "MLOps Pipeline Trigger Function Service Account"
  description  = "Runtime identity used by Cloud Function to launch Vertex AI Pipeline jobs."

  depends_on = [google_project_service.required_apis]
}

resource "google_service_account" "cloud_build" {
  account_id   = "mlops-cloud-build"
  display_name = "MLOps Cloud Build Service Account"
  description  = "Identity recommended for the GitHub Cloud Build trigger."

  depends_on = [google_project_service.required_apis]
}

locals {
  vertex_pipeline_project_roles = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataViewer",
    "roles/bigquery.jobUser",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])

  cloud_function_project_roles = toset([
    "roles/aiplatform.user",
    "roles/logging.logWriter"
  ])

  cloud_build_project_roles = toset([
    "roles/aiplatform.admin",
    "roles/artifactregistry.admin",
    "roles/bigquery.admin",
    "roles/cloudbuild.builds.builder",
    "roles/cloudfunctions.developer",
    "roles/cloudscheduler.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",
    "roles/logging.configWriter",
    "roles/monitoring.admin",
    "roles/pubsub.admin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/serviceusage.serviceUsageAdmin",
    "roles/storage.admin"
  ])
}

resource "google_project_iam_member" "vertex_pipeline_roles" {
  for_each = local.vertex_pipeline_project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.vertex_pipeline.email}"
}

resource "google_project_iam_member" "cloud_function_roles" {
  for_each = local.cloud_function_project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_function.email}"
}

resource "google_project_iam_member" "cloud_build_roles" {
  for_each = local.cloud_build_project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_storage_bucket_iam_member" "vertex_pipeline_artifacts" {
  bucket = google_storage_bucket.mlops_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vertex_pipeline.email}"
}

resource "google_storage_bucket_iam_member" "cloud_function_template_reader" {
  bucket = google_storage_bucket.mlops_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_function.email}"
}

resource "google_service_account_iam_member" "function_can_use_vertex_pipeline_sa" {
  service_account_id = google_service_account.vertex_pipeline.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cloud_function.email}"
}

resource "google_service_account_iam_member" "cloud_build_can_use_vertex_pipeline_sa" {
  service_account_id = google_service_account.vertex_pipeline.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_service_account_iam_member" "cloud_build_can_use_function_sa" {
  service_account_id = google_service_account.cloud_function.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cloud_build.email}"
}

