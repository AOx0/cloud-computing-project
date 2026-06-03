resource "google_vertex_ai_endpoint" "mlops_endpoint" {
  name         = var.vertex_endpoint_name
  display_name = "mlops-generic-endpoint"
  description  = "Reusable Vertex AI endpoint for the currently approved model."
  location     = var.region
  region       = var.region
  labels       = local.labels

  depends_on = [google_project_service.required_apis]
}

