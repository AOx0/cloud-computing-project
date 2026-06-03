resource "google_logging_metric" "pipeline_failures" {
  name   = "mlops_pipeline_failures"
  filter = "resource.type=\"aiplatform.googleapis.com/PipelineJob\" AND severity>=ERROR"

  metric_descriptor {
    metric_kind  = "DELTA"
    value_type   = "INT64"
    unit         = "1"
    display_name = "MLOps pipeline failures"
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_alert_policy" "pipeline_failure_alert" {
  display_name = "MLOps pipeline failure alert"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Pipeline failures detected"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.pipeline_failures.name}\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  documentation {
    content   = "A Vertex AI Pipeline job emitted an ERROR log. Check Vertex AI Pipelines and Cloud Logging for the failed run."
    mime_type = "text/markdown"
  }

  depends_on = [google_project_service.required_apis]
}

