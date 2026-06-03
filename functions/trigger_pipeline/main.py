from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Any

from google.cloud import aiplatform


def _event_payload(cloud_event: Any) -> dict[str, Any]:
    message = cloud_event.data.get("message", {}) if hasattr(cloud_event, "data") else {}
    encoded_data = message.get("data")
    if not encoded_data:
        return {}

    decoded = base64.b64decode(encoded_data).decode("utf-8")
    if not decoded:
        return {}
    return json.loads(decoded)


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def trigger_pipeline(cloud_event: Any) -> None:
    """Launch a Vertex AI Pipeline job from a Pub/Sub CloudEvent."""

    payload = _event_payload(cloud_event)
    project_id = _env("PROJECT_ID")
    region = _env("REGION")
    pipeline_root = payload.get("pipeline_root", _env("PIPELINE_ROOT"))
    template_path = payload.get("template_path", _env("TEMPLATE_PATH"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    parameter_values = {
        "project_id": project_id,
        "region": region,
        "bq_source_table": payload.get("bq_source_table", _env("BQ_SOURCE_TABLE")),
        "pipeline_root": pipeline_root,
        "model_display_name": payload.get("model_display_name", _env("MODEL_DISPLAY_NAME")),
        "endpoint_resource_name": payload.get(
            "endpoint_resource_name", _env("ENDPOINT_RESOURCE_NAME")
        ),
        "deploy_threshold": float(payload.get("deploy_threshold", _env("DEPLOY_THRESHOLD"))),
        "metric_name": payload.get("metric_name", _env("METRIC_NAME")),
        "serving_container_image_uri": payload.get(
            "serving_container_image_uri", _env("SERVING_CONTAINER_IMAGE_URI")
        ),
        "service_account": payload.get("service_account", _env("VERTEX_PIPELINE_SA")),
    }

    aiplatform.init(project=project_id, location=region, staging_bucket=pipeline_root)

    job = aiplatform.PipelineJob(
        display_name=f"mlops-triggered-run-{timestamp}",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        enable_caching=False,
    )
    job.submit(service_account=_env("VERTEX_PIPELINE_SA"))

    print(
        json.dumps(
            {
                "message": "submitted Vertex AI Pipeline job",
                "resource_name": job.resource_name,
                "template_path": template_path,
                "parameters": parameter_values,
            }
        )
    )

