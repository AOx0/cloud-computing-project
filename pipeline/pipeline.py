from kfp import dsl

from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp

from pipeline.components.data_validation import validate_data_component
from pipeline.components.evaluate import evaluate_model_component


@dsl.pipeline(name="jigsaw-toxic-comment-mlops-pipeline")
def mlops_pipeline(
    project_id: str,
    region: str,
    gcs_data_uri: str,
    pipeline_root: str,
    model_display_name: str = "jigsaw-toxic-svc-embed",
    endpoint_display_name: str = "jigsaw-toxic-endpoint",
    deploy_threshold: float = 0.95,
    serving_container_image_uri: str = "",
    service_account: str = "",
    synthetic_api_key_secret: str = "synthetic-api-key",
) -> None:
    """MLOps pipeline for Jigsaw Toxic Comment Classification.

    Full workflow:
      1. Validate data from GCS (row count, label presence, no corruption)
      2. Train LinearSVC + char_wb TF-IDF + nomic-embed on Vertex AI custom job
      3. Evaluate model (AUC macro, per-label AUC, deploy gate)
      4. Upload model to Vertex AI Model Registry
      5. Create endpoint
      6. Deploy model to endpoint
    """

    # Step 1: Data validation
    validation_task = validate_data_component(
        project_id=project_id,
        gcs_data_uri=gcs_data_uri,
    )

    # Step 2: Custom training job on Vertex AI
    # This runs our training container which:
    #   - Downloads data from GCS
    #   - Computes TF-IDF char_wb features
    #   - Computes nomic-embed embeddings via Synthetic API
    #   - Trains 6 CalibratedClassifierCV(LinearSVC) models
    #   - Uploads model artifacts to GCS
    training_task = CustomTrainingJobOp(
        project=project_id,
        location=region,
        display_name="jigsaw-toxic-training",
        worker_pool_specs=[
            {
                "containerSpec": {
                    "imageUri": serving_container_image_uri,
                    "args": [
                        "--mode", "train",
                        "--gcs-data-uri", gcs_data_uri,
                        "--gcs-output-uri", f"{pipeline_root}/model_artifacts",
                        "--synthetic-api-key-secret", synthetic_api_key_secret,
                    ],
                    "env": [
                        {"name": "AIP_MODEL_DIR", "value": f"{pipeline_root}/model_artifacts"},
                    ],
                },
                "replicaCount": "1",
                "machineSpec": {
                    "machineType": "n1-standard-4",
                },
            }
        ],
    ).after(validation_task)

    # Step 3: Evaluate model
    evaluate_task = evaluate_model_component(
        project_id=project_id,
        gcs_model_uri=f"{pipeline_root}/model_artifacts",
        gcs_data_uri=gcs_data_uri,
        deploy_threshold=deploy_threshold,
    ).after(training_task)

    # Steps 4-6: Upload, create endpoint, deploy
    with dsl.If(
        evaluate_task.outputs["deploy_decision"] == True,
        name="deploy-if-auc-passes",
    ):
        unmanaged_model = dsl.importer(
            artifact_uri=f"{pipeline_root}/model_artifacts",
            artifact_class=artifact_types.UnmanagedContainerModel,
            metadata={
                "containerSpec": {
                    "imageUri": serving_container_image_uri,
                    "predictRoute": "/predict",
                    "healthRoute": "/health",
                    "ports": [{"containerPort": 8080}],
                    "env": [
                        {"name": "SYNTHETIC_API_KEY", "valueFrom": {"secretKeyRef": {"name": synthetic_api_key_secret, "key": "latest"}}},
                    ],
                },
            },
        )

        upload_task = ModelUploadOp(
            project=project_id,
            location=region,
            display_name=model_display_name,
            description=(
                "LinearSVC + char_wb TF-IDF + nomic-embed for Jigsaw toxic comment "
                "classification. AUC macro 0.9903, trained on 159k Wikipedia comments."
            ),
            unmanaged_container_model=unmanaged_model.outputs["artifact"],
            labels={"source": "mlops_pipeline", "model": "linearsvc_charwb_embed"},
        )

        endpoint_task = EndpointCreateOp(
            project=project_id,
            location=region,
            display_name=endpoint_display_name,
        )

        ModelDeployOp(
            model=upload_task.outputs["model"],
            endpoint=endpoint_task.outputs["endpoint"],
            deployed_model_display_name=f"{model_display_name}-deployed",
            automatic_resources_min_replica_count=1,
            automatic_resources_max_replica_count=1,
            service_account=service_account,
            enable_access_logging=True,
        )
