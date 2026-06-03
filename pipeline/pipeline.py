from kfp import dsl

from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp

from pipeline.components.data_validation import validate_data_component
from pipeline.components.deploy import import_existing_endpoint_component
from pipeline.components.evaluate import evaluate_model_component
from pipeline.components.preprocessing import preprocess_data_component
from pipeline.components.train import train_model_component


@dsl.pipeline(name="model-agnostic-mlops-gcp-pipeline")
def mlops_pipeline(
    project_id: str,
    region: str,
    bq_source_table: str,
    pipeline_root: str,
    model_display_name: str = "mlops-placeholder-sklearn",
    endpoint_display_name: str = "mlops-generic-endpoint",
    endpoint_resource_name: str = "",
    deploy_threshold: float = 0.75,
    metric_name: str = "accuracy",
    label_column: str = "label",
    serving_container_image_uri: str = "",
    service_account: str = "",
) -> None:
    """Model-agnostic MLOps pipeline.

    The `pipeline_root` parameter is passed at job submission time by Vertex AI.
    It is also kept here as an explicit parameter for documentation and
    reproducibility across environments.
    """

    validation_task = validate_data_component(
        project_id=project_id,
        bq_source_table=bq_source_table,
        label_column=label_column,
    )

    preprocess_task = preprocess_data_component(
        project_id=project_id,
        bq_source_table=bq_source_table,
        label_column=label_column,
    ).after(validation_task)

    train_task = train_model_component(
        train_data=preprocess_task.outputs["train_data"],
        label_column=label_column,
    )

    evaluate_task = evaluate_model_component(
        model_artifact=train_task.outputs["model_artifact"],
        test_data=preprocess_task.outputs["test_data"],
        label_column=label_column,
        metric_name=metric_name,
        deploy_threshold=deploy_threshold,
    )

    with dsl.If(evaluate_task.outputs["deploy_decision"] == True, name="deploy-if-metric-passes"):
        unmanaged_model = dsl.importer(
            artifact_uri=train_task.outputs["model_artifact_uri"],
            artifact_class=artifact_types.UnmanagedContainerModel,
            metadata={
                "containerSpec": {
                    "imageUri": serving_container_image_uri,
                    "predictRoute": "/predict",
                    "healthRoute": "/health",
                    "ports": [{"containerPort": 8080}],
                }
            },
        )

        upload_task = ModelUploadOp(
            project=project_id,
            location=region,
            display_name=model_display_name,
            description=(
                "Model uploaded by the model-agnostic academic MLOps pipeline. "
                "The current sklearn model is a replaceable placeholder."
            ),
            unmanaged_container_model=unmanaged_model.outputs["artifact"],
            labels={"source": "mlops_pipeline", "stage": "candidate"},
        )

        endpoint_task = import_existing_endpoint_component(
            endpoint_resource_name=endpoint_resource_name
        )

        ModelDeployOp(
            model=upload_task.outputs["model"],
            endpoint=endpoint_task.outputs["endpoint"],
            deployed_model_display_name=f"{model_display_name}-deployed",
            automatic_resources_min_replica_count=1,
            automatic_resources_max_replica_count=1,
            service_account=service_account,
            enable_access_logging=True,
        ).after(upload_task)
