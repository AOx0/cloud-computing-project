from kfp import dsl
from kfp.dsl import Output

from google_cloud_pipeline_components.types import artifact_types


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-pipeline-components>=2.19.0,<3.0.0"],
)
def import_existing_endpoint_component(
    endpoint_resource_name: str,
    endpoint: Output[artifact_types.VertexEndpoint],
) -> None:
    """Create a VertexEndpoint artifact reference for a Terraform-created endpoint."""

    if not endpoint_resource_name:
        raise ValueError("endpoint_resource_name is required for deployment.")

    endpoint.metadata["resourceName"] = endpoint_resource_name
    endpoint.uri = endpoint_resource_name

