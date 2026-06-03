# Monitoring Strategy

## Data drift

Data drift happens when the production input distribution changes compared with training data. For example, numeric feature ranges, category frequencies, text length, image quality, or missing-value patterns can shift.

Use Vertex AI Model Monitoring when the final model schema is known. Configure baseline data from training and compare production requests against that baseline.

## Prediction drift

Prediction drift happens when output distributions change. Examples:

- a classifier starts predicting one class too often
- a regressor shifts toward higher or lower values
- an NLP model produces unexpected label distributions

Prediction drift does not always mean the model is wrong, but it is a signal that the environment changed and the model should be reviewed.

## Model degradation

Model degradation happens when real-world performance gets worse over time. It requires ground truth or delayed labels. The architecture should store predictions, labels when available, and evaluation metrics so the team can compare current performance with previous model versions.

## Vertex AI Model Monitoring

Use Vertex AI Model Monitoring for:

- feature skew between training and serving data
- feature drift over time
- alerting when drift exceeds thresholds
- production request sampling

The exact monitoring config depends on the final model type, input schema, and prediction objective.

## Cloud Logging

Cloud Logging should capture:

- Cloud Build logs
- Terraform execution logs
- Cloud Function trigger logs
- Vertex AI Pipeline job logs
- component logs
- endpoint access logs
- prediction container errors

Logs should include run ids, model display names, dataset ids, metric values, deployment decisions, and endpoint resource names.

## Cloud Monitoring alerts

Recommended alerts:

- Vertex AI Pipeline failure
- Cloud Function execution errors
- endpoint 5xx errors
- endpoint latency above SLO
- endpoint traffic drops unexpectedly
- model drift threshold exceeded
- prediction container restart or health check failures

The Terraform prototype creates a basic log-based metric and alert for pipeline failures. Production environments should add notification channels and SLO-based policies.

## Retraining policy

Retraining can be triggered by:

- fixed schedule
- new data availability
- drift alert
- metric degradation
- business event
- manual approval

The demo uses weekly retraining through Cloud Scheduler. A production workflow can combine scheduled retraining with event-based retraining.

## Rollback strategy

Keep previous model versions in Vertex AI Model Registry. If the new model causes errors, latency issues, drift, or business metric degradation:

1. reduce traffic to the new deployed model
2. redeploy or shift traffic to the previous approved model
3. inspect logs and evaluation artifacts
4. block future deployment until the root cause is fixed

For high-risk systems, use canary or shadow deployment before sending all production traffic to the new model.

