#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# set -e

echo "Starting training job with remote MLflow tracking..."

# --- MLflow Configuration for Remote Server ---

# export MLFLOW_TRACKING_URI="http://your-mlflow-server.com:5000"

# export MLFLOW_TRACKING_USERNAME="your_mlflow_username"
# export MLFLOW_TRACKING_PASSWORD="your_mlflow_password"

# export MLFLOW_TRACKING_TOKEN="your_mlflow_api_token"


# export MLFLOW_S3_ENDPOINT_URL="https://s3.amazonaws.com" # For standard AWS S3, usually not needed if IAM roles are set up
# export MLFLOW_ARTIFACT_LOCATION="s3://your-mlflow-artifacts-bucket/project_name"


# --- AWS Credentials (if artifacts go directly to S3) ---
# export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
# export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
# export AWS_DEFAULT_REGION="us-east-1" # Or your region

# --- Execute training script ---

python3 train_iam.py --exp-name iam \
--max-lr 1e-3 \
--train-bs 128 \
--val-bs 8 \
--weight-decay 0.5 \
--img-size 512 64 \
--total-iter 100000 \
#--data_dir /opt/ml/input/data/training


echo "Training job completed."


#python3 test_iam.py --exp-name iam \
