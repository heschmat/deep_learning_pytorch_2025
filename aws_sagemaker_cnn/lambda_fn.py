import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set this in the Lambda environment variables or hardcode for now
# ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "your-endpoint-name")
ENDPOINT_NAME = 'dog-breed-endpoint-x2'

sagemaker_runtime = boto3.client("sagemaker-runtime")

def lambda_handler(event, context):
    logger.info(f"📦 Event: {json.dumps(event)}")

    try:
        # API Gateway sends body as a stringified JSON by default
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        # You expect exactly this format: {"image_url": "https://..."}
        if "image_url" not in body:
            return _response(400, {"error": "'image_url' is required in payload"})

        payload = json.dumps({"image_url": body["image_url"]})

        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload
        )

        result = response["Body"].read().decode("utf-8")
        logger.info(f"✅ SageMaker Response: {result}")
        return _response(200, json.loads(result))

    except Exception as e:
        logger.exception("❌ Error invoking endpoint")
        return _response(500, {"error": str(e)})

def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }
