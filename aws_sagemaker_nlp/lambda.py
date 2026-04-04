import json
import boto3

# Make sure to replace with a running endpoint
# o.w. An error occurred (404) when calling the InvokeEndpoint operation
ENDPOINT = 'news-distilbert-endpoint-v4'

# N.B. AccessDeniedException
# not authorized to perform: sagemaker:InvokeEndpoint on resource


def lambda_handler(event, context):
    sm_client = boto3.client('sagemaker-runtime')
    print(json.dumps(event))
    #body = json.loads(event['body'])

    # Handle both:
    # 1) API Gateway payloads with event["body"]
    # 2) direct JSON objects already at top level
    if "body" in event:
        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)
    else:
        body = event

    headline = body['headline']

    # payload format should match as per `input_fn` setup.
    payload = json.dumps({'inputs': headline})

    response = sm_client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='application/json',
        Body=payload.encode("utf-8"),
    )

    result = json.loads(
        response['Body'].read().decode()
    )
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }


"""
curl -i -X POST \
  https://g454stuhzj.execute-api.us-east-1.amazonaws.com/dev \
  -H "Content-Type: application/json" \
  -d '{"headline":"Global Health Officials Warn of Rising Antibiotic Resistance as New Superbug Cases Emerge"}'

"""
