import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

text_to_embed = "What's docling?"
payload = {
    "inputText": text_to_embed,
    "normalize": True,
    "dimensions": 1024
}

response = bedrock_runtime.invoke_model(
    body=json.dumps(payload),
    modelId="amazon.titan-embed-text-v2:0",
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
embedding = response_body.get("embedding")
print(f"👉 Embedding for '{text_to_embed}': {embedding}")
