import json
import time
import boto3
from typing import Optional
from src.llm.base import LLMProvider
from config.settings import AWS_REGION, BEDROCK_MODEL_ID
from src.logger import setup_logger

logger = setup_logger(__name__)

class BedrockProvider(LLMProvider):
    """Wrapper for AWS Bedrock models."""
    
    def __init__(self, model_id: str = BEDROCK_MODEL_ID, region: str = AWS_REGION):
        self.model_id = model_id
        self.region = region
        try:
            self.client = boto3.client("bedrock-runtime", region_name=self.region)
            logger.info(f"Initialized BedrockProvider with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        start_time = time.time()
        
        message_list = [{"role": "user", "content": [{"text": prompt}]}]
        
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": {"maxTokens": 300, "topP": 0.1, "topK": 20, "temperature": 0},
        }

        if system_prompt:
             request_body["system"] = [{"text": system_prompt}]
             
        try:
            logger.info(f"Invoking Bedrock model {self.model_id}...")
            response = self.client.invoke_model(
                modelId=self.model_id, 
                body=json.dumps(request_body)
            )
            result = json.loads(response["body"].read())
            elapsed = time.time() - start_time
            logger.info(f"Received response from Bedrock in {elapsed:.2f}s")
            return result['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {e}")
            raise 

    def evaluate(self, prompt: str) -> str:
        """Simple invocation for evaluation tasks (no system prompt usually)."""
        # Could reuse generate, but sometimes eval needs less overhead/different params
        return self.generate(prompt)
