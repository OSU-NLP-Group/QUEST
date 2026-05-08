import os
from typing import Optional, Dict, Any
import requests
import logging
from litellm import completion

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Read API keys and model configurations from environment variables
READ_API_KEY = os.environ.get("JINA_API_KEY", "")
FACT_Model = os.environ.get("FACT_MODEL", "gpt-5-mini")
Model = os.environ.get("DEFAULT_MODEL", "gpt-5-mini")

class AIClient:

    def __init__(self, api_key: Optional[str] = None, model: str = Model):
        """
        Initialize AI Client with litellm support for multiple models.

        Args:
            api_key: API key (optional, will be read from environment based on model)
            model: Model name in litellm format (e.g., "gemini/gemini-2.5-pro", "gpt-4", "claude-3-sonnet")
        """
        self.model = model
        self.api_key = api_key

        # Store additional API configuration
        self.api_config = {}

    def _generate_vertexai(self, user_prompt: str, system_prompt: str, model_name: str) -> str:
        """Generate text using Vertex AI SDK directly (for vertexai/* models)."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            import google.auth
        except ImportError:
            raise ImportError(
                "vertexai package not installed. Run: pip install google-cloud-aiplatform"
            )

        project_id = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise RuntimeError("Set VERTEXAI_PROJECT or GOOGLE_CLOUD_PROJECT for vertexai/* models.")
        location = os.environ.get("VERTEXAI_LOCATION", "us-central1")

        credentials, _ = google.auth.default(quota_project_id=project_id)
        vertexai.init(project=project_id, location=location, credentials=credentials)

        init_kwargs = {}
        if system_prompt:
            init_kwargs["system_instruction"] = system_prompt

        gemini_model = GenerativeModel(model_name, **init_kwargs)
        response = gemini_model.generate_content(
            user_prompt,
            generation_config=GenerationConfig(temperature=1.0),
        )
        return response.text

    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        """
        Generate text response using litellm, or Vertex AI SDK for vertexai/* models.

        Args:
            user_prompt: User's prompt text
            system_prompt: System prompt (optional)
            model: Override model for this specific call

        Returns:
            Generated text response
        """
        model_to_use = model or self.model

        # Route vertexai/* models to the Vertex AI SDK directly
        if model_to_use.startswith("vertexai/"):
            model_name = model_to_use[len("vertexai/"):]
            try:
                return self._generate_vertexai(user_prompt, system_prompt, model_name)
            except Exception as e:
                raise Exception(f"Failed to generate content with model {model_to_use}: {str(e)}")

        # Build messages
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add user prompt
        messages.append({
            "role": "user",
            "content": user_prompt
        })

        # Prepare litellm call parameters
        call_kwargs = {
            "model": model_to_use,
            "messages": messages,
            "timeout": 600,
            "temperature": 1.0,
            "num_retries": 2
        }

        # Add model-specific API configuration
        if model_to_use.startswith("azure/"):
            # Azure OpenAI configuration
            api_key = self.api_key or os.environ.get("AZURE_API_KEY")
            api_base = os.environ.get("AZURE_API_BASE")
            api_version = os.environ.get("AZURE_API_VERSION")
            if api_key:
                call_kwargs["api_key"] = api_key
            if api_base:
                call_kwargs["api_base"] = api_base
            if api_version:
                call_kwargs["api_version"] = api_version

        elif model_to_use.startswith("bedrock/"):
            # AWS Bedrock configuration
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            aws_region_name = os.environ.get("AWS_REGION_NAME")
            if aws_access_key_id:
                call_kwargs["aws_access_key_id"] = aws_access_key_id
            if aws_secret_access_key:
                call_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_region_name:
                call_kwargs["aws_region_name"] = aws_region_name

        elif model_to_use.startswith("gemini/"):
            # Google Gemini configuration
            api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key

        elif model_to_use.startswith("gpt-") or model_to_use.startswith("openai/"):
            # OpenAI configuration
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key

        elif model_to_use.startswith("claude-") or model_to_use.startswith("anthropic/"):
            # Anthropic configuration
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key

        elif model_to_use.startswith("deepseek/"):
            # DeepSeek configuration
            api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key
        else:
            # For other models, try to use generic API key
            if self.api_key:
                call_kwargs["api_key"] = self.api_key

        try:
            # Call litellm completion
            response = completion(**call_kwargs)
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Failed to generate content with model {model_to_use}: {str(e)}")

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                'Authorization': self.api_key,
                'X-Timeout': "60000",
                "X-With-Generated-Alt": "true",
            }
            response = requests.get(jina_url, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

            response_dict = response.json()

            return {
                'url': response_dict['data']['url'],
                'title': response_dict['data']['title'],
                'description': response_dict['data']['description'],
                'content': response_dict['data']['content'],
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            logger.error(str(e))
            return {
                'url': url,
                'content': '',
                'error': str(e)
            }
        
jina_tool = WebScrapingJinaTool()

def scrape_url(url: str) -> Dict[str, Any]:
    return jina_tool(url)

def call_model(user_prompt: str, model: str = FACT_Model) -> str:
    """
    Convenience function to call a model with a user prompt.

    Args:
        user_prompt: User's prompt text
        model: Model name in litellm format (default: FACT_Model)

    Returns:
        Generated text response
    """
    client = AIClient(model=model)
    return client.generate(user_prompt)

if __name__ == "__main__":
    url = ""
    result = scrape_url(url)
    print(result)
