"""AI model client implementations for Gemini and OpenAI."""

import os
import json
import asyncio
import logging
import time
import random
import re
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRY_ATTEMPTS = 10
BASE_RETRY_DELAY = 2.0


def is_retryable_error(error_msg: str, error_type: type = None) -> bool:
    """
    Determine if an error is worth retrying based on the error message and type.
    
    Args:
        error_msg: The error message string
        error_type: The exception type (optional)
        
    Returns:
        True if the error should be retried, False otherwise
    """
    error_msg_lower = str(error_msg).lower()
    
    # Retryable HTTP status codes and patterns
    retryable_patterns = [
        r'50[0-9]', r'429', r'502', r'503', r'504',
        'timeout', 'overloaded', 'temporarily', 'unavailable',
        'rate limit', 'quota.*exceeded', 'too many requests',
        'connection', 'network', 'socket', 'dns',
        'internal server error', 'bad gateway', 'service unavailable',
        'gateway timeout', 'request timeout'
    ]
    
    # Non-retryable patterns
    non_retryable_patterns = [
        r'40[0-4]', r'401', r'403', r'404',
        'unauthorized', 'invalid.*key', 'invalid.*token',
        'not found', 'bad request', 'invalid.*request',
        'validation.*error', 'invalid.*format',
        'permission.*denied', 'access.*denied',
        'model.*not.*found', 'does not exist'
    ]
    
    # Check for non-retryable patterns first
    for pattern in non_retryable_patterns:
        if re.search(pattern, error_msg_lower):
            return False
    
    # Check for retryable patterns
    for pattern in retryable_patterns:
        if re.search(pattern, error_msg_lower):
            return True
    
    # Default: retry unknown errors
    return True


async def retry_async(func, *args, max_attempts: int = MAX_RETRY_ATTEMPTS, **kwargs):
    """Async retry decorator with exponential backoff and smart error detection."""
    last_exception = None
    delay = BASE_RETRY_DELAY
    
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                logger.info(f"🔄 Retry attempt {attempt + 1}/{max_attempts} for {func.__name__}")
            
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"✅ Success on attempt {attempt + 1}/{max_attempts}")
            
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            if not is_retryable_error(error_msg, type(e)):
                logger.error(f"❌ Permanent error in {func.__name__}: {error_msg}")
                raise e
            
            if attempt == max_attempts - 1:
                logger.error(f"❌ Final failure in {func.__name__} after {max_attempts} attempts: {error_msg}")
                break
            
            logger.warning(f"⚠️ Attempt {attempt + 1}/{max_attempts} failed in {func.__name__}: {error_msg}")
            logger.info(f"⏳ Waiting {delay:.1f}s before retry...")
            
            await asyncio.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)
    
    raise last_exception


class BaseAIClient(ABC):
    """Abstract base class for AI clients"""
    
    @abstractmethod
    async def generate_with_schema_async(self, user_prompt: str, system_prompt: str = "", schema: Dict = None) -> Dict[str, Any]:
        """Generate response with JSON schema enforcement (async)"""
        pass
    
    @abstractmethod
    async def generate_async(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate text response (async)"""
        pass


class GeminiClient(BaseAIClient):
    """Gemini AI client implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI library not available. Install with: uv add google-genai")

        # Prefer Vertex AI if project is configured (matches DRB eval setup);
        # fall back to API-key auth otherwise.
        vertexai_project = os.environ.get("VERTEXAI_PROJECT")
        vertexai_location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if vertexai_project:
            self.client = genai.Client(
                vertexai=True,
                project=vertexai_project,
                location=vertexai_location,
                http_options={'timeout': 600000},
            )
        elif self.api_key:
            self.client = genai.Client(api_key=self.api_key, http_options={'timeout': 600000})
        else:
            raise ValueError(
                "No Gemini credentials found! Either set VERTEXAI_PROJECT "
                "(for Vertex AI auth via gcloud) or GEMINI_API_KEY."
            )
        self.model = model
        
    def _supports_thinking(self) -> bool:
        """Check if the model supports thinking"""
        thinking_models = ["gemini-2.5-pro", "gemini-2.5-flash"]
        return any(thinking_model in self.model.lower() for thinking_model in thinking_models)
    
    async def generate_with_schema_async(self, user_prompt: str, system_prompt: str = "", schema: Dict = None) -> Dict[str, Any]:
        """Generate response with JSON schema enforcement (async) with retry logic"""
        return await retry_async(self._generate_with_schema_async_impl, user_prompt, system_prompt, schema)
    
    async def _generate_with_schema_async_impl(self, user_prompt: str, system_prompt: str = "", schema: Dict = None) -> Dict[str, Any]:
        """Internal implementation for generate_with_schema_async"""
        schema_prompt = ""
        if schema:
            schema_prompt = f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(schema, indent=2)}"
        
        full_user_prompt = user_prompt + schema_prompt
        response_text = await self.generate_async(full_user_prompt, system_prompt)
        
        try:
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "")
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "")
            
            # Fix common escape sequence issues
            response_text = response_text.replace("\\+", "+").replace("\\&", "&").replace("\\%", "%")
            response_text = response_text.replace("\\:", ":").replace("\\=", "=").replace("\\?", "?")
            response_text = response_text.replace("\\#", "#").replace("\\~", "~")
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            raise
    
    async def generate_async(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate text response (async) with retry logic"""
        return await retry_async(self._generate_async_impl, user_prompt, system_prompt)
    
    async def _generate_async_impl(self, user_prompt: str, system_prompt: str = "") -> str:
        """Internal implementation for generate_async"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, user_prompt, system_prompt)
    
    def _generate_sync(self, user_prompt: str, system_prompt: str = "") -> str:
        """Synchronous generate method"""
        full_prompt = user_prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        contents = [{"role": "user", "parts": [{"text": full_prompt}]}]
        
        try:
            if self._supports_thinking():
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=16000)
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents
                )
            return response.text
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")


class OpenAIClient(BaseAIClient):
    """OpenAI client implementation (supports Azure OpenAI when AZURE_OPENAI_ENDPOINT is set)"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: uv add openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided! Please set OPENAI_API_KEY environment variable.")

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            self.client = openai.AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=azure_endpoint,
            )
            # Azure's `model` parameter is the deployment name, not the upstream model id.
            self.model = os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT") or model
        else:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.model = model
    
    async def generate_with_schema_async(self, user_prompt: str, system_prompt: str = "", schema: Dict = None) -> Dict[str, Any]:
        """Generate response with JSON schema enforcement (async) with retry logic"""
        return await retry_async(self._generate_with_schema_async_impl, user_prompt, system_prompt, schema)
    
    async def _generate_with_schema_async_impl(self, user_prompt: str, system_prompt: str = "", schema: Dict = None) -> Dict[str, Any]:
        """Internal implementation for generate_with_schema_async"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
        
        response_format = None
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "grading_response",
                    "schema": schema
                }
            }
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")
    
    async def generate_async(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate text response (async) with retry logic"""
        return await retry_async(self._generate_async_impl, user_prompt, system_prompt)
    
    async def _generate_async_impl(self, user_prompt: str, system_prompt: str = "") -> str:
        """Internal implementation for generate_async"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")


def create_ai_client(provider: str = "gemini", **kwargs) -> BaseAIClient:
    """Factory function to create AI clients"""
    if provider.lower() == "gemini":
        return GeminiClient(**kwargs)
    elif provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'gemini' or 'openai'")

