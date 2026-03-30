#!/usr/bin/env python3
"""
Test script to verify different LLM model configurations.
Usage: python test_models.py --model <model_name>
"""

import argparse
import os
from utils.api import AIClient


def test_model(model_name: str):
    """Test a specific model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}\n")

    try:
        # Initialize client
        client = AIClient(model=model_name)
        print(f"✓ Client initialized successfully")

        # Simple test prompt
        test_prompt = "What is 2+2? Please answer with just the number."
        print(f"\nSending test prompt: '{test_prompt}'")

        # Generate response
        response = client.generate(user_prompt=test_prompt)
        print(f"\n✓ Response received:")
        print(f"  {response}")

        print(f"\n{'='*60}")
        print(f"✓ Test passed for {model_name}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed for {model_name}")
        print(f"  Error: {str(e)}")
        print(f"\n{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test LLM model configurations')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name in litellm format (e.g., "gpt-4", "gemini/gemini-2.5-pro")')

    args = parser.parse_args()

    # Print environment info
    print("\nEnvironment Variables Check:")
    print("-" * 60)

    env_vars = [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION_NAME"
    ]

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # Mask API keys for security
            if "KEY" in var or "SECRET" in var:
                masked = value[:8] + "..." if len(value) > 8 else "***"
                print(f"  {var}: {masked}")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: Not set")

    print("-" * 60)

    # Run test
    success = test_model(args.model)

    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
