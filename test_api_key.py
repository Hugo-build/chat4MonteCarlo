#!/usr/bin/env python3
"""
Simple test script to verify if the OpenAI API key in .env works.
"""
import os
import sys
from dotenv import load_dotenv

def test_api_key():
    """Test if the OpenAI API key is valid and working."""
    
    # Load environment variables
    print("=" * 60)
    print("Testing OpenAI API Key from .env")
    print("=" * 60)
    
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Check if API key exists
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("\nPlease make sure you have a .env file with:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Mask API key for display
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
    print(f"\n✓ API Key found: {masked_key}")
    
    # Detect API key type and suggest correct base URL
    key_type = "Unknown"
    expected_base_url = None
    
    if api_key.startswith("sk-or-"):
        key_type = "OpenRouter"
        expected_base_url = "https://openrouter.ai/api/v1"
    elif api_key.startswith("sk-proj-") or api_key.startswith("sk-"):
        key_type = "OpenAI"
        expected_base_url = None  # Use default
    elif api_key.startswith("AIzaSy"):
        key_type = "Google Gemini"
        expected_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    print(f"✓ Detected API Key Type: {key_type}")
    
    if base_url:
        print(f"✓ Custom Base URL: {base_url}")
        
        # Check for mismatch
        if expected_base_url and base_url != expected_base_url:
            print(f"\n⚠️  WARNING: API key type ({key_type}) may not match the base URL!")
            print(f"   Expected base URL for {key_type}: {expected_base_url}")
    else:
        print(f"✓ Using default OpenAI base URL")
        if expected_base_url and key_type != "OpenAI":
            print(f"\n⚠️  WARNING: {key_type} key requires custom base URL!")
            print(f"   Add to .env: OPENAI_BASE_URL={expected_base_url}")
    
    print(f"✓ Model: {model}")
    
    # Try to import openai
    try:
        from openai import OpenAI
        print("\n✓ OpenAI library imported successfully")
    except ImportError:
        print("\n❌ ERROR: OpenAI library not installed")
        print("Please install it with: pip install openai")
        return False
    
    # Test the API with a simple request
    print("\n" + "-" * 60)
    print("Testing API connection...")
    print("-" * 60)
    
    try:
        # Initialize client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = OpenAI(**client_kwargs)
        
        # Make a simple test request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'API test successful' and nothing else."}
            ],
            max_tokens=20
        )
        
        # Check response
        if response and response.choices:
            message = response.choices[0].message.content
            print(f"\n✅ SUCCESS! API is working correctly")
            print(f"\nTest Response: {message}")
            print(f"\nModel used: {response.model}")
            print(f"Tokens used: {response.usage.total_tokens}")
            print("\n" + "=" * 60)
            print("✅ Your API key is valid and working!")
            print("=" * 60)
            return True
        else:
            print("\n⚠️ WARNING: Got a response but it's empty")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: API test failed")
        print(f"\nError message: {str(e)}")
        
        # Check for common misconfigurations
        error_str = str(e).lower()
        
        if "api key not valid" in error_str or "invalid" in error_str:
            print("\n🔍 DIAGNOSIS:")
            if api_key.startswith("sk-or-") and base_url and "openrouter" not in base_url:
                print("  ❌ MISMATCH DETECTED: You're using an OpenRouter API key with a non-OpenRouter endpoint!")
                print("\n  💡 SOLUTION: Update your .env file:")
                print("     OPENAI_BASE_URL=https://openrouter.ai/api/v1")
                print("     OPENAI_MODEL=google/gemini-2.0-flash-exp:free")
                print("\n  📚 See available models: https://openrouter.ai/models")
            elif api_key.startswith("AIzaSy") and base_url and "generativelanguage" not in base_url:
                print("  ❌ MISMATCH DETECTED: You're using a Google API key with a non-Google endpoint!")
                print("\n  💡 SOLUTION: Update your .env file:")
                print("     OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/")
            else:
                print("  Possible reasons:")
                print("    1. Invalid API key")
                print("    2. API key doesn't have proper permissions")
                print("    3. API key/base URL mismatch")
        else:
            print("\nPossible reasons:")
            print("  1. Network connection issues")
            print("  2. Quota exceeded or billing issues")
            print("  3. Service temporarily unavailable")
            if base_url:
                print("  4. Custom base URL might be incorrect")
        
        return False

if __name__ == "__main__":
    success = test_api_key()
    sys.exit(0 if success else 1)

