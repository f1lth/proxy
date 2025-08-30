import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")
HOTKEY = os.environ.get("HOTKEY")

async def test_verify_endpoint():
    """Get test data then call /verify endpoint."""

    # Define request and headers
    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "x-hotkey": HOTKEY
    }
    
    # Get a signed response from the proxy
    async with httpx.AsyncClient() as client:
        proxy_response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request,
            headers=headers
        )
        signed_data = proxy_response.json()
    
    # Send to /verify
    async with httpx.AsyncClient() as client:
        verify_response = await client.post(
            f"{BASE_URL}/verify",
            json=signed_data  # The actual response data
        )
        result = verify_response.json()
        
    if result["valid"]:
        print(f"Valid signature from hotkey: {result['hotkey']}")
    else:
        print(f"Invalid: {result['error']}")

if __name__ == "__main__":
    asyncio.run(test_verify_endpoint())