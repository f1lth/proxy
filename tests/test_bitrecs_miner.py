import os
import time
import json
import httpx
import asyncio
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from app.main import SignedResponse


load_dotenv()
BASE_URL = "http://127.0.0.1:8000"
OPENAI_KEY = os.environ.get("OPENAI_KEY")
HOTKEY = os.environ.get("HOTKEY")

async def call_proxy(
    request: dict, 
    headers: dict
) -> SignedResponse:
    """Call v1/chat/completions with an openai key."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request,
            headers=headers
        )
        return response.json()

async def main():
    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", 
               "x-hotkey": HOTKEY, 
               "x-provider": "CHAT_GPT"}
    
    response = await call_proxy(request, headers)
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(main())