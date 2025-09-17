import os
import time
import json
import httpx
import pytest
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from app.main import SignedResponse

load_dotenv()
BASE_URL = "http://127.0.0.1:8000"
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")
HOTKEY = os.environ.get("HOTKEY")

async def call_proxy(
    request: dict, 
    headers: dict
) -> SignedResponse:
    """Call v1/chat/completions with an openai key."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request,
            headers=headers
        )
        return response.json()

@pytest.mark.asyncio
async def test_call_proxy():
    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}",
               "x-hotkey": HOTKEY,
               "x-provider": "OPENROUTER"}

    response = await call_proxy(request, headers)
    assert response is not None