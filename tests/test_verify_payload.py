import os
import base64
import httpx
import json
import pytest
from typing import Dict, Any
from dotenv import load_dotenv
# from cryptography.hazmat.primitives import serialization
# from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


load_dotenv()
BASE_URL = "http://localhost:8000"
HOTKEY = os.environ.get("HOTKEY")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")

async def get_public_key() -> Ed25519PublicKey:
    """Get public key from proxy server."""
    #self.proxy_key : bytes = get_proxy_public_key(self.proxy_url)
    #self.public_key = Ed25519PublicKey.from_public_bytes(self.proxy_key)
    async with httpx.AsyncClient(timeout=30.0) as client:
        public_key_response = await client.get(f"{BASE_URL}/public_key")
        public_key_response.raise_for_status()
        public_key_string = json.loads(public_key_response.text)["public_key"]
        raw_bytes = bytes.fromhex(public_key_string)
        return Ed25519PublicKey.from_public_bytes(raw_bytes)
        

async def call_proxy_server(
    request: dict, 
    openrouter_key: str, 
    hotkey: str
) -> Dict[str, Any]:
    """Call the proxy server with a request."""
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "x-hotkey": hotkey,
        "x-provider": "OPENROUTER"
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

def verify_signature(
    response: Dict[str, Any], 
    public_key: Ed25519PublicKey
) -> bool:
    """Verify the signature of the response."""
    proof = response["proof"]
    signature_b64 = response["signature"]

    print(f"Proof: {proof}")
    print(f"Signature (base64): {signature_b64}")

    signature_bytes = base64.b64decode(signature_b64)
    serialized_proof = json.dumps(proof, sort_keys=True).encode()  # Added sort_keys=True

    try:
        public_key.verify(signature_bytes, serialized_proof)
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

@pytest.mark.asyncio
async def test_verify_signature():
    """Fetch the public key, make request, sign, then verify."""
    public_key = await get_public_key()

    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }

    # Returns a SignedResponse object with openrouter package, proof, signature
    response = await call_proxy_server(request, OPENROUTER_KEY, HOTKEY)

    assert verify_signature(response, public_key), "Signature verification failed"


async def main():
    public_key = await get_public_key()

    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }

    # Returns a SignedResponse object with openrouter package, proof, signature
    response = await call_proxy_server(request, OPENROUTER_KEY, HOTKEY)

    assert verify_signature(response, public_key), "Signature verification failed"

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
