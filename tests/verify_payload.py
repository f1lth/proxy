import os
import base64
import httpx
import json
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from get_sig import call_proxy
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

load_dotenv()
BASE_URL = "http://127.0.0.1:8000"
HOTKEY = os.environ.get("HOTKEY")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")

async def get_public_key():
    """Get public key from proxy server."""
    async with httpx.AsyncClient() as client:
        pubkey_response = await client.get(f"{BASE_URL}/pubkey")
        pubkey_response.raise_for_status()
        pubkey_string = json.loads(pubkey_response.text)["PUBKEY"]
        return serialization.load_pem_public_key(pubkey_string.encode())

async def call_proxy_server(
    request: dict, 
    openrouter_key: str, 
    hotkey: str
) -> Dict[str, Any]:
    """Call the proxy server with a request."""
    headers = { 
        "Authorization": f"Bearer {openrouter_key}", 
        "x-hotkey": hotkey
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

def def verify_signature(
    response: Dict[str, Any], 
    public_key: ed25519.Ed25519PublicKey
) -> bool:
    """Verify the signature of the response."""
    proof = response["proof"]
    signature_b64 = response["signature"]

    signature_bytes = base64.b64decode(signature_b64)
    serialized_proof = json.dumps(proof).encode()

    try:
        public_key.verify(signature_bytes, serialized_proof)
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

async def main():
    """Fetch the public key, make request, sign, then verify."""
    public_key = await get_public_key()
    
    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }
    
    # Returns a SignedResponse object with openrouter package, proof, signature
    response = await call_proxy_server(request, OPENROUTER_KEY, HOTKEY)
    
    if verify_signature(response, public_key):
        print("Signature verified!")
    else:
        print("Verification failed!")

if __name__ == "__main__":
    asyncio.run(main())
