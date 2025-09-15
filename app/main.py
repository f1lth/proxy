import os
import json
import uuid
import hashlib
import base64
import httpx
import uvicorn
import logging
from typing import Union, Dict
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Header, HTTPException
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from bitrecs.llms.factory import LLM, LLMFactory


load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

private_key_path = os.environ.get("PRIVATE_KEY_PATH")
public_key_path = os.environ.get("PUBLIC_KEY_PATH")

with open(private_key_path, "rb") as f:
    PRIVATE_SIGNING_KEY = serialization.load_pem_private_key(
        f.read(),
        password=None,
    )

with open(public_key_path, "rb") as f:
    PUBLIC_SIGNING_KEY = f.read()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    
    # Optional params
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    
    # OpenRouter params
    provider: dict | None = None
    transforms: list[str] | None = None
    route: str | None = None
    class Config:
        extra = "allow"

class SignedResponse(BaseModel):
    response: dict
    proof: dict
    signature: str

@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
    logger.info("Server shutting down")

@app.get("/health")
async def health():
    return {"status": "healthy"}
 
@app.get("/pubkey")
async def pubkey(request: Request) -> Dict[str, str]:
    return {"PUBKEY": PUBLIC_SIGNING_KEY.decode()}

# @app.post("/v1/chat/completions", response_model=SignedResponse)
# @limiter.limit("60/minute")
# async def forward_proxy_request(
#     request: Request,
#     completion_request: ChatCompletionRequest,
#     authorization: str = Header(),  
#     x_hotkey: str = Header()
# ) -> SignedResponse:
#     request_id = str(uuid.uuid4())
#     logger.info(f"Request {request_id} from hotkey: {x_hotkey}, model: {completion_request.model}")
    
#     try:
#         # Filter openai or openrouter
#         if authorization.startswith("Bearer sk-or-"):
#             url = "https://openrouter.ai/api/v1/chat/completions"
#         elif authorization.startswith("Bearer sk-"):
#             url = "https://api.openai.com/v1/chat/completions"
#         else:
#             logger.warning(f"Unknown API key format for request {request_id}")
#             raise HTTPException(400, "Unknown API key format")

#         payload = completion_request.dict(exclude_unset=True)

#         # Send the request to openai or openrouter
#         response = await client.post(
#             url,
#             json=payload,
#             headers={"Authorization": authorization}
#         )
        
#         if response.status_code != 200:
#             logger.error(f"Upstream error for request {request_id}: {response.status_code}")
#             raise HTTPException(status_code=response.status_code, detail=response.text)

#         # Form the proof payload
#         proof = {}
#         proof["timestamp"] = datetime.utcnow().isoformat()        
#         proof["request_hash"] = hashlib.sha256(json.dumps(completion_request.dict()).encode()).hexdigest()
#         proof["response_hash"] = hashlib.sha256(response.content).hexdigest()
#         proof["hotkey"] = x_hotkey
#         proof["model"] = completion_request.model
#         proof["unique_id"] = request_id

#         # Sign the proof
#         serialized_proof = json.dumps(proof).encode()
#         signature = PRIVATE_SIGNING_KEY.sign(serialized_proof)
        
#         logger.info(f"Request {request_id} completed successfully")
        
#         # Return SignedResponse
#         return SignedResponse(
#             response=response.json(),
#             proof=proof,
#             signature=base64.b64encode(signature).decode()
#         )

#     except httpx.TimeoutException:
#         logger.error(f"Timeout for request {request_id}")
#         raise HTTPException(504, "Upstream timeout")
#     except httpx.HTTPError as e:
#         logger.error(f"HTTP error for request {request_id}: {str(e)}")
#         raise HTTPException(502, f"Upstream error: {str(e)}")
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         logger.error(f"Unexpected error for request {request_id}: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/v1/chat/completions", response_model=SignedResponse)
@limiter.limit("60/minute")
async def forward_proxy_request(
    request: Request,
    completion_request: ChatCompletionRequest,
    authorization: str = Header(),  
    x_hotkey: str = Header(),
    x_provider: str = Header()
) -> SignedResponse:
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} from hotkey: {x_hotkey}, model: {completion_request.model}")
    
    try:

        match x_provider:
            case "CHAT_GPT":
                url = "https://api.openai.com/v1/chat/completions"
            case "OPENROUTER":
                url = "https://openrouter.ai/api/v1/chat/completions"
            case "GEMINI":
                url = "https://generativelanguage.googleapis.com/v1beta/openai"
            case "CHUTES":
                url = "https://llm.chutes.ai/v1/chat/completions"
            case _:
                logger.warning(f"Unknown provider for request {request_id}")
                raise HTTPException(400, "Unknown provider")

        payload = completion_request.dict(exclude_unset=True)

        # Send the request to openai or openrouter
        response = await client.post(
            url,
            json=payload,
            headers={"Authorization": authorization}
        )
        
        if response.status_code != 200:
            logger.error(f"Upstream error for request {request_id}: {response.status_code}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Form the proof payload
        proof = {}
        proof["timestamp"] = datetime.utcnow().isoformat()        
        proof["request_hash"] = hashlib.sha256(json.dumps(completion_request.dict()).encode()).hexdigest()
        proof["response_hash"] = hashlib.sha256(response.content).hexdigest()
        proof["hotkey"] = x_hotkey
        proof["model"] = completion_request.model
        proof["unique_id"] = request_id

        # Sign the proof
        serialized_proof = json.dumps(proof).encode()
        signature = PRIVATE_SIGNING_KEY.sign(serialized_proof)
        
        logger.info(f"Request {request_id} completed successfully")
        
        # Return SignedResponse
        return SignedResponse(
            response=response.json(),
            proof=proof,
            signature=base64.b64encode(signature).decode()
        )

    except httpx.TimeoutException:
        logger.error(f"Timeout for request {request_id}")
        raise HTTPException(504, "Upstream timeout")
    except httpx.HTTPError as e:
        logger.error(f"HTTP error for request {request_id}: {str(e)}")
        raise HTTPException(502, f"Upstream error: {str(e)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/verify")
async def verify_endpoint(
    response: SignedResponse
) -> Dict[str, Union[bool, str]]:
    try:
        public_key = serialization.load_pem_public_key(PUBLIC_SIGNING_KEY)
        public_key.verify(
            base64.b64decode(response.signature), 
            json.dumps(response.proof).encode()
        )
        return {
            "valid": True,
            "hotkey": response.proof.get("hotkey"),
            "timestamp": response.proof.get("timestamp"),
            "model": response.proof.get("model")
        }
    except Exception as e:
        logger.warning(f"Verification failed: {str(e)}")
        return {
            "valid": False,
            "error": "Invalid signature"
        }