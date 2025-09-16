import os
import gc
import json
import uuid
import hashlib
import base64
import httpx
import uvicorn
import asyncio
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

import bittensor as bt
from bitrecs.llms.factory import LLM, LLMFactory

global metagraph

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
async def startup():
    asyncio.create_task(update_metagraph_data())

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
    
    # first make sure hotkey has stake in the metagraph , and the request ip matches that hotkeys's axon ip
    if not await check_hotkey_stake(x_hotkey):
        logger.warning(f"Hotkey {x_hotkey} does not have stake in the metagraph")
        raise HTTPException(400, "INVALID REQUEST HOTKEY MISMATCH")
    if not await check_request_ip(request.client.host, x_hotkey):
        logger.warning(f"Request IP {request.client.host} does not match hotkey {x_hotkey}'s axon IP")
        raise HTTPException(400, "INVALID REQUEST IP MISMATCH")

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
            case "GROQ":
                url = "https://api.groq.com/openai/v1/chat/completions"
            case "CEREBRAS":
                url = "https://api.cerebras.ai/v1/chat/completions"
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


async def update_metagraph_data():
    while True:
        global metagraph
        result = await get_metagraph_data()
        if result is not None:
            metagraph = result
        await asyncio.sleep(300)


async def get_metagraph_data() -> dict:
    """Get the metagraph data."""
    try:
        network = "finney"
        netuid = 122

        logger.info(f'Fetching metagraph data for {network}:{netuid}...')
        subnet = bt.metagraph(netuid=netuid, network=network)

        # Extract all relevant data from metagraph
        data = {
            'uids': [],
            'network_info': {
                'netuid': netuid,
                'network': network,
                'block': int(subnet.block) if hasattr(subnet, 'block') else None,
                'total_neurons': len(subnet.uids),
                'timestamp': datetime.now().isoformat()
            },
            'aggregated_stats': {}
        }

        for i, uid in enumerate(subnet.uids.tolist()):
            try:
                # Safer data extraction with bounds checking
                def safe_get_value(tensor, uid, default=0.0):
                    try:
                        if tensor is not None and len(tensor) > uid:
                            value = tensor[uid]
                            return float(value) if not np.isnan(value) and np.isfinite(value) else default
                    except Exception:
                        pass
                    return default
                
                try:
                    stake = float(subnet.S[uid])
                except Exception:
                    stake = 0.0

                hotkey = ''
                coldkey = ''
                axon_ip = ''
                axon_port = 0
                
                try:
                    if hasattr(subnet, 'hotkeys') and len(subnet.hotkeys) > uid:
                        hotkey = str(subnet.hotkeys[uid])
                except Exception:
                    pass
                
                try:
                    if hasattr(subnet, 'coldkeys') and len(subnet.coldkeys) > uid:
                        coldkey = str(subnet.coldkeys[uid])
                except Exception:
                    pass
                
                # Extract axon information 
                try:
                    if hasattr(subnet, 'axons') and len(subnet.axons) > uid:
                        axon = subnet.axons[uid]
                        if hasattr(axon, 'ip'):
                            axon_ip = str(axon.ip)
                        if hasattr(axon, 'port'):
                            axon_port = int(axon.port) if axon.port is not None else 0
                except Exception as e:
                    logger.debug(f'Error extracting axon info for uid {uid}: {e}')
                    pass
                
                neuron_data = {
                    'uid': int(uid),
                    'hotkey': hotkey,
                    'stake': stake,
                    'axon_ip': axon_ip,
                    'axon_port': axon_port,
                }
                data['uids'].append(neuron_data)
                
            except Exception as e:
                logger.error(f'Error processing uid {uid}: {e}')
                continue

        total_neurons = len(data['uids'])
        data['aggregated_stats'] = {
            'total_neurons': total_neurons,
        }
        
        logger.info(f'Successfully processed {total_neurons} neurons')
        gc.collect()

        metagraph = {
            'uids': data['uids'],  # original list
            'by_hotkey': {neuron['hotkey']: neuron for neuron in data['uids']},  # O(1) lookup
            'by_ip': {neuron['axon_ip']: neuron for neuron in data['uids']},     # O(1) lookup
            'network_info': data['network_info'],
            'aggregated_stats': data['aggregated_stats']
        }
        
        return metagraph
        
    except Exception as e:
        logger.error(f'Error fetching metagraph data: {e}')
        logger.error(f'Traceback: {traceback.format_exc()}')
        return None


async def check_hotkey_stake(
    metagraph: dict, 
    hotkey: str, 
    stake: float
) -> bool:
    """Check if hotkey has stake in the metagraph."""
    if metagraph is None or hotkey is None or stake is None:
        return False
    neuron = metagraph['by_hotkey'].get(hotkey)
    return neuron['stake'] > stake if neuron else False

async def check_request_ip(
    metagraph: dict, 
    hotkey: str, 
    request_ip: str, 
) -> bool:
    """Check if request IP matches hotkey's axon IP."""
    if metagraph is None or hotkey is None or request_ip is None:
        return false
    neuron = metagraph['by_hotkey'].get(hotkey)
    return neuron['axon_ip'] == request_ip if neuron else False