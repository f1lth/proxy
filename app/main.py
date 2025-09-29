import os
import gc
import json
import uuid
import hashlib
import base64
from fastapi.responses import JSONResponse
import httpx
import asyncio
import logging
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import bittensor as bt
from contextlib import asynccontextmanager
from typing import Union, Dict
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Header, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives import serialization


global metagraph

# Simple 5-minute cache for metagraph data
metagraph_cache = None
metagraph_cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes in seconds

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Server starting up")
    logger.info("Creating metagraph data task")
    asyncio.create_task(update_metagraph_data())
    yield
    # Shutdown
    await client.aclose()
    logger.info("Server shutting down")

app = FastAPI(lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

B64_PRIVATE_KEY = os.environ.get("B64_PRIVATE_KEY")
if not B64_PRIVATE_KEY:
    raise ValueError("B64_PRIVATE_KEY environment variable not set")
base64_key = base64.b64decode(B64_PRIVATE_KEY)
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(base64_key)
PUBLIC_KEY = PRIVATE_KEY.public_key()

#private_key_path = os.environ.get("PRIVATE_KEY_PATH")
#public_key_path = os.environ.get("PUBLIC_KEY_PATH")

# with open(private_key_path, "rb") as f:
#     PRIVATE_SIGNING_KEY = serialization.load_pem_private_key(
#         f.read(),
#         password=None,
#     )

# with open(public_key_path, "rb") as f:
#     PUBLIC_SIGNING_KEY = f.read()

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

    model_config = ConfigDict(extra="allow")

class SignedResponse(BaseModel):
    response: dict
    proof: dict
    signature: str
    timestamp: str
    ttl: str


@app.get("/health")
async def health():
    node_count = len(metagraph['uids']) if metagraph else 0
    return {"status": "healthy", "nodes": node_count}
 
# @app.get("/public_key")
# async def public_key(request: Request) -> Dict[str, str]:
#     return {"public_key": PUBLIC_KEY.decode()}

@app.get("/public_key")
@limiter.limit("180/minute")
async def get_public_key(request: Request):
    public_key_raw_bytes = PUBLIC_KEY.public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw
    )
    public_key_hex = public_key_raw_bytes.hex()
    return JSONResponse(status_code=200, content={"public_key": public_key_hex})


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
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Request {request_id} from hotkey: {x_hotkey}, IP: {client_ip}, model: {completion_request.model}")

    # First make sure hotkey has stake in the metagraph, and the request ip matches that hotkey's axon ip
    if 1==2:
        if not await check_hotkey_stake(metagraph, x_hotkey, 100):  # Minimum 100 TAO stake
            logger.warning(f"Hotkey {x_hotkey} does not have sufficient stake in the metagraph")
            raise HTTPException(400, "INVALID REQUEST: INSUFFICIENT STAKE")
    if 1==2:
        if not await check_request_ip(metagraph, x_hotkey, client_ip):
            logger.warning(f"Request IP {client_ip} does not match hotkey {x_hotkey}'s axon IP")
            raise HTTPException(400, "INVALID REQUEST: IP MISMATCH")
    
    try:
        match x_provider:
            case "OPENAI":
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

        # Core proof (what gets signed - NO time data)
        proof = {
            "request_hash": hashlib.sha256(json.dumps(completion_request.dict()).encode()).hexdigest(),
            "response_hash": hashlib.sha256(response.content).hexdigest(),
            "hotkey": x_hotkey,
            "model": completion_request.model,
            "unique_id": request_id
        }

        # Time metadata (NOT signed)
        timestamp = datetime.utcnow().isoformat()
        ttl = (datetime.utcnow() + timedelta(minutes=5)).isoformat()

        # Sign only the core proof
        serialized_proof = json.dumps(proof, sort_keys=True).encode()
        signature = PRIVATE_KEY.sign(serialized_proof)

        logger.info(f"Request {request_id} completed successfully")

        # Return SignedResponse
        return SignedResponse(
            response=response.json(),
            proof=proof,
            signature=base64.b64encode(signature).decode(),
            timestamp=timestamp,
            ttl=ttl
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
        public_key : Ed25519PublicKey = serialization.load_pem_public_key(PUBLIC_KEY)
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
    """Get the metagraph data with 5-minute cache."""
    global metagraph_cache, metagraph_cache_timestamp

    # Check if we have cached data and it's still valid
    current_time = time.time()
    if (metagraph_cache is not None and
        metagraph_cache_timestamp is not None and
        (current_time - metagraph_cache_timestamp) < CACHE_DURATION):
        logger.info("Returning cached metagraph data")
        return metagraph_cache

    try:
        # network = "finney"
        # netuid = 122
        network = "test"
        netuid = 296

        logger.info(f'Fetching fresh metagraph data for {network}:{netuid}...')
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
                    logger.info(f'Error extracting axon info for uid {uid}: {e}')
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

        metagraph_result = {
            'uids': data['uids'],  # original list
            'by_hotkey': {neuron['hotkey']: neuron for neuron in data['uids']},  # O(1) lookup
            'by_ip': {neuron['axon_ip']: neuron for neuron in data['uids']},     # O(1) lookup
            'network_info': data['network_info'],
            'aggregated_stats': data['aggregated_stats']
        }

        # Update cache
        metagraph_cache = metagraph_result
        metagraph_cache_timestamp = time.time()

        return metagraph_result
        
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
        return False
    neuron = metagraph['by_hotkey'].get(hotkey)
    return neuron['axon_ip'] == request_ip if neuron else False