# OpenRouter Proxy - Quick Setup

## Generate ED25519 Keys
```bash
openssl genpkey -algorithm Ed25519 -out private.pem
openssl pkey -in private.pem -pubout -out public.pem
```

## Configure Environment

### `proxy/.env` (for the server)
```
private=./private.pem
public=./public.pem
```

### `proxy/tests/.env` (for testing)
```
OPENROUTER_KEY=sk-or-v1-xxxxx
OPENAI_KEY=sk-xxxxx
HOTKEY=your-miner-hotkey
```

## Run
```bash
cd proxy
uv sync
uv run uvicorn app.main:app --reload
```

## Test
```bash
uv run tests/get_sig.py      # Test OpenRouter signing
uv run tests/try_verify.py   # Verify signatures
```

## Endpoints
Server runs on `http://127.0.0.1:8000`
* `POST /v1/chat/completions` - Proxy endpoint
* `GET /pubkey` - Get public key