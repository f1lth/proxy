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
from app.main import get_metagraph_data

async def main():
    mg = await get_metagraph_data()
    print(mg)

if __name__ == "__main__":
    asyncio.run(main())