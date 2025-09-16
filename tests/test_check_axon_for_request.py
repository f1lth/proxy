import os
import time
import json
import httpx
import asyncio
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from app.main import SignedResponse, get_metagraph_data, check_request_ip

async def main():
    global metagraph 
    metagraph = await get_metagraph_data()
    '''{'uid': 253, 'hotkey': '5DPqHhDjKpjiPhBuQxmcHtuKB8dEjKcAXw5ywFSQNqd2x8aB', 'stake': 0.0, 'axon_ip': '195.189.96.71', 'axon_port': 32000}, 
    {'uid': 254, 'hotkey': '5Ci4LgPLz36sviMhVshJ7dwfPvbf8f3jfXY4CTfJmDDFbPNe', 'stake': 0.0, 'axon_ip': '84.32.59.227', 'axon_port': 61060},'''

    print('these IPs should match 16/09/25 should eval true:', await check_request_ip(metagraph, "5DPqHhDjKpjiPhBuQxmcHtuKB8dEjKcAXw5ywFSQNqd2x8aB", "195.189.96.71"))
    print('these IPs should match 16/09/25 should eval true:', await check_request_ip(metagraph, "5Ci4LgPLz36sviMhVshJ7dwfPvbf8f3jfXY4CTfJmDDFbPNe", "84.32.59.227"))
    print('these IPs should do not match 16/09/25 should eval false', await check_request_ip(metagraph, "5Ci4LgPLz36sviMhVshJ7dwfPvbf8f3jfXY4CTfJmDDFbPNe", "83.42.19.117"))

    
if __name__ == "__main__":
    asyncio.run(main())