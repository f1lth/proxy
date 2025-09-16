import os
import time
import json
import httpx
import asyncio
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from app.main import SignedResponse, get_metagraph_data, check_hotkey_stake

async def main():
    global metagraph 
    metagraph = await get_metagraph_data()
    # {'uid': 30, 'hotkey': '5CQMsvYCLACHJVMdcg1sZpDtEnGzVU947fgiwkxC8JYAY5e4', 'stake': 0.0, 'axon_ip': '84.32.64.254', 'axon_port': 53846}
    # {'uid': 53, 'hotkey': '5CXEbmzg7SD9dAsxep8MpjE28PbHxPotE63UnzLqu9VB99Tr', 'stake': 112105.5390625, 'axon_ip': '68.183.201.235', 'axon_port': 8091},
    print('Should be false random miner hotkey does not have 100 stake:', await check_hotkey_stake(metagraph, "5CQMsvYCLACHJVMdcg1sZpDtEnGzVU947fgiwkxC8JYAY5e4", 100))
    print('Should be true uid53 our bitrecs val hotkey has 100 stake:', await check_hotkey_stake(metagraph, "5CXEbmzg7SD9dAsxep8MpjE28PbHxPotE63UnzLqu9VB99Tr", 100))

    
if __name__ == "__main__":
    asyncio.run(main())