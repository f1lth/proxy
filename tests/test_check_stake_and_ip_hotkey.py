import os
import time
import json
import httpx
import pytest
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from app.main import SignedResponse, get_metagraph_data, check_hotkey_stake, check_request_ip

@pytest.mark.asyncio
async def test_stake_and_ip_checks():
    metagraph = await get_metagraph_data()

    # {'uid': 30, 'hotkey': '5CQMsvYCLACHJVMdcg1sZpDtEnGzVU947fgiwkxC8JYAY5e4', 'stake': 0.0, 'axon_ip': '84.32.64.254', 'axon_port': 53846}
    # {'uid': 53, 'hotkey': '5CXEbmzg7SD9dAsxep8MpjE28PbHxPotE63UnzLqu9VB99Tr', 'stake': 112105.5390625, 'axon_ip': '68.183.201.235', 'axon_port': 8091},
    assert(not await check_hotkey_stake(metagraph, "5CQMsvYCLACHJVMdcg1sZpDtEnGzVU947fgiwkxC8JYAY5e4", 100))
    assert(await check_hotkey_stake(metagraph, "5CXEbmzg7SD9dAsxep8MpjE28PbHxPotE63UnzLqu9VB99Tr", 100))

    #{'uid': 253, 'hotkey': '5DPqHhDjKpjiPhBuQxmcHtuKB8dEjKcAXw5ywFSQNqd2x8aB', 'stake': 0.0, 'axon_ip': '195.189.96.71', 'axon_port': 32000} 
    #{'uid': 254, 'hotkey': '5Ci4LgPLz36sviMhVshJ7dwfPvbf8f3jfXY4CTfJmDDFbPNe', 'stake': 0.0, 'axon_ip': '84.32.59.227', 'axon_port': 61060}
    assert(await check_request_ip(metagraph, "5DPqHhDjKpjiPhBuQxmcHtuKB8dEjKcAXw5ywFSQNqd2x8aB", "195.189.96.71"))
    assert(not await check_request_ip(metagraph, "5Ci4LgPLz36sviMhVshJ7dwfPvbf8f3jfXY4CTfJmDDFbPNe", "83.42.19.117"))


