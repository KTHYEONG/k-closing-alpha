import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from src.api.kis.client import KisApiClient

async def main():
    async with aiohttp.ClientSession() as session:
        client = KisApiClient()
        await client.ensure_token(session)
        code = "003010" # 혜인
        
        res_detail = await client.get_current_price(session, code)
        print("=== res_detail ===")
        print(res_detail)
        
        res_strength = await client.get_trade_strength(session, code)
        print("=== res_strength ===")
        print(res_strength)

if __name__ == "__main__":
    asyncio.run(main())
