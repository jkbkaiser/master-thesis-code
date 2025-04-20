import asyncio
from time import time

import aiohttp

URL = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"  # or any img

async def fetch(session, sem):
    async with sem:
        async with session.get(URL) as r:
            await r.read()

async def main():
    sem = asyncio.Semaphore(300)
    connector = aiohttp.TCPConnector(limit=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, sem) for _ in range(1000)]
        start = time()
        await asyncio.gather(*tasks)
        print(f"Fetched 1000 images in {time() - start:.2f}s")

asyncio.run(main())
