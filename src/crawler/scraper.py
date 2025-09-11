import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict
import os

BASE_URL = "https://fastapi.tiangolo.com"
RAW_DATA_DIR = "data/raw"

class DocumentPage:
    def __init__(self, url: str, html: str):
        self.url = url
        self.html = html

class FastAPIScraper:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.visited = set()
        self.to_visit = set([base_url])
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

    async def fetch(self, session, url):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    async def crawl(self, max_pages: int = 100):
        async with aiohttp.ClientSession() as session:
            pages = []
            while self.to_visit and len(self.visited) < max_pages:
                url = self.to_visit.pop()
                if url in self.visited:
                    continue
                html = await self.fetch(session, url)
                if html:
                    pages.append(DocumentPage(url, html))
                    self.visited.add(url)
                    self.save_raw_content(url, html)
                    for link in self.extract_links(html):
                        full_url = self.normalize_url(link)
                        if full_url and full_url.startswith(self.base_url) and full_url not in self.visited:
                            self.to_visit.add(full_url)
            return pages

    def extract_links(self, html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        return [a.get("href") for a in soup.find_all("a", href=True)]

    def normalize_url(self, link: str) -> str:
        if link.startswith("http"):
            return link
        if link.startswith("/"):
            return self.base_url + link
        return None

    def save_raw_content(self, url: str, html: str):
        filename = url.replace(self.base_url, "").replace("/", "_").strip("_") or "index"
        path = os.path.join(RAW_DATA_DIR, f"{filename}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

# Example usage (for script):
# if __name__ == "__main__":
#     scraper = FastAPIScraper()
#     asyncio.run(scraper.crawl(max_pages=50))
