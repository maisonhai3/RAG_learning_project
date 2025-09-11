import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.crawler.scraper import FastAPIScraper
import pytest
import asyncio


@pytest.mark.asyncio
async def test_crawl_fastapi_docs():
    scraper = FastAPIScraper()
    pages = await scraper.crawl(max_pages=2)
    assert len(pages) > 0
    for page in pages:
        assert page.url.startswith(scraper.base_url)
        assert '<html' in page.html.lower()
