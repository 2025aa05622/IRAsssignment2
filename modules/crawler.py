"""
crawler.py

Web crawler for the End-to-End Information Retrieval System.
"""

import random
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime
from urllib.parse import urljoin


import requests
from bs4 import BeautifulSoup
from config import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PAGES,
    REQUEST_TIMEOUT,
    USER_AGENT,
)
from modules.database import db
from modules.duplicate_detector import (
    generate_sha256,
    is_near_duplicate,
)

HEADERS = {"User-Agent": USER_AGENT}

import logging

# logging.basicConfig(filename="app.log", level=logging.INFO)
# logging.info("crawler started")


class WebCrawler:

    def __init__(self):

        self.visited_urls = set()
        self.skipped_urls = set()
        self.total_pages = 0

    # ----------------------------------------------------

    def download_page(self, url):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                return None

            # st.subheader("response.text " + response.text)
            # logging.info(response.text)
            return response.text

        except Exception:
            return None

    # ----------------------------------------------------

    def extract_metadata(self, html, url):

        soup = BeautifulSoup(html, "lxml")

        title = ""

        if soup.title:
            title = soup.title.get_text(strip=True)

        author = ""

        author_tag = soup.find("meta", attrs={"name": "author"})

        if author_tag:
            author = author_tag.get("content", "")

        language = ""

        html_tag = soup.find("html")

        if html_tag:
            language = html_tag.get("lang", "")

        publish_date = ""

        date_tag = soup.find(
            "meta",
            attrs={"property": "article:published_time"},
        )

        if date_tag:
            publish_date = date_tag.get("content", "")

        return {
            "url": url,
            "title": title,
            "author": author,
            "language": language,
            "publish_date": publish_date,
        }

    # ----------------------------------------------------

    def extract_text(self, html):

        soup = BeautifulSoup(html, "lxml")

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "aside",
            ]
        ):
            tag.decompose()

        text = soup.get_text(" ")

        text = " ".join(text.split())

        return text

    # ----------------------------------------------------

    def extract_links(self, html, base_url, max_links=5):
        """
        Extract outgoing links from a page.
        Restricts links to max_links randomly selected links.
        """

        soup = BeautifulSoup(html, "html.parser")

        links = set()

        for anchor in soup.find_all("a", href=True):

            href = anchor["href"]

            # Convert relative URL to absolute URL
            url = urljoin(base_url, href)

            parsed = urlparse(url)

            # Keep only http/https links
            if parsed.scheme not in ["http", "https"]:
                continue

            # Remove fragments (#section)
            clean_url = (
                f"{parsed.scheme}://"
                f"{parsed.netloc}"
                f"{parsed.path}"
            )

            links.add(clean_url)


        links = list(links)


        # Randomly select maximum links
        if len(links) > max_links:
            links = random.sample(
                links,
                max_links
            )


        return links

    # ----------------------------------------------------

    def crawl(
        self,
        seed_urls,
        max_depth=DEFAULT_MAX_DEPTH,
        max_pages=DEFAULT_MAX_PAGES,
    ):

        self.visited_urls = set()
        self.total_pages = 0

        queue = deque((url, 0) for url in seed_urls)
        crawled_documents = []

        # logging.info("Crawler started")

        while queue and self.total_pages < max_pages:

            url, depth = queue.popleft()

            if depth > max_depth:
                continue

            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)

            # logging.info(f"Crawling {url} (depth={depth})")

            html = self.download_page(url)

            if html is None:
                continue

            metadata = self.extract_metadata(html, url)
            text = self.extract_text(html)

            if len(text) < 100:
                continue

            # Generates a number from 0 to 5 inclusive
            random_num = random.randint(0, 3)

            # Extract links only once
            links = self.extract_links(html, url, random_num)

            crawled_documents.append(
                {
                    "document_id": self.total_pages + 1,
                    "url": url,
                    "title": metadata["title"],
                    "content": text,
                    "links": links,
                }
            )

            self.total_pages += 1

            # logging.info(f"{url} -> {len(links)} outgoing links")

            if depth < max_depth:

                for link in links:

                    if link not in self.visited_urls:
                        queue.append((link, depth + 1))

        # logging.info(
        #     f"Finished crawling. Documents={len(crawled_documents)}"
        # )

        return {
            "pages": self.total_pages,
            "visited": len(self.visited_urls),
            "Documents": crawled_documents,
        }