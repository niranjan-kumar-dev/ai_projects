"""Polite same-domain website crawler.

Breadth-first search starting from one URL:

1. Read robots.txt (and obey it, unless disabled).
2. Seed the queue with sitemap.xml URLs when available, otherwise the start URL.
3. Pop a URL, fetch it, keep it if it is HTML, push its same-domain links.
4. Stop at `max_pages` or when the queue is empty.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Iterator
from urllib import robotparser
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chat_with_website.config import settings
from chat_with_website.ingestion.models import CrawledPage
from chat_with_website.ingestion.url_utils import (
    has_skipped_extension,
    is_same_domain,
    normalize_url,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]  # (message, done, total)


class Crawler:
    MAX_BODY_BYTES = 5 * 1024 * 1024  # ignore HTML documents above 5 MB

    def __init__(
        self,
        start_url: str,
        max_pages: int | None = None,
        max_depth: int | None = None,
        delay: float | None = None,
        respect_robots: bool | None = None,
        timeout: float | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.start_url = normalize_url(start_url)
        if self.start_url is None:
            raise ValueError(f"Invalid start URL: {start_url}")
        self.max_pages = max_pages or settings.crawl_max_pages
        self.max_depth = max_depth if max_depth is not None else settings.crawl_max_depth
        self.delay = delay if delay is not None else settings.crawl_delay_seconds
        self.respect_robots = respect_robots if respect_robots is not None else settings.crawl_respect_robots
        self.timeout = timeout or settings.crawl_timeout_seconds
        self.user_agent = user_agent or settings.user_agent

        self.session = self._build_session()
        self.robots = self._load_robots() if self.respect_robots else None
        self.seen: set[str] = set()

    # ------------------------------------------------------------------ setup
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.8",
        })
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _site_root(self) -> str:
        parts = urlsplit(self.start_url)
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

    def _load_robots(self) -> robotparser.RobotFileParser | None:
        rp = robotparser.RobotFileParser()
        robots_url = self._site_root() + "/robots.txt"
        try:
            resp = self.session.get(robots_url, timeout=self.timeout)
            if resp.status_code >= 400:
                log.debug("No robots.txt at %s (status %s)", robots_url, resp.status_code)
                return None
            rp.parse(resp.text.splitlines())
            log.debug("Loaded robots.txt from %s", robots_url)
            return rp
        except requests.RequestException as exc:
            log.debug("Could not fetch robots.txt: %s", exc)
            return None

    def _allowed(self, url: str) -> bool:
        if self.robots is None:
            return True
        return self.robots.can_fetch(self.user_agent, url)

    # ---------------------------------------------------------------- sitemap
    def _sitemap_urls(self) -> list[str]:
        """Collect page URLs from sitemap.xml (and nested sitemaps), best-effort."""
        candidates = []
        if self.robots is not None and self.robots.site_maps():
            candidates.extend(self.robots.site_maps())
        candidates.append(self._site_root() + "/sitemap.xml")

        urls: list[str] = []
        visited_maps: set[str] = set()
        queue = deque(candidates)
        while queue and len(urls) < self.max_pages * 4:
            sm = queue.popleft()
            if sm in visited_maps:
                continue
            visited_maps.add(sm)
            try:
                resp = self.session.get(sm, timeout=self.timeout)
                looks_like_xml = (
                    "xml" in resp.headers.get("Content-Type", "").lower()
                    or resp.text.lstrip().startswith("<?xml")
                )
                if resp.status_code != 200 or not looks_like_xml:
                    continue
                root = ElementTree.fromstring(resp.content)
            except (requests.RequestException, ElementTree.ParseError) as exc:
                log.debug("Sitemap %s unusable: %s", sm, exc)
                continue

            tag = root.tag.lower()
            for loc in root.iter():
                if not loc.tag.lower().endswith("loc") or not loc.text:
                    continue
                target = loc.text.strip()
                if tag.endswith("sitemapindex"):
                    queue.append(target)
                else:
                    urls.append(target)
        if urls:
            log.info("Found %d URLs in sitemap(s)", len(urls))
        return urls

    # ------------------------------------------------------------------ crawl
    def _fetch(self, url: str) -> requests.Response | None:
        try:
            # stream=True lets us look at the headers before downloading the body
            resp = self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True)
            ctype = resp.headers.get("Content-Type", "").lower()
            length = int(resp.headers.get("Content-Length") or 0)
            if resp.status_code != 200 or not ("text/html" in ctype or "application/xhtml" in ctype):
                log.debug("Skipping %s (status %s, type %s)", url, resp.status_code, ctype or "?")
                resp.close()
                return None
            if length > self.MAX_BODY_BYTES:
                log.debug("Skipping %s (body %d bytes too large)", url, length)
                resp.close()
                return None
            _ = resp.content  # download the body now that we know it is HTML
            return resp
        except requests.RequestException as exc:
            log.warning("Fetch failed %s: %s", url, exc)
            return None

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            link = normalize_url(a["href"], base=base_url)
            if link is None or has_skipped_extension(link) or not is_same_domain(link, self.start_url):
                continue
            links.append(link)
        return links

    def crawl(self, progress: ProgressCallback | None = None) -> Iterator[CrawledPage]:
        """Yield crawled HTML pages until the page limit is reached."""
        queue: deque[tuple[str, int]] = deque()
        for u in self._sitemap_urls():
            n = normalize_url(u)
            if n and is_same_domain(n, self.start_url) and not has_skipped_extension(n):
                queue.append((n, 0))
        queue.appendleft((self.start_url, 0))

        pages = 0
        while queue and pages < self.max_pages:
            url, depth = queue.popleft()
            if url in self.seen:
                continue
            self.seen.add(url)
            if not self._allowed(url):
                log.info("robots.txt disallows %s", url)
                continue

            if progress:
                progress(f"Fetching {url}", pages, self.max_pages)
            resp = self._fetch(url)
            if resp is None:
                continue
            final_url = normalize_url(resp.url) or url
            if final_url != url:
                if final_url in self.seen or not is_same_domain(final_url, self.start_url):
                    continue
                self.seen.add(final_url)

            html = resp.text
            pages += 1
            yield CrawledPage(url=final_url, html=html, status_code=resp.status_code, depth=depth)

            if depth < self.max_depth:
                for link in self._extract_links(html, final_url):
                    if link not in self.seen:
                        queue.append((link, depth + 1))

            if self.delay:
                time.sleep(self.delay)

        log.info("Crawl finished: %d pages fetched, %d URLs seen", pages, len(self.seen))
