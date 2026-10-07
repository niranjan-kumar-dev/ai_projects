"""URL helpers: normalisation, same-domain checks and SSRF protection."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

# File types that are never HTML pages worth indexing.
SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".gz", ".tar",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tiff",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".webm", ".wav",
    ".css", ".js", ".json", ".xml", ".rss", ".atom", ".woff", ".woff2", ".ttf", ".eot",
    ".exe", ".dmg", ".apk", ".msi",
}

# Query parameters that only track visitors and do not change page content.
TRACKING_PARAMS_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "_ga")


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Return a canonical form of `url` so the same page is only crawled once.

    - resolves relative links against `base`
    - lower-cases scheme and host
    - removes the fragment (#section) and tracking query params
    - removes default ports and trailing slashes (except for the root path)
    Returns None for non-http(s) links (mailto:, tel:, javascript:, ...).
    """
    if base:
        url = urljoin(base, url.strip())
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None

    host = parts.hostname.lower() if parts.hostname else ""
    port = parts.port
    if port and not ((parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(TRACKING_PARAMS_PREFIXES)
    ]
    query.sort()
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))


def registrable_host(url: str) -> str:
    """Host without a leading 'www.' so www.example.com and example.com match."""
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_same_domain(url: str, root_url: str) -> bool:
    return registrable_host(url) == registrable_host(root_url) and registrable_host(url) != ""


def has_skipped_extension(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def is_private_host(hostname: str) -> bool:
    """True if the host resolves to a loopback / private / link-local address.

    Crawling such hosts from a server would let a user reach internal services
    (SSRF).  We refuse them unless CRAWL_ALLOW_PRIVATE_HOSTS is set.
    """
    if hostname in ("localhost",) or hostname.endswith(".local"):
        return True
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # unresolvable -> treat as unsafe
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def validate_start_url(url: str, allow_private: bool = False) -> str:
    """Validate a user-supplied URL and return its normalised form, or raise ValueError."""
    if not url or not url.strip():
        raise ValueError("URL is empty")
    candidate = url.strip()
    if "://" not in candidate:
        candidate = "https://" + candidate
    normalized = normalize_url(candidate)
    if normalized is None:
        raise ValueError(f"Only http(s) URLs are supported: {url!r}")
    host = urlsplit(normalized).hostname or ""
    if not allow_private and is_private_host(host):
        raise ValueError(f"Refusing to crawl private/local host: {host}")
    return normalized
