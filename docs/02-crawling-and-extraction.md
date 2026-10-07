# Step 2 – Crawling and extraction

## What we are building

`ingest.py <url>` must turn a website into clean text, one entry per page.
That is two jobs:

1. **Crawl** – discover and download the site's pages.
2. **Extract** – keep the readable content, throw away everything else.

Garbage in, garbage out: every menu item we fail to remove becomes a chunk that can
be retrieved instead of the real answer.

## Crawler (`ingestion/crawler.py`)

A **breadth-first search** over same-domain links:

```
queue = [start_url] (+ URLs from sitemap.xml if available)
while queue and pages < max_pages:
    url = queue.pop_front()
    skip if seen / disallowed by robots.txt / not HTML
    fetch; yield page
    push every same-domain <a href> link with depth+1
```

Good-citizen rules built in:

| Rule | Why |
|---|---|
| Obey `robots.txt` (`urllib.robotparser`) | Site owners say what may be crawled. |
| `CRAWL_DELAY_SECONDS` between requests | Don't hammer small servers. |
| `max_pages` (50) and `max_depth` (3) | Bound time and cost; configurable per website. |
| Identify yourself with `USER_AGENT` | Lets site owners contact you; many sites block blank agents. |
| Only `text/html` responses, body ≤ 5 MB | PDFs/images are skipped by extension *and* by Content-Type. |
| Retries with back-off on 429/5xx | Transient errors shouldn't kill a crawl. |

**Sitemap first.** If `/sitemap.xml` (or one listed in robots.txt) exists, its URLs seed the
queue, so deep pages are found even when they are not linked from the home page.

### URL normalisation (`url_utils.py`)
`/About/`, `/about?utm_source=x` and `/about#team` are one page. `normalize_url` lower-cases
the host, drops fragments and tracking parameters, strips trailing slashes and sorts the
query string, so the `seen` set catches duplicates.

### Security: SSRF
If this app runs on a server, a user could type `http://localhost:5432` or an internal IP
and make *your server* fetch internal services. `validate_start_url` resolves the host and
refuses private, loopback and link-local addresses (`CRAWL_ALLOW_PRIVATE_HOSTS=true` only
for local testing).

## Extractor (`ingestion/extractor.py`)

Per page:
1. Parse with BeautifulSoup + lxml.
2. `decompose()` noise tags (`script style nav header footer aside form iframe svg ...`)
   and selectors like `[role=navigation]`, `[class*=cookie]`, `[class*=sidebar]`.
3. Pick the main region: first of `main`, `[role=main]`, `article`, `#content` … with
   more than 200 characters; else `<body>`.
4. Insert newlines around block tags so paragraphs survive `get_text()`, then collapse whitespace.
5. Title = `<title>` → `og:title` → first `<h1>` → URL.
6. `content_hash = sha256(text)` for change detection later.

Across pages (`remove_boilerplate`): any line that appears on ≥ 30 % of the crawled pages
(minimum 3) is boilerplate – "Subscribe to our newsletter", a phone number in a footer div the
tag-based rules missed – and is removed from all pages. Pages left with fewer than
`MIN_PAGE_WORDS` (50) words are dropped.

> **Why not `WebBaseLoader`?** LangChain's loader is fine for one URL but gives no control over
> link following, robots.txt, content-type checks or boilerplate. The ~150 lines here are the
> part of a RAG system that most decides answer quality, so we own them.

> **JavaScript-rendered sites** (React/Next apps that ship an empty `<body>`) come back with
> almost no text. Stage 7 discusses adding Playwright for those; most marketing/company sites
> are server-rendered and work as-is.

## Test this step

```powershell
pytest tests/test_url_utils.py tests/test_extractor.py -q
python scripts/ingest.py https://www.python.org/about/ --dry-run --max-pages 6
```
`--dry-run` crawls and cleans but never touches the database. For every page you should see
the URL, title, word count and the first 200 characters of clean text. Read them and check
that menus and footers are gone. Add `-v` to see which URLs were skipped and why.

Try your own company website the same way. If a page you care about is missing, check
`robots.txt`, whether it is linked within 3 clicks, and whether its text is rendered by JavaScript.

Next: [Step 3 – Chunking](03-chunking.md)
