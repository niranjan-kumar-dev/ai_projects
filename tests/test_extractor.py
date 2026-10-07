from chat_with_website.ingestion.extractor import extract_page, remove_boilerplate
from chat_with_website.ingestion.models import CrawledPage

HTML = """
<html><head><title>Acme Corp - Services</title>
<meta property="og:title" content="Our Services | Acme">
<script>alert('x')</script><style>p{color:red}</style></head>
<body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<div class="cookie-banner">We use cookies</div>
<main>
  <h1>Our Services</h1>
  <p>We provide web development and cloud consulting for small businesses in Hyderabad and beyond.</p>
  <p>Our second service is managed hosting with 24/7 monitoring and daily backups included for every plan.</p>
  <ul><li>Web development</li><li>Cloud consulting</li></ul>
</main>
<footer>© 2026 Acme Corp. All rights reserved.</footer>
</body></html>
"""


def _page(html: str, url: str = "https://acme.example/services") -> CrawledPage:
    return CrawledPage(url=url, html=html, status_code=200, depth=0)


def test_extract_removes_noise_and_keeps_main_content():
    page = extract_page(_page(HTML))
    assert page.title == "Acme Corp - Services"
    assert "alert" not in page.text
    assert "cookies" not in page.text
    assert "All rights reserved" not in page.text
    assert "Home" not in page.text.split("\n")[0]
    assert "web development and cloud consulting" in page.text
    assert "managed hosting" in page.text
    assert page.word_count > 20
    assert len(page.content_hash) == 64


def test_extract_keeps_paragraph_boundaries():
    page = extract_page(_page(HTML))
    assert "\n" in page.text
    assert "Our Services" in page.lines[0]


def test_remove_boilerplate_drops_repeated_lines():
    repeated = "<p>Call us today on 555-0100 for a free quote!</p>"
    pages = [
        extract_page(_page(f"<html><body><main><p>Page {i} unique content about topic {i} with enough words to count.</p>{repeated}</main></body></html>", url=f"https://x.example/p{i}"))
        for i in range(5)
    ]
    cleaned = remove_boilerplate(pages, ratio=0.3)
    for p in cleaned:
        assert "free quote" not in p.text
        assert "unique content" in p.text
