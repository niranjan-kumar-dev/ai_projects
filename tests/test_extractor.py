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


IMG_HTML = """
<html><head><title>Acme - Product</title>
<meta property="og:image" content="/static/og.png"></head>
<body>
<header><a href="/"><img class="site-logo" src="/img/logo.svg" alt="Acme Corp"></a></header>
<main>
  <h1>Dashboard</h1>
  <p>Our dashboard shows every metric that matters for your team, updated in real time, with alerts.</p>
  <figure><img src="/shots/dashboard.png" alt="Dashboard overview" width="800"><figcaption>The main dashboard view</figcaption></figure>
  <h2>Pricing</h2>
  <img data-src="https://cdn.example/pricing.webp" alt="">
  <img src="/t/pixel.gif" width="1" height="1" alt="tracker">
  <img src="/icons/star.png" width="16" height="16" alt="star">
  <img src="data:image/png;base64,AAAA" alt="inline">
  <img srcset="/shots/team-480.jpg 480w, /shots/team-960.jpg 960w" alt="Our team">
  <p>More words so the main region is long enough to be chosen as the content root of this page.</p>
</main>
</body></html>
"""


def test_extract_images_collects_logo_og_and_content_images():
    page = extract_page(_page(IMG_HTML, url="https://acme.example/product"))
    by_url = {i.url: i for i in page.images}

    assert "https://acme.example/img/logo.svg" in by_url          # logo from <header>, despite header removal
    assert by_url["https://acme.example/img/logo.svg"].alt == "Acme Corp"
    assert by_url["https://acme.example/img/logo.svg"].context == "site logo"

    assert by_url["https://acme.example/static/og.png"].context == "page preview image"

    shot = by_url["https://acme.example/shots/dashboard.png"]      # relative URL resolved
    assert shot.alt == "Dashboard overview" and shot.caption == "The main dashboard view"
    assert shot.context == "Dashboard"

    assert by_url["https://cdn.example/pricing.webp"].context == "Pricing"   # lazy data-src, heading context
    assert "https://acme.example/shots/team-480.jpg" in by_url               # first srcset candidate

    assert not any("pixel" in u or "star.png" in u or u.startswith("data:") for u in by_url)
    assert "Dashboard overview" not in page.text  # alt text is not leaked into the page text


def test_content_hash_changes_when_images_change():
    a = extract_page(_page(IMG_HTML, url="https://acme.example/product"))
    b = extract_page(_page(IMG_HTML.replace("dashboard.png", "dashboard-v2.png"), url="https://acme.example/product"))
    assert a.text == b.text and a.content_hash != b.content_hash


def test_remove_boilerplate_keeps_images():
    pages = [
        extract_page(_page(IMG_HTML.replace("Dashboard", f"Dashboard {i}"), url=f"https://acme.example/p{i}"))
        for i in range(4)
    ]
    cleaned = remove_boilerplate(pages, ratio=0.3)
    assert all(len(p.images) == len(pages[0].images) for p in cleaned)


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
