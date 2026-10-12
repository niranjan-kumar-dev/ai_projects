from chat_with_website.ingestion.extractor import extract_page, image_filename_words, remove_boilerplate
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


AUTHOR_HTML = """
<html><head><title>5 Tips</title></head><body><main>
<h1>5 Tips for Productive Meetings</h1>
<p>Meetings eat time. Here are five concrete tips that keep every meeting short, focused and useful for the team.</p>
<h4>Tags:</h4>
<div class="blog_post_author">
  <div class="blog_author_img_title"><img src="/blog//images/AuthorImage/Suyash.jpg" class="img-fluid" alt="Author image"></div>
  <div><h5>Suyash Shreekant</h5><p>Suyash, a co-founder of DictaAI, has been crafting creative solutions in the media world for years.</p></div>
</div>
<p>Some more text to make sure the main region is comfortably longer than the two hundred character threshold.</p>
</main></body></html>
"""


def test_author_photo_gets_nearby_text_and_filename():
    page = extract_page(_page(AUTHOR_HTML, url="https://www.dictaai.com/blog/5-tips"))
    photo = next(i for i in page.images if i.url.endswith("/Suyash.jpg"))
    assert photo.alt == "Author image"
    assert photo.filename == "Suyash"
    assert photo.nearby.startswith("Suyash Shreekant Suyash, a co-founder of DictaAI")
    assert len(photo.nearby) <= 160
    assert photo.context == "Tags:"
    assert photo.description.startswith("Image: Author image. Near: Suyash Shreekant")
    assert photo.description.endswith("Section: Tags:. File: Suyash")


def test_image_filename_words():
    assert image_filename_words("https://x/img/AuthorImage/Suyash.jpg") == "Suyash"
    assert image_filename_words("https://x/theme2/images/human-transcriptionists-1.svg") == "human transcriptionists"
    assert image_filename_words("https://x/i/5%20Tips%20for%20Meetings.png") == "Tips for Meetings"
    assert image_filename_words("https://x/i/logo.svg") == ""
    assert image_filename_words("https://x/i/a1b2c3d4e5f6.png") == ""
    assert image_filename_words("https://x/i/1x1.gif") == ""


CLIENTS_HTML = """
<html><head><title>Clients</title></head><body>
<header><a href="/"><img src="/img/logo.svg" alt="Acme Corp" class="site-logo"></a></header>
<main>
  <h1>Trusted by</h1>
  <p>Hundreds of companies rely on Acme every day for their mission-critical transcription work worldwide.</p>
  <div class="client-logos">
    <img src="/clients/duke-logo.png" alt="Duke University" width="200">
    <img src="/clients/nyu-logo.png" alt="NYU" width="200">
  </div>
  <p>More words so the main region is comfortably longer than the two hundred character threshold used.</p>
</main></body></html>
"""


def test_client_logo_carousel_is_content_not_site_logo():
    page = extract_page(_page(CLIENTS_HTML, url="https://acme.example/clients"))
    by_url = {i.url.rsplit("/", 1)[-1]: i for i in page.images}
    assert by_url["logo.svg"].context == "site logo"
    assert by_url["duke-logo.png"].context == "Trusted by" and by_url["duke-logo.png"].alt == "Duke University"
    assert by_url["nyu-logo.png"].context == "Trusted by"


def test_remove_boilerplate_keeps_repeated_image_once():
    pages = [
        extract_page(_page(CLIENTS_HTML.replace("Trusted by", f"Trusted by {i}"), url=f"https://acme.example/p{i}"))
        for i in range(5)
    ]
    cleaned = remove_boilerplate(pages, ratio=0.3)
    logo_pages = [p.url for p in cleaned if any(i.url.endswith("/logo.svg") for i in p.images)]
    assert logo_pages == ["https://acme.example/p0"]  # site logo indexed once, not five times
    assert all(any(i.url.endswith("/duke-logo.png") for i in p.images) for p in cleaned[:1])
    assert cleaned[1].content_hash != pages[1].content_hash  # image list changed -> re-index


def test_logo_does_not_pick_up_nav_text():
    page = extract_page(_page(IMG_HTML, url="https://acme.example/product"))
    logo = next(i for i in page.images if i.url.endswith("/logo.svg"))
    assert logo.nearby == "" and logo.context == "site logo"


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
    # the first page keeps every image; images repeated on all pages are indexed only there
    assert len(cleaned[0].images) == len(pages[0].images)
    all_urls = [i.url for p in cleaned for i in p.images]
    assert len(all_urls) == len(set(all_urls))


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
