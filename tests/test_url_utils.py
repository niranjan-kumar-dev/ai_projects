import pytest

from chat_with_website.ingestion.url_utils import (
    has_skipped_extension,
    is_same_domain,
    normalize_url,
    validate_start_url,
)


def test_normalize_strips_fragment_tracking_and_trailing_slash():
    url = "HTTPS://Example.com/About/?utm_source=x&b=2&a=1#team"
    assert normalize_url(url) == "https://example.com/About?a=1&b=2"


def test_normalize_resolves_relative_links():
    assert normalize_url("../contact", base="https://example.com/about/team/") == "https://example.com/about/contact"


def test_normalize_rejects_non_http():
    assert normalize_url("mailto:hi@example.com") is None
    assert normalize_url("javascript:void(0)") is None
    assert normalize_url("tel:+123") is None


def test_normalize_keeps_root_slash_and_drops_default_port():
    assert normalize_url("https://example.com:443/") == "https://example.com/"
    assert normalize_url("http://example.com:8080/x/") == "http://example.com:8080/x"


def test_same_domain_ignores_www():
    assert is_same_domain("https://www.example.com/a", "https://example.com")
    assert not is_same_domain("https://blog.example.com/a", "https://example.com")


def test_skipped_extensions():
    assert has_skipped_extension("https://example.com/file.PDF")
    assert not has_skipped_extension("https://example.com/page.html")


def test_validate_start_url_adds_scheme():
    assert validate_start_url("example.com/about") == "https://example.com/about"


def test_validate_start_url_blocks_private_hosts():
    with pytest.raises(ValueError):
        validate_start_url("http://localhost:8000")
    with pytest.raises(ValueError):
        validate_start_url("http://127.0.0.1/admin")
    assert validate_start_url("http://127.0.0.1/admin", allow_private=True) == "http://127.0.0.1/admin"
