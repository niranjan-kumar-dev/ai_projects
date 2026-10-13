"""Branding helpers: logo validation, colours, text limits, starter questions, defaults, CSS."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from chat_with_website.ui.branding import (
    DEFAULT_DESCRIPTION,
    DEFAULT_STARTERS,
    MAX_LOGO_BYTES,
    accent_css,
    branding_of,
    clean_text,
    contrast_ratio,
    parse_starters,
    process_logo,
    starter_text,
    text_on,
    validate_branding,
    validate_color,
)


def _image(fmt: str, size=(800, 400), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (200, 30, 30)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_process_logo_accepts_png_jpeg_webp_and_normalises_to_small_png(fmt):
    out = process_logo(_image(fmt))
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.mode == "RGBA"
    assert max(img.size) == 256 and img.size == (256, 128)  # aspect ratio kept


def test_process_logo_rejects_by_content_not_name():
    with pytest.raises(ValueError, match="not a valid image"):
        process_logo(b"hello, I am a text file renamed to logo.png")
    with pytest.raises(ValueError, match="Unsupported image type"):
        process_logo(_image("GIF", mode="P"))
    with pytest.raises(ValueError, match="smaller than"):
        process_logo(b"\x89PNG" + b"0" * MAX_LOGO_BYTES)
    with pytest.raises(ValueError, match="empty"):
        process_logo(b"")


def test_colour_validation_and_contrast():
    assert validate_color("#ABC") == "#aabbcc" and validate_color(" 2563eb ") == "#2563eb"
    for bad in ("red", "#12345", "#ggg", "", None):
        with pytest.raises(ValueError):
            validate_color(bad)
    assert text_on("#ffffff") == "#000000" and text_on("#1d4ed8") == "#ffffff"
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#212121", "#2a2a2a") < 1.5


def test_text_and_starter_rules():
    assert clean_text("  Hello   world  ", 5) == "Hello" and clean_text("   ", 10) is None
    assert parse_starters("What services?\n\nwhat services?\nPricing") == ["What services?", "Pricing?"]
    with pytest.raises(ValueError, match="At most 4"):
        parse_starters("a?\nb?\nc?\nd?\ne?")
    with pytest.raises(ValueError, match="120 characters"):
        parse_starters("x" * 121)
    assert parse_starters("") == []
    assert starter_text(":material/help: What is this?") == "What is this?" and starter_text("Plain?") == "Plain?"


def test_validate_branding_cleans_everything():
    out = validate_branding({"display_name": " Acme  Bot ", "welcome_heading": "", "description": "d",
                             "accent_color": "#FFF", "starter_questions": "One?\nTwo", "input_placeholder": None})
    assert out == {"display_name": "Acme Bot", "welcome_heading": None, "description": "d",
                   "input_placeholder": None, "accent_color": "#ffffff", "starter_questions": ["One?", "Two?"]}
    assert validate_branding({"accent_color": "  "})["accent_color"] is None


def test_branding_defaults_for_an_unbranded_row_and_resolved_values():
    b = branding_of({"id": 1, "name": "acme support"})
    assert (b.display_name, b.heading, b.description) == ("Acme Support", "Ask anything about Acme Support", DEFAULT_DESCRIPTION)
    assert b.accent is None and b.starters == DEFAULT_STARTERS and b.placeholder == "Message Acme Support…"
    assert b.has_logo is False and b.logo_version == ""
    b2 = branding_of({"id": 1, "name": "acme", "display_name": "Acme Helper", "welcome_heading": "Hello!", "description": "We help.",
                      "accent_color": "#00FF00", "starter_questions": ["A?", "", "B?"], "input_placeholder": "Type…",
                      "logo_updated_at": "2026-10-13", "has_logo": True})
    assert (b2.display_name, b2.heading, b2.description, b2.accent) == ("Acme Helper", "Hello!", "We help.", "#00ff00")
    assert b2.starters == ["A?", "B?"] and b2.placeholder == "Type…" and b2.has_logo and b2.logo_version == "2026-10-13"
    assert branding_of({"name": "x", "accent_color": "not-a-colour"}).accent is None  # bad stored value -> default


def test_accent_css_contains_only_the_colour_and_known_hooks():
    css = accent_css("#2563EB")
    assert css.startswith("<style>") and "#2563eb" in css and "stBaseButton-primary" in css
    assert "<script" not in css and "{" in css
    with pytest.raises(ValueError):
        accent_css("javascript:alert(1)")
