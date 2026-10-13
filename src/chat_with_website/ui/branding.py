"""Per-chatbot branding: validation, defaults and the small bits of CSS the Chat page injects.

Pure Python (Pillow for images, no Streamlit), so every rule is unit-tested. The Chatbots page
calls the validators before writing; the Chat page calls `branding_of` every run to resolve a
chatbot row into what to show, so switching chatbots can never leak another bot's look.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

from chat_with_website.rag.followups import clean_followups

MAX_LOGO_BYTES = 2 * 1024 * 1024
LOGO_SIZE = 256  # logos are normalised to at most 256x256 PNG
ALLOWED_LOGO_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_DISPLAY_NAME = 60
MAX_HEADING = 120
MAX_DESCRIPTION = 300
MAX_PLACEHOLDER = 80
MAX_STARTERS = 4
MAX_STARTER_CHARS = 120

DEFAULT_DESCRIPTION = "Answers come only from the website content indexed for this chatbot."
DEFAULT_STARTERS = [
    ":material/help: What is this website about?",
    ":material/image: Show me the logo",
    ":material/mail: How can I contact you?",
]
DARK_BACKGROUND = "#212121"  # [theme.dark] backgroundColor in .streamlit/config.toml

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")
_LIMITS = {
    "display_name": MAX_DISPLAY_NAME,
    "welcome_heading": MAX_HEADING,
    "description": MAX_DESCRIPTION,
    "input_placeholder": MAX_PLACEHOLDER,
}


# ------------------------------------------------------------------ validation
def process_logo(data: bytes) -> bytes:
    """Validate an uploaded image by its *content* and normalise it to a small PNG.

    Accepts PNG, JPEG and WebP up to MAX_LOGO_BYTES; anything else (including a text file
    renamed to .png) raises ValueError with a message for the user. Re-encoding as PNG drops
    metadata and keeps the stored logo small (fits an avatar / a 72 px welcome image).
    """
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_LOGO_BYTES:
        raise ValueError(f"The logo must be smaller than {MAX_LOGO_BYTES // (1024 * 1024)} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or "").upper()
        probe.verify()  # checks the file structure without decoding everything
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("That file is not a valid image. Upload a PNG, JPG or WebP file.") from exc
    if fmt not in ALLOWED_LOGO_FORMATS:
        raise ValueError(f"Unsupported image type {fmt or 'unknown'}. Upload a PNG, JPG or WebP file.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGBA")
        img.thumbnail((LOGO_SIZE, LOGO_SIZE))
    except (OSError, ValueError) as exc:
        raise ValueError("The image could not be decoded. Try another file.") from exc
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def validate_color(value: str | None) -> str:
    """'#ABC' / '#aabbcc' -> '#aabbcc'; anything else raises ValueError."""
    m = _HEX_RE.match((value or "").strip())
    if not m:
        raise ValueError("The accent colour must be a hex value like #2563eb.")
    h = m.group(1).lower()
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"#{h}"


def _luminance(color: str) -> float:
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two '#rrggbb' colours (1..21)."""
    la, lb = _luminance(validate_color(a)), _luminance(validate_color(b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def text_on(color: str) -> str:
    """Black or white, whichever reads better on `color`."""
    return "#000000" if contrast_ratio(color, "#000000") >= contrast_ratio(color, "#ffffff") else "#ffffff"


def clean_text(value: str | None, limit: int) -> str | None:
    """Collapse whitespace, trim, cut to `limit`; empty -> None (= use the default)."""
    text = " ".join((value or "").split()).strip()
    return text[:limit] if text else None


def parse_starters(text: str | None) -> list[str]:
    """One question per line (the order is the display order); max 4, each <= 120 chars."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    too_long = [ln for ln in lines if len(ln) > MAX_STARTER_CHARS]
    if too_long:
        raise ValueError(f"Starter questions must be at most {MAX_STARTER_CHARS} characters each.")
    if len(lines) > MAX_STARTERS:
        raise ValueError(f"At most {MAX_STARTERS} starter questions are allowed.")
    return clean_followups(lines, limit=MAX_STARTERS)


def validate_branding(fields: dict) -> dict:
    """Clean every text field, validate the colour, parse the starters; raises ValueError."""
    out = {k: clean_text(fields.get(k), lim) for k, lim in _LIMITS.items()}
    out["accent_color"] = validate_color(fields["accent_color"]) if (fields.get("accent_color") or "").strip() else None
    out["starter_questions"] = parse_starters(fields.get("starter_questions")) or None
    return out


# -------------------------------------------------------------------- resolve
@dataclass
class Branding:
    display_name: str
    heading: str
    description: str
    accent: str | None  # None = the theme's primary colour
    starters: list[str] = field(default_factory=list)
    placeholder: str = ""
    has_logo: bool = False
    logo_version: str = ""  # logo_updated_at as text; part of the logo cache key


def branding_of(bot: dict) -> Branding:
    """Resolve a chatbot row (with or without branding columns) into what the Chat page shows."""
    display = (bot.get("display_name") or "").strip() or str(bot.get("name") or "").title()
    accent = None
    if bot.get("accent_color"):
        try:
            accent = validate_color(bot["accent_color"])
        except ValueError:
            accent = None
    starters = [s for s in (bot.get("starter_questions") or []) if isinstance(s, str) and s.strip()]
    stamp = bot.get("logo_updated_at")
    return Branding(
        display_name=display,
        heading=(bot.get("welcome_heading") or "").strip() or f"Ask anything about {display}",
        description=(bot.get("description") or "").strip() or DEFAULT_DESCRIPTION,
        accent=accent,
        starters=starters[:MAX_STARTERS] or list(DEFAULT_STARTERS),
        placeholder=(bot.get("input_placeholder") or "").strip() or f"Message {display}…",
        has_logo=bool(stamp) and bool(bot.get("has_logo", True)),
        logo_version=str(stamp) if stamp else "",
    )


def starter_text(starter: str) -> str:
    """Drop the ':material/…:' icon prefix the default chips carry."""
    return re.sub(r"^:material/[a-z_]+:\s*", "", starter)


def accent_css(accent: str) -> str:
    """CSS that applies the (validated) accent colour to a curated set of elements.

    Only `data-testid` hooks are used and only a colour value ever lands in the stylesheet,
    so user-provided text can never reach it.
    """
    accent = validate_color(accent)
    fg = text_on(accent)
    # `!important`: Streamlit's own (emotion) rules have the same specificity and may be injected
    # after this stylesheet; without it the theme colour wins at random. The rules are scoped to
    # data-testid hooks and the Chat page emits them only for the selected chatbot.
    return (
        "<style>"
        # New chat / Save / primary buttons
        f"[data-testid='stBaseButton-primary']{{background-color:{accent}!important;border-color:{accent}!important;color:{fg}!important}}"
        "[data-testid='stBaseButton-primary']:hover{filter:brightness(1.12)}"
        # starter chips + thumbs (st.pills / st.feedback are button groups)
        f"[data-testid='stBaseButton-pills']{{border-color:{accent}!important;color:{accent}!important}}"
        f"[data-testid='stBaseButton-pillsActive']{{background-color:{accent}!important;border-color:{accent}!important;color:{fg}!important}}"
        # the open conversation in the sidebar
        f"[data-testid='stSidebar'] [data-testid='stBaseButton-secondary']{{border-color:{accent}!important}}"
        # chat input focus ring + send button
        f"[data-testid='stChatInput']:focus-within{{border-color:{accent}!important;box-shadow:0 0 0 1px {accent}!important}}"
        f"[data-testid='stChatInputSubmitButton'] button{{color:{accent}!important}}"
        # links (citations, sources) and keyboard focus outlines
        f"[data-testid='stChatMessage'] a{{color:{accent}!important}}"
        f"button:focus-visible{{outline-color:{accent}!important}}"
        "</style>"
    )
