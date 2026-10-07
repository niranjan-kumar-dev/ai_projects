from chat_with_website.ingestion.chunker import build_splitter, chunk_page
from chat_with_website.ingestion.extractor import content_hash
from chat_with_website.ingestion.models import PageContent


def _page(text: str) -> PageContent:
    return PageContent(url="https://x.example/a", title="Title A", text=text, content_hash=content_hash(text))


def test_chunks_respect_size_and_carry_metadata():
    text = "\n\n".join(f"Paragraph {i}. " + ("Lorem ipsum dolor sit amet. " * 12) for i in range(10))
    chunks = chunk_page(_page(text), website_id=7, page_id=42, splitter=build_splitter(chunk_size=400, chunk_overlap=50))
    assert len(chunks) > 3
    for i, c in enumerate(chunks):
        assert len(c.content) <= 400
        assert c.chunk_index == i
        assert c.metadata["chunk_id"] == f"42:{i}"
        assert c.metadata["website_id"] == 7
        assert c.metadata["page_id"] == 42
        assert c.metadata["url"] == "https://x.example/a"
        assert c.metadata["title"] == "Title A"
        assert c.metadata["source"] == "web"
        assert c.metadata["chunk_count"] == len(chunks)


def test_embed_text_prepends_title():
    chunks = chunk_page(_page("Some content that is long enough to survive the minimum size filter here."), website_id=1, page_id=1)
    assert chunks[0].embed_text.startswith("Title A\n\n")


def test_overlap_shares_text_between_neighbours():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_page(_page(text), website_id=1, page_id=1, splitter=build_splitter(chunk_size=200, chunk_overlap=60))
    first_tail = chunks[0].content.split()[-3:]
    assert all(w in chunks[1].content for w in first_tail)
