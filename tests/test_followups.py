"""Follow-up suggestions: parsing/validation helpers and the chain step with a fake model."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from chat_with_website.config import settings
from chat_with_website.rag.chain import Prepared, RagAnswer, RagChain, Source
from chat_with_website.rag.followups import clean_followups, parse_followups


def test_parse_accepts_bare_fenced_or_prose_json_and_falls_back_to_lines():
    assert parse_followups('["A?", "B?"]') == ["A?", "B?"]
    assert parse_followups('Sure:\n```json\n["A?"]\n```') == ["A?"]
    assert parse_followups("- What is it?\n2. How much?\nnot a question") == ["What is it?", "How much?"]
    assert parse_followups("") == [] and parse_followups("[not json") == []
    assert parse_followups('[{"q": 1}]') == []  # wrong shape -> nothing, no exception


def test_clean_dedupes_normalises_and_drops_already_asked():
    raw = ["  What services do you offer ", "what services do you offer?", "", "x" * 130,
           "How much does it cost", "What is this website about?", "Where are you based?"]
    out = clean_followups(raw, asked=["What is this website about?"], limit=3)
    assert out == ["What services do you offer?", "How much does it cost?", "Where are you based?"]
    assert clean_followups([], asked=[]) == []
    assert clean_followups(["A?", "B?", "C?", "D?"], limit=2) == ["A?", "B?"]


def _chain(replies: list[str]) -> RagChain:
    llm = FakeListChatModel(responses=replies)
    return RagChain(retriever=None, llm=llm, chatbot_name="Acme")  # retriever unused here


def _prepared(question="What do you build?") -> Prepared:
    docs = [Document(page_content="We build websites and apps.", metadata={"title": "Home", "url": "https://a/"})]
    return Prepared(question=question, history=[], standalone=question, documents=docs)


def test_suggest_followups_uses_one_call_and_validates(monkeypatch):
    monkeypatch.setattr(settings, "followup_count", 3)
    chain = _chain(['["Do you build apps?", "do you build apps?", "What do you build?", "Where are you based?"]'])
    result = RagAnswer(answer="Websites [1].", sources=[Source(n=1, url="https://a/", title="Home", score=0.5, cited=True)])
    out = chain.suggest_followups(_prepared(), result, [HumanMessage(content="Hi"), AIMessage(content="Hello")])
    # duplicate dropped, the question just asked dropped, order kept
    assert out == ["Do you build apps?", "Where are you based?"]


def test_suggest_followups_skips_ungrounded_disabled_and_failures(monkeypatch):
    monkeypatch.setattr(settings, "followup_count", 3)
    grounded = RagAnswer(answer="x", sources=[], grounded=True)
    assert _chain(['["A?"]']).suggest_followups(Prepared(question="q", history=[], standalone="q", documents=[]), grounded) == []
    assert _chain(['["A?"]']).suggest_followups(_prepared(), RagAnswer(answer="x", grounded=False)) == []
    monkeypatch.setattr(settings, "followup_count", 0)
    assert _chain(['["A?"]']).suggest_followups(_prepared(), grounded) == []
    monkeypatch.setattr(settings, "followup_count", 3)
    assert _chain(["garbage with no questions"]).suggest_followups(_prepared(), RagAnswer(answer="x", grounded=True)) == []

    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("api down")

    chain = RagChain(retriever=None, llm=Boom(responses=["x"]))
    assert chain.suggest_followups(_prepared(), RagAnswer(answer="x", grounded=True)) == []
