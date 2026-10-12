"""Chain tests with a fake LLM and fake retriever - no database, no Ollama."""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever

from chat_with_website.rag.chain import RagChain, build_sources, format_context, trim_history
from chat_with_website.rag.prompts import NOT_ENOUGH_INFO, is_greeting

OUT_OF_SCOPE_ACME = (
    "Sorry, I couldn't find that in Acme Support's website content. "
    "I'm the Acme Support assistant - ask me anything about Acme Support and I'll answer from its website content."
)

DOCS = [
    Document(page_content="We offer web development and cloud consulting.", metadata={"url": "https://a.example/services", "title": "Services", "score": 0.81}),
    Document(page_content="Call us at 555-0100 or email hi@a.example.", metadata={"url": "https://a.example/contact", "title": "Contact", "score": 0.64}),
]


class FakeRetriever(BaseRetriever):
    docs: list = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None):
        return self.docs


def test_format_context_numbers_passages():
    ctx = format_context(DOCS)
    assert ctx.startswith("[1] Services (https://a.example/services)\n")
    assert "[2] Contact (https://a.example/contact)" in ctx


def test_build_sources_marks_cited():
    sources = build_sources(DOCS, "We do web dev [1] and you can call [2][1].")
    assert [s.cited for s in sources] == [True, True]
    sources = build_sources(DOCS, "Only the first one [1].")
    assert [s.cited for s in sources] == [True, False]
    assert sources[1].url == "https://a.example/contact"


def test_trim_history_keeps_last_turns():
    hist = [HumanMessage(content=str(i)) if i % 2 == 0 else AIMessage(content=str(i)) for i in range(20)]
    assert len(trim_history(hist, turns=3)) == 6
    assert trim_history(hist, turns=3)[0].content == "14"


def test_chain_without_history_skips_rewrite_and_cites():
    llm = FakeListChatModel(responses=["They offer web development and cloud consulting [1]."])
    chain = RagChain(FakeRetriever(docs=DOCS), llm)
    r = chain.invoke("What services do you provide?")
    assert r.standalone_question == "What services do you provide?"  # no rewrite without history
    assert r.grounded
    assert r.cited_sources[0].url == "https://a.example/services"


def test_chain_rewrites_followup_with_history():
    llm = FakeListChatModel(responses=["What is included in cloud consulting?", "Cloud consulting includes [1]."])
    chain = RagChain(FakeRetriever(docs=DOCS), llm)
    history = [HumanMessage(content="What services?"), AIMessage(content="Web dev and cloud consulting.")]
    r = chain.invoke("Tell me more about the second one.", history)
    assert r.standalone_question == "What is included in cloud consulting?"
    assert r.answer == "Cloud consulting includes [1]."


def test_chain_short_circuits_when_nothing_retrieved():
    llm = FakeListChatModel(responses=["SHOULD NOT BE CALLED"])
    chain = RagChain(FakeRetriever(docs=[]), llm, chatbot_name="Acme Support")
    r = chain.invoke("Who won the world cup?")
    assert r.answer == OUT_OF_SCOPE_ACME
    assert not r.grounded and r.sources == []


def test_chain_out_of_scope_without_name():
    chain = RagChain(FakeRetriever(docs=[]), FakeListChatModel(responses=["x"]))
    r = chain.invoke("Who won the world cup?")
    assert r.answer == "Sorry, I couldn't find that in the indexed website content. Ask me anything about this website and I'll answer from its content."


def test_is_greeting():
    for text in ["Hi", "hello!", "Hey there :)", "Good morning", "HII", "namaste", "hi team"]:
        assert is_greeting(text), text
    for text in ["Hi, what services do you offer?", "hello world program", "", "What is your refund policy?"]:
        assert not is_greeting(text), text


def test_chain_greets_with_chatbot_name_without_retrieval_or_llm():
    llm = FakeListChatModel(responses=["SHOULD NOT BE CALLED"])
    chain = RagChain(FakeRetriever(docs=DOCS), llm, chatbot_name="Acme Support")
    r = chain.invoke("Hi")
    assert r.answer == "Hi! I'm the Acme Support assistant. Ask me anything about Acme Support and I'll answer from its website content."
    assert r.sources == [] and not r.grounded


def test_chain_greets_without_name():
    chain = RagChain(FakeRetriever(docs=DOCS), FakeListChatModel(responses=["x"]))
    assert chain.invoke("hello").answer == "Hi! Ask me anything about this website and I'll answer from its content."


def test_chain_drops_sources_when_model_declines():
    llm = FakeListChatModel(responses=[NOT_ENOUGH_INFO + " The site does list services, though."])
    chain = RagChain(FakeRetriever(docs=DOCS), llm, chatbot_name="Acme Support")
    r = chain.invoke("What is your refund policy?")
    assert not r.grounded
    assert r.sources == []
    assert r.answer == OUT_OF_SCOPE_ACME  # model's sentinel text is replaced for the user
