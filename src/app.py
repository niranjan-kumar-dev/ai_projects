import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

# Providers:
#   "ollama" -> free, local (HuggingFace embeddings + Ollama chat) - for testing
#   "openai" -> paid OpenAI API (needs OPENAI_API_KEY with credits) - for future use
PROVIDERS = {
    "Ollama (free, testing)": "ollama",
    "OpenAI (paid)": "openai",
}
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(provider):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


def get_llm(provider):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI()

    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL)


def get_vectorstore_from_url(url, provider):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the document chunks
    vectorstore = Chroma.from_documents(documents_chunks, embedding=get_embeddings(provider))

    return vectorstore


def get_context_retriever_chain(vector_store, provider):
    llm = get_llm(provider)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    ])
    retriever_chain = create_history_aware_retriever(
        retriever=retriever,
        llm=llm,
        prompt=prompt)
    return retriever_chain


def get_conversation_rag_chain(retriever_chain, provider):
    llm = get_llm(provider)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt=prompt)
    retriever_aware_rag_chain = create_retrieval_chain(
        retriever=retriever_chain,
        combine_docs_chain=stuff_documents_chain
    )
    return retriever_aware_rag_chain


#app configuration
st.set_page_config(page_title="Chat with website", page_icon="🤖", layout="wide")
st.title("Chat with website")


#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    labels = list(PROVIDERS.keys())
    default_index = list(PROVIDERS.values()).index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in PROVIDERS.values() else 0
    provider = PROVIDERS[st.selectbox("Model provider", labels, index=default_index)]

if website_url is None or website_url == "":
    st.info("Please enter a website URL to start the conversation.")
else:
    #session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I assist you today?")
        ]

    # rebuild the vector store when the URL or provider changes
    # (different providers use different embedding sizes)
    if st.session_state.get("loaded_source") != (website_url, provider):
        with st.spinner("Loading website..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url, provider)
        st.session_state.loaded_source = (website_url, provider)

    #create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, provider)
    conversation_rag_chain = get_conversation_rag_chain(retriever_chain, provider)

    #user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = conversation_rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        })["answer"]
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    #Conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)
