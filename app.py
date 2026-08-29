"""
Streamlit RAG Chat Application
- Upload a PDF book
- Build a vector store from the PDF
- Chat with the document using Mistral AI
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ── Configuration ───────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"

# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 RAG Chat – Talk to your Book",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Chat – Talk to your Book")
st.caption("Upload a PDF, build a knowledge base, and ask questions about its content.")


# ── Helper Functions ────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings_model():
    return MistralAIEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource
def get_llm():
    return ChatMistralAI(model=CHAT_MODEL)


def build_vector_store(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            st.error("The PDF appears to be empty or could not be read.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        embeddings_model = get_embeddings_model()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=CHROMA_DIR,
        )
        return vector_store

    finally:
        os.unlink(tmp_path)


def load_existing_vector_store():
    if os.path.exists(CHROMA_DIR):
        embeddings_model = get_embeddings_model()
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings_model,
        )
    return None


def get_answer(vector_store, question: str) -> str:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": question})
    llm = get_llm()
    response = llm.invoke(final_prompt)
    return response.content


# ── Session State Initialization ────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_existing_vector_store()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "book_name" not in st.session_state:
    st.session_state.book_name = None

# ── Sidebar – PDF Upload ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload your Book")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF book to build the knowledge base.",
    )

    if uploaded_file is not None:
        if st.button("🔨 Build Knowledge Base", use_container_width=True):
            with st.spinner("Reading PDF, chunking, and creating embeddings… This may take a minute."):
                vector_store = build_vector_store(uploaded_file)

            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.book_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"✅ Knowledge base built from **{uploaded_file.name}**!")

    st.divider()

    if st.session_state.vector_store is not None:
        st.success("🟢 Knowledge base is ready!")
        if st.session_state.book_name:
            st.info(f"📖 Current book: **{st.session_state.book_name}**")
    else:
        st.warning("🟡 No knowledge base loaded. Upload a PDF to get started.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Chat Area ──────────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your book…"):
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF and build the knowledge base first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = get_answer(st.session_state.vector_store, question)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})