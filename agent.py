# =========================
# CONFIG 🔥
# =========================
USE_LOCAL = True   # True → Local (Ollama + FAISS)
                  # False → Cloud (Gemini + Chroma)

# =========================
# IMPORTS
# =========================
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# conditional import
if USE_LOCAL:
    import faiss
else:
    from langchain_community.vectorstores import Chroma

load_dotenv()

# =========================
# PAGE CONFIG 💎
# =========================
st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

# =========================
# CSS 💎
# =========================
st.markdown("""
<style>
.main { background-color: #0E1117; }
.stChatMessage { border-radius: 12px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
for key in ["messages", "documents", "bm25", "index", "vector_store"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# =========================
# LLM LOADER
# =========================
@st.cache_resource
def load_llm():
    if USE_LOCAL:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3", temperature=0)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        API_KEY = st.secrets.get("API_KEY") or os.getenv("API_KEY")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=API_KEY
        )

llm = load_llm()

# =========================
# MODELS (LOCAL ONLY)
# =========================
@st.cache_resource
def load_embedding():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedding_model = load_embedding()
reranker = load_reranker()

# =========================
# SEARCH FUNCTIONS
# =========================
def vector_search(query, k=5):
    query_vec = embedding_model.encode([query])
    _, indices = st.session_state.index.search(query_vec, k)
    return [st.session_state.documents[i] for i in indices[0]]

def hybrid_search(query):
    vector_docs = vector_search(query, k=5)

    tokenized = query.lower().split()
    scores = st.session_state.bm25.get_scores(tokenized)
    top_idx = np.argsort(scores)[-5:]

    bm25_docs = [st.session_state.documents[i] for i in top_idx]

    return vector_docs + bm25_docs

def rerank(query, docs, top_n=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# =========================
# SIDEBAR 💎
# =========================
with st.sidebar:
    st.title("⚙️ Controls")

    mode = "🟢 Local (Ollama + FAISS)" if USE_LOCAL else "☁️ Cloud (Gemini + Chroma)"
    st.markdown(f"### Mode: {mode}")

    if st.button("🧹 Reset Chat"):
        st.session_state.messages = []

    if st.button("🗑 Reset Knowledge Base"):
        st.session_state.documents = None
        st.session_state.index = None
        st.session_state.vector_store = None
        st.session_state.bm25 = None

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150
            )
            docs = splitter.split_documents(documents)[:15]

            texts = [doc.page_content for doc in docs]

            if USE_LOCAL:
                embeddings = embedding_model.encode(texts)

                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embeddings)

                st.session_state.index = index
                st.session_state.documents = docs

            else:
                st.session_state.vector_store = Chroma.from_documents(
                    docs,
                    embedding_model
                )

            tokenized = [t.lower().split() for t in texts]
            st.session_state.bm25 = BM25Okapi(tokenized)

        st.success("✅ PDF Ready!")

# =========================
# MAIN UI 💎
# =========================
st.title("🤖 AI PDF Assistant")
st.caption("Ask anything from your document 🚀")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================
# CHAT
# =========================
if question := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):

            if USE_LOCAL and st.session_state.index is None:
                answer = "⚠️ Upload PDF first."

            elif not USE_LOCAL and st.session_state.vector_store is None:
                answer = "⚠️ Upload PDF first."

            else:
                if USE_LOCAL:
                    docs = hybrid_search(question)
                else:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    docs = retriever.invoke(question)

                docs = list({doc.page_content: doc for doc in docs}.values())
                docs = rerank(question, docs, top_n=3)

                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = f"""
                Answer ONLY using the context below.

                If not found, say "Not found in document."

                Context:
                {context}

                Question:
                {question}
                """

                response = llm.invoke(prompt)
                answer = response.content

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})