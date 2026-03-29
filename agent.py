# =========================
# CONFIG 🔥
# =========================
USE_LOCAL = False   # True → Advanced | False → Simple (deploy)

# =========================
# IMPORTS
# =========================
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = None

if "index" not in st.session_state:
    st.session_state.index = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

# =========================
# LLM
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
            model="gemini-2.5-flash",
            google_api_key=API_KEY,
            temperature=0.2,
            max_output_tokens=200
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
# LOCAL SEARCH
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

def rerank(query, docs, top_n=2):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Controls")

    mode = "🟢 Advanced (Local)" if USE_LOCAL else "☁️ Simple (Cloud)"
    st.markdown(f"### Mode: {mode}")

    if st.button("Reset Chat"):
        st.session_state.messages = []

    if st.button("Reset Knowledge Base"):
        st.session_state.documents = None
        st.session_state.index = None
        st.session_state.vector_store = None
        st.session_state.bm25 = None

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

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
                import faiss

                embeddings = embedding_model.encode(texts)
                dim = embeddings.shape[1]

                index = faiss.IndexFlatL2(dim)
                index.add(embeddings)

                st.session_state.index = index
                st.session_state.documents = docs

                tokenized = [t.lower().split() for t in texts]
                st.session_state.bm25 = BM25Okapi(tokenized)

            else:
                from langchain_community.embeddings import HuggingFaceEmbeddings

                hf_embed = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_store = Chroma.from_documents(
                    docs,
                    hf_embed
                )

        st.success("PDF Ready!")

# =========================
# UI
# =========================
st.title("🤖 AI PDF Assistant")

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================
# CHAT LOGIC
# =========================
if question := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if USE_LOCAL:
                if st.session_state.index is None:
                    answer = "Upload PDF first."
                else:
                    docs = hybrid_search(question)
                    docs = list({d.page_content: d for d in docs}.values())
                    docs = rerank(question, docs)

            else:
                if st.session_state.vector_store is None:
                    answer = "Upload PDF first."
                    docs = []
                else:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 2}
                    )
                    docs = retriever.invoke(question)

            if 'docs' in locals() and docs:
                context = "\n\n".join([doc.page_content[:200] for doc in docs])

                prompt = f"""
                Answer ONLY using the context.

                Context:
                {context}

                Question:
                {question}
                """

                # =========================
        # LLM CALL (FINAL FIX)
        # =========================
        st.info("🚀 Generating answer...")

        answer = "⚠️ Failed to generate response. Please try again."

        try:
            # 🔥 LIMIT PROMPT SIZE
            safe_prompt = prompt[:2000]

            response = llm.invoke(
                safe_prompt,
                config={"timeout": 15}  # 🔥 HARD TIMEOUT
            )

            if response and hasattr(response, "content"):
                answer = response.content
            else:
                answer = "⚠️ Empty response from AI."

        except Exception as e:
            answer = f"❌ Error: {str(e)}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})