# =========================
# CONFIG 🔥
# =========================
USE_LOCAL = False   # True → Ollama | False → Gemini

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

# ✅ NEW GEMINI SDK
from google import genai

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

# =========================
# SESSION STATE
# =========================
for key in ["messages", "documents", "index", "vector_store", "bm25"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# =========================
# LOCAL MODELS (UNCHANGED)
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
# GEMINI CLIENT (FIXED 🔥)
# =========================
if not USE_LOCAL:
    client = genai.Client(
        api_key=st.secrets.get("API_KEY") or os.getenv("API_KEY")
    )

# =========================
# LOCAL SEARCH (UNCHANGED)
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

    mode = "🟢 Advanced (Local)" if USE_LOCAL else "☁️ Simple (Gemini)"
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

        st.success("✅ PDF Ready!")

# =========================
# UI
# =========================
st.title("🤖 AI PDF Assistant")

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================
# CHAT
# =========================
if question := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # =========================
            # LOCAL (UNCHANGED)
            # =========================
            if USE_LOCAL:
                if st.session_state.index is None:
                    answer = "Upload PDF first."
                else:
                    docs = hybrid_search(question)
                    docs = list({d.page_content: d for d in docs}.values())
                    docs = rerank(question, docs)

                    context = "\n\n".join([doc.page_content[:200] for doc in docs])

                    prompt = f"""
Answer based only on the context.

Context:
{context}

Question:
{question}
"""

                    from langchain_community.chat_models import ChatOllama
                    llm = ChatOllama(model="llama3", temperature=0)

                    response = llm.invoke(prompt)
                    answer = response.content

            # =========================
            # CLOUD (FIXED GEMINI 🔥)
            # =========================
            else:
                if st.session_state.vector_store is None:
                    answer = "Upload PDF first."
                else:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 1}
                    )
                    docs = retriever.invoke(question)

                    context = "\n\n".join([doc.page_content[:150] for doc in docs])

                    prompt = f"""
You are a strict AI assistant.

Answer ONLY from the given context.
If answer is not present, say: "Not found in document."

Keep answer concise.

Context:
{context}

Question:
{question}
"""

                    try:
                        response = client.models.generate_content(
                            model="gemini-1.5-flash-001",
                            contents=prompt
                        )

                        answer = response.text if response.text else "⚠️ No response."

                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})