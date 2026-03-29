# =========================
# CONFIG 🔥
# =========================
USE_LOCAL = False   # True → Ollama | False → Groq

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

from groq import Groq

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
# LOCAL MODELS
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
# GROQ CLIENT 🔥
# =========================
if not USE_LOCAL:
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def query_llm(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert resume analyst.

Guidelines:
- Focus on work experience and timeline
- Combine multiple sections if needed
- Infer logically when year is implicit
- NEVER ignore job roles if present
- Be concise and confident
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"❌ Groq Error: {str(e)}"

# =========================
# SIDEBAR (PDF UPLOAD)
# =========================
with st.sidebar:
    st.title("⚙️ Controls")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(documents)[:40]

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
st.title("🤖 AI Resume Assistant")

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================
# CHAT
# =========================
if question := st.chat_input("Ask about resume..."):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.vector_store is None:
                answer = "Upload PDF first."
            else:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 10}
                )

                # 🔥 QUERY STEERING
                enhanced_query = f"""
{question}

Focus on:
- work experience
- jobs
- roles
- employment timeline
- years like 2023, 2024, 2025
"""

                docs = retriever.invoke(enhanced_query)

                # 🔥 FILTER RELEVANT CHUNKS
                filtered_docs = []
                for doc in docs:
                    text = doc.page_content.lower()

                    if any(keyword in text for keyword in [
                        "experience", "developer", "engineer",
                        "worked", "training", "2023", "2024", "2025"
                    ]):
                        filtered_docs.append(doc)

                if filtered_docs:
                    docs = filtered_docs

                # 🔥 MERGE CONTEXT
                unique_docs = list({doc.page_content: doc for doc in docs}.values())
                context = "\n\n".join([doc.page_content for doc in unique_docs])

                # 🔥 FINAL PROMPT
                prompt = f"""
Analyze the resume context carefully.

- Extract experience based on timeline
- Combine multiple roles if needed
- Answer clearly

Context:
{context}

Question:
{question}
"""

                answer = query_llm(prompt)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})