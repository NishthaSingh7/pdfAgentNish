# =========================
# IMPORTS
# =========================
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Resume Assistant", page_icon="🤖", layout="wide")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Controls")

    USE_LOCAL = st.toggle("🟢 Use Local LLM (Ollama)", value=False)

    mode = "🟢 Local (Ollama)" if USE_LOCAL else "☁️ Cloud (Groq)"
    st.markdown(f"### Mode: {mode}")

    st.divider()

    uploaded_file = st.file_uploader("📄 Upload Resume PDF", type="pdf")

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

            docs = splitter.split_documents(documents)

            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vector_store = Chroma.from_documents(docs, embedder)

        st.success("✅ Resume Loaded!")

# =========================
# GROQ SETUP
# =========================
if not USE_LOCAL:
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def query_llm(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert resume analyst.

- Focus on experience and timeline
- Combine multiple roles if needed
- Answer clearly
"""
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Groq Error: {str(e)}"

# =========================
# UI HEADER
# =========================
st.title("🤖 AI Resume Assistant")
st.caption("Ask anything about the uploaded resume — powered by RAG + LLM")

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
if question := st.chat_input("Ask about resume..."):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.vector_store is None:
                answer = "⚠️ Please upload a PDF first."
                sources = []
            else:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 10}
                )

                # 🔥 Query Enhancement
                enhanced_query = f"""
{question}

Focus on:
- work experience
- jobs
- roles
- timeline
- years like 2023, 2024, 2025
"""

                docs = retriever.invoke(enhanced_query)

                # 🔥 Filter Relevant Chunks
                filtered_docs = []
                for doc in docs:
                    text = doc.page_content.lower()
                    if any(word in text for word in [
                        "experience", "developer", "engineer",
                        "worked", "training", "2023", "2024", "2025"
                    ]):
                        filtered_docs.append(doc)

                if filtered_docs:
                    docs = filtered_docs

                # 🔥 Context
                unique_docs = list({d.page_content: d for d in docs}.values())
                context = "\n\n".join([d.page_content for d in unique_docs])

                # 🔥 Prompt
                prompt = f"""
Analyze resume and answer clearly.

Context:
{context}

Question:
{question}
"""

                # =========================
                # LLM CALL
                # =========================
                if USE_LOCAL:
                    from langchain_community.chat_models import ChatOllama

                    llm = ChatOllama(model="llama3", temperature=0)
                    response = llm.invoke(prompt)
                    answer = response.content
                else:
                    answer = query_llm(prompt)

                # 🔥 CITATIONS
                sources = unique_docs[:3]

        st.markdown(answer)

        # =========================
        # SHOW CITATIONS
        # =========================
        if sources:
            with st.expander("📄 Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})